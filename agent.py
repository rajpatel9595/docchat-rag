"""
Agentic RAG
-----------
An LLM agent that reasons about what to search, searches multiple times
with different strategies, evaluates results, and synthesizes a final answer.

Instead of a single retrieve→answer pass, the agent:
  1. Breaks complex questions into sub-questions
  2. Picks the right search strategy (semantic vs keyword vs page lookup)
  3. Searches multiple times if needed
  4. Evaluates whether it has enough context
  5. Synthesizes a comprehensive, grounded answer
"""

from __future__ import annotations
import os
import pickle
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from retriever import AdvancedRetriever
from config import (
    GROQ_MODEL, LLM_TEMPERATURE, MAX_HISTORY_TURNS, STORES_DIR,
    MAX_AGENT_ITERATIONS,
    AGENT_MAX_DOCS_PER_CALL, AGENT_MAX_CHARS_PER_DOC, AGENT_MAX_TOTAL_CHARS,
)

load_dotenv()

AGENT_SYSTEM = """You are a meticulous research assistant. You answer questions \
about a single document by calling the search tools below. You never invent facts.

## How to think
1. **Plan first.** Read the question. If it has multiple parts ("compare X and Y",
   "list all Z and explain why"), break it into sub-questions and search each one.
2. **Search 2–4 times** with *different* phrasings or keywords. The first search
   is often not the best. Vary the angle: synonyms, related concepts, specific
   names. Don't repeat the same query twice.
3. **Pick the right tool**:
   - `semantic_search` — your default. Use for ideas, themes, explanations,
     "what does the author say about X", conceptual questions.
   - `keyword_search` — use for exact names, numbers, dates, jargon, or when
     semantic search returned nothing relevant.
   - `get_page` — use only when a previous result cited a specific page and
     you want the surrounding context.
4. **Stop searching** when you have enough concrete evidence to answer. Do NOT
   keep searching forever — you have a hard limit on tool calls.
5. **Synthesize**, don't just dump quotes. Write a clear, organized answer that
   directly addresses the question, weaving in evidence.

## Rules — non-negotiable
- Use ONLY information returned by the tools. If the tools don't contain the
  answer, say: *"The document doesn't cover this."* Do not guess.
- Cite every claim with a page number like `[Page 42]`. Multiple pages are fine.
- Quote the document verbatim for direct claims (in "quotes"), but paraphrase
  longer passages.
- If two passages disagree, surface the disagreement instead of picking one.
- Be specific and concrete. No vague filler ("the document discusses many
  things"). Lead with the answer, then the evidence.

## Worked example
Question: *"What does the author say about meditation and brain plasticity?"*
Plan: two concepts — "meditation" + "brain plasticity / neuroplasticity".
- Call `semantic_search("meditation and neuroplasticity")`
- Call `semantic_search("how meditation rewires the brain")`
- If a result mentions a study on page 87, call `get_page(87)` for context.
- Synthesize: lead with the author's claim, support with quotes + page cites."""


def _format_search_results(
    docs: list,
    max_chars_per_doc: int = AGENT_MAX_CHARS_PER_DOC,
    max_docs: int = AGENT_MAX_DOCS_PER_CALL,
) -> str:
    """Format documents into a compact string for the agent to read."""
    if not docs:
        return "No results found. Try a different phrasing or the keyword_search tool."
    parts = []
    for doc in docs[:max_docs]:
        page = doc.metadata.get("page", 0) + 1
        text = doc.page_content[:max_chars_per_doc]
        if len(doc.page_content) > max_chars_per_doc:
            text += "..."
        parts.append(f"[Page {page}] {text}")
    result = "\n\n---\n\n".join(parts)
    if len(result) > AGENT_MAX_TOTAL_CHARS:
        result = result[:AGENT_MAX_TOTAL_CHARS] + "\n...(truncated)"
    return result


class AgenticRAG:
    def __init__(self, doc_name: str):
        self.doc_name = doc_name
        self.retriever = AdvancedRetriever(doc_name)
        self.chat_history: list[dict] = []

        # Load chunks for page lookup
        bm25_path = os.path.join(STORES_DIR, doc_name, "bm25_chunks.pkl")
        with open(bm25_path, "rb") as f:
            self.all_chunks = pickle.load(f)

        # LLM
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=os.getenv("GROQ_API_KEY"),
        )

        # Build tools and agent
        self.tools = self._build_tools()
        self.executor = self._build_agent()

    def _build_tools(self) -> list:
        retriever = self.retriever
        all_chunks = self.all_chunks

        @tool
        def semantic_search(query: str) -> str:
            """Search the document by meaning (vector + BM25 hybrid, reranked).
            Best for conceptual questions, ideas, themes, and finding passages
            related to a topic. This is your DEFAULT tool — use it first.
            Vary the query phrasing across multiple calls for better coverage."""
            try:
                docs = retriever.lite_retrieve(query, top_k=AGENT_MAX_DOCS_PER_CALL)
                return _format_search_results(docs)
            except Exception as e:
                return f"semantic_search failed: {e}. Try keyword_search instead."

        @tool
        def keyword_search(keywords: str) -> str:
            """Search for exact keywords, names, numbers, or specific phrases
            using BM25 lexical matching. Use this when you need an EXACT term
            (a person's name, a number, a date, a technical term) that semantic
            search might miss or rank poorly."""
            try:
                docs = retriever._bm25_search(keywords, k=AGENT_MAX_DOCS_PER_CALL * 2)
                return _format_search_results(docs)
            except Exception as e:
                return f"keyword_search failed: {e}."

        @tool
        def get_page(page_number: int) -> str:
            """Read all content on a specific page (and the pages immediately
            before and after for surrounding context). Use this only AFTER a
            previous search returned a [Page N] citation you want to expand."""
            try:
                target = int(page_number)
            except (TypeError, ValueError):
                return f"Invalid page number: {page_number!r}. Pass an integer."
            window = {target - 1, target, target + 1}
            results = [
                doc for doc in all_chunks
                if (doc.metadata.get("page", -1) + 1) in window
            ]
            if not results:
                return f"No content found for page {target}."
            # Sort by page so adjacent context reads in order
            results.sort(key=lambda d: d.metadata.get("page", 0))
            return _format_search_results(results, max_docs=AGENT_MAX_DOCS_PER_CALL)

        return [semantic_search, keyword_search, get_page]

    def _build_agent(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages([
            ("system", AGENT_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            return_intermediate_steps=True,
            max_iterations=MAX_AGENT_ITERATIONS,
            handle_parsing_errors=True,
        )

    def _get_chat_messages(self) -> list:
        """Convert chat history to LangChain messages."""
        messages = []
        for turn in self.chat_history[-MAX_HISTORY_TURNS:]:
            messages.append(HumanMessage(content=turn["question"]))
            messages.append(AIMessage(content=turn["answer"]))
        return messages

    def ask(self, question: str) -> dict:
        """
        Run the agent. Returns:
        {
            "answer": str,
            "steps": [{"tool": str, "input": str, "output": str}, ...],
            "num_searches": int,
        }
        """
        try:
            result = self.executor.invoke({
                "input": question,
                "chat_history": self._get_chat_messages(),
            })
        except Exception as e:
            return {
                "answer": f"The agent hit an error and could not complete: {e}",
                "steps": [],
                "num_searches": 0,
            }

        # Parse intermediate steps
        steps = []
        for action, observation in result.get("intermediate_steps", []):
            obs_str = str(observation)
            steps.append({
                "tool": action.tool,
                "input": action.tool_input if isinstance(action.tool_input, str)
                         else str(action.tool_input),
                "output": obs_str[:500] if len(obs_str) > 500 else obs_str,
            })

        answer = result.get("output") or "The agent did not produce an answer."
        self.chat_history.append({"question": question, "answer": answer})

        return {
            "answer": answer,
            "steps": steps,
            "num_searches": len(steps),
        }

    def clear_history(self):
        self.chat_history = []
