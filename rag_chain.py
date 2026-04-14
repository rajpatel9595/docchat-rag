"""
RAG Chain with Conversation Memory + Answer Verification
---------------------------------------------------------
Orchestrates:
  1. Standalone question rewriting (makes follow-ups work)
  2. Advanced retrieval (HyDE + hybrid + multi-query + rerank)
  3. Context-aware answer generation with streaming
  4. Answer verification — second LLM pass flags unsupported claims
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import AdvancedRetriever
from config import GROQ_MODEL, LLM_TEMPERATURE, MAX_HISTORY_TURNS

load_dotenv()

# --- Prompts ---

CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Given the conversation history and a follow-up question,
rephrase the follow-up into a standalone question that captures the full intent.
If the question is already standalone, return it unchanged.

Chat history:
{chat_history}

Follow-up question: {question}

Standalone question:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a knowledgeable assistant answering questions based on the provided document.

Rules:
- Answer using ONLY the context below. Do not use outside knowledge.
- Quote the document directly when possible (use quotation marks).
- Synthesize information from multiple passages when relevant.
- Cite page numbers in brackets like [Page 42] when referencing specific passages.
- If the context doesn't contain enough information, say:
  "The document doesn't directly address this topic."
- Be specific, concrete, and well-structured.

Context from the document:
{context}

Conversation so far:
{chat_history}

Question: {question}

Answer:"""
)

VERIFY_PROMPT = ChatPromptTemplate.from_template(
    """You are a fact-checking assistant. Your job is to verify whether an answer
is fully supported by the provided source passages.

Source passages:
{context}

Answer to verify:
{answer}

Instructions:
- Check each claim in the answer against the source passages.
- If the answer is fully supported, respond with exactly: VERIFIED
- If any claim is NOT supported by the sources, respond with:
  ISSUES: [brief description of unsupported claims]

Verdict:"""
)


def format_docs(docs):
    """Format retrieved documents with page numbers."""
    return "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1}] {doc.page_content}"
        for doc in docs
    )


def format_chat_history(history: list[dict]) -> str:
    if not history:
        return "(no prior conversation)"
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        lines.append(f"Human: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


class RAGChain:
    def __init__(self, doc_name: str):
        self.doc_name = doc_name
        self.retriever = AdvancedRetriever(doc_name)
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.condense_chain = CONDENSE_PROMPT | self.llm | StrOutputParser()
        self.answer_chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        self.verify_chain = VERIFY_PROMPT | self.llm | StrOutputParser()
        self.chat_history: list[dict] = []

    def _condense_question(self, question: str) -> str:
        if not self.chat_history:
            return question
        history_str = format_chat_history(self.chat_history)
        return self.condense_chain.invoke({
            "chat_history": history_str,
            "question": question,
        })

    def _verify_answer(self, answer: str, context: str) -> str | None:
        """
        Run a verification pass. Returns None if verified,
        or a string describing issues if claims are unsupported.
        """
        verdict = self.verify_chain.invoke({
            "context": context,
            "answer": answer,
        }).strip()
        if verdict.upper().startswith("VERIFIED"):
            return None
        return verdict

    def ask(self, question: str) -> dict:
        """
        Full RAG pipeline (non-streaming).
        Returns dict with: answer, docs, verified, issues
        """
        standalone = self._condense_question(question)
        docs = self.retriever.retrieve(standalone)
        context = format_docs(docs)
        history_str = format_chat_history(self.chat_history)

        answer = self.answer_chain.invoke({
            "context": context,
            "chat_history": history_str,
            "question": standalone,
        })

        # Verify
        issues = self._verify_answer(answer, context)

        self.chat_history.append({"question": question, "answer": answer})
        return {
            "answer": answer,
            "docs": docs,
            "verified": issues is None,
            "issues": issues,
        }

    def stream(self, question: str):
        """
        Full RAG pipeline (streaming).
        Yields: (chunk, None, None) for text
        Final yield: (None, docs, verification_result)
        """
        standalone = self._condense_question(question)
        docs = self.retriever.retrieve(standalone)
        context = format_docs(docs)
        history_str = format_chat_history(self.chat_history)

        full_answer = []
        for chunk in self.answer_chain.stream({
            "context": context,
            "chat_history": history_str,
            "question": standalone,
        }):
            full_answer.append(chunk)
            yield chunk, None, None

        answer = "".join(full_answer)
        self.chat_history.append({"question": question, "answer": answer})

        # Verify (runs after streaming completes)
        issues = self._verify_answer(answer, context)
        verification = {"verified": issues is None, "issues": issues}

        yield None, docs, verification

    def clear_history(self):
        self.chat_history = []
