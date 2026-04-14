"""
Advanced Retrieval Pipeline
----------------------------
Full pipeline per question:
  1. HyDE — generate a hypothetical answer, embed it for better semantic matching
  2. Multi-query — LLM generates query variations for broader recall
  3. Hybrid search — BM25 (keyword) + Vector (semantic) per query
  4. Reciprocal Rank Fusion — merge keyword and vector results
  5. Cross-encoder rerank — score every chunk, return top-K
"""

from __future__ import annotations
import os
import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from config import (
    EMBED_MODEL, STORES_DIR,
    GROQ_MODEL, LLM_TEMPERATURE,
    RETRIEVAL_FETCH_K, BM25_FETCH_K,
    NUM_QUERY_VARIATIONS, RERANK_TOP_K,
    RERANK_MODEL, RRF_K,
)


MULTI_QUERY_PROMPT = """Generate {n} different search queries to find relevant information for this question.
Each query should use different keywords or approach the topic from a different angle.
Return ONLY the queries, one per line. No numbering, no explanation.

Question: {question}"""

HYDE_PROMPT = """Write a short passage (3-5 sentences) that would answer the following question.
Write it as if you are quoting from an authoritative book on the subject.
Do not say "I don't know" — write a plausible, detailed passage.

Question: {question}

Passage:"""


class AdvancedRetriever:
    def __init__(self, doc_name: str):
        self.doc_name = doc_name
        store_dir = os.path.join(STORES_DIR, doc_name)
        chroma_dir = os.path.join(store_dir, "chroma")
        bm25_path = os.path.join(store_dir, "bm25_chunks.pkl")

        # Vector store
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.vectorstore = Chroma(
            persist_directory=chroma_dir,
            collection_name=doc_name,
            embedding_function=self.embeddings,
        )

        # BM25 keyword index
        with open(bm25_path, "rb") as f:
            self.bm25_chunks = pickle.load(f)
        tokenized = [doc.page_content.lower().split() for doc in self.bm25_chunks]
        self.bm25 = BM25Okapi(tokenized)

        # LLM for query generation
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )

        # Cross-encoder reranker
        self.reranker = CrossEncoder(RERANK_MODEL)

    # ---- Step 1: HyDE ----

    def _generate_hypothetical_answer(self, question: str) -> str:
        """Generate a hypothetical passage that answers the question."""
        prompt = HYDE_PROMPT.format(question=question)
        response = self.llm.invoke(prompt)
        return response.content.strip()

    # ---- Step 2: Multi-Query ----

    def _generate_query_variations(self, question: str) -> list[str]:
        """Generate multiple phrasings of the question."""
        prompt = MULTI_QUERY_PROMPT.format(n=NUM_QUERY_VARIATIONS, question=question)
        response = self.llm.invoke(prompt)
        variations = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip()
        ]
        return variations[:NUM_QUERY_VARIATIONS]

    # ---- Step 3: Hybrid Search (Vector + BM25) ----

    def _vector_search(self, query: str, k: int = RETRIEVAL_FETCH_K) -> list:
        """Semantic similarity search via embeddings."""
        return self.vectorstore.similarity_search(query, k=k)

    def _vector_search_by_embedding(self, text: str, k: int = RETRIEVAL_FETCH_K) -> list:
        """Search using the embedding of arbitrary text (for HyDE)."""
        embedding = self.embeddings.embed_query(text)
        return self.vectorstore.similarity_search_by_vector(embedding, k=k)

    def _bm25_search(self, query: str, k: int = BM25_FETCH_K) -> list:
        """Keyword search via BM25."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.bm25_chunks[i] for i in top_indices if scores[i] > 0]

    # ---- Step 4: Reciprocal Rank Fusion ----

    def _reciprocal_rank_fusion(self, rankings: list[list]) -> list:
        """
        Merge multiple ranked lists using RRF.
        Each ranking is a list of Documents (ordered by relevance).
        Returns a single merged + deduplicated list.
        """
        doc_scores = {}   # content_hash -> cumulative RRF score
        doc_map = {}       # content_hash -> Document

        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                key = hash(doc.page_content)
                doc_map[key] = doc
                doc_scores[key] = doc_scores.get(key, 0) + 1.0 / (rank + RRF_K)

        sorted_keys = sorted(doc_scores, key=doc_scores.get, reverse=True)
        return [doc_map[k] for k in sorted_keys]

    # ---- Step 5: Cross-Encoder Rerank ----

    def _rerank(self, question: str, docs: list) -> list:
        """Score every chunk against the question, return top-K."""
        if not docs:
            return []
        pairs = [(question, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:RERANK_TOP_K]]

    # ---- Full Pipeline ----

    def lite_retrieve(self, question: str, top_k: int | None = None) -> list:
        """
        Fast retrieval for the agent: hybrid (vector + BM25) on the raw query,
        RRF merge, cross-encoder rerank. NO HyDE, NO multi-query — the agent
        already plays that role by calling this tool with multiple phrasings.

        ~10x faster than `retrieve()` because it skips ~5 LLM calls.
        """
        vector_results = self._vector_search(question)
        bm25_results = self._bm25_search(question)
        fused = self._reciprocal_rank_fusion([vector_results, bm25_results])
        reranked = self._rerank(question, fused)
        if top_k is not None:
            return reranked[:top_k]
        return reranked

    def retrieve(self, question: str) -> list:
        """
        Full retrieval:
          HyDE → multi-query → hybrid search (vector+BM25) per query
          → RRF merge → cross-encoder rerank → top-K
        """
        # HyDE: generate hypothetical answer
        hyde_passage = self._generate_hypothetical_answer(question)

        # Multi-query: original + HyDE + variations
        variations = self._generate_query_variations(question)
        all_queries = [question] + variations

        # Collect all rankings
        all_rankings = []

        # HyDE vector search (embed the hypothetical passage)
        hyde_results = self._vector_search_by_embedding(hyde_passage)
        all_rankings.append(hyde_results)

        # For each query: vector + BM25
        for query in all_queries:
            vector_results = self._vector_search(query)
            bm25_results = self._bm25_search(query)
            all_rankings.append(vector_results)
            all_rankings.append(bm25_results)

        # Merge with Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(all_rankings)

        # Rerank with cross-encoder
        reranked = self._rerank(question, fused)

        return reranked
