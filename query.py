"""
CLI Chat Interface
------------------
Interactive terminal chat with document selection.

Usage:
    python query.py                   # pick from ingested docs
    python query.py book_of_elon      # direct doc name
"""

from __future__ import annotations
import sys
from dotenv import load_dotenv
from rag_chain import RAGChain
from ingest import list_ingested

load_dotenv()


def pick_document() -> str | None:
    """Let user choose from ingested documents."""
    docs = list_ingested()
    if not docs:
        print("No documents ingested yet. Run 'python ingest.py' first.")
        return None

    print("\nIngested documents:\n")
    for i, name in enumerate(docs, 1):
        print(f"  {i}. {name}")

    print()
    choice = input("Pick a document (number): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(docs):
            return docs[idx]
    print("Invalid choice.")
    return None


def main():
    # Accept doc name as argument or pick interactively
    if len(sys.argv) > 1:
        doc_name = sys.argv[1]
    else:
        doc_name = pick_document()

    if not doc_name:
        return

    print(f"\nLoading RAG pipeline for '{doc_name}'...")
    rag = RAGChain(doc_name)
    print("Ready! Commands: 'quit' to exit, 'clear' to reset conversation\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "clear":
            rag.clear_history()
            print("[Conversation cleared]\n")
            continue

        print()
        docs = None
        verification = None

        for chunk, source_docs, verif in rag.stream(question):
            if chunk is not None:
                print(chunk, end="", flush=True)
            if source_docs is not None:
                docs = source_docs
            if verif is not None:
                verification = verif

        print("\n")

        # Sources
        if docs:
            pages = sorted(set(doc.metadata.get("page", 0) + 1 for doc in docs))
            print(f"  Sources: pages {pages}")

        # Verification
        if verification:
            if verification["verified"]:
                print("  Verified: All claims supported by sources")
            else:
                print(f"  Warning: {verification['issues']}")
        print()


if __name__ == "__main__":
    main()
