"""
Generic Ingestion Pipeline
----------------------------
Pick any file → load → clean → chunk → embed → store (vector + BM25).

Usage:
    python ingest.py                          # interactive file picker
    python ingest.py /path/to/file.pdf        # direct path
"""

import os
import sys
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from document_loader import load_file, get_doc_name, list_downloads
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL,
    STORES_DIR, DOWNLOADS_DIR,
)

load_dotenv()


def ingest_file(file_path: str) -> str:
    """
    Ingest a single file into the RAG store.
    Returns the doc_name (used to load it later).
    """
    doc_name = get_doc_name(file_path)
    store_dir = os.path.join(STORES_DIR, doc_name)
    chroma_dir = os.path.join(store_dir, "chroma")
    bm25_path = os.path.join(store_dir, "bm25_chunks.pkl")

    # Check if already ingested
    if os.path.exists(chroma_dir) and os.path.exists(bm25_path):
        print(f"  '{doc_name}' is already ingested. Skipping.")
        print(f"  (Delete '{store_dir}' to re-ingest)")
        return doc_name

    os.makedirs(store_dir, exist_ok=True)

    print(f"1/4  Loading '{os.path.basename(file_path)}'...")
    documents = load_file(file_path)
    print(f"     {len(documents)} sections loaded")

    print("2/4  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"     {len(chunks)} chunks created")

    print("3/4  Embedding and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name=doc_name,
    )

    print("4/4  Building BM25 keyword index...")
    with open(bm25_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\nDone! '{doc_name}' is ready for querying.")
    return doc_name


def list_ingested() -> list[str]:
    """Return names of all ingested documents."""
    if not os.path.exists(STORES_DIR):
        return []
    return [
        name for name in sorted(os.listdir(STORES_DIR))
        if os.path.isdir(os.path.join(STORES_DIR, name))
        and os.path.exists(os.path.join(STORES_DIR, name, "chroma"))
    ]


def interactive_picker():
    """Let user pick a file from Downloads interactively."""
    files = list_downloads(DOWNLOADS_DIR)
    if not files:
        print(f"No supported files found in {DOWNLOADS_DIR}")
        return None

    print(f"\nFiles in {DOWNLOADS_DIR}:\n")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")

    print()
    choice = input("Enter number (or full path to any file): ").strip()

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            return os.path.join(DOWNLOADS_DIR, files[idx])
        print("Invalid number.")
        return None
    elif os.path.isfile(choice):
        return choice
    else:
        print(f"File not found: {choice}")
        return None


def main():
    # Accept file path as argument or pick interactively
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = interactive_picker()

    if file_path and os.path.isfile(file_path):
        ingest_file(file_path)
    else:
        print("No file selected.")


if __name__ == "__main__":
    main()
