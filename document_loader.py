"""
Universal Document Loader
--------------------------
Loads PDF, TXT, DOCX, and CSV files into LangChain Documents.
"""

from __future__ import annotations
import os
import re
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)
from config import SUPPORTED_EXTENSIONS


def clean_text(text: str) -> str:
    """Clean extracted document text."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def load_file(file_path: str) -> list:
    """Load a file and return cleaned LangChain Documents."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        ".csv": CSVLoader,
    }

    loader = loaders[ext](file_path)
    documents = loader.load()

    # Clean and add source metadata
    file_name = os.path.basename(file_path)
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["source_file"] = file_name

    # Remove empty documents
    documents = [doc for doc in documents if len(doc.page_content) > 30]
    return documents


def get_doc_name(file_path: str) -> str:
    """Generate a clean collection name from a file path."""
    name = os.path.splitext(os.path.basename(file_path))[0]
    # Sanitize: only keep alphanumeric, hyphens, underscores
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_").lower()
    # ChromaDB requires collection names 3-63 chars, start/end with alphanumeric
    if len(name) < 3:
        name = name + "_doc"
    return name[:63]


def list_downloads(downloads_dir: str) -> list[str]:
    """List supported files in the downloads directory.

    Returns an empty list if the directory is missing or unreadable
    (e.g. macOS TCC denies access to ~/Downloads).
    """
    try:
        entries = sorted(os.listdir(downloads_dir))
    except (PermissionError, FileNotFoundError, OSError):
        return []

    files = []
    for f in entries:
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(f)
    return files
