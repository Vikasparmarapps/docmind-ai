# ============================================================
# rag/loader.py  —  Loads documents from PDF, TXT, or URL
# ============================================================
# This file has ONE job: take a file or URL → return text docs

import os
import tempfile

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)


def load_pdf(file) -> list:
    """
    Load a PDF file uploaded via Streamlit.
    Saves it temporarily, reads it, then deletes the temp file.
    Returns a list of LangChain Document objects (one per page).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    docs = PyPDFLoader(tmp_path).load()
    os.unlink(tmp_path)   # delete temp file after reading
    return docs


def load_txt(file) -> list:
    """
    Load a plain text (.txt) file uploaded via Streamlit.
    Returns a list of LangChain Document objects.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    docs = TextLoader(tmp_path, encoding="utf-8").load()
    os.unlink(tmp_path)
    return docs


def load_url(url: str) -> list:
    """
    Scrape a webpage and load its text content.
    Returns a list of LangChain Document objects.
    """
    return WebBaseLoader(url).load()
