# ============================================================
# rag/vectorstore.py  —  Handles ChromaDB (vector database)
# ============================================================
# This file has ONE job: store and retrieve document chunks
#
# What is a vector database?
#   Normal database → stores text as-is, searches by exact match
#   Vector database → converts text to numbers (embeddings),
#                     searches by meaning/similarity
#
# Example:
#   "What is ML?" and "Define machine learning" are different strings
#   but a vector DB knows they mean the same thing → returns same results

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

import streamlit as st
from config import CHROMA_DIR, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """
    Load the embedding model once and cache it.
    all-MiniLM-L6-v2 = a small 80MB model that converts text → 384 numbers
    These numbers represent the "meaning" of the text.
    Cached so it only loads once, not on every page refresh.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(embeddings):
    """
    Connect to (or create) the ChromaDB database on disk.
    CHROMA_DIR is where the database files are saved.
    """
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


def store_documents(docs: list, source_name: str, embeddings) -> tuple:
    """
    Takes raw documents, splits them into chunks, embeds them,
    and saves everything to ChromaDB.

    Why split into chunks?
      LLMs have token limits. A 50-page PDF is too big to send at once.
      We split it into 800-token chunks, store all chunks,
      then at query time only retrieve the 6 most relevant chunks.

    Returns:
      (vectorstore, raw_text_chunks)
      raw_text_chunks are used later for Q&A generation
    """
    # Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Tag each chunk so we know which file it came from
    for chunk in chunks:
        chunk.metadata["source_name"] = source_name

    # Save to ChromaDB (auto-persisted in Chroma 0.4+)
    vs = get_vectorstore(embeddings)
    vs.add_documents(chunks)

    # Also return raw text for the Q&A generator
    raw_texts = [chunk.page_content for chunk in chunks]

    return vs, raw_texts
