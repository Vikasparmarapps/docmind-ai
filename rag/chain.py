# ============================================================
# rag/chain.py  —  Handles asking questions to your documents
# ============================================================
# This file has ONE job: take a question + documents → return answer
#
# How RAG works (simple explanation):
#
#   Normal LLM:  You ask a question → LLM answers from its training memory
#                Problem: it may hallucinate or not know your document
#
#   RAG:  You ask a question
#         → search your document chunks for the most relevant pieces
#         → send those pieces + your question to the LLM
#         → LLM answers using YOUR document as reference
#         Result: accurate, grounded answers from your actual files

try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    from langchain_community.llms import Ollama

from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

from config import OLLAMA_MODEL


# This is the instruction we give the LLM before every question.
# It tells the LLM HOW to behave when answering.
PROMPT_TEMPLATE = """You are a knowledgeable AI assistant. Answer the user's question using the context passages below.

Instructions:
- Read ALL context passages carefully before answering
- Give a clear, complete, helpful answer based on what the context says
- You may combine information from multiple passages
- If the context genuinely has no relevant information at all, only then say you couldn't find it
- Do NOT refuse to answer if the context contains relevant information — even partial

Context passages:
{context}

Question: {question}

Answer (be specific and helpful):"""


def build_chain(vectorstore):
    """
    Builds the RAG pipeline (chain).

    What this does step by step:
      1. User asks a question
      2. The question is converted to a vector (embedding)
      3. ChromaDB finds the 6 most relevant document chunks
         using MMR (picks diverse chunks, not duplicates)
      4. Those chunks + the question are sent to Ollama (local LLM)
      5. Ollama reads the chunks and writes an answer
      6. The answer + source documents are returned

    MMR = Maximal Marginal Relevance
      Instead of returning the 6 most similar chunks (which may all say
      the same thing), MMR picks 6 DIVERSE chunks covering different
      parts of the answer. Better coverage = better answers.
    """
    llm = Ollama(model=OLLAMA_MODEL, temperature=0.2)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # "stuff" = stuff all chunks into one prompt
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,              # return 6 final chunks
                "fetch_k": 12,       # fetch 12 first, then pick best 6 via MMR
            },
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def ask(question: str, vectorstore) -> dict:
    """
    Ask a question against the loaded documents.

    Returns a dict:
      {
        "answer":  "The answer text...",
        "sources": [list of source Document objects with metadata]
      }
    """
    chain = build_chain(vectorstore)
    result = chain.invoke({"query": question})

    return {
        "answer":  result["result"],
        "sources": result.get("source_documents", []),
    }
