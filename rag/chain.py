from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from config import USE_CLOUD, OLLAMA_MODEL, GROQ_MODEL, GROQ_API_KEY# ============================================================


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


def get_llm():
    """
    Returns the correct LLM based on environment.
    GROQ_API_KEY set → Groq (cloud)
    No key           → Ollama (local)
    """
    if USE_CLOUD:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.2,
        )
    else:
        try:
            from langchain_ollama import OllamaLLM as Ollama
        except ImportError:
            from langchain_community.llms import Ollama
        return Ollama(model=OLLAMA_MODEL, temperature=0.2)


def build_chain(vectorstore):
    """Builds the RAG pipeline using the appropriate LLM."""
    llm = get_llm()

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 6},
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def ask(question: str, vectorstore) -> dict:
    """Ask a question against the loaded documents."""
    chain = build_chain(vectorstore)
    result = chain.invoke({"query": question})
    return {
        "answer":  result["result"],
        "sources": result.get("source_documents", []),
    }