<div align="center">

# 🧠 DocMind AI

### RAG-Powered Document Chatbot with Auto Q&A Generation

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-FF6B35?style=for-the-badge)](https://trychroma.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge)](https://ollama.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**Ask questions about any document. Auto-generate Q&A pairs. Export to PDF or DOCX.**
*100% local — no API keys, no internet required after setup.*

[📺 Demo Video](#demo) · [⚡ Quick Start](#quick-start) · [🏗️ Architecture](#architecture) · [📦 Features](#features)

---

![DocMind AI Screenshot](assets/screenshot.png)

</div>

---

## 📦 Features

| Feature | Description |
|---|---|
| 💬 **Document Chat** | Ask natural language questions about uploaded PDFs, TXT files, or websites |
| ⚡ **Auto Q&A Generator** | AI automatically generates question-answer pairs from your document |
| 🔢 **Smart Batch Inference** | Auto-detects model context limits and batches generation accordingly |
| 📥 **Export Q&A** | Download generated Q&A as **PDF**, **DOCX**, or **TXT** |
| 🌐 **Multi-language** | Generate Q&A in English, Hindi, Spanish, French, German, Arabic |
| 🎯 **Difficulty Levels** | Easy / Medium / Hard Q&A difficulty selector |
| 📂 **Multi-source Ingestion** | PDF + TXT + URL all supported |
| 🔒 **100% Local & Private** | No data leaves your machine |
| 🗂️ **Persistent Vector DB** | ChromaDB persists between sessions automatically |

---

## 🏗️ Architecture

```
                        ┌─────────────────────────────────┐
                        │         USER INTERFACE           │
                        │   Streamlit  (No Sidebar UI)     │
                        └──────────┬──────────┬───────────┘
                                   │          │
                    ┌──────────────▼──┐   ┌───▼──────────────┐
                    │   💬 Chat Tab   │   │  ⚡ Auto Q&A Tab  │
                    └──────────┬──────┘   └───┬──────────────┘
                               │              │
              ┌────────────────▼──────────────▼────────────────┐
              │                 RAG PIPELINE                     │
              │                                                  │
              │  Documents (PDF/TXT/URL)                         │
              │       │                                          │
              │       ▼                                          │
              │  ┌─────────────────┐                            │
              │  │  Text Splitter  │  RecursiveCharacterSplitter │
              │  │  chunk=800tok   │  overlap=100tok             │
              │  └────────┬────────┘                            │
              │           ▼                                      │
              │  ┌─────────────────┐                            │
              │  │   Embeddings    │  all-MiniLM-L6-v2 (local)  │
              │  │   HuggingFace   │  384-dim vectors            │
              │  └────────┬────────┘                            │
              │           ▼                                      │
              │  ┌─────────────────┐                            │
              │  │   ChromaDB      │  Vector store (on disk)    │
              │  │   Vector Store  │  MMR similarity search     │
              │  └────────┬────────┘                            │
              │           │                                      │
              │    At query time:                                │
              │    Question → Embed → Top-6 chunks (MMR)        │
              │           ▼                                      │
              │  ┌─────────────────┐                            │
              │  │  Ollama (Local) │  llama2 / llama3 / mistral │
              │  │  LLM Inference  │  RetrievalQA Chain         │
              │  └────────┬────────┘                            │
              │           ▼                                      │
              │     Answer + Source Citations                    │
              └──────────────────────────────────────────────────┘
                               │
              ┌────────────────▼───────────────────┐
              │         EXPORT ENGINE               │
              │  ReportLab (PDF) · python-docx      │
              │  Styled Q&A with branding           │
              └────────────────────────────────────┘
```

### Auto Q&A Batch Strategy

```
User requests 12 pairs  →  llama2 7b (4096 token context)

  Batch 1: chunks[0:6]  → generates Q1–Q4   ✅
  Batch 2: chunks[3:9]  → generates Q5–Q8   ✅
  Batch 3: chunks[6:12] → generates Q9–Q12  ✅
  
  Merge + Deduplicate → 12 unique Q&A pairs  🎯
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/vikasparmar/docmind-ai.git
cd docmind-ai
```

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Ollama & pull a model
```bash
# Install Ollama → https://ollama.ai
ollama pull llama2            # or: mistral, llama3, gemma2
ollama serve                  # keep this running
```

### 4. Run the app
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 📁 Project Structure

```
docmind-ai/
│
├── app.py                  ← Main application (UI + RAG + Export)
├── requirements.txt        ← Dependencies
├── README.md               ← This file
├── DEPLOY.md               ← Deployment guide (local/Docker/cloud)
│
├── chroma_db/              ← Auto-created: vector database
└── assets/
    └── screenshot.png      ← App screenshot for README
```

### 💡 Better Architecture (Production Upgrade Path)

The current single-file approach is great for a portfolio project. For production, here's how you'd split it:

```
docmind-ai/
├── app.py                  ← Streamlit UI only
├── config.py               ← All constants & model limits
├── rag/
│   ├── loader.py           ← PDF/TXT/URL loaders
│   ├── vectorstore.py      ← ChromaDB operations
│   ├── chain.py            ← RAG chain & prompt
│   └── generator.py        ← Auto Q&A batch generator
├── export/
│   ├── pdf_export.py       ← ReportLab PDF generation
│   └── docx_export.py      ← python-docx generation
├── tests/
│   ├── test_rag.py
│   └── test_export.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🤖 Supported Models

| Model | Context | Max Q&A | Speed | Quality |
|---|---|---|---|---|
| `tinyllama` | 2K | 5 | ⚡⚡⚡ | ⭐⭐ |
| `phi3` | 4K | 10 | ⚡⚡⚡ | ⭐⭐⭐ |
| `llama2` | 4K | 10 | ⚡⚡ | ⭐⭐⭐ |
| `mistral` | 8K | 20 | ⚡⚡ | ⭐⭐⭐⭐ |
| `llama3` | 8K | 20 | ⚡⚡ | ⭐⭐⭐⭐ |
| `gemma2` | 8K | 20 | ⚡⚡ | ⭐⭐⭐⭐ |
| `mixtral` | 32K | 20 | ⚡ | ⭐⭐⭐⭐⭐ |
| `llama3.1` | 128K | 20 | ⚡ | ⭐⭐⭐⭐⭐ |

---

## 🧠 Key Concepts Demonstrated

- **RAG (Retrieval-Augmented Generation)** — grounding LLM answers in real documents
- **Vector Embeddings** — semantic search with `all-MiniLM-L6-v2`
- **MMR Retrieval** — Maximal Marginal Relevance for diverse, non-redundant chunks
- **Batched LLM Inference** — handling small context window models intelligently
- **Prompt Engineering** — structured prompts for JSON output, difficulty control, multilingual
- **Local LLM Deployment** — Ollama integration, no cloud dependency
- **Export Pipeline** — ReportLab PDF + python-docx generation

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Ollama (llama2 / llama3 / mistral) |
| Orchestration | LangChain 0.2+ |
| Vector DB | ChromaDB |
| Embeddings | HuggingFace sentence-transformers |
| PDF Export | ReportLab |
| DOCX Export | python-docx |
| Language | Python 3.10+ |

---

## 📊 How RAG Works (Simple Explanation)

```
Traditional LLM:  Question → LLM memory → Answer (may hallucinate)

RAG:  Question → Search your docs → Relevant chunks → LLM + context → Accurate answer
```

RAG is like giving the LLM an open-book exam instead of asking it to recall from memory.

---

## 🚀 Deployment

See **[DEPLOY.md](DEPLOY.md)** for full instructions:
- Local development
- Docker container
- Streamlit Cloud (with Gemini API swap)
- Hugging Face Spaces
- VPS / cloud server

---

## 👤 Author

**Vikas Parmar** — AI/ML Developer

[![Gmail](https://img.shields.io/badge/Gmail-vikasparmar444@gmail.com-EA4335?style=flat&logo=gmail)](mailto:vikasparmar444@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-vikasparmar-181717?style=flat&logo=github)](https://github.com/vikasparmar)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

**⭐ Star this repo if it helped you!**

*Built with LangChain · ChromaDB · Ollama · Streamlit*

</div>
