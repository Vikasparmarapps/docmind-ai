# 🚀 DocMind AI — Deployment Guide

This file explains how to run DocMind AI in different environments.
Choose the option that fits your situation.

---

## Option 1 — Local Machine (Recommended for Development)

This is the simplest way. Everything runs on your own computer.
No internet needed after setup. Completely free.

**Step 1 — Install Ollama**
```
Download from: https://ollama.ai
```

**Step 2 — Pull a model**
```bash
ollama pull llama2       # small, works on most computers (4GB RAM)
ollama pull mistral      # better quality (8GB RAM)
ollama pull llama3       # best quality (8GB RAM)
```

**Step 3 — Create virtual environment**
```bash
cd docmind_v2
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**Step 4 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 5 — Run the app**
```bash
# Terminal 1 — keep this running
ollama serve

# Terminal 2 — start the app
streamlit run app.py
```

**Step 6 — Open in browser**
```
http://localhost:8501
```

---

## Option 2 — Docker (Run anywhere without installing Python)

Docker packages your entire app into a container.
Useful when you want to share the app with someone who doesn't have Python installed.

**Step 1 — Create Dockerfile**

Create a file called `Dockerfile` in your project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

**Step 2 — Build the Docker image**
```bash
docker build -t docmind-ai .
```

**Step 3 — Run the container**
```bash
docker run -p 8501:8501 docmind-ai
```

**Step 4 — Open in browser**
```
http://localhost:8501
```

> ⚠️ Note: Ollama must still be running on your host machine.
> Inside the Dockerfile, point Ollama to host:
> Set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in your config.

---

## Option 3 — Streamlit Cloud (Free, shareable link)

Streamlit Cloud gives you a public URL like `yourapp.streamlit.app`.
Free tier available. Good for portfolio demos.

> ⚠️ Important: Streamlit Cloud cannot run Ollama (no GPU/heavy models).
> You must swap Ollama with a cloud LLM API (Gemini or OpenAI).

**Step 1 — Swap Ollama for Gemini API in `rag/chain.py` and `rag/generator.py`**

```python
# Remove this:
from langchain_ollama import OllamaLLM as Ollama
llm = Ollama(model=OLLAMA_MODEL, temperature=0.2)

# Replace with this:
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",     # free tier: 1500 requests/day
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.2,
)
```

**Step 2 — Add to requirements.txt**
```
langchain-google-genai>=1.0.0
```

**Step 3 — Push to GitHub**
```bash
git add .
git commit -m "Add Streamlit Cloud deployment"
git push
```

**Step 4 — Deploy on Streamlit Cloud**
```
1. Go to https://share.streamlit.io
2. Click "New app"
3. Connect your GitHub repo
4. Set main file: app.py
5. Add secret: GOOGLE_API_KEY = your_key_here
6. Click Deploy
```

**Get a free Gemini API key:**
```
https://aistudio.google.com/app/apikey
```

---

## Option 4 — Hugging Face Spaces (Free, great for AI portfolio)

HF Spaces is the best platform to showcase AI projects.
Recruiters and engineers check HF Spaces.

> Same Ollama → Gemini swap as Option 3 applies here too.

**Step 1 — Create a Space**
```
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose SDK: Streamlit
4. Choose visibility: Public
```

**Step 2 — Add your secret key**
```
Settings → Variables and Secrets → Add GOOGLE_API_KEY
```

**Step 3 — Push your code**
```bash
git remote add hf https://huggingface.co/spaces/vikasparmar/docmind-ai
git push hf main
```

---

## Option 5 — VPS / Cloud Server (24/7 uptime, ~$5/month)

Use this if you want the app always running at a custom domain.
DigitalOcean, AWS EC2, or Hetzner all work.

**Step 1 — SSH into your server**
```bash
ssh root@your-server-ip
```

**Step 2 — Install dependencies**
```bash
apt update && apt install python3-pip python3-venv -y
curl -fsSL https://ollama.ai/install.sh | sh
```

**Step 3 — Clone and setup**
```bash
git clone https://github.com/vikasparmar/docmind-ai.git
cd docmind-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Step 4 — Run Ollama as a background service**
```bash
ollama pull llama2
ollama serve &
```

**Step 5 — Run Streamlit as a background service**
```bash
nohup streamlit run app.py --server.port 8501 &
```

**Step 6 — Access at**
```
http://your-server-ip:8501
```

---

## 📊 Deployment Options Compared

| Option | Cost | Speed | Best For |
|---|---|---|---|
| Local Machine | Free | Fast | Development & demo video |
| Docker | Free | Fast | Sharing with teammates |
| Streamlit Cloud | Free | Medium | Portfolio demo link |
| Hugging Face Spaces | Free | Medium | AI portfolio showcase |
| VPS Server | ~$5/mo | Fast | Always-on production |

---

## 🔑 Environment Variables Reference

| Variable | Where Used | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Cloud deployments | Gemini API key (replaces Ollama) |
| `OLLAMA_BASE_URL` | Docker | Points to Ollama host |

Store secrets in:
- **Local** → `.env` file (never commit this)
- **Streamlit Cloud** → App Settings → Secrets
- **HF Spaces** → Settings → Variables and Secrets
- **VPS** → `export GOOGLE_API_KEY=xxx` in `.bashrc`

---

## ❓ Common Issues

**Ollama not responding**
```bash
# Make sure ollama is running
ollama serve

# Check if it's running
curl http://localhost:11434
```

**Port already in use**
```bash
streamlit run app.py --server.port 8502
```

**ChromaDB errors after update**
```bash
# Delete and recreate the database
rm -rf chroma_db/
# Then re-upload your documents in the app
```

**Model too slow**
```
Switch to a smaller model in config.py:
OLLAMA_MODEL = "tinyllama"   # fastest, lowest quality
OLLAMA_MODEL = "phi3"        # good balance
OLLAMA_MODEL = "mistral"     # best quality for 8GB RAM
```
