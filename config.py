# ============================================================
# config.py — All settings. Change only this file.
# ============================================================
#
# TWO modes — auto detected:
#
#   LOCAL MODE  → Ollama  (no API key needed, runs on your machine)
#   CLOUD MODE  → Groq    (set GROQ_API_KEY in .env or Streamlit secrets)
#
# How to run locally with Groq:
#   1. Create .env file in project root:
#      GROQ_API_KEY=gsk_your-key-here
#   2. pip install python-dotenv
#   3. streamlit run app.py
#
# How to deploy on Streamlit Cloud:
#   Settings → Secrets → add:
#   GROQ_API_KEY = "gsk_your-key-here"

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file if it exists

# ── Detect mode
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
USE_CLOUD    = bool(GROQ_API_KEY)

# ── Local settings (Ollama)
OLLAMA_MODEL = "llama2:7b"   # change to "llama3", "mistral", etc.

# ── Cloud settings (Groq)
GROQ_MODEL = "llama-3.1-8b-instant"   # fast + free
# Other options:
# "llama-3.3-70b-versatile"  → better quality, slower
# "mixtral-8x7b-32768"       → good balance

# ── Shared settings
CHROMA_DIR    = "./chroma_db"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50

# ── Model limits (used for Q&A batch generation)
MODEL_LIMITS = {
    "tinyllama"              : (2048,   5,  2, "🔴"),
    "phi"                    : (2048,   5,  2, "🔴"),
    "phi3"                   : (4096,  10,  4, "🟡"),
    "llama2"                 : (4096,  10,  4, "🟡"),
    "llama2:7b"              : (4096,  10,  4, "🟡"),
    "mistral"                : (8192,  20,  6, "🟢"),
    "llama3"                 : (8192,  20,  6, "🟢"),
    "gemma2"                 : (8192,  20,  6, "🟢"),
    "mixtral"                : (32768, 20,  8, "🟢"),
    "llama3.1"               : (131072,20, 10, "🟢"),
    # Groq models
    "llama-3.1-8b-instant"   : (131072, 20, 8, "🟢"),
    "llama-3.3-70b-versatile": (131072, 20, 8, "🟢"),
    "mixtral-8x7b-32768"     : (32768,  20, 6, "🟢"),
}

DEFAULT_LIMITS = (4096, 10, 4, "🟡")


def get_model_limits():
    """Returns (context_tokens, max_pairs, batch_size, emoji) for active model."""
    active = GROQ_MODEL if USE_CLOUD else OLLAMA_MODEL
    key = active.lower().strip()
    if key in MODEL_LIMITS:
        return MODEL_LIMITS[key]
    for k, v in MODEL_LIMITS.items():
        if key.startswith(k):
            return v
    return DEFAULT_LIMITS


if USE_CLOUD:
    print(f"🌐 Cloud Mode — Groq ({GROQ_MODEL})")
else:
    print(f"💻 Local Mode — Ollama ({OLLAMA_MODEL})")