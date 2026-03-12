# ============================================================
# config.py — All settings. Change only this file.
# ============================================================
#
# TWO modes — auto detected:
#
#   LOCAL MODE  → Ollama  (no API key needed, runs on your machine)
#   CLOUD MODE  → Gemini  (set GOOGLE_API_KEY in Streamlit secrets)
#
# How to deploy on Streamlit Cloud:
#   Settings → Secrets → add:
#   GOOGLE_API_KEY = "your-gemini-api-key"

import os

# ── Detect mode
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
USE_CLOUD      = bool(GOOGLE_API_KEY)

# ── Local settings (Ollama)
OLLAMA_MODEL = "llama2:7b"   # change to "llama3", "mistral", etc.

# ── Cloud settings (Gemini)
GEMINI_MODEL = "gemini-1.5-flash"   # free: 1500 req/day

# ── Shared settings
CHROMA_DIR    = "./chroma_db"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# ── Model limits (used for Q&A batch generation)
MODEL_LIMITS = {
    "tinyllama"  : (2048,   5,  2, "🔴"),
    "phi"        : (2048,   5,  2, "🔴"),
    "phi3"       : (4096,  10,  4, "🟡"),
    "llama2"     : (4096,  10,  4, "🟡"),
    "llama2:7b"  : (4096,  10,  4, "🟡"),
    "mistral"    : (8192,  20,  6, "🟢"),
    "llama3"     : (8192,  20,  6, "🟢"),
    "gemma2"     : (8192,  20,  6, "🟢"),
    "mixtral"    : (32768, 20,  8, "🟢"),
    "llama3.1"   : (131072,20, 10, "🟢"),
    # Gemini has large context — treat as high capacity
    "gemini-1.5-flash" : (131072, 20, 10, "🟢"),
    "gemini-1.5-pro"   : (131072, 20, 10, "🟢"),
}

DEFAULT_LIMITS = (4096, 10, 4, "🟡")


def get_model_limits():
    """Returns (context_tokens, max_pairs, batch_size, emoji) for active model."""
    active = GEMINI_MODEL if USE_CLOUD else OLLAMA_MODEL
    key = active.lower().strip()
    if key in MODEL_LIMITS:
        return MODEL_LIMITS[key]
    for k, v in MODEL_LIMITS.items():
        if key.startswith(k):
            return v
    return DEFAULT_LIMITS


if USE_CLOUD:
    print(f"🌐 Cloud Mode — Gemini ({GEMINI_MODEL})")
else:
    print(f"💻 Local Mode — Ollama ({OLLAMA_MODEL})")