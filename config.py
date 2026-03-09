# ============================================================
# config.py  —  All settings live here. Change only this file.
# ============================================================

# ── Vector database storage folder
CHROMA_DIR = "./chroma_db"

# ── Embedding model (runs locally, no API key needed)
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Your Ollama model — change this to switch models
OLLAMA_MODEL = "llama2:7b"   # options: "llama3", "mistral", "gemma2", etc.

# ── Text chunking settings
CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 100   # overlap between chunks (avoids missing info at edges)

# ── Model limits table
# Format: model_name → (context_tokens, max_safe_qa_pairs, batch_size, status_emoji)
#
# Why this matters:
#   llama2 7b has only 4096 tokens → can generate max ~10 Q&A pairs safely
#   llama3 has 8192 tokens → can generate up to 20 pairs
#   If you ask for more than the model can handle, it silently stops early
#
MODEL_LIMITS = {
    # ── Very small context (avoid high pair counts)
    "tinyllama"  : (2048,   5,  2, "🔴"),
    "phi"        : (2048,   5,  2, "🔴"),
    # ── Medium context (batching auto-handles limits)
    "phi3"       : (4096,  10,  4, "🟡"),
    "llama2"     : (4096,  10,  4, "🟡"),
    "llama2:7b"  : (4096,  10,  4, "🟡"),
    "llama2:13b" : (4096,  12,  4, "🟡"),
    "llama2:70b" : (4096,  12,  4, "🟡"),
    # ── Large context (most capable)
    "mistral"    : (8192,  20,  6, "🟢"),
    "mistral:7b" : (8192,  20,  6, "🟢"),
    "llama3"     : (8192,  20,  6, "🟢"),
    "llama3:8b"  : (8192,  20,  6, "🟢"),
    "gemma"      : (8192,  20,  6, "🟢"),
    "gemma2"     : (8192,  20,  6, "🟢"),
    "codellama"  : (16384, 20,  8, "🟢"),
    "mixtral"    : (32768, 20,  8, "🟢"),
    "qwen2"      : (131072,20, 10, "🟢"),
    "llama3.1"   : (131072,20, 10, "🟢"),
    "llama3.2"   : (131072,20, 10, "🟢"),
    "llama3:70b" : (131072,20, 10, "🟢"),
    "deepseek-r1": (65536, 20, 10, "🟢"),
}

# Fallback if your model is not in the list above
DEFAULT_LIMITS = (4096, 10, 4, "🟡")


def get_model_limits():
    """
    Returns (context_tokens, max_pairs, batch_size, emoji) for OLLAMA_MODEL.
    Falls back to DEFAULT_LIMITS if model not found.
    """
    key = OLLAMA_MODEL.lower().strip()
    if key in MODEL_LIMITS:
        return MODEL_LIMITS[key]
    # Try prefix match: "llama3:instruct" matches "llama3"
    for k, v in MODEL_LIMITS.items():
        if key.startswith(k):
            return v
    return DEFAULT_LIMITS
