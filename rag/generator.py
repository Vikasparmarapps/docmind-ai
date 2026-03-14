# ============================================================
# rag/generator.py  —  Auto Q&A pair generation
# ============================================================
# This file has ONE job: take document chunks → generate Q&A pairs
#
# The big problem we solve here:
#
#   llama2 7b has only 4096 tokens of "working memory"
#   If you ask for 12 Q&A pairs in one shot, it runs out of space
#   and silently stops after 4 pairs.
#
#   Solution: BATCHING
#   Instead of asking for 12 at once, we ask for 4 at a time, 3 times.
#   Each batch uses a different section of the document (for variety).
#   Finally we merge all batches and remove any duplicate questions.
#
#   Batch 1 → Q1–Q4   (uses doc section 1)
#   Batch 2 → Q5–Q8   (uses doc section 2)
#   Batch 3 → Q9–Q12  (uses doc section 3)
#   Merge + deduplicate → 12 unique Q&A pairs ✅

import re
import json

import streamlit as st

from config import USE_CLOUD, OLLAMA_MODEL, GROQ_MODEL, GROQ_API_KEY, get_model_limits


def _get_llm():
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
            temperature=0.4,
        )
    else:
        try:
            from langchain_ollama import OllamaLLM as Ollama
        except ImportError:
            from langchain_community.llms import Ollama
        return Ollama(model=OLLAMA_MODEL, temperature=0.4)


# Maps difficulty level to a description the LLM understands
DIFFICULTY_MAP = {
    "Easy":   "simple factual questions a beginner can answer",
    "Medium": "moderate analytical questions requiring understanding",
    "Hard":   "deep inferential or critical-thinking questions",
}


def _parse_pairs(raw: str, limit: int) -> list:
    """
    Parse Q&A pairs from the LLM's raw text output.

    The LLM is supposed to return JSON like:
      [{"q": "Question?", "a": "Answer."}]

    But LLMs sometimes add extra text, markdown fences (```), or
    return plain text instead of JSON. This function handles all cases:
      1. Try to parse as JSON first (clean approach)
      2. If JSON fails, fall back to line-by-line parsing
         looking for lines that start with Q1. / A1. / Question: etc.
    """
    # ── Attempt 1: JSON parsing
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r"\[.*\]", clean, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            return [
                {
                    "q": str(p.get("q", "")).strip(),
                    "a": str(p.get("a", "")).strip(),
                }
                for p in parsed
                if isinstance(p, dict) and p.get("q") and p.get("a")
            ][:limit]
    except Exception:
        pass

    # ── Attempt 2: Line-by-line fallback
    pairs, lines, i = [], raw.strip().split("\n"), 0
    while i < len(lines) and len(pairs) < limit:
        line = lines[i].strip()
        if re.match(r"^[Qq\d][\.\):]", line) or line.lower().startswith("question"):
            q = re.sub(r"^[Qq\d\.\)\:\s]+", "", line).strip()
            a = ""
            if i + 1 < len(lines):
                a = re.sub(r"^[Aa\d\.\)\:\s]+", "", lines[i + 1]).strip()
                i += 2
            else:
                i += 1
            if q:
                pairs.append({"q": q, "a": a})
        else:
            i += 1

    return pairs[:limit]


def _generate_single_batch(
    batch_size: int,
    context: str,
    difficulty: str,
    language: str,
    offset: int,
) -> list:
    """
    Ask the LLM to generate ONE small batch of Q&A pairs.
    """
    lang_note = (
        "" if language == "English"
        else f"Write ALL questions and answers in {language}."
    )
    difficulty_desc = DIFFICULTY_MAP.get(difficulty, "moderate")

    prompt = (
        f"You are an expert educator. Read the document excerpt below and generate "
        f"EXACTLY {batch_size} question-answer pairs numbered {offset + 1} to {offset + batch_size}.\n"
        f"Difficulty: {difficulty_desc}\n"
        f"{lang_note}\n"
        f"Rules:\n"
        f"- Each question must be answerable from the text\n"
        f"- Answers: 1-3 sentences, concise\n"
        f"- Output ONLY a JSON array, no markdown, no extra text\n"
        f"- Format: [{{\"q\": \"Question?\", \"a\": \"Answer.\"}}]\n\n"
        f"Document excerpt:\n{context}\n\n"
        f"JSON output ({batch_size} pairs):"
    )

    llm = _get_llm()
    raw = llm.invoke(prompt)
    raw_output = raw.content if hasattr(raw, "content") else raw

    return _parse_pairs(raw_output, batch_size)


def generate_qa(n: int, difficulty: str, language: str, raw_chunks: list) -> list:
    """
    Generate n Q&A pairs from the loaded document chunks.

    Uses batching to work around small context window models (e.g. llama2 7b).
    Shows a progress bar in Streamlit during generation.

    Args:
        n          : how many Q&A pairs to generate
        difficulty : "Easy", "Medium", or "Hard"
        language   : "English", "Hindi", "Spanish", etc.
        raw_chunks : list of text strings (document chunks from ChromaDB)

    Returns:
        list of dicts: [{"q": "...", "a": "..."}, ...]
    """
    if not raw_chunks:
        return []

    ctx_tokens, _max_pairs, batch_size, _emoji = get_model_limits()
    context_chars = min(1800, max(800, ctx_tokens // 4))

    all_pairs     = []
    batch_num     = 0
    total_batches = (n + batch_size - 1) // batch_size

    progress = st.progress(0, text="Starting generation...")

    while len(all_pairs) < n:
        remaining       = n - len(all_pairs)
        current_batch_n = min(batch_size, remaining)

        chunk_offset = (batch_num * 3) % max(1, len(raw_chunks))
        rotated      = raw_chunks[chunk_offset:] + raw_chunks[:chunk_offset]
        context      = "\n\n".join(rotated[:6])[:context_chars]

        progress_pct = min(batch_num / total_batches, 0.95)
        progress.progress(
            progress_pct,
            text=f"Generating batch {batch_num + 1}/{total_batches} "
                 f"({len(all_pairs)}/{n} pairs done)...",
        )

        batch_pairs = _generate_single_batch(
            current_batch_n, context, difficulty, language, len(all_pairs)
        )

        if not batch_pairs:
            break

        all_pairs.extend(batch_pairs)
        batch_num += 1

    progress.progress(1.0, text=f"Done! {len(all_pairs)} pairs generated.")

    # Remove duplicate questions
    seen, unique = set(), []
    for pair in all_pairs:
        key = pair["q"].lower()[:60]
        if key not in seen:
            seen.add(key)
            unique.append(pair)

    return unique[:n]