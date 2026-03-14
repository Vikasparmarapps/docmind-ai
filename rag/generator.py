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
    # ── Attempt 1: Clean markdown fences and try direct JSON parse
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            return [
                {"q": str(p.get("q", "")).strip(), "a": str(p.get("a", "")).strip()}
                for p in parsed
                if isinstance(p, dict) and p.get("q") and p.get("a")
            ][:limit]
    except Exception:
        pass

    # ── Attempt 2: Find JSON array anywhere in the text (handles extra prose)
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r"\[.*?\]", clean, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [
                    {"q": str(p.get("q", "")).strip(), "a": str(p.get("a", "")).strip()}
                    for p in parsed
                    if isinstance(p, dict) and p.get("q") and p.get("a")
                ][:limit]
    except Exception:
        pass

    # ── Attempt 3: Find individual JSON objects {q:..., a:...} scattered in text
    try:
        objects = re.findall(r'\{[^{}]*"q"\s*:\s*"[^"]+?"[^{}]*"a"\s*:\s*"[^"]+?"[^{}]*\}', raw, re.DOTALL)
        if objects:
            pairs = []
            for obj in objects[:limit]:
                parsed = json.loads(obj)
                if parsed.get("q") and parsed.get("a"):
                    pairs.append({"q": parsed["q"].strip(), "a": parsed["a"].strip()})
            if pairs:
                return pairs
    except Exception:
        pass

    # ── Attempt 4: Line-by-line fallback for plain text output
    # Handles formats like:
    #   Q1: What is...   A1: The answer...
    #   Question: ...    Answer: ...
    #   1. What is...    Answer: ...
    pairs, lines, i = [], raw.strip().split("\n"), 0
    while i < len(lines) and len(pairs) < limit:
        line = lines[i].strip()
        is_question = (
            re.match(r"^[Qq]\d*[\.\):\s]", line) or
            re.match(r"^\d+[\.\)]\s", line) or
            line.lower().startswith("question")
        )
        if is_question:
            q = re.sub(r"^[Qq\d\.\)\:\s]+", "", line).strip()
            q = re.sub(r"^[Qq]uestion\s*\d*\s*[:]\s*", "", q, flags=re.IGNORECASE).strip()
            a = ""
            # Look for answer on next non-empty line
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line:
                    a = re.sub(r"^[Aa\d\.\)\:\s]+", "", next_line).strip()
                    a = re.sub(r"^[Aa]nswer\s*\d*\s*[:]\s*", "", a, flags=re.IGNORECASE).strip()
                    i = j + 1
                    break
                j += 1
            else:
                i += 1
            if q:
                pairs.append({"q": q, "a": a})
        else:
            i += 1

    if pairs:
        return pairs[:limit]

    # ── Attempt 5: Last resort — split by numbered patterns
    chunks = re.split(r'\n\s*\d+[\.\)]\s+', raw)
    pairs = []
    for chunk in chunks[1:limit+1]:   # skip first empty split
        lines_c = [l.strip() for l in chunk.strip().split("\n") if l.strip()]
        if len(lines_c) >= 2:
            pairs.append({"q": lines_c[0], "a": " ".join(lines_c[1:])})
        elif len(lines_c) == 1:
            pairs.append({"q": lines_c[0], "a": ""})

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
        f"Generate EXACTLY {batch_size} question-answer pairs from the document below.\n"
        f"Difficulty: {difficulty_desc}\n"
        f"{lang_note}\n"
        f"STRICT RULES:\n"
        f"- Output ONLY a valid JSON array. Nothing else. No intro text, no explanation.\n"
        f"- Start your response with [ and end with ]\n"
        f"- Each item: {{\"q\": \"Question here?\", \"a\": \"Answer here.\"}}\n"
        f"- Questions must be answerable from the text\n"
        f"- Answers: 1-3 sentences\n\n"
        f"Document:\n{context}\n\n"
        f"JSON array only:"
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