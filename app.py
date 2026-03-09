# ============================================================
# app.py  —  Streamlit UI (the only file you run)
# ============================================================
# This file has ONE job: draw the interface and respond to clicks.
# All the real logic lives in other files:
#
#   config.py              → settings (model name, chunk size, etc.)
#   rag/loader.py          → loading PDF / TXT / URL
#   rag/vectorstore.py     → saving & searching document chunks
#   rag/chain.py           → answering questions (RAG pipeline)
#   rag/generator.py       → generating Q&A pairs automatically
#   export/pdf_export.py   → downloading Q&A as PDF
#   export/docx_export.py  → downloading Q&A as Word file
#
# To run:
#   ollama serve           ← in one terminal
#   streamlit run app.py   ← in another terminal

import os
import datetime
import streamlit as st
import streamlit.components.v1 as components

# ── Suppress noisy TensorFlow startup messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ.setdefault("USER_AGENT", "DocMindAI/1.0")

# ── Import our own modules
from config import OLLAMA_MODEL, get_model_limits

from rag.loader      import load_pdf, load_txt, load_url
from rag.vectorstore import get_embeddings, get_vectorstore, store_documents
from rag.chain       import ask
from rag.generator   import generate_qa

from export.pdf_export  import export_pdf
from export.docx_export import export_docx


# ════════════════════════════════════════════
# PAGE SETUP
# ════════════════════════════════════════════
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Inject all CSS via zero-height iframe so Streamlit doesn't parse it as markdown
_CSS = (
    "<link href='https://fonts.googleapis.com/css2?"
    "family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600"
    "&display=swap' rel='stylesheet'>"
    "<style>"
    "section[data-testid='stSidebar']{display:none!important}"
    "button[kind='header']{display:none!important}"
    "#MainMenu,footer,header{display:none!important}"
    ".stApp{background:#080c14}"
    ".block-container{max-width:820px!important;padding:0 24px 120px!important}"
    "html,body{font-family:'Sora',sans-serif;color:#e2e8f0}"
    ".hero{text-align:center;padding:52px 0 28px;position:relative}"
    ".hero-glow{position:absolute;top:0;left:50%;transform:translateX(-50%);"
    "width:320px;height:120px;"
    "background:radial-gradient(ellipse,rgba(0,229,255,.18) 0%,transparent 70%);"
    "pointer-events:none}"
    ".hero h1{font-size:2.4rem;font-weight:700;letter-spacing:-1px;"
    "background:linear-gradient(135deg,#00e5ff 0%,#7c3aed 100%);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 6px}"
    ".hero p{color:#64748b;font-size:.95rem;margin:0}"
    ".section-card{background:#0f1623;border:1px solid #1e293b;border-radius:16px;"
    "padding:22px 24px;margin-bottom:18px;transition:border-color .2s}"
    ".section-card:hover{border-color:#00e5ff44}"
    ".section-title{font-size:.7rem;font-weight:600;letter-spacing:.12em;"
    "color:#00e5ff;text-transform:uppercase;margin-bottom:14px}"
    "[data-testid='stFileUploader']{background:#080c14!important;"
    "border:1.5px dashed #1e3a5f!important;border-radius:10px!important;padding:8px!important}"
    "[data-testid='stFileUploader']:hover{border-color:#00e5ff88!important}"
    "input[type='text'],.stTextInput input{background:#0a1020!important;"
    "border:1.5px solid #1e293b!important;border-radius:10px!important;"
    "color:#e2e8f0!important;font-family:'JetBrains Mono',monospace!important;"
    "font-size:.9rem!important;padding:12px 16px!important}"
    "input[type='text']:focus,.stTextInput input:focus{border-color:#00e5ff!important;"
    "box-shadow:0 0 0 3px rgba(0,229,255,.1)!important}"
    ".stButton>button{background:linear-gradient(135deg,#00c4d4,#7c3aed)!important;"
    "border:none!important;border-radius:10px!important;color:#fff!important;"
    "font-family:'Sora',sans-serif!important;font-weight:600!important;"
    "font-size:.88rem!important;padding:10px 20px!important;"
    "transition:opacity .2s,transform .15s!important}"
    ".stButton>button:hover{opacity:.88!important;transform:translateY(-1px)!important}"
    ".stDownloadButton>button{background:#0f1623!important;"
    "border:1.5px solid #00e5ff55!important;border-radius:10px!important;"
    "color:#00e5ff!important;font-family:'Sora',sans-serif!important;"
    "font-weight:600!important;font-size:.85rem!important;"
    "padding:9px 18px!important;transition:all .2s!important}"
    ".stDownloadButton>button:hover{background:#00e5ff14!important;border-color:#00e5ff!important}"
    ".stSlider [data-testid='stThumbValue']{color:#00e5ff!important}"
    "[data-baseweb='select']>div{background:#0a1020!important;"
    "border:1.5px solid #1e293b!important;border-radius:10px!important;color:#e2e8f0!important}"
    ".msg-wrap{display:flex;align-items:flex-start;gap:12px;margin:12px 0}"
    ".msg-wrap.user{flex-direction:row-reverse}"
    ".avatar{width:34px;height:34px;border-radius:50%;display:flex;"
    "align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;font-weight:700}"
    ".avatar.user-av{background:linear-gradient(135deg,#7c3aed,#a78bfa)}"
    ".avatar.bot-av{background:linear-gradient(135deg,#00c4d4,#0ea5e9);"
    "font-family:'JetBrains Mono',monospace;font-size:.75rem}"
    ".bubble{max-width:78%;padding:13px 17px;border-radius:16px;font-size:.93rem;line-height:1.65}"
    ".bubble.user-b{background:linear-gradient(135deg,#312e81,#1e1b4b);"
    "border:1px solid #4338ca55;border-bottom-right-radius:4px;color:#c7d2fe}"
    ".bubble.bot-b{background:#0f1623;border:1px solid #1e293b;"
    "border-bottom-left-radius:4px;color:#cbd5e1}"
    ".qa-card{background:#0a1020;border:1px solid #1e293b;border-left:3px solid #00e5ff;"
    "border-radius:12px;padding:16px 20px;margin:10px 0;transition:border-color .2s}"
    ".qa-card:hover{border-color:#00e5ff}"
    ".qa-num{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#00e5ff;"
    "letter-spacing:.1em;margin-bottom:5px}"
    ".qa-q{font-weight:600;font-size:.97rem;color:#f1f5f9;margin-bottom:8px}"
    ".qa-a{font-size:.9rem;color:#94a3b8;line-height:1.65}"
    ".src-pill{display:inline-block;background:#0a1020;border:1px solid #1e293b;"
    "border-radius:20px;padding:3px 12px;font-size:.75rem;color:#64748b;margin:3px 4px 3px 0}"
    ".doc-pill{display:inline-block;background:#0a2218;border:1px solid #064e3b44;"
    "border-radius:20px;padding:4px 12px;font-size:.78rem;color:#34d399;margin:3px 4px 3px 0}"
    ".fancy-hr{border:none;height:1px;"
    "background:linear-gradient(90deg,transparent,#1e293b,transparent);margin:24px 0}"
    ".status-badge{display:inline-flex;align-items:center;gap:6px;background:#0a2218;"
    "border:1px solid #065f4655;border-radius:20px;padding:4px 12px;font-size:.78rem;color:#34d399}"
    ".status-dot{width:7px;height:7px;border-radius:50%;background:#34d399;"
    "animation:pulse 2s infinite}"
    "@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}"
    ".stSpinner>div{border-top-color:#00e5ff!important}"
    "[data-baseweb='tab-list']{background:#0f1623!important;border-radius:12px!important;"
    "padding:4px!important;border:1px solid #1e293b!important;gap:2px!important}"
    "[data-baseweb='tab']{border-radius:9px!important;color:#64748b!important;"
    "font-family:'Sora',sans-serif!important;font-size:.87rem!important;"
    "font-weight:600!important;padding:8px 20px!important}"
    "[aria-selected='true'][data-baseweb='tab']{"
    "background:linear-gradient(135deg,#00c4d455,#7c3aed55)!important;color:#e2e8f0!important}"
    "[data-testid='stExpander']{background:#0a1020!important;"
    "border:1px solid #1e293b!important;border-radius:10px!important}"
    "[data-testid='stAlert']{border-radius:10px!important}"
    "</style>"
)
components.html(_CSS, height=0)


# ════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════
# Streamlit re-runs the entire script on every interaction.
# st.session_state is the only way to keep data between re-runs.
for key, default in [
    ("chat_history", []),      # list of {role, content, sources} dicts
    ("vectorstore",  None),    # ChromaDB connection object
    ("docs_loaded",  []),      # list of loaded file/URL names (for display)
    ("raw_chunks",   []),      # list of raw text strings (for Q&A generation)
    ("generated_qa", []),      # list of {q, a} dicts from last generation
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ════════════════════════════════════════════
# STARTUP: reconnect to existing ChromaDB
# ════════════════════════════════════════════
# If the user uploaded documents in a previous session,
# the ChromaDB folder still exists on disk.
# We reconnect to it silently so the app feels stateful.
emb = get_embeddings()
if st.session_state.vectorstore is None and os.path.exists("./chroma_db"):
    try:
        st.session_state.vectorstore = get_vectorstore(emb)
    except Exception:
        pass


# ════════════════════════════════════════════
# HERO HEADER
# ════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-glow"></div>
  <h1>🧠 DocMind AI</h1>
  <p>Upload documents · Ask questions · Auto-generate Q&amp;A</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════
tab_chat, tab_docs, tab_qa = st.tabs([
    "💬  Chat",
    "📂  Documents",
    "⚡  Auto Q&A",
])


# ────────────────────────────────────────────
# TAB 1: DOCUMENTS
# ────────────────────────────────────────────
with tab_docs:

    # Show which files are currently loaded
    if st.session_state.docs_loaded:
        badges = " ".join([
            f'<span class="doc-pill">{d}</span>'
            for d in st.session_state.docs_loaded
        ])
        st.markdown(f"""
        <div style="margin-bottom:18px">
          <span class="status-badge">
            <span class="status-dot"></span>Knowledge base active
          </span>
          &nbsp;&nbsp;{badges}
        </div>
        """, unsafe_allow_html=True)

    # ── Upload PDF
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Upload PDF</div>', unsafe_allow_html=True)
    pdf_file = st.file_uploader("PDF", type=["pdf"], key="pdf_up", label_visibility="collapsed")
    if pdf_file and st.button("Add PDF →", key="add_pdf"):
        with st.spinner("Chunking PDF..."):
            try:
                docs = load_pdf(pdf_file)
                vs, chunks = store_documents(docs, pdf_file.name, emb)
                st.session_state.vectorstore = vs
                st.session_state.raw_chunks.extend(chunks)
                st.session_state.docs_loaded.append(f"📄 {pdf_file.name}")
                st.success(f"✅ Added {len(chunks)} chunks from {pdf_file.name}")
            except Exception as e:
                st.error(str(e))
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Upload TXT
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 Upload TXT</div>', unsafe_allow_html=True)
    txt_file = st.file_uploader("TXT", type=["txt"], key="txt_up", label_visibility="collapsed")
    if txt_file and st.button("Add TXT →", key="add_txt"):
        with st.spinner("Processing..."):
            try:
                docs = load_txt(txt_file)
                vs, chunks = store_documents(docs, txt_file.name, emb)
                st.session_state.vectorstore = vs
                st.session_state.raw_chunks.extend(chunks)
                st.session_state.docs_loaded.append(f"📝 {txt_file.name}")
                st.success(f"✅ Added {len(chunks)} chunks from {txt_file.name}")
            except Exception as e:
                st.error(str(e))
    st.markdown('</div>', unsafe_allow_html=True)

    # ── URL
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌐 Add Website URL</div>', unsafe_allow_html=True)
    url_input = st.text_input("URL", placeholder="https://example.com/article", label_visibility="collapsed")
    if st.button("Scrape & Add →", key="add_url") and url_input:
        with st.spinner("Scraping webpage..."):
            try:
                domain = url_input.split("/")[2]
                docs   = load_url(url_input)
                vs, chunks = store_documents(docs, domain, emb)
                st.session_state.vectorstore = vs
                st.session_state.raw_chunks.extend(chunks)
                st.session_state.docs_loaded.append(f"🌐 {domain}")
                st.success(f"✅ Scraped {len(chunks)} chunks from {domain}")
            except Exception as e:
                st.error(str(e))
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Clear all
    if st.session_state.docs_loaded:
        st.markdown('<hr class="fancy-hr">', unsafe_allow_html=True)
        if st.button("🗑️ Clear all documents & reset", key="clear_all"):
            import shutil
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
            st.session_state.update(
                vectorstore=None, docs_loaded=[], chat_history=[],
                raw_chunks=[], generated_qa=[]
            )
            st.success("Cleared!")
            st.rerun()


# ────────────────────────────────────────────
# TAB 2: CHAT
# ────────────────────────────────────────────
with tab_chat:

    if not st.session_state.docs_loaded:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
          <div style="font-size:3rem;margin-bottom:12px">📂</div>
          <div style="color:#475569;font-size:1rem">
            Go to the <b style="color:#00e5ff">Documents</b> tab and upload something first
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Render chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-wrap user">
                  <div class="avatar user-av">V</div>
                  <div class="bubble user-b">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-wrap">
                  <div class="avatar bot-av">AI</div>
                  <div class="bubble bot-b">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)

                # Show source documents in a collapsible section
                if msg.get("sources"):
                    with st.expander("📚 View sources", expanded=False):
                        seen = set()
                        for src in msg["sources"]:
                            name  = src.metadata.get("source_name", src.metadata.get("source", "Unknown"))
                            page  = src.metadata.get("page", "")
                            key   = f"{name}-{page}"
                            if key not in seen:
                                seen.add(key)
                                label = name + (f" · p.{page + 1}" if page != "" else "")
                                st.markdown(
                                    f'<span class="src-pill">📌 {label}</span>',
                                    unsafe_allow_html=True,
                                )
                                st.caption(src.page_content[:260] + "...")

        st.markdown('<hr class="fancy-hr">', unsafe_allow_html=True)

        # ── Question input
        col_input, col_send = st.columns([6, 1])
        with col_input:
            question = st.text_input(
                "q",
                placeholder="Ask anything about your documents...",
                label_visibility="collapsed",
                key="chat_input",
            )
        with col_send:
            send = st.button("Send →", use_container_width=True)

        # ── Handle send
        if send and question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                try:
                    result = ask(question, st.session_state.vectorstore)
                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": f"❌ {e} — Is Ollama running? Run: `ollama serve`",
                        "sources": [],
                    })
            st.rerun()

        # ── Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()


# ────────────────────────────────────────────
# TAB 3: AUTO Q&A GENERATOR
# ────────────────────────────────────────────
with tab_qa:

    if not st.session_state.raw_chunks:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
          <div style="font-size:3rem;margin-bottom:12px">⚡</div>
          <div style="color:#475569;font-size:1rem">
            Upload a document first from the <b style="color:#00e5ff">Documents</b> tab
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚙️ Generator Settings</div>', unsafe_allow_html=True)

        # ── Model info banner (color-coded by context size)
        ctx_tok, max_pairs, batch_sz, emoji = get_model_limits()
        color_bg  = {"🟢": "#064e3b", "🟡": "#78350f", "🔴": "#7f1d1d"}.get(emoji, "#1e293b")
        color_fg  = {"🟢": "#34d399", "🟡": "#fbbf24", "🔴": "#f87171"}.get(emoji, "#e2e8f0")
        warn_text = {
            "🔴": "⚠️ Very limited context — keep pairs low",
            "🟡": "ℹ️ Medium context — batching enabled automatically",
            "🟢": "✅ Large context — high pair counts supported",
        }.get(emoji, "")

        st.markdown(f"""
        <div style='background:{color_bg};border:1px solid {color_fg}33;border-radius:10px;
             padding:10px 16px;margin-bottom:14px;display:flex;align-items:center;gap:10px'>
          <span style='font-size:1.3rem'>{emoji}</span>
          <div>
            <span style='color:{color_fg};font-weight:700;font-size:.9rem'>
              Model: {OLLAMA_MODEL}
            </span>
            <span style='color:#64748b;font-size:.8rem;margin-left:10px'>
              {ctx_tok:,} token context · max {max_pairs} pairs recommended
            </span>
            <br>
            <span style='color:{color_fg};font-size:.78rem'>{warn_text}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            num_q = st.slider(
                "Number of Q&A pairs",
                min_value=1,
                max_value=max_pairs,
                value=min(5, max_pairs),
                step=1,
                help=f"Max {max_pairs} recommended for {OLLAMA_MODEL} ({ctx_tok:,} token context)",
            )
            if num_q == max_pairs:
                st.caption(f"⚠️ At model limit — may be slow ({(num_q + batch_sz - 1) // batch_sz} batches)")
        with col2:
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
        with col3:
            language = st.selectbox(
                "Language",
                ["English", "Hindi", "Spanish", "French", "German", "Arabic"],
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Generate button
        if st.button(f"⚡ Generate {num_q} Q&A Pairs", use_container_width=True):
            try:
                pairs = generate_qa(
                    n=num_q,
                    difficulty=difficulty,
                    language=language,
                    raw_chunks=st.session_state.raw_chunks,
                )
                if pairs:
                    st.session_state.generated_qa = pairs
                    st.success(f"✅ {len(pairs)} pairs generated!")
                else:
                    st.error("Could not parse output. Try again.")
            except Exception as e:
                st.error(f"❌ {e} — Is Ollama running? `ollama serve`")

        # ── Display generated Q&A cards
        if st.session_state.generated_qa:
            pairs = st.session_state.generated_qa
            st.markdown('<hr class="fancy-hr">', unsafe_allow_html=True)
            st.markdown(
                f"<div style='color:#64748b;font-size:.8rem;margin-bottom:12px'>"
                f"GENERATED — {len(pairs)} PAIRS · {difficulty.upper()} · {language.upper()}"
                f"</div>",
                unsafe_allow_html=True,
            )

            for i, p in enumerate(pairs, 1):
                st.markdown(f"""
                <div class="qa-card">
                  <div class="qa-num">Q {i:02d}</div>
                  <div class="qa-q">{p['q']}</div>
                  <div class="qa-a">💡 {p['a']}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Export buttons
            st.markdown('<hr class="fancy-hr">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📥 Download</div>', unsafe_allow_html=True)

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            col_pdf, col_docx, col_txt = st.columns(3)

            with col_pdf:
                try:
                    st.download_button(
                        "⬇️ PDF", export_pdf(pairs),
                        file_name=f"qa_{ts}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"PDF error: {e}")

            with col_docx:
                try:
                    st.download_button(
                        "⬇️ DOCX", export_docx(pairs),
                        file_name=f"qa_{ts}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"DOCX error: {e}")

            with col_txt:
                txt_content = "\n\n".join([
                    f"Q{i}. {p['q']}\nA{i}. {p['a']}"
                    for i, p in enumerate(pairs, 1)
                ])
                st.download_button(
                    "⬇️ TXT", txt_content.encode(),
                    file_name=f"qa_{ts}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            if st.button("🔄 Regenerate", key="regen"):
                st.session_state.generated_qa = []
                st.rerun()
