"""
DocChat RAG — Polished UI
"""

from __future__ import annotations
import os
import time
import shutil
import pickle
import streamlit as st
from dotenv import load_dotenv
from rag_chain import RAGChain
from agent import AgenticRAG
from ingest import ingest_file, list_ingested
from document_loader import list_downloads
from config import DOWNLOADS_DIR, STORES_DIR

load_dotenv()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ Page Config ━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="DocChat RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ API Key Validation ━━━━━━━━━━━━━━━━━━━━

_REQUIRED_KEYS = ("GROQ_API_KEY", "GOOGLE_API_KEY")
_missing_keys = [k for k in _REQUIRED_KEYS if not os.getenv(k)]
if _missing_keys:
    st.error(
        "**Missing API keys:** "
        + ", ".join(f"`{k}`" for k in _missing_keys)
        + "\n\nCreate a `.env` file in the project root (copy `.env.example`) "
        "and add your keys:\n\n"
        "- Groq: https://console.groq.com/keys\n"
        "- Google AI Studio: https://aistudio.google.com/app/apikey\n\n"
        "Then restart the app."
    )
    st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ CSS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
/* ── Layout ── */
.block-container { max-width: 880px; padding-top: 1.5rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* ── Brand ── */
.brand {
    text-align: center;
    padding: 1.2rem 0 0.6rem;
}
.brand-name {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #7c5cfc, #b44aff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.brand-sub {
    font-size: 0.7rem;
    color: #484f58;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Section headers in sidebar ── */
.sidebar-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #484f58;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 1rem 0 0.4rem;
}

/* ── Hero / landing ── */
.hero-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem 2rem;
    text-align: center;
}
.hero-icon {
    width: 80px; height: 80px;
    border-radius: 24px;
    background: linear-gradient(135deg, #7c5cfc, #b44aff);
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(124,92,252,0.3);
}
.hero-icon svg { width: 40px; height: 40px; fill: white; }
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #e6edf3;
    letter-spacing: -1px;
    margin-bottom: 0.5rem;
}
.hero-desc {
    font-size: 1rem;
    color: #484f58;
    max-width: 420px;
    line-height: 1.6;
}

/* ── Stat pills ── */
.stats-row {
    display: flex;
    gap: 12px;
    justify-content: center;
    margin-top: 2rem;
}
.stat-pill {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 0.9rem 1.4rem;
    text-align: center;
    min-width: 120px;
}
.stat-pill .val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e6edf3;
}
.stat-pill .lbl {
    font-size: 0.68rem;
    color: #484f58;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 2px;
}
.stat-pill.accent {
    border-color: #7c5cfc;
    background: linear-gradient(135deg, rgba(124,92,252,0.1), rgba(180,74,255,0.1));
}
.stat-pill.accent .val { color: #b44aff; }

/* ── Feature cards on landing ── */
.features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 2.5rem;
    max-width: 650px;
}
.feat-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.2s;
}
.feat-card:hover { border-color: #7c5cfc; }
.feat-card .feat-icon {
    font-size: 1.3rem;
    margin-bottom: 0.4rem;
}
.feat-card .feat-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 0.2rem;
}
.feat-card .feat-desc {
    font-size: 0.72rem;
    color: #484f58;
    line-height: 1.4;
}

/* ── Doc header when chatting ── */
.doc-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.8rem 0;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid #21262d;
}
.doc-avatar {
    width: 44px; height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #7c5cfc, #b44aff);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; color: white; font-weight: 700;
    flex-shrink: 0;
}
.doc-info .doc-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e6edf3;
}
.doc-info .doc-meta {
    font-size: 0.75rem;
    color: #484f58;
}

/* ── Chat messages ── */
.stChatMessage { border-radius: 14px !important; }

/* ── Verification ── */
.verif {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 8px;
}
.verif.ok {
    background: rgba(63,185,80,0.12);
    color: #3fb950;
    border: 1px solid rgba(63,185,80,0.2);
}
.verif.warn {
    background: rgba(210,153,34,0.12);
    color: #d29922;
    border: 1px solid rgba(210,153,34,0.2);
}

/* ── Source cards in expander ── */
.src-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 8px;
    font-size: 0.82rem;
    color: #8b949e;
    line-height: 1.55;
}
.src-card .src-page {
    display: inline-block;
    background: #7c5cfc;
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    margin-bottom: 6px;
    letter-spacing: 0.5px;
}

/* ── Timing ── */
.timing-pill {
    display: inline-block;
    font-size: 0.72rem;
    color: #484f58;
    margin-top: 6px;
}

/* ── Formats badges on landing ── */
.format-row {
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-top: 1.5rem;
}
.format-badge {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 6px 16px;
    font-size: 0.78rem;
    color: #8b949e;
    font-weight: 500;
}

/* ── Pipeline steps in sidebar ── */
.pipe-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    font-size: 0.78rem;
    color: #8b949e;
}
.pipe-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #7c5cfc;
    flex-shrink: 0;
}

/* ── Agent steps ── */
.agent-step {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 6px;
    font-size: 0.8rem;
}
.agent-step .step-tool {
    display: inline-block;
    background: #7c5cfc;
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    margin-right: 8px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.agent-step .step-query {
    color: #e6edf3;
    font-weight: 500;
}
.agent-step .step-preview {
    color: #484f58;
    font-size: 0.72rem;
    margin-top: 4px;
    line-height: 1.4;
}

/* ── Mode toggle ── */
.mode-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 8px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.mode-badge.standard {
    background: rgba(63,185,80,0.12);
    color: #3fb950;
    border: 1px solid rgba(63,185,80,0.2);
}
.mode-badge.agentic {
    background: rgba(124,92,252,0.15);
    color: #b44aff;
    border: 1px solid rgba(124,92,252,0.3);
}

/* hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None
if "rag" not in st.session_state:
    st.session_state.rag = None
if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = "Standard"


def pretty_name(name: str) -> str:
    """Turn 'becoming-supernatural-free-pdf-download' → 'Becoming Supernatural'"""
    n = name.replace("-", " ").replace("_", " ")
    # Strip common junk suffixes
    for junk in ["free pdf download", "free pdf", "pdf download", "pdf", "download", "free", "doc"]:
        n = n.removesuffix(f" {junk}")
    return n.strip().title()


def get_chunk_count(doc_name: str) -> int:
    bm25_path = os.path.join(STORES_DIR, doc_name, "bm25_chunks.pkl")
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            return len(pickle.load(f))
    return 0


def render_sources_html(docs) -> str:
    html = ""
    for doc in docs:
        page = doc.metadata.get("page", 0) + 1
        text = doc.page_content[:300].replace("\n", " ").replace("<", "&lt;")
        html += f'<div class="src-card"><span class="src-page">PAGE {page}</span><div>{text}...</div></div>'
    return html


def render_agent_steps_html(steps: list) -> str:
    """Render agent tool-call steps as styled HTML."""
    if not steps:
        return ""
    html = ""
    for i, step in enumerate(steps, 1):
        tool = step["tool"].replace("_", " ").title()
        query = step["input"].replace("<", "&lt;")
        preview = step["output"][:200].replace("<", "&lt;").replace("\n", " ")
        html += f'''<div class="agent-step">
            <span class="step-tool">{tool}</span>
            <span class="step-query">{query}</span>
            <div class="step-preview">{preview}...</div>
        </div>'''
    return html


def render_verif_html(verification) -> str:
    if not verification:
        return ""
    if verification["verified"]:
        return '<div class="verif ok">Verified against sources</div>'
    return f'<div class="verif warn">{verification["issues"]}</div>'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ Sidebar ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:

    # Brand
    st.markdown("""
    <div class="brand">
        <div class="brand-name">DocChat RAG</div>
        <div class="brand-sub">Advanced Retrieval Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Add Document ──
    st.markdown('<div class="sidebar-label">Add Document</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload",
        type=["pdf", "txt", "docx", "csv"],
        label_visibility="collapsed",
    )
    if uploaded:
        if st.button("Ingest uploaded file", use_container_width=True, type="primary"):
            temp_path = os.path.join("/tmp", uploaded.name)
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded.getvalue())
                with st.spinner("Ingesting..."):
                    doc_name = ingest_file(temp_path)
                st.toast(f"'{pretty_name(doc_name)}' is ready!", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    with st.expander("Browse Downloads"):
        downloads_readable = os.access(DOWNLOADS_DIR, os.R_OK)
        available_files = list_downloads(DOWNLOADS_DIR) if downloads_readable else []
        if available_files:
            selected_file = st.selectbox(
                "f", options=[""] + available_files,
                format_func=lambda x: "Select a file..." if x == "" else x,
                label_visibility="collapsed",
            )
            if selected_file and st.button("Ingest", use_container_width=True):
                path = os.path.join(DOWNLOADS_DIR, selected_file)
                try:
                    with st.spinner("Ingesting..."):
                        doc_name = ingest_file(path)
                    st.toast(f"'{pretty_name(doc_name)}' is ready!", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
        elif not downloads_readable:
            st.caption(
                "macOS is blocking access to ~/Downloads. "
                "Grant your terminal app Full Disk Access in "
                "System Settings → Privacy & Security, or use "
                "**Upload** / **Paste file path** above."
            )
        else:
            st.caption("No supported files found.")

    with st.expander("Paste file path"):
        custom_path = st.text_input("path", label_visibility="collapsed", placeholder="/path/to/file.pdf")
        if custom_path and st.button("Ingest path", use_container_width=True):
            if os.path.isfile(custom_path):
                try:
                    with st.spinner("Ingesting..."):
                        doc_name = ingest_file(custom_path)
                    st.toast("Ready!", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
            else:
                st.error("File not found.")

    st.markdown("---")

    # ── Documents ──
    ingested = list_ingested()

    st.markdown('<div class="sidebar-label">Your Documents</div>', unsafe_allow_html=True)

    if ingested:
        display_names = {d: pretty_name(d) for d in ingested}
        chosen = st.selectbox(
            "doc",
            options=ingested,
            format_func=lambda x: display_names[x],
            label_visibility="collapsed",
        )

        chunks = get_chunk_count(chosen)
        st.caption(f"{chunks} chunks indexed")

        # ── Mode Toggle ──
        st.markdown('<div class="sidebar-label">Mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            "mode",
            options=["Standard", "Agentic"],
            horizontal=True,
            label_visibility="collapsed",
            index=0 if st.session_state.rag_mode == "Standard" else 1,
        )
        if mode != st.session_state.rag_mode:
            st.session_state.rag_mode = mode
            st.session_state.rag = None  # force reload with new mode
            st.session_state.messages = []

        if mode == "Standard":
            st.caption("Single-pass retrieval with verification.")
        else:
            st.caption("AI agent reasons & searches multiple times.")

        if chosen != st.session_state.active_doc:
            st.session_state.active_doc = chosen
            st.session_state.messages = []
            st.session_state.rag = None

        if st.session_state.rag is None and chosen:
            with st.spinner(f"Loading pipeline..."):
                if st.session_state.rag_mode == "Agentic":
                    st.session_state.rag = AgenticRAG(chosen)
                else:
                    st.session_state.rag = RAGChain(chosen)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.rag:
                    st.session_state.rag.clear_history()
                st.rerun()
        with c2:
            if st.button("Delete", use_container_width=True):
                shutil.rmtree(os.path.join(STORES_DIR, chosen), ignore_errors=True)
                st.session_state.active_doc = None
                st.session_state.rag = None
                st.session_state.messages = []
                st.rerun()
    else:
        st.caption("No documents yet.")

    st.markdown("---")

    # ── Pipeline ──
    st.markdown('<div class="sidebar-label">Pipeline</div>', unsafe_allow_html=True)
    if st.session_state.rag_mode == "Agentic":
        steps = [
            "Agent reasoning loop",
            "Semantic search tool",
            "Keyword search tool",
            "Page lookup tool",
            "Multi-step planning",
            "Adaptive re-searching",
            "Grounded answer synthesis",
        ]
    else:
        steps = [
            "HyDE hypothetical embedding",
            "Multi-query generation",
            "BM25 keyword search",
            "Vector semantic search",
            "Reciprocal rank fusion",
            "Cross-encoder reranking",
            "LLM answer generation",
            "Hallucination verification",
        ]
    for s in steps:
        st.markdown(f'<div class="pipe-step"><div class="pipe-dot"></div>{s}</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ Main Area ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if not st.session_state.rag:

    # ── Landing Page ──
    st.markdown("""
    <div class="hero-container">
        <div class="hero-icon">
            <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm-1 2l5 5h-5V4zM6 20V4h5v7h7v9H6z"/><path d="M8 14h8v2H8zm0-3h8v2H8z"/></svg>
        </div>
        <div class="hero-title">Chat with any document</div>
        <div class="hero-desc">
            Upload a PDF, TXT, DOCX, or CSV file and ask questions.
            Two modes: Standard RAG with verification, or Agentic RAG with multi-step reasoning.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    num_docs = len(ingested) if ingested else 0
    total_chunks = sum(get_chunk_count(d) for d in ingested) if ingested else 0

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-pill">
            <div class="val">{num_docs}</div>
            <div class="lbl">Documents</div>
        </div>
        <div class="stat-pill">
            <div class="val">{total_chunks:,}</div>
            <div class="lbl">Chunks</div>
        </div>
        <div class="stat-pill accent">
            <div class="val">2</div>
            <div class="lbl">RAG Modes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Features
    st.markdown("""
    <div class="features" style="grid-template-columns: repeat(4, 1fr); max-width: 860px;">
        <div class="feat-card">
            <div class="feat-icon">HyDE</div>
            <div class="feat-title">Hypothetical Answers</div>
            <div class="feat-desc">Generates a hypothetical answer to embed for better semantic matching</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">BM25</div>
            <div class="feat-title">Hybrid Search</div>
            <div class="feat-desc">Combines keyword search with vector search for complete coverage</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">CE</div>
            <div class="feat-title">Cross-Encoder</div>
            <div class="feat-desc">Reranks results with a cross-encoder for precision scoring</div>
        </div>
        <div class="feat-card" style="border-color: rgba(124,92,252,0.3); background: linear-gradient(135deg, rgba(124,92,252,0.08), rgba(180,74,255,0.08));">
            <div class="feat-icon" style="color: #b44aff;">Agent</div>
            <div class="feat-title">Agentic RAG</div>
            <div class="feat-desc">AI agent reasons, plans, and searches multiple times for complex questions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Formats
    st.markdown("""
    <div class="format-row">
        <div class="format-badge">PDF</div>
        <div class="format-badge">TXT</div>
        <div class="format-badge">DOCX</div>
        <div class="format-badge">CSV</div>
    </div>
    """, unsafe_allow_html=True)

else:

    # ── Document header ──
    name = pretty_name(st.session_state.active_doc)
    initial = name[0] if name else "D"
    chunks = get_chunk_count(st.session_state.active_doc)
    msgs = len(st.session_state.messages)
    is_agentic = st.session_state.rag_mode == "Agentic"
    mode_cls = "agentic" if is_agentic else "standard"
    mode_label = "Agentic" if is_agentic else "Standard"

    st.markdown(f"""
    <div class="doc-header">
        <div class="doc-avatar">{initial}</div>
        <div class="doc-info">
            <div class="doc-title">{name} <span class="mode-badge {mode_cls}">{mode_label}</span></div>
            <div class="doc-meta">{chunks:,} chunks &nbsp;&middot;&nbsp; {msgs} messages</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chat history ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("verification"):
                st.markdown(render_verif_html(msg["verification"]), unsafe_allow_html=True)
            if msg.get("timing"):
                st.markdown(f'<div class="timing-pill">{msg["timing"]}</div>', unsafe_allow_html=True)
            if msg.get("agent_steps"):
                with st.expander(f"Agent reasoning — {len(msg['agent_steps'])} tool calls"):
                    st.markdown(render_agent_steps_html(msg["agent_steps"]), unsafe_allow_html=True)
            if msg.get("source_docs"):
                with st.expander(f"{len(msg['source_docs'])} sources"):
                    st.markdown(render_sources_html(msg["source_docs"]), unsafe_allow_html=True)

    # ── Chat input ──
    if question := st.chat_input("Ask anything about this document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            t0 = time.time()

            if is_agentic:
                # ── Agentic mode ──
                with st.status("Agent is thinking...", expanded=True) as status:
                    result = st.session_state.rag.ask(question)

                    # Show each tool call as it happened
                    for i, step in enumerate(result["steps"], 1):
                        tool_name = step["tool"].replace("_", " ").title()
                        st.write(f"**{i}. {tool_name}** → `{step['input']}`")

                    searches = result["num_searches"]
                    status.update(
                        label=f"Agent completed — {searches} tool call{'s' if searches != 1 else ''}",
                        state="complete",
                        expanded=False,
                    )

                answer = result["answer"]
                st.markdown(answer)

                elapsed = time.time() - t0
                timing_str = f"{elapsed:.1f}s"
                st.markdown(f'<div class="timing-pill">Answered in {timing_str} &middot; {searches} searches</div>', unsafe_allow_html=True)

                if result["steps"]:
                    with st.expander(f"Agent reasoning — {len(result['steps'])} tool calls"):
                        st.markdown(render_agent_steps_html(result["steps"]), unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "agent_steps": result["steps"],
                    "timing": f"Answered in {timing_str} · {searches} searches",
                })

            else:
                # ── Standard mode ──
                placeholder = st.empty()
                full_response = []
                docs = None
                verification = None

                for chunk, source_docs, verif in st.session_state.rag.stream(question):
                    if chunk is not None:
                        full_response.append(chunk)
                        placeholder.markdown("".join(full_response) + " **|**")
                    if source_docs is not None:
                        docs = source_docs
                    if verif is not None:
                        verification = verif

                elapsed = time.time() - t0
                answer = "".join(full_response)
                placeholder.markdown(answer)

                if verification:
                    st.markdown(render_verif_html(verification), unsafe_allow_html=True)

                timing_str = f"{elapsed:.1f}s"
                st.markdown(f'<div class="timing-pill">Answered in {timing_str}</div>', unsafe_allow_html=True)

                if docs:
                    with st.expander(f"{len(docs)} sources"):
                        st.markdown(render_sources_html(docs), unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "source_docs": docs,
                    "verification": verification,
                    "timing": f"Answered in {timing_str}",
                })
