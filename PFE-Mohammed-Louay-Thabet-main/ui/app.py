# ui/app.py
#
# Enhanced Streamlit dashboard for the Olive RAG System.
# Features:
#   - Chat with DeepSeek-R1 (via Ollama)
#   - Multilingual UI: English / French / Arabic (RTL)
#   - Data Ingestion tab: trigger full pipeline from browser
#   - RAG Viewer tab: browse Qdrant knowledge base
#   - DeepSeek-R1 chain-of-thought display (collapsible)

import sys
from pathlib import Path

# Make src/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from src.agent import ask
from ui.components import (
    TRANSLATIONS,
    DEFAULT_REGION,
    new_session_id,
    save_current_session,
    t,
    language_selector,
    region_selector,
    inject_rtl_css,
    render_thinking,
    render_rag_viewer,
    render_ingestion_panel,
    render_llm_selector,
    render_history_panel,
)

# ---------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Olive RAG — Médenine",
    page_icon="🫒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "region" not in st.session_state:
    st.session_state["region"] = DEFAULT_REGION
if "backend" not in st.session_state:
    st.session_state["backend"] = "llama3.1:8b"
if "session_id" not in st.session_state:
    st.session_state["session_id"] = new_session_id()

# ---------------------------------------------------------------
# Inject global CSS
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Olive-themed accent colours */
    :root {
        --olive-dark:  #4a5e1f;
        --olive-mid:   #6b8e23;
        --olive-light: #b8cc6e;
        --sand:        #f5f0e8;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: var(--sand);
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, var(--olive-dark), var(--olive-mid));
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    /* Chat message bubbles */
    .stChatMessage {
        border-radius: 12px;
    }

    /* Route badge */
    .route-badge {
        display: inline-block;
        background: var(--olive-light);
        color: var(--olive-dark);
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* RTL support for Arabic */
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, 'Arabic Typesetting', sans-serif;
        line-height: 1.8;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

inject_rtl_css()

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    language_selector()
    region_selector()
    st.divider()
    render_llm_selector()
    st.divider()

    st.markdown(f"## 🫒 {t('sidebar_title')}")
    region = st.session_state.get("region", DEFAULT_REGION)
    st.markdown(f"**📍 {region}, Tunisia**")
    st.divider()

    st.markdown(f"### 📊 {t('sources_title')}")
    st.markdown(f"- 🗄️ {t('src_timescale')}")
    st.markdown(f"- 📄 {t('src_qdrant')}")
    st.markdown(f"- 🌐 {t('src_web')}")
    st.divider()

    st.markdown(f"### 💡 {t('examples_title')}")
    for ex in TRANSLATIONS[st.session_state.get("lang", "en")]["examples"]:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["example_query"] = ex

    st.divider()
    col_clear, col_new = st.sidebar.columns(2)
    with col_clear:
        if st.button(f"🗑️ {t('clear_chat')}", use_container_width=True):
            st.session_state["messages"]   = []
            st.session_state["session_id"] = new_session_id()
            st.rerun()
    with col_new:
        if st.button(f"✨ {t('new_session')}", use_container_width=True):
            st.session_state["messages"]   = []
            st.session_state["session_id"] = new_session_id()
            st.rerun()

    # System status
    st.divider()
    st.markdown("### 🔧 System")
    from src.config import QDRANT_COLLECTION
    st.caption(f"Session: `{st.session_state['session_id']}`")
    st.caption(f"Collection: `{QDRANT_COLLECTION}`")

# ---------------------------------------------------------------
# Main area — tab layout
# ---------------------------------------------------------------
tab_chat, tab_ingest, tab_data, tab_history = st.tabs([
    "💬 Chat",
    f"⚙️ {t('ingestion_title')}",
    f"📚 {t('rag_viewer_title')}",
    f"📜 {t('history_title')}",
])

# ============================================================
# TAB 1 — Chat
# ============================================================
with tab_chat:
    st.markdown(
        f"""
        <div class="main-header">
            <h2 style="margin:0">🫒 {t('main_title')}</h2>
            <p style="margin:4px 0 0 0; opacity:0.85">{t('caption')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if st.session_state.get("lang") == "ar" and msg["role"] == "assistant":
                st.markdown(
                    f'<div class="rtl-text">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg["content"])

            if msg.get("thinking"):
                render_thinking(msg["thinking"])

            if msg["role"] == "assistant":
                st.markdown(
                    '<span class="route-badge">🔍 BM25 + Semantic + Metadata</span>',
                    unsafe_allow_html=True,
                )

    # Handle example button clicks
    query = None
    if "example_query" in st.session_state:
        query = st.session_state.pop("example_query")

    # Chat input
    user_input = st.chat_input(t("chat_placeholder"))
    if user_input:
        query = user_input

    # Process query
    if query:
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            if st.session_state.get("lang") == "ar":
                st.markdown(
                    f'<div class="rtl-text">{query}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(query)

        # Generate answer
        with st.chat_message("assistant"):
            backend = st.session_state.get("backend", "llama3.1:8b")
            with st.spinner("🧠 Thinking... (BM25 + Semantic + Metadata)"):
                try:
                    result   = ask(query, backend=backend)
                    answer   = result["answer"]
                    thinking = result.get("thinking", "")
                except Exception as e:
                    answer   = f"Error: {str(e)}"
                    thinking = ""

            # Display answer
            if st.session_state.get("lang") == "ar":
                st.markdown(
                    f'<div class="rtl-text">{answer}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(answer)

            # Show thinking block if present
            render_thinking(thinking)

            # Search method badge
            st.markdown(
                '<span class="route-badge">🔍 BM25 + Semantic + Metadata</span>',
                unsafe_allow_html=True,
            )

        # Save to history
        st.session_state["messages"].append({
            "role":     "assistant",
            "content":  answer,
            "thinking": thinking,
        })
        save_current_session()

# ============================================================
# TAB 2 — Data Ingestion
# ============================================================
with tab_ingest:
    st.markdown(
        f"""
        <div class="main-header">
            <h2 style="margin:0">⚙️ {t('ingestion_title')}</h2>
            <p style="margin:4px 0 0 0; opacity:0.85">
                Trigger the full data ingestion pipeline from this panel.
                Steps: Excel → TimescaleDB | PDF chunks → Qdrant | Web scraping → Qdrant
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        render_ingestion_panel()

    with col_right:
        st.markdown("### 📋 Pipeline Steps")
        st.markdown("""
        | Step | Input | Output |
        |---|---|---|
        | **1. Tabular** | `data/raw/*.xlsx` | TimescaleDB `olive_monthly` |
        | **2. Documents** | `data/processed/olive_chunks.jsonl` | Qdrant collection |
        | **3. Web scraper** | URLs (defaults + custom) | Qdrant collection |
        """)
        st.markdown("---")
        st.markdown("### 📁 Notebooks (manual run)")
        st.markdown("""
        For step-by-step exploration, open these notebooks in Jupyter:
        - `notebooks/01_tabular_processing.ipynb`
        - `notebooks/02_document_ingestion.ipynb`
        - `notebooks/03_web_scraper.ipynb`

```bash
        jupyter notebook notebooks/
```
        """)

# ============================================================
# TAB 3 — RAG Data Viewer
# ============================================================
with tab_data:
    st.markdown(
        f"""
        <div class="main-header">
            <h2 style="margin:0">📚 {t('rag_viewer_title')}</h2>
            <p style="margin:4px 0 0 0; opacity:0.85">
                Browse all text chunks stored in the Qdrant vector database.
                Filter by language, content type, and domain.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_rag_viewer()

# ============================================================
# TAB 4 — Chat History
# ============================================================
with tab_history:
    st.markdown(
        f"""
        <div class="main-header">
            <h2 style="margin:0">📜 {t('history_title')}</h2>
            <p style="margin:4px 0 0 0; opacity:0.85">
                Browse, resume, or delete your past chat sessions.
                Sessions are saved automatically after each answer.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_history_panel()