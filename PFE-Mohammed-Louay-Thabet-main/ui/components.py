# ui/components.py
#
# Reusable UI components for the Olive RAG Streamlit dashboard.
# Handles: multilingual labels, Arabic RTL injection, RAG data viewer,
# ingestion panel, thinking block display, LLM selector, chat history.

import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

# ── Chat history storage ────────────────────────────────────────────────────
_HISTORY_DIR = Path(__file__).resolve().parents[1] / "data" / "chat_history"


def _history_dir() -> Path:
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return _HISTORY_DIR


def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:6]


def save_current_session():
    """Persist the current chat session to disk as a JSON file."""
    messages = st.session_state.get("messages", [])
    if not messages:
        return
    session_id = st.session_state.get("session_id")
    if not session_id:
        return
    title = next(
        (m["content"][:70] for m in messages if m["role"] == "user"),
        "Untitled",
    )
    payload = {
        "id":         session_id,
        "title":      title,
        "backend":    st.session_state.get("backend", "llama3.1:8b"),
        "lang":       st.session_state.get("lang", "en"),
        "saved_at":   datetime.now().isoformat(timespec="seconds"),
        "messages":   messages,
    }
    dest = _history_dir() / f"{session_id}.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_all_sessions() -> list[dict]:
    sessions = []
    for f in sorted(_history_dir().glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_file"] = str(f)
            sessions.append(data)
        except Exception:
            pass
    return sessions


def _delete_session(session_id: str):
    f = _history_dir() / f"{session_id}.json"
    if f.exists():
        f.unlink()


# ---------------------------------------------------------------
# Translations — English / French / Arabic
# ---------------------------------------------------------------

TRANSLATIONS = {
    "en": {
        "page_title":         "Olive RAG — Médenine",
        "main_title":         "Olive Production Assistant",
        "caption":            "Ask questions about olive production, climate, and agronomy in Médenine.",
        "sidebar_title":      "Olive RAG System",
        "sidebar_region":     "Region: Médenine, Tunisia",
        "sources_title":      "Data sources",
        "src_timescale":      "Monthly climate + production (TimescaleDB)",
        "src_qdrant":         "Research documents (Qdrant)",
        "src_web":            "Web knowledge base (Qdrant)",
        "examples_title":     "Example questions",
        "clear_chat":         "Clear chat",
        "new_session":        "New session",
        "chat_placeholder":   "Ask about olive production in Médenine...",
        "route_label":        "Route",
        "thinking_label":     "Reasoning (DeepSeek-R1)",
        "thinking_toggle":    "Show chain-of-thought",
        "ingestion_title":    "Data Ingestion",
        "run_ingestion":      "Run Full Ingestion Pipeline",
        "ingestion_running":  "Running ingestion...",
        "ingestion_done":     "Ingestion complete",
        "add_urls":           "Add URLs to scrape (one per line)",
        "rag_viewer_title":   "RAG Knowledge Base",
        "view_rag":           "Browse Ingested Data",
        "filter_lang":        "Filter by language",
        "filter_type":        "Filter by content type",
        "filter_domain":      "Filter by domain",
        "chunk_preview":      "Chunk preview",
        "total_chunks":       "Total chunks in Qdrant",
        "no_chunks":          "No chunks found. Run ingestion first.",
        "llm_backend":        "LLM Backend",
        "history_title":      "Chat History",
        "no_history":         "No saved sessions yet. Start a conversation to save history automatically.",
        "load_session":       "Load",
        "delete_session":     "Delete",
        "history_qa_count":   "Q&A",
        "history_resume":     "Resume",
        "examples": [
            "What was the olive production in 2019?",
            "Which years had the lowest production?",
            "How do olive trees resist drought?",
            "What is the impact of warm winters on olive oil quality?",
            "Compare production between 2010 and 2020",
        ],
    },
    "fr": {
        "page_title":         "Olive RAG — Médenine",
        "main_title":         "Assistant Production Oléicole",
        "caption":            "Posez vos questions sur la production d'olives, le climat et l'agronomie à Médenine.",
        "sidebar_title":      "Système RAG Olive",
        "sidebar_region":     "Région : Médenine, Tunisie",
        "sidebar_data":       "Données : 1990–2025",
        "sources_title":      "Sources de données",
        "src_timescale":      "Climat mensuel + production (TimescaleDB)",
        "src_qdrant":         "Documents de recherche (Qdrant)",
        "src_web":            "Base de connaissances web (Qdrant)",
        "examples_title":     "Questions exemples",
        "clear_chat":         "Effacer la conversation",
        "new_session":        "Nouvelle session",
        "chat_placeholder":   "Posez votre question sur les olives à Médenine...",
        "route_label":        "Itinéraire",
        "thinking_label":     "Raisonnement (DeepSeek-R1)",
        "thinking_toggle":    "Afficher le raisonnement",
        "ingestion_title":    "Ingestion des données",
        "run_ingestion":      "Lancer le pipeline d'ingestion",
        "ingestion_running":  "Ingestion en cours...",
        "ingestion_done":     "Ingestion terminée",
        "add_urls":           "Ajouter des URLs à scraper (une par ligne)",
        "rag_viewer_title":   "Base de connaissances RAG",
        "view_rag":           "Parcourir les données ingérées",
        "filter_lang":        "Filtrer par langue",
        "filter_type":        "Filtrer par type de contenu",
        "filter_domain":      "Filtrer par domaine",
        "chunk_preview":      "Aperçu du chunk",
        "total_chunks":       "Total de chunks dans Qdrant",
        "no_chunks":          "Aucun chunk trouvé. Lancez l'ingestion d'abord.",
        "llm_backend":        "Modèle LLM",
        "history_title":      "Historique",
        "no_history":         "Aucune session sauvegardée. Commencez une conversation pour sauvegarder l'historique.",
        "load_session":       "Charger",
        "delete_session":     "Supprimer",
        "history_qa_count":   "Q&R",
        "history_resume":     "Reprendre",
        "examples": [
            "Quelle était la production d'olives en 2019 ?",
            "Quelles années ont eu la plus faible production ?",
            "Comment les oliviers résistent-ils à la sécheresse ?",
            "Quel est l'impact des hivers chauds sur la qualité de l'huile d'olive ?",
            "Comparer la production entre 2010 et 2020",
        ],
    },
    "ar": {
        "page_title":         "نظام RAG للزيتون — مدنين",
        "main_title":         "مساعد إنتاج الزيتون",
        "caption":            "اطرح أسئلتك حول إنتاج الزيتون والمناخ والزراعة في مدنين.",
        "sidebar_title":      "نظام RAG للزيتون",
        "sidebar_region":     "المنطقة: مدنين، تونس",
        "sidebar_data":       "البيانات: 1990–2025",
        "sources_title":      "مصادر البيانات",
        "src_timescale":      "المناخ الشهري + الإنتاج (TimescaleDB)",
        "src_qdrant":         "وثائق البحث (Qdrant)",
        "src_web":            "قاعدة المعرفة على الويب (Qdrant)",
        "examples_title":     "أسئلة مثالية",
        "clear_chat":         "مسح المحادثة",
        "new_session":        "جلسة جديدة",
        "chat_placeholder":   "اسأل عن إنتاج الزيتون في مدنين...",
        "route_label":        "المسار",
        "thinking_label":     "التفكير (DeepSeek-R1)",
        "thinking_toggle":    "عرض سلسلة التفكير",
        "ingestion_title":    "استيعاب البيانات",
        "run_ingestion":      "تشغيل خط أنابيب الاستيعاب",
        "ingestion_running":  "جارٍ الاستيعاب...",
        "ingestion_done":     "اكتمل الاستيعاب",
        "add_urls":           "أضف روابط للاستخراج (رابط في كل سطر)",
        "rag_viewer_title":   "قاعدة معرفة RAG",
        "view_rag":           "تصفح البيانات المستوعبة",
        "filter_lang":        "تصفية حسب اللغة",
        "filter_type":        "تصفية حسب نوع المحتوى",
        "filter_domain":      "تصفية حسب المجال",
        "chunk_preview":      "معاينة المقطع",
        "total_chunks":       "إجمالي المقاطع في Qdrant",
        "no_chunks":          "لم يتم العثور على مقاطع. شغّل الاستيعاب أولاً.",
        "llm_backend":        "نموذج اللغة",
        "history_title":      "سجل المحادثات",
        "no_history":         "لا توجد جلسات محفوظة. ابدأ محادثة لحفظ السجل تلقائياً.",
        "load_session":       "تحميل",
        "delete_session":     "حذف",
        "history_qa_count":   "س&ج",
        "history_resume":     "استئناف",
        "examples": [
            "ما كان إنتاج الزيتون عام 2019؟",
            "أي السنوات شهدت أدنى إنتاج؟",
            "كيف تتحمل أشجار الزيتون الجفاف؟",
            "ما تأثير الشتاء الدافئ على جودة زيت الزيتون؟",
            "قارن الإنتاج بين 2010 و2020",
        ],
    },
}


def t(key: str) -> str:
    """Returns the translated string for the current session language."""
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


# ---------------------------------------------------------------
# Arabic RTL CSS injection
# ---------------------------------------------------------------

def inject_rtl_css():
    """Injects CSS to support Arabic right-to-left layout."""
    st.markdown(
        """
        <style>
        /* RTL layout for Arabic */
        .rtl-text {
            direction: rtl;
            text-align: right;
            font-family: 'Segoe UI', Tahoma, 'Arabic Typesetting', sans-serif;
        }
        /* Make chat messages RTL when language is Arabic */
        .arabic-mode .stChatMessage {
            direction: rtl;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def maybe_rtl(text: str) -> str:
    """Wraps text in RTL div for Arabic, returns plain text otherwise."""
    if st.session_state.get("lang") == "ar":
        return f'<div class="rtl-text">{text}</div>'
    return text


# ---------------------------------------------------------------
# Tunisian regions
# ---------------------------------------------------------------

TUNISIAN_REGIONS = [
    "Ariana", "Béja", "Ben Arous", "Bizerte", "Gabès", "Gafsa",
    "Jendouba", "Kairouan", "Kasserine", "Kébili", "Le Kef", "Mahdia",
    "Manouba", "Médenine", "Monastir", "Nabeul", "Sfax", "Sidi Bouzid",
    "Siliana", "Sousse", "Tataouine", "Tozeur", "Tunis", "Zaghouan",
]

DEFAULT_REGION = "Médenine"


# ---------------------------------------------------------------
# Language selector
# ---------------------------------------------------------------

def language_selector():
    """Renders the language selector in the sidebar."""
    lang_options = {"English": "en", "Français": "fr", "العربية": "ar"}
    selected_label = st.sidebar.selectbox(
        "🌐 Language / Langue / اللغة",
        options=list(lang_options.keys()),
        index=list(lang_options.values()).index(
            st.session_state.get("lang", "en")
        ),
    )
    st.session_state["lang"] = lang_options[selected_label]


def region_selector():
    """Renders the Tunisian region selector in the sidebar."""
    default_idx = TUNISIAN_REGIONS.index(DEFAULT_REGION)
    selected = st.sidebar.selectbox(
        "📍 Region / Région / المنطقة",
        options=TUNISIAN_REGIONS,
        index=TUNISIAN_REGIONS.index(
            st.session_state.get("region", DEFAULT_REGION)
        ) if st.session_state.get("region", DEFAULT_REGION) in TUNISIAN_REGIONS else default_idx,
        key="region_selector",
    )
    st.session_state["region"] = selected


# ---------------------------------------------------------------
# DeepSeek-R1 thinking block
# ---------------------------------------------------------------

def render_thinking(thinking: str):
    """Shows the chain-of-thought in a collapsible expander."""
    if thinking:
        with st.expander(f"💭 {t('thinking_toggle')}", expanded=False):
            st.caption(t("thinking_label"))
            st.markdown(f"```\n{thinking[:2000]}{'...' if len(thinking) > 2000 else ''}\n```")


# ---------------------------------------------------------------
# RAG Knowledge Base Viewer
# ---------------------------------------------------------------

def render_rag_viewer():
    """
    Fetches chunks from Qdrant and displays them in a searchable,
    filterable dataframe. Shows source, language, content type,
    and a text preview for each chunk.
    """
    from src.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

    st.subheader(f"📚 {t('rag_viewer_title')}")

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        info   = client.get_collection(QDRANT_COLLECTION)
        total  = info.points_count

        st.metric(t("total_chunks"), total)

        if total == 0:
            st.info(t("no_chunks"))
            return

        # Sidebar filters
        col1, col2, col3 = st.columns(3)
        with col1:
            lang_filter = st.selectbox(
                t("filter_lang"),
                ["All", "en", "fr", "ar"],
                key="rag_lang_filter",
            )
        with col2:
            type_filter = st.selectbox(
                t("filter_type"),
                ["All", "text", "table", "web"],
                key="rag_type_filter",
            )
        with col3:
            domain_filter = st.selectbox(
                t("filter_domain"),
                ["All", "olive_agronomy", "related_domain"],
                key="rag_domain_filter",
            )

        # Build Qdrant filter
        must = []
        if lang_filter != "All":
            must.append(FieldCondition(key="language", match=MatchValue(value=lang_filter)))
        if type_filter != "All":
            must.append(FieldCondition(key="content_type", match=MatchValue(value=type_filter)))
        if domain_filter != "All":
            must.append(FieldCondition(key="domain", match=MatchValue(value=domain_filter)))

        q_filter = Filter(must=must) if must else None

        # Scroll through Qdrant points (max 200 for display)
        points, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=q_filter,
            limit=200,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            st.info(t("no_chunks"))
            return

        # Build dataframe for display
        rows = []
        for p in points:
            pl = p.payload
            rows.append({
                "ID":       p.id,
                "Source":   pl.get("source_pdf", "?"),
                "Page":     pl.get("page", "?"),
                "Domain":   pl.get("domain", "?"),
                "Language": pl.get("language", "?"),
                "Type":     pl.get("content_type", "?"),
                "Chars":    pl.get("char_count", len(pl.get("text", ""))),
                "Preview":  (pl.get("text", "")[:120] + "...") if len(pl.get("text", "")) > 120 else pl.get("text", ""),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=400)

        # Detail view: click-to-expand a chunk
        st.markdown("---")
        st.markdown(f"**{t('chunk_preview')}**")
        chunk_idx = st.number_input(
            "Chunk index (0-based from table above)",
            min_value=0,
            max_value=max(0, len(rows) - 1),
            value=0,
            key="chunk_detail_idx",
        )
        if rows:
            selected = points[chunk_idx]
            st.markdown(f"**Source:** {selected.payload.get('source_pdf', '?')} | "
                        f"**Page:** {selected.payload.get('page', '?')} | "
                        f"**Lang:** {selected.payload.get('language', '?')}")
            if st.session_state.get("lang") == "ar":
                st.markdown(
                    f'<div class="rtl-text">{selected.payload.get("text", "")}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.text_area(
                    "Full text",
                    value=selected.payload.get("text", ""),
                    height=200,
                    disabled=True,
                    key="chunk_detail_text",
                )

    except Exception as e:
        st.error(f"Could not connect to Qdrant: {e}")
        st.info("Make sure `docker-compose up -d qdrant` is running.")


# ---------------------------------------------------------------
# Ingestion Panel
# ---------------------------------------------------------------

def render_ingestion_panel():
    """
    Full ingestion panel:
      - PDF upload  → saved to data/raw/, extracted to olive_chunks.jsonl
      - Excel upload → saved to data/raw/ (production/climate/rainfall)
      - URL input   → web scraper
      - Single button triggers the complete 4-step pipeline
    """
    import tempfile
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[1]
    RAW_DIR  = BASE_DIR / "data" / "raw"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    st.subheader(f"⚙️ {t('ingestion_title')}")

    # ── PDF upload ──────────────────────────────────────────────
    st.markdown("#### 📄 Upload PDF Documents")
    pdf_files = st.file_uploader(
        "Drop PDF files here (research papers, reports, agronomy docs)",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    saved_pdf_paths = []
    if pdf_files:
        for uf in pdf_files:
            dest = RAW_DIR / uf.name
            dest.write_bytes(uf.read())
            saved_pdf_paths.append(dest)
        st.caption(f"✅ {len(pdf_files)} PDF(s) ready: {', '.join(f.name for f in pdf_files)}")

    # ── Excel upload ─────────────────────────────────────────────
    st.markdown("#### 📊 Upload Raw Data Files (Excel)")
    excel_files = st.file_uploader(
        "Drop Excel files here (production.xlsx, climate.xlsx, rainfall.xlsx)",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        key="excel_uploader",
    )
    if excel_files:
        for uf in excel_files:
            dest = RAW_DIR / uf.name
            dest.write_bytes(uf.read())
        st.caption(f"✅ {len(excel_files)} Excel file(s) saved to data/raw/")

    # ── URL input ────────────────────────────────────────────────
    st.markdown("#### 🌐 Web URLs to Scrape")
    urls_text = st.text_area(
        t("add_urls"),
        height=80,
        placeholder="https://www.fao.org/olive/...\nhttps://www.onagri.nat.tn/...",
        key="scraper_urls_input",
    )
    extra_urls = [u.strip() for u in urls_text.splitlines() if u.strip().startswith("http")]
    if extra_urls:
        st.caption(f"✅ {len(extra_urls)} URL(s) will be scraped")

    st.markdown("---")

    if st.button(f"🚀 {t('run_ingestion')}", use_container_width=True, type="primary"):
        from src.ingestion.pipeline import run_full_pipeline

        step_labels = {
            "pdf_extraction": "📄 PDF Extraction",
            "tabular":        "🗄️ Tabular Data (Excel → DB)",
            "documents":      "📚 Documents → Qdrant",
            "web_scraper":    "🌐 Web Scraper",
        }

        with st.status(t("ingestion_running"), expanded=True) as status_box:
            results = run_full_pipeline(
                extra_urls=extra_urls if extra_urls else None,
                pdf_paths=saved_pdf_paths if saved_pdf_paths else None,
            )
            status_box.update(label=t("ingestion_done"), state="complete")

        for r in results:
            key  = r.get("step", "unknown")
            ok   = r.get("ok", False)
            msg  = r.get("message", "")
            label = step_labels.get(key, key)
            if ok:
                st.success(f"✅ **{label}**: {msg}")
            else:
                st.warning(f"⚠️ **{label}**: {msg}")


# ---------------------------------------------------------------
# LLM Backend Selector
# ---------------------------------------------------------------

def render_llm_selector():
    """Sidebar selectbox for choosing the active LLM backend."""
    from src.agent import LLM_BACKENDS, DEFAULT_BACKEND

    options = list(LLM_BACKENDS.keys())
    labels  = {k: v["label"] for k, v in LLM_BACKENDS.items()}

    current = st.session_state.get("backend", DEFAULT_BACKEND)
    if current not in options:
        current = DEFAULT_BACKEND

    selected = st.sidebar.selectbox(
        f"🤖 {t('llm_backend')}",
        options=options,
        format_func=lambda k: labels[k],
        index=options.index(current),
        key="llm_backend_selector",
    )
    st.session_state["backend"] = selected

    # Show a small badge: local GPU vs cloud
    backend_type = LLM_BACKENDS[selected]["type"]
    badge = "🖥️ Local GPU" if backend_type == "ollama" else "☁️ Cloud API"
    st.sidebar.caption(badge)


# ---------------------------------------------------------------
# Chat History Panel
# ---------------------------------------------------------------

def render_history_panel():
    """
    Full-page history browser.
    Lists all saved sessions; each row has Load and Delete buttons.
    Loading a session restores messages + backend into the current session.
    """
    sessions = _load_all_sessions()

    if not sessions:
        st.info(t("no_history"))
        return

    st.markdown(f"**{len(sessions)}** session(s) saved")
    st.markdown("---")

    for session in sessions:
        sid        = session.get("id", "?")
        title      = session.get("title", "Untitled")
        backend    = session.get("backend", "?")
        lang       = session.get("lang", "en").upper()
        saved_at   = session.get("saved_at", sid)
        messages   = session.get("messages", [])
        n_qa       = sum(1 for m in messages if m["role"] == "user")

        col_info, col_load, col_del = st.columns([6, 1, 1])

        with col_info:
            st.markdown(
                f"**{title}**  \n"
                f"`{backend}` &nbsp;·&nbsp; {lang} &nbsp;·&nbsp; "
                f"{n_qa} {t('history_qa_count')} &nbsp;·&nbsp; {saved_at}"
            )

        with col_load:
            if st.button(t("load_session"), key=f"load_{sid}", use_container_width=True):
                st.session_state["messages"] = messages
                st.session_state["backend"]  = session.get("backend", "llama3.1:8b")
                st.session_state["lang"]     = session.get("lang", "en")
                st.session_state["session_id"] = sid
                st.rerun()

        with col_del:
            if st.button("🗑️", key=f"del_{sid}", use_container_width=True,
                         help=t("delete_session")):
                _delete_session(sid)
                # If deleting the active session, reset to a fresh one
                if st.session_state.get("session_id") == sid:
                    st.session_state["messages"]   = []
                    st.session_state["session_id"] = new_session_id()
                st.rerun()

        st.divider()
