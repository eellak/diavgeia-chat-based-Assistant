"""
DiavgeiaAssistant — Modern UI (v2)

A polished Streamlit interface for the RAG-based Diaugeia assistant.
The original streamlit_app_demo.py is preserved unchanged as a checkpoint.
"""
import sys
from pathlib import Path

# Add parent directory to Python path to import diaygeia package
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import uuid
import time
from datetime import datetime
import json
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from diaygeia.bot.response import DiaygeiaBot, get_user_history
from diaygeia import grounding

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

_user_avatar_path = SCRIPT_DIR / "assets" / "user_icon.png"
_bot_avatar_path = SCRIPT_DIR / "assets" / "bot_icon.png"
USER_AVATAR = str(_user_avatar_path) if _user_avatar_path.exists() else "👤"
BOT_AVATAR = str(_bot_avatar_path) if _bot_avatar_path.exists() else "🤖"

CONVERSATIONS_FILE = SCRIPT_DIR / "conversations.json"

# Theme palette — single source of truth
PRIMARY = "#4F46E5"
PRIMARY_DARK = "#3730A3"
ACCENT = "#7C3AED"
SOFT_BG = "#F5F7FB"
CARD_BG = "#FFFFFF"
TEXT_MAIN = "#1F2937"
TEXT_MUTED = "#6B7280"
BORDER = "#E5E7EB"

# Suggested starter prompts shown on the empty chat state.
# Retrieval is BM25 (lexical) over ~32k random Diavgeia decisions, so prompts
# must work as keyword matches — no numeric ranges, no semantic intent, no
# topic-specific assumptions (the corpus is not curated by ministry/sector).
SUGGESTED_PROMPTS = [
    ("📑", "Εξήγησέ μου τι είναι ο αριθμός ΑΔΑ και πώς λειτουργεί"),
    ("🔎", "Δείξε μου την απόφαση με ΑΔΑ ΕΒΚΕΟΞΤΖ-ΑΓΞ"),
    ("📝", "Βρες μια απόφαση ανάληψης υποχρέωσης και συνόψισέ την"),
    ("🏛️", "Αναζήτησε αποφάσεις σχετικές με δημοτικό λιμενικό ταμείο"),
]

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="DiavgeiaAssistant — Έξυπνη αναζήτηση στη Διαύγεια",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "DiavgeiaAssistant — RAG assistant over Greek Diavgeia open data.",
        "Get Help": "https://diavgeia.gov.gr",
    },
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
    /* Import a nicer font (falls back gracefully) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background: linear-gradient(180deg, {SOFT_BG} 0%, #ffffff 100%);
    }}

    /* --- Hero header --- */
    .hero {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.18);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }}
    .hero::after {{
        content: '';
        position: absolute;
        right: -40px; top: -40px;
        width: 220px; height: 220px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }}
    .hero h1 {{
        color: white !important;
        font-size: 28px;
        margin: 0 0 6px 0;
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    .hero p {{
        margin: 0;
        opacity: 0.92;
        font-size: 15px;
    }}
    .hero .pill {{
        display: inline-block;
        background: rgba(255,255,255,0.18);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 500;
        margin-bottom: 10px;
    }}

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {{
        background: #FAFBFD;
        border-right: 1px solid {BORDER};
    }}
    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 20px;
        font-weight: 700;
        color: {PRIMARY};
        margin: 4px 0 18px 0;
    }}
    .sidebar-brand-logo {{
        width: 36px; height: 36px;
        background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        color: white; font-size: 18px;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }}
    .nav-section-label {{
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {TEXT_MUTED};
        margin: 16px 4px 6px 4px;
    }}

    /* --- Buttons --- */
    .stButton>button {{
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        border: 1px solid {BORDER} !important;
    }}
    .stButton>button[kind="primary"] {{
        background: linear-gradient(135deg, {PRIMARY}, {ACCENT}) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
    }}
    .stButton>button[kind="primary"]:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.35) !important;
    }}
    .stButton>button[kind="secondary"]:hover {{
        border-color: {PRIMARY} !important;
        color: {PRIMARY} !important;
        background: #F4F4FB !important;
    }}

    /* --- Chat input --- */
    .stChatInput {{
        background: white !important;
        border-radius: 14px !important;
    }}
    .stChatInput > div {{
        background: white !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06) !important;
    }}
    .stChatInput textarea {{
        background: white !important;
        color: {TEXT_MAIN} !important;
        border: 1.5px solid {BORDER} !important;
        border-radius: 12px !important;
        font-size: 15px !important;
    }}
    .stChatInput textarea:focus, .stChatInput textarea:focus-visible {{
        border-color: {PRIMARY} !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
    }}
    .stChatInput button {{
        background: linear-gradient(135deg, {PRIMARY}, {ACCENT}) !important;
        border: none !important;
        color: white !important;
        border-radius: 10px !important;
    }}
    .stChatInput button:hover {{
        transform: scale(1.05);
    }}

    /* --- Chat messages --- */
    .stChatMessage {{
        background: {CARD_BG} !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        margin: 8px 0 !important;
        border: 1px solid {BORDER} !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    }}
    .stChatMessage [data-testid="stMarkdownContainer"] {{
        color: {TEXT_MAIN} !important;
        line-height: 1.6;
    }}
    /* Visual cue for user messages (lavender bubble) */
    div[data-testid="stChatMessage"]:has(img[src*="user_icon"]) {{
        background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%) !important;
        border-color: #DDD6FE !important;
    }}

    /* --- Stat cards --- */
    .stat-card {{
        background: white;
        padding: 22px 20px;
        border-radius: 16px;
        border: 1px solid {BORDER};
        text-align: left;
        margin: 6px 0;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .stat-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }}
    .stat-card .icon-bubble {{
        width: 40px; height: 40px;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px;
        margin-bottom: 12px;
        background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
        color: white;
    }}
    .stat-number {{
        font-size: 30px;
        font-weight: 700;
        color: {TEXT_MAIN};
        line-height: 1;
    }}
    .stat-label {{
        font-size: 13px;
        color: {TEXT_MUTED};
        margin-top: 6px;
        font-weight: 500;
    }}

    /* --- Suggested prompts --- */
    .suggested-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 12px;
        margin: 20px 0;
    }}
    .suggested-chip {{
        background: white;
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 14px 16px;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 14px;
        color: {TEXT_MAIN};
    }}
    .suggested-chip:hover {{
        border-color: {PRIMARY};
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.12);
    }}

    /* --- Empty state --- */
    .empty-state {{
        text-align: center;
        padding: 40px 20px;
    }}
    .empty-state-icon {{
        font-size: 56px;
        margin-bottom: 16px;
    }}
    .empty-state-title {{
        font-size: 22px;
        font-weight: 600;
        color: {TEXT_MAIN};
        margin-bottom: 8px;
    }}
    .empty-state-sub {{
        color: {TEXT_MUTED};
        font-size: 15px;
        max-width: 520px;
        margin: 0 auto;
    }}

    /* --- Conversation list items --- */
    .conv-item {{
        padding: 10px 12px;
        border-radius: 10px;
        margin-bottom: 4px;
        cursor: pointer;
        border: 1px solid transparent;
        transition: all 0.15s ease;
    }}
    .conv-item:hover {{ background: #F4F4FB; }}
    .conv-item.active {{
        background: linear-gradient(135deg, rgba(79,70,229,0.08), rgba(124,58,237,0.06));
        border-color: rgba(79,70,229,0.25);
    }}

    /* --- Expanders --- */
    .streamlit-expanderHeader {{
        background: #FAFBFD !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
    }}
    .streamlit-expanderContent {{
        background: #FAFBFD !important;
        padding: 14px !important;
        border-radius: 10px !important;
        border: 1px solid {BORDER} !important;
    }}
    .streamlit-expanderContent pre {{
        background: white !important;
        padding: 12px !important;
        border-radius: 8px !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT_MAIN} !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        font-size: 12.5px !important;
    }}

    /* --- Titles --- */
    h1, h2, h3 {{ color: {TEXT_MAIN}; letter-spacing: -0.3px; }}

    /* --- Status pill --- */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 500;
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }}
    .status-dot {{
        width: 8px; height: 8px;
        background: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 0 3px rgba(16,185,129,0.18);
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50%      {{ opacity: 0.6; }}
    }}

    /* --- Footer --- */
    .footer-bar {{
        text-align: center;
        color: {TEXT_MUTED};
        font-size: 13px;
        padding: 24px 0 10px 0;
        margin-top: 28px;
        border-top: 1px solid {BORDER};
    }}

    /* Hide Streamlit chrome we don't need */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Session state init
# ----------------------------------------------------------------------------
if "session_id" not in st.session_state:
    try:
        query_params = st.experimental_get_query_params()
        if "session_id" in query_params:
            st.session_state.session_id = query_params["session_id"][0]
        else:
            new_session_id = str(uuid.uuid4())
            st.session_state.session_id = new_session_id
            st.experimental_set_query_params(session_id=new_session_id)
    except Exception:
        st.session_state.session_id = str(uuid.uuid4())

if "conversations" not in st.session_state:
    if CONVERSATIONS_FILE.exists():
        with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
            st.session_state.conversations = json.load(f)
    else:
        st.session_state.conversations = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(int(time.time()))

if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot_instance" not in st.session_state:
    st.session_state.bot_instance = DiaygeiaBot(name="diaygeia")

if "page" not in st.session_state:
    st.session_state.page = "chat"

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def save_conversations():
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.conversations, f, ensure_ascii=False, indent=2)

def save_current_conversation():
    if len(st.session_state.messages) > 0:
        title = "Νέα Συνομιλία"
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                title = msg["content"][:60] + ("…" if len(msg["content"]) > 60 else "")
                break
        st.session_state.conversations[st.session_state.current_chat_id] = {
            "title": title,
            "messages": st.session_state.messages.copy(),
            "timestamp": int(time.time()),
            "session_id": st.session_state.session_id,
        }
        save_conversations()

def load_conversation(chat_id):
    save_current_conversation()
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.conversations[chat_id]["messages"].copy()
    if "session_id" in st.session_state.conversations[chat_id]:
        st.session_state.session_id = st.session_state.conversations[chat_id]["session_id"]
    else:
        st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

def delete_conversation(chat_id):
    if chat_id in st.session_state.conversations:
        del st.session_state.conversations[chat_id]
        save_conversations()
        if chat_id == st.session_state.current_chat_id:
            new_chat()
        else:
            st.rerun()

def new_chat():
    save_current_conversation()
    st.session_state.current_chat_id = str(int(time.time()))
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.pending_prompt = None
    st.rerun()

def calculate_statistics():
    total_conversations = len(st.session_state.conversations)
    total_messages = sum(len(c["messages"]) for c in st.session_state.conversations.values())
    total_user_messages = sum(
        sum(1 for m in c["messages"] if m["role"] == "user")
        for c in st.session_state.conversations.values()
    )
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "total_user_messages": total_user_messages,
        "avg_messages_per_conversation": total_messages / total_conversations if total_conversations else 0,
    }

def relative_time(ts: int) -> str:
    """Human-friendly relative timestamp in Greek."""
    if not ts:
        return ""
    delta = time.time() - ts
    if delta < 60: return "μόλις τώρα"
    if delta < 3600: return f"πριν {int(delta // 60)}'"
    if delta < 86400: return f"πριν {int(delta // 3600)} ώρες"
    if delta < 86400 * 7: return f"πριν {int(delta // 86400)} ημέρες"
    return datetime.fromtimestamp(ts).strftime("%d/%m/%Y")

def run_bot_turn(prompt: str):
    """Run a single chat turn — appends both user & assistant messages and saves."""
    st.session_state.messages.append({"role": "user", "content": prompt})

    _spacer, _user_col = st.columns([1, 3])
    with _user_col:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 _Σκέφτομαι…_")
        try:
            bot = st.session_state.bot_instance
            user_history_data = bot.session_manager.get_user_session(st.session_state.session_id)
            history = get_user_history(user_history_data)

            k = st.session_state.get("num_results", 3)
            context = ""            # raw retrieved-doc context (RAG path only)
            context_results = []    # retrieved docs, for the sources / evidence panel
            # Structured path first: countable questions ("how many / by type / by org /
            # by date") get an exact aggregation answer; None -> fall back to RAG.
            response = bot.try_structured(prompt)
            if response is None:
                # Tier 0 retrieval (multi-field + subject boost) + generation.
                context_results = bot.get_context_multi(prompt, k=k)
                if not context_results:
                    response = grounding.INSUFFICIENT   # nothing retrieved -> don't hallucinate
                else:
                    context = "\n\n".join([str(item) for sublist in context_results for item in sublist.values()])
                    response = bot.get_llm_response(prompt, history, context)
                    # cite the retrieved ΑΔΑs as clickable source links
                    response = grounding.linkify_adas(response, [r["id"] for r in context_results])

            bot.session_manager.update_user_session(
                st.session_state.session_id,
                user_history_data,
                prompt,
                response,
            )

            if st.session_state.get("enable_streaming", True):
                full_response = ""
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.03)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(response)
            else:
                message_placeholder.markdown(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "context": context,
                "sources": context_results,
            })
            save_current_conversation()
        except Exception as e:
            error_message = f"❌ **Σφάλμα:** {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-logo">🏛️</div>
            <div>DiavgeiaAssistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # New chat
    if st.button("➕  Νέα Συνομιλία", use_container_width=True, type="primary"):
        new_chat()

    # Navigation
    st.markdown('<div class="nav-section-label">Πλοήγηση</div>', unsafe_allow_html=True)
    nav_cols = st.columns(3)
    pages = [("💬", "chat", "Chat"), ("📊", "stats", "Στατ."), ("ℹ️", "info", "Info")]
    for col, (icon, key, label) in zip(nav_cols, pages):
        with col:
            is_active = st.session_state.page == key
            if st.button(
                f"{icon}\n{label}",
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.page = key
                st.rerun()

    # Conversations
    st.markdown('<div class="nav-section-label">Πρόσφατες Συνομιλίες</div>', unsafe_allow_html=True)

    sorted_conversations = sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1].get("timestamp", 0),
        reverse=True,
    )

    if not sorted_conversations:
        st.caption("Δεν υπάρχουν ακόμα συνομιλίες.")

    for chat_id, conv_data in sorted_conversations[:12]:
        is_current = chat_id == st.session_state.current_chat_id
        title = conv_data.get("title", "Νέα Συνομιλία")
        ts_label = relative_time(conv_data.get("timestamp", 0))
        col1, col2 = st.columns([6, 1])
        with col1:
            label = f"{'🟣  ' if is_current else '💬  '}{title[:30]}{'…' if len(title) > 30 else ''}"
            if ts_label:
                label += f"\n   · {ts_label}"
            if st.button(
                label,
                key=f"conv_{chat_id}",
                use_container_width=True,
                type="primary" if is_current else "secondary",
            ):
                st.session_state.page = "chat"
                load_conversation(chat_id)
        with col2:
            if st.button("🗑️", key=f"del_{chat_id}", help="Διαγραφή συνομιλίας"):
                delete_conversation(chat_id)

    # Settings
    st.markdown('<div class="nav-section-label">Ρυθμίσεις</div>', unsafe_allow_html=True)
    with st.expander("⚙️  Προτιμήσεις"):
        num_results = st.slider("Αποτελέσματα ανά αναζήτηση", 1, 10, 3)
        show_context = st.checkbox("Εμφάνιση context (πηγών)", value=False)
        enable_streaming = st.checkbox("Streaming απαντήσεων", value=True)
        st.session_state.num_results = num_results
        st.session_state.show_context = show_context
        st.session_state.enable_streaming = enable_streaming

    st.markdown(
        f"""
        <div style='margin-top: 16px;'>
            <span class="status-pill"><span class="status-dot"></span> Σύστημα ενεργό</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------------
if st.session_state.page == "chat":
    st.markdown(
        f"""
        <div class="hero">
            <div class="pill">🤖 RAG · Powered by Gemini & Elasticsearch</div>
            <h1>Έξυπνη αναζήτηση στη Διαύγεια</h1>
            <p>Κάνε ερωτήσεις σε φυσική γλώσσα για έγγραφα, αποφάσεις και δαπάνες του ελληνικού δημοσίου.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Empty state with suggested prompts
    if len(st.session_state.messages) == 0 and st.session_state.pending_prompt is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">💬</div>
                <div class="empty-state-title">Πώς μπορώ να σε βοηθήσω;</div>
                <div class="empty-state-sub">
                    Ξεκίνα γράφοντας μια ερώτηση στο πεδίο παρακάτω,
                    ή διάλεξε μία από τις προτεινόμενες ερωτήσεις.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggested prompts as a 2x2 grid of Streamlit buttons (so they actually click)
        sp_cols = st.columns(2)
        for idx, (emoji, prompt_text) in enumerate(SUGGESTED_PROMPTS):
            with sp_cols[idx % 2]:
                if st.button(
                    f"{emoji}  {prompt_text}",
                    key=f"suggest_{idx}",
                    use_container_width=True,
                ):
                    st.session_state.pending_prompt = prompt_text
                    st.rerun()

    # Render existing messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            _spacer, _user_col = st.columns([1, 3])
            with _user_col:
                with st.chat_message("user", avatar=USER_AVATAR):
                    st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                st.markdown(message["content"])
                if st.session_state.get("show_context", False) and (
                    message.get("sources") or message.get("context")
                ):
                    with st.expander("📄  Πηγές & context που χρησιμοποιήθηκαν"):
                        if message.get("sources"):
                            st.markdown(grounding.sources_markdown(message["sources"]))
                        else:
                            st.code(message["context"], language=None)

    # Handle pending prompt from suggested chips
    if st.session_state.pending_prompt:
        pending = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        run_bot_turn(pending)
        st.rerun()

    # Chat input
    if prompt := st.chat_input("Γράψε την ερώτησή σου εδώ…"):
        run_bot_turn(prompt)

elif st.session_state.page == "stats":
    st.markdown(
        f"""
        <div class="hero">
            <div class="pill">📊 Αναλυτικά</div>
            <h1>Στατιστικά & Δραστηριότητα</h1>
            <p>Δες πόσο χρησιμοποιείς το DiavgeiaAssistant και τι μοτίβα εμφανίζονται.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = calculate_statistics()
    cards = [
        ("💬", "Συνομιλίες", stats["total_conversations"]),
        ("✉️", "Συνολικά Μηνύματα", stats["total_messages"]),
        ("❓", "Ερωτήσεις σου", stats["total_user_messages"]),
        ("📈", "Μ.Ο. ανά συνομιλία", f"{stats['avg_messages_per_conversation']:.1f}"),
    ]
    cols = st.columns(4)
    for col, (icon, label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="icon-bubble">{icon}</div>
                    <div class="stat-number">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br/>", unsafe_allow_html=True)

    if st.session_state.conversations:
        # Timeline
        st.subheader("📈  Χρονολόγιο δραστηριότητας")
        timeline_data = []
        for conv in st.session_state.conversations.values():
            ts = conv.get("timestamp", 0)
            timeline_data.append({
                "Ημερομηνία": datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                "Συνομιλίες": 1,
                "Μηνύματα": len(conv["messages"]),
            })
        df = pd.DataFrame(timeline_data)
        df_grouped = df.groupby("Ημερομηνία").sum().reset_index()
        fig = px.area(
            df_grouped,
            x="Ημερομηνία",
            y=["Συνομιλίες", "Μηνύματα"],
            labels={"value": "Πλήθος", "variable": "Τύπος"},
            color_discrete_sequence=[PRIMARY, ACCENT],
        )
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10),
            font=dict(family="Inter", color=TEXT_MAIN),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("📝  Μήκος μηνυμάτων")
            rows = []
            for c in st.session_state.conversations.values():
                for m in c["messages"]:
                    rows.append({
                        "Τύπος": "Χρήστης" if m["role"] == "user" else "Assistant",
                        "Μήκος": len(m["content"]),
                    })
            df_l = pd.DataFrame(rows)
            if not df_l.empty:
                fig2 = px.box(
                    df_l, x="Τύπος", y="Μήκος", color="Τύπος",
                    color_discrete_sequence=[PRIMARY, ACCENT],
                )
                fig2.update_layout(
                    plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(l=10, r=10, t=20, b=10),
                    font=dict(family="Inter", color=TEXT_MAIN),
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)

        with right:
            st.subheader("🥧  Μηνύματα ανά ρόλο")
            user_total = stats["total_user_messages"]
            bot_total = stats["total_messages"] - user_total
            if user_total + bot_total > 0:
                pie = go.Figure(data=[
                    go.Pie(
                        labels=["Χρήστης", "Assistant"],
                        values=[user_total, bot_total],
                        hole=0.55,
                        marker=dict(colors=[PRIMARY, ACCENT]),
                    )
                ])
                pie.update_layout(
                    margin=dict(l=10, r=10, t=20, b=10),
                    font=dict(family="Inter", color=TEXT_MAIN),
                    showlegend=True,
                )
                st.plotly_chart(pie, use_container_width=True)
    else:
        st.info("Δεν υπάρχουν ακόμα συνομιλίες — ξεκίνα μια νέα από την καρτέλα Chat!")

elif st.session_state.page == "info":
    st.markdown(
        f"""
        <div class="hero">
            <div class="pill">ℹ️ Σχετικά</div>
            <h1>Σχετικά με το DiavgeiaAssistant</h1>
            <p>Ένας RAG-based βοηθός για ευκολότερη πλοήγηση στα ανοιχτά δεδομένα της Διαύγειας.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_about, tab_how, tab_tech, tab_contact = st.tabs(
        ["📖 Επισκόπηση", "🚀 Πώς να ξεκινήσεις", "🛠️ Τεχνολογίες", "📧 Επαφή"]
    )

    with tab_about:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                """
                ### 🏛️ Τι είναι το DiavgeiaAssistant;

                Το **DiavgeiaAssistant** είναι ένα έξυπνο σύστημα ερωτο-απαντήσεων που χρησιμοποιεί
                **RAG (Retrieval-Augmented Generation)** για να συνδυάζει αναζήτηση σε
                πραγματικά έγγραφα της **Ελληνικής Διαύγειας** με τη δύναμη ενός
                μεγάλου γλωσσικού μοντέλου.

                ### 🎯 Χαρακτηριστικά

                - **🔍 Ακριβής αναζήτηση** σε εκατομμύρια αποφάσεις μέσω Elasticsearch
                - **🤖 Φυσική συνομιλία** στα ελληνικά μέσω Gemini
                - **📎 Αναφορές ΑΔΑ** για κάθε απάντηση
                - **💾 Ιστορικό συνομιλιών** που μένει στη συσκευή σου
                - **🎨 Καθαρό, σύγχρονο UI** βασισμένο σε Streamlit
                """
            )
        with col2:
            st.markdown("### 📊 System Status")
            st.success("✅ Elasticsearch")
            st.success("✅ Gemini API")
            st.success("✅ Session Manager")
            st.markdown("### 🔗 Χρήσιμα")
            st.markdown(
                """
                - [Διαύγεια](https://diavgeia.gov.gr)
                - [Google Gemini](https://ai.google.dev/)
                - [Vertex AI](https://cloud.google.com/vertex-ai)
                """
            )

    with tab_how:
        st.markdown(
            """
            ### 🚀 Πώς να ξεκινήσεις σε 3 βήματα

            1. **Νέα Συνομιλία** — κάνε κλικ στο κουμπί στο sidebar για να ξεκινήσεις από την αρχή
            2. **Διάλεξε μια προτεινόμενη ερώτηση** ή γράψε τη δική σου στο πεδίο εισαγωγής
            3. **Δες την απάντηση** — μπορείς να ενεργοποιήσεις το «Εμφάνιση context» για να δεις
               από ποια έγγραφα προέρχεται

            ### 💡 Συμβουλές για καλύτερα αποτελέσματα

            - Να είσαι **συγκεκριμένος** (π.χ. αναφέρε υπουργείο, έτος, θεματική)
            - Για διευκρινίσεις, **συνέχισε στην ίδια συνομιλία** — ο Assistant θυμάται το context
            - Χρησιμοποίησε **ορολογία της Διαύγειας** (ΑΔΑ, φορέας, απόφαση) για ακριβέστερα hits
            """
        )

    with tab_tech:
        st.markdown(
            """
            ### 🛠️ Stack

            | Συστατικό | Τεχνολογία |
            |---|---|
            | Retrieval | Elasticsearch |
            | LLM | Google Gemini (Vertex AI) |
            | Orchestration | Python · custom RAG pipeline |
            | UI | Streamlit |
            | Data | [Diavgeia Open Data](https://diavgeia.gov.gr) |

            ### 🔒 Ιδιωτικότητα
            Οι συνομιλίες αποθηκεύονται **τοπικά** (σε `conversations.json`) και δεν αποστέλλονται
            σε τρίτους πέρα από τα queries προς το LLM.
            """
        )

    with tab_contact:
        st.markdown(
            """
            ### 📧 Επικοινωνία

            - **GitHub**: άνοιξε issue στο repo
            - **Project**: Διαύγεια Chat-based Assistant
            """
        )

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="footer-bar">
        Powered by <b>Google Gemini</b> · <b>Elasticsearch</b> ·
        <a href='https://diavgeia.gov.gr' target='_blank' style='color:{PRIMARY}; text-decoration:none;'>Διαύγεια</a>
    </div>
    """,
    unsafe_allow_html=True,
)
