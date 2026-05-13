"""
Enhanced Streamlit Demo for DiavgeiaBot - RAG-based Q&A Chatbot
Includes all features + statistics and analytics
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
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from diaygeia.bot.response import DiaygeiaBot, get_user_history
from diaygeia.domain import Conversation

# Load environment variables from project root
load_dotenv(PROJECT_ROOT / ".env")

# Define avatar paths with safe fallback for Docker
_user_avatar_path = SCRIPT_DIR / "assets" / "user_icon.png"
_bot_avatar_path = SCRIPT_DIR / "assets" / "bot_icon.png"

USER_AVATAR = str(_user_avatar_path) if _user_avatar_path.exists() else "👤"
BOT_AVATAR = str(_bot_avatar_path) if _bot_avatar_path.exists() else "🤖"

# Page configuration
st.set_page_config(
    page_title="DiavgeiaBot - Ελληνική Διαύγεια",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #ffffff;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #ffffff;
    }
    
    /* Fix chat input styling */
    .stChatInput {
        background-color: white !important;
    }
    
    .stChatInput > div {
        background-color: white !important;
        border-radius: 8px !important;
    }
    
    .stChatInput textarea {
        background-color: white !important;
        color: #333 !important;
        border: 1px solid #4F46E5 !important;
    }
    
    /* Change focus border color to blue */
    .stChatInput textarea:focus {
        border-color: #4F46E5 !important;
        outline: none !important;
        box-shadow: 0 0 0 1px #4F46E5 !important;
    }
    
    .stChatInput textarea:focus-visible {
        border-color: #4F46E5 !important;
        outline: none !important;
        box-shadow: 0 0 0 1px #4F46E5 !important;
    }
    
    /* Chat input send button - change from red to blue */
    .stChatInput button {
        background-color: #4F46E5 !important;
        border-color: #4F46E5 !important;
        color: white !important;
    }
    
    .stChatInput button:hover {
        background-color: #4338CA !important;
        border-color: #4338CA !important;
    }
    
    .stChatInput button:active {
        background-color: #3730A3 !important;
        border-color: #3730A3 !important;
    }
    
    .stChatInput button:focus {
        box-shadow: 0 0 0 0.2rem rgba(79, 70, 229, 0.5) !important;
    }
    
    /* Override any red focus rings */
    .stChatInput *:focus {
        outline-color: #4F46E5 !important;
    }
    
    .stChatInput div[data-baseweb="base-input"]:focus-within {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 1px #4F46E5 !important;
    }
    
    /* Chat messages - make them visible */
    .stChatMessage {
        background-color: #f3f4f6 !important;
        border-radius: 10px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
    }
    
    /* Message content should be visible */
    .stChatMessage [data-testid="stMarkdownContainer"] {
        color: #333 !important;
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background-color: #e9f3ff !important;
    }
    
    /* Bot message */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #ffffff !important;
        border: 1px solid #eee !important;
    }
    
    /* Buttons - remove red colors and use blue */
    .stButton>button {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button[kind="primary"] {
        background-color: #4F46E5 !important;
        border-color: #4F46E5 !important;
        color: white !important;
    }
    
    .stButton>button[kind="primary"]:hover {
        background-color: #4338CA !important;
        border-color: #4338CA !important;
    }
    
    .stButton>button[kind="primary"]:active {
        background-color: #3730A3 !important;
        border-color: #3730A3 !important;
    }
    
    .stButton>button[kind="primary"]:focus {
        box-shadow: 0 0 0 0.2rem rgba(79, 70, 229, 0.5) !important;
    }
    
    /* Secondary buttons */
    .stButton>button[kind="secondary"]:hover {
        border-color: #4F46E5 !important;
        color: #4F46E5 !important;
    }
    
    .stButton>button[kind="secondary"]:active {
        border-color: #3730A3 !important;
        color: #3730A3 !important;
    }
    
    .stButton>button[kind="secondary"]:focus {
        box-shadow: 0 0 0 0.2rem rgba(79, 70, 229, 0.25) !important;
    }
    
    /* Title styling */
    h1 {
        color: #4F46E5;
        font-weight: 600;
    }
    
    /* Sidebar title */
    .sidebar-title {
        font-size: 24px;
        font-weight: 600;
        color: #4F46E5;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Stats card */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    
    .stat-number {
        font-size: 36px;
        font-weight: bold;
    }
    
    .stat-label {
        font-size: 14px;
        opacity: 0.9;
    }
    
    /* Fix expander content visibility */
    .streamlit-expanderContent {
        background-color: #f8f9fa !important;
        padding: 15px !important;
        border-radius: 5px !important;
    }
    
    .streamlit-expanderContent pre {
        background-color: #fff !important;
        padding: 10px !important;
        border: 1px solid #4F46E5 !important;
        border-radius: 5px !important;
        color: #333 !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }
    
    /* Make text in expander visible */
    .streamlit-expanderContent p,
    .streamlit-expanderContent div {
        color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
# Use a persistent session ID (doesn't change on page reload)
if "session_id" not in st.session_state:
    # Try to get from query params first, otherwise create new
    try:
        query_params = st.experimental_get_query_params()
        if "session_id" in query_params:
            st.session_state.session_id = query_params["session_id"][0]
        else:
            # Create new session and add to URL
            new_session_id = str(uuid.uuid4())
            st.session_state.session_id = new_session_id
            st.experimental_set_query_params(session_id=new_session_id)
    except:
        # Fallback: just use a random session ID
        st.session_state.session_id = str(uuid.uuid4())

# Set up conversations file path in streamlit_ui folder
CONVERSATIONS_FILE = SCRIPT_DIR / "conversations.json"

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

# Helper functions
def save_conversations():
    """Save conversations to file"""
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.conversations, f, ensure_ascii=False, indent=2)

def save_current_conversation():
    """Save the current conversation"""
    if len(st.session_state.messages) > 0:
        title = "Νέα Συνομιλία"
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                title = msg["content"][:50] + ("..." if len(msg["content"]) > 50 else "")
                break
        
        st.session_state.conversations[st.session_state.current_chat_id] = {
            "title": title,
            "messages": st.session_state.messages.copy(),
            "timestamp": int(time.time()),
            "session_id": st.session_state.session_id  # Save the session_id with the conversation
        }
        save_conversations()

def load_conversation(chat_id):
    """Load a conversation"""
    save_current_conversation()
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.conversations[chat_id]["messages"].copy()
    # Restore the session_id associated with this conversation, or create new one if not exists
    if "session_id" in st.session_state.conversations[chat_id]:
        st.session_state.session_id = st.session_state.conversations[chat_id]["session_id"]
    else:
        st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

def delete_conversation(chat_id):
    """Delete a conversation"""
    if chat_id in st.session_state.conversations:
        del st.session_state.conversations[chat_id]
        save_conversations()
        
        if chat_id == st.session_state.current_chat_id:
            new_chat()
        else:
            st.rerun()

def new_chat():
    """Create a new chat"""
    save_current_conversation()
    st.session_state.current_chat_id = str(int(time.time()))
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

def calculate_statistics():
    """Calculate statistics from conversations"""
    total_conversations = len(st.session_state.conversations)
    total_messages = sum(len(conv["messages"]) for conv in st.session_state.conversations.values())
    total_user_messages = sum(
        sum(1 for msg in conv["messages"] if msg["role"] == "user")
        for conv in st.session_state.conversations.values()
    )
    
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "total_user_messages": total_user_messages,
        "avg_messages_per_conversation": total_messages / total_conversations if total_conversations > 0 else 0
    }

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏛️ DiavgeiaBot</div>', unsafe_allow_html=True)
    
    # Navigation
    st.markdown("### 📱 Πλοήγηση")
    if st.button("💬 Chat", use_container_width=True, type="primary" if st.session_state.page == "chat" else "secondary"):
        st.session_state.page = "chat"
        st.rerun()
    
    if st.button("📊 Στατιστικά", use_container_width=True, type="primary" if st.session_state.page == "stats" else "secondary"):
        st.session_state.page = "stats"
        st.rerun()
    
    if st.button("ℹ️ Πληροφορίες", use_container_width=True, type="primary" if st.session_state.page == "info" else "secondary"):
        st.session_state.page = "info"
        st.rerun()
    
    st.markdown("---")
    
    # New Chat Button
    if st.button("➕ Νέα Συνομιλία", use_container_width=True, type="primary"):
        new_chat()
    
    st.markdown("---")
    
    # Conversations list
    st.subheader("Συνομιλίες")
    
    sorted_conversations = sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1].get("timestamp", 0),
        reverse=True
    )
    
    for chat_id, conv_data in sorted_conversations[:10]:  # Show last 10
        col1, col2 = st.columns([5, 1])
        
        with col1:
            is_current = chat_id == st.session_state.current_chat_id
            button_type = "primary" if is_current else "secondary"
            
            if st.button(
                f"💬 {conv_data['title'][:25]}...",
                key=f"conv_{chat_id}",
                use_container_width=True,
                type=button_type if is_current else "secondary"
            ):
                st.session_state.page = "chat"
                load_conversation(chat_id)
        
        with col2:
            if st.button("🗑️", key=f"del_{chat_id}"):
                delete_conversation(chat_id)
    
    st.markdown("---")
    
    # Settings
    with st.expander("⚙️ Ρυθμίσεις"):
        num_results = st.slider(
            "Αριθμός αποτελεσμάτων",
            min_value=1,
            max_value=10,
            value=1
        )
        
        show_context = st.checkbox(
            "Εμφάνιση context",
            value=False
        )
        
        enable_streaming = st.checkbox(
            "Streaming απαντήσεων",
            value=True
        )
        
        st.session_state.num_results = num_results
        st.session_state.show_context = show_context
        st.session_state.enable_streaming = enable_streaming

# Main content area
if st.session_state.page == "chat":
    # Chat page
    st.title("🏛️ DiavgeiaBot - Ρωτήστε για τη Διαύγεια")
    st.markdown("*Κάντε ερωτήσεις για έγγραφα και αποφάσεις της Ελληνικής Διαύγειας*")
    
    # Display chat messages
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            
            if (message["role"] == "assistant" and 
                st.session_state.get("show_context", False) and 
                "context" in message):
                with st.expander("📄 Context που χρησιμοποιήθηκε"):
                    st.code(message["context"], language=None)
    
    # Chat input
    if prompt := st.chat_input("Γράψε το μήνυμά σου εδώ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Σκέφτομαι...")
            
            try:
                bot = st.session_state.bot_instance
                user_history_data = bot.session_manager.get_user_session(st.session_state.session_id)
                history = get_user_history(user_history_data)
                str_history = "\n".join([item['content'] for item in history])
                
                context_results = bot.get_context(str_history + " " + prompt)
                context = "\n\n".join([str(item) for sublist in context_results for item in sublist.values()])
                
                response = bot.get_llm_response(prompt, history, context)
                
                bot.session_manager.update_user_session(
                    st.session_state.session_id, 
                    user_history_data, 
                    prompt, 
                    response
                )
                
                # Streaming effect
                if st.session_state.get("enable_streaming", True):
                    full_response = ""
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)  # Slower typing effect (was 0.02)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(response)
                else:
                    message_placeholder.markdown(response)
                
                assistant_message = {
                    "role": "assistant", 
                    "content": response,
                    "context": context
                }
                st.session_state.messages.append(assistant_message)
                save_current_conversation()
                
            except Exception as e:
                error_message = f"❌ Σφάλμα: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

elif st.session_state.page == "stats":
    # Statistics page
    st.title("📊 Στατιστικά & Ανάλυση")
    
    stats = calculate_statistics()
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_conversations']}</div>
            <div class="stat-label">Συνομιλίες</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_messages']}</div>
            <div class="stat-label">Μηνύματα</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_user_messages']}</div>
            <div class="stat-label">Ερωτήσεις</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['avg_messages_per_conversation']:.1f}</div>
            <div class="stat-label">Μέσος όρος/Συνομιλία</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Conversations timeline
    if st.session_state.conversations:
        st.subheader("📈 Χρονολόγιο Συνομιλιών")
        
        # Prepare data for timeline
        timeline_data = []
        for chat_id, conv in st.session_state.conversations.items():
            timestamp = conv.get("timestamp", 0)
            date = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            timeline_data.append({
                "Ημερομηνία": date,
                "Συνομιλίες": 1,
                "Μηνύματα": len(conv["messages"])
            })
        
        df = pd.DataFrame(timeline_data)
        df_grouped = df.groupby("Ημερομηνία").sum().reset_index()
        
        fig = px.line(df_grouped, x="Ημερομηνία", y=["Συνομιλίες", "Μηνύματα"],
                     title="Δραστηριότητα ανά ημέρα",
                     labels={"value": "Πλήθος", "variable": "Τύπος"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Message length analysis
        st.subheader("📝 Ανάλυση Μήκους Μηνυμάτων")
        
        message_lengths = []
        for conv in st.session_state.conversations.values():
            for msg in conv["messages"]:
                if msg["role"] == "user":
                    message_lengths.append({
                        "Τύπος": "Χρήστης",
                        "Μήκος": len(msg["content"])
                    })
                else:
                    message_lengths.append({
                        "Τύπος": "Bot",
                        "Μήκος": len(msg["content"])
                    })
        
        if message_lengths:
            df_lengths = pd.DataFrame(message_lengths)
            fig = px.box(df_lengths, x="Τύπος", y="Μήκος",
                        title="Κατανομή μήκους μηνυμάτων",
                        color="Τύπος")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Δεν υπάρχουν δεδομένα για εμφάνιση στατιστικών. Ξεκινήστε μια συνομιλία!")

elif st.session_state.page == "info":
    # Info page
    st.title("ℹ️ Πληροφορίες για το DiavgeiaBot")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🏛️ Τι είναι το DiavgeiaBot;
        
        Το **DiavgeiaBot** είναι ένα έξυπνο chatbot που χρησιμοποιεί την τεχνολογία 
        **RAG (Retrieval-Augmented Generation)** για να απαντά σε ερωτήσεις σχετικά με 
        έγγραφα και αποφάσεις της Ελληνικής Διαύγειας.
        
        ### 🎯 Χαρακτηριστικά
        
        - **🔍 Έξυπνη Αναζήτηση**: Χρησιμοποιεί Elasticsearch για να βρει σχετικά έγγραφα
        - **🤖 AI-Powered**: Ενσωματώνει Gemini για φυσική συνομιλία
        - **📝 Αναφορές ΑΔΑ**: Παρέχει αναφορές σε συγκεκριμένα έγγραφα
        - **💾 Ιστορικό**: Διατηρεί το ιστορικό των συνομιλιών σας
        - **🎨 Modern UI**: Όμορφο και εύχρηστο περιβάλλον με Streamlit
        
        ### 🛠️ Τεχνολογίες
        
        - **Backend**: Python, Flask, Elasticsearch
        - **AI/ML**: Google Gemini (Vertex AI), RAG Architecture
        - **Frontend**: Streamlit
        - **Data Source**: Diavgeia Open Data
        
        ### 📖 Πώς να το χρησιμοποιήσετε
        
        1. **Ξεκινήστε μια νέα συνομιλία** με το κουμπί "Νέα Συνομιλία"
        2. **Γράψτε την ερώτησή σας** στο πεδίο εισαγωγής
        3. **Περιμένετε την απάντηση** από το bot
        4. **Δείτε το context** (προαιρετικό) για να δείτε τα έγγραφα που χρησιμοποιήθηκαν
        
        ### 🔒 Ιδιωτικότητα
        
        Οι συνομιλίες σας αποθηκεύονται τοπικά στη συσκευή σας και δεν κοινοποιούνται 
        σε τρίτους.
        """)
    
    with col2:
        st.markdown("""
        ### 📊 System Status
        """)
        
        # System status indicators
        st.success("✅ Elasticsearch: Online")
        st.success("✅ Gemini API: Connected")
        st.success("✅ Session Manager: Active")
        
        st.markdown("---")
        
        st.markdown("""
        ### 🔗 Links
        
        - [Διαύγεια](https://diavgeia.gov.gr)
        - [Google Gemini](https://ai.google.dev/)
        - [Vertex AI](https://cloud.google.com/vertex-ai)
        - [Streamlit](https://streamlit.io)
        - [Elasticsearch](https://www.elastic.co)
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📧 Contact
        
        Για ερωτήσεις και υποστήριξη:
        - GitHub Issues
        - Email: georgiosantoniou@mail.ntua.gr
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 14px; padding: 20px;'>
        Powered by Google Gemini (Vertex AI) & Elasticsearch | 
        <a href='https://diavgeia.gov.gr' target='_blank'>Διαύγεια</a> 
    </div>
    """,
    unsafe_allow_html=True
)
