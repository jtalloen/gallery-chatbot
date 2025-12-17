import streamlit as st
from PIL import Image
import base64
import google.generativeai as genai
import os
import PyPDF2
import docx
import time
import tomli

# Compatibility shim for st.experimental_rerun
try:
    rerun = st.experimental_rerun
except AttributeError:
    from streamlit.runtime.scriptrunner import RerunException
    def rerun():
        raise RerunException(None)

# Set page config for custom theme
st.set_page_config(
    page_title="Gallery263 Chatbot",
    page_icon="logo263.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Read and encode logo as base64 for HTML embedding
with open("logo263.png", "rb") as image_file:
    logo_base64 = base64.b64encode(image_file.read()).decode()

# Combined CSS, logo, and header HTML with embedded base64 image
st.markdown(
    f"""
    <style>
        body {{
            background-color: #ffffff;
        }}
        .stApp {{
            background-color: #ffffff;
            max-width: 100vw;
            margin-left: 0;
            margin-right: 0;
        }}
        .st-bw {{
            color: #1a3d2b;
        }}
        .stForm button, .stButton>button {{
            background-color: #d6f13a;
            color: #1a3d2b;
            border-radius: 8px;
            font-weight: bold;
            margin-left: auto;
            margin-right: 0;
        }}
        .stForm button:hover, .stButton>button:hover {{
            background-color: #b8d82a;
            color: #1a3d2b;
        }}
        .stTextInput>div>div>input {{
            background-color: #eaf5e1;
            color: #1a3d2b;
        }}
        .stForm {{
            background: #1a3d2b;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(214, 241, 58, 0.08);
            margin-bottom: 1.5rem;
        }}
        .logo263-header {{
            width: 100%;
            display: flex;
            justify-content: left;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }}
        .gallery263-title {{
            color: #1a3d2b;
            text-align: center;
            font-size: 2.25rem;
            font-weight: bold;
            margin: 0;
        }}
        .gallery263-subtitle {{
            color: #1a3d2b;
            text-align: center;
            font-size: 1.15rem;
            margin: 0 0 1.5rem 0;
        }}
        .chat-bubble {{
            margin-bottom: 4px;
            padding: 8px;
            border-radius: 6px;
            word-break: break-word;
        }}
        .chat-bubble.user {{
            color: #1a3d2b;
            background: #eaf5e1;
            text-align: right;
            margin-left: 0;
        }}
        .chat-bubble.bot {{
            color: #1a3d2b;
            background: #d6f13a;
            text-align: left;
            margin-left: 0;
            margin-right: 0;
        }}
        .custom-hr {{
            border: none;
            margin: 2rem 0 2rem 0;
            width: 100%;
        }}
    </style>

    <div class='logo263-header'>
        <img src='data:image/png;base64,{logo_base64}' width='120px'/>
    </div>

    <h1 class='gallery263-title'>
        Gallery263 AI Assistant
    </h1>
    
    <p class='gallery263-subtitle'>
        More grants to write? Let's do it!
    </p>

    """,
    unsafe_allow_html=True
)

# =====================
# Gemini API Key Setup
# =====================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    st.warning("Please set your Gemini API key in Streamlit secrets or as the GEMINI_API_KEY environment variable.")

genai.configure(api_key=GEMINI_API_KEY)

# =====================
# Sidebar: File Upload & Model Tools
# =====================

st.sidebar.header("Context Files")
# File uploader for LLM context
uploaded_files = st.sidebar.file_uploader(
    "Upload files for LLM context (PDF, TXT, DOCX)",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    key="context_files"
)
if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded.")
    # You can process and store these files for LLM context here

# =====================
# File Extraction Helpers
# =====================
def extract_text_from_file(uploaded_file):
    """Extract text from PDF, DOCX, or TXT file."""
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() or '' for page in reader.pages)
        return text
    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    else:
        return ""

# =====================
# Context Assembly (Gallery Context from secrets + Uploaded Files)
# =====================
context_texts = []
# Add Gallery263 context from Streamlit secrets
try:
    gallery_context = st.secrets["GALLERY263_CONTEXT"]
    context_texts.append(f"[Gallery Context]\n{gallery_context}")
except Exception as e:
    context_texts.append(f"[Gallery Context]\n[Could not read context from secrets: {e}]")

# Add uploaded files as context
if uploaded_files:
    for file in uploaded_files:
        try:
            text = extract_text_from_file(file)
            if text:
                context_texts.append(f"File: {file.name}\n{text}")
        except Exception as e:
            context_texts.append(f"File: {file.name}\n[Could not extract text: {e}]")

# =====================
# Gemini Chat Function
# =====================
def extract_gemini_text(response):
    """Robustly extract text from a Gemini response or chunk. Returns empty string if no text."""
    try:
        return response.candidates[0].content.parts[0].text
    except Exception:
        return ""

def gemini_chat(messages, context_texts=None):
    """Send chat history and context to Gemini LLM and return the response."""
    if not GEMINI_API_KEY:
        return "[Gemini API key not set]"
    model = genai.GenerativeModel('models/gemma-3-27b-it')
    # Add context from files as system prompt if available
    if context_texts:
        context_prompt = "\n\n".join(context_texts)
        messages = [{"role": "user", "parts": [f"[Context for LLM]\n{context_prompt}"]}] + messages
    chat = model.start_chat(history=messages[:-1]) if len(messages) > 1 else model.start_chat()
    response = chat.send_message(messages[-1]["parts"][0])
    return extract_gemini_text(response) if response else "[No response received from Gemini.]"

def gemini_chat_stream(messages, context_texts=None):
    """Stream chat history and context to Gemini LLM and yield the response as it arrives."""
    if not GEMINI_API_KEY:
        yield "[Gemini API key not set]"
        return
    model = genai.GenerativeModel('models/gemma-3-27b-it')
    # Add context from files as system prompt if available
    if context_texts:
        context_prompt = "\n\n".join(context_texts)
        messages = [{"role": "user", "parts": [f"[Context for LLM]\n{context_prompt}"]}] + messages
    chat = model.start_chat(history=messages[:-1]) if len(messages) > 1 else model.start_chat()
    response = chat.send_message(messages[-1]["parts"][0], stream=True)
    for chunk in response:
        text = extract_gemini_text(chunk)
        if text is not None:
            yield text

# =====================
# Chatbot UI & Logic
# =====================
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Display chat history above the input box
for sender, msg in st.session_state['chat_history']:
    if sender == "user":
        st.markdown(f"<div class='chat-bubble user'>{msg}</div>", unsafe_allow_html=True)
    elif sender == "bot":
        st.markdown(f"<div class='chat-bubble bot'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble'>{msg}</div>", unsafe_allow_html=True)

# Streaming response placeholder (shows before the input form)
stream_placeholder = st.empty()
bot_stream = st.empty()

# User input below chat history and streaming
st.markdown('<br>', unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([7.5, 1])
    with col1:
        user_input = st.text_input(
            label="hidden_label",
            key="input_box",
            label_visibility="collapsed",
            placeholder="Type your prompt here..."
        )
    with col2:
        send_clicked = st.form_submit_button("Send", use_container_width=True)

if send_clicked and user_input:
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    # Show the user's prompt as a placeholder bubble above the streaming response
    stream_placeholder.markdown(f"<div class='chat-bubble user'>{user_input}</div>", unsafe_allow_html=True)
    # Build full message history for Gemini
    gemini_messages = []
    for sender, msg in st.session_state['chat_history']:
        if sender == "user":
            gemini_messages.append({"role": "user", "parts": [msg]})
        elif sender == "bot":
            gemini_messages.append({"role": "model", "parts": [msg]})
    gemini_messages.append({"role": "user", "parts": [user_input]})
    # Stream the bot reply above the input form, styled as a green chat bubble
    streamed_text = ""
    for chunk in gemini_chat_stream(gemini_messages, context_texts):
        if chunk:
            streamed_text += chunk
            bot_stream.markdown(f"<div class='chat-bubble bot'>{streamed_text}</div>", unsafe_allow_html=True)
    if not streamed_text:
        streamed_text = "[No response received from Gemini. The model may have finished early or returned no content.]"
        bot_stream.markdown(f"<div class='chat-bubble bot'>{streamed_text}</div>", unsafe_allow_html=True)
    st.session_state['chat_history'].append(("user", user_input))
    st.session_state['chat_history'].append(("bot", streamed_text if streamed_text else "[No response received from Gemini.]") )
    rerun()
