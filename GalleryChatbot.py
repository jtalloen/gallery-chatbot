import streamlit as st
from PIL import Image
import base64
from google import genai
from huggingface_hub import InferenceClient
import os
import pypdf
import docx
import time
import re
import requests
import json

###############################################
######### FALLBACK MODEL CONFIGURATION ########
###############################################

# Toggle this to enable/disable HuggingFace Inference API fallback when Gemini rate limits are hit
USE_FALLBACK_MODEL = True  # Set to False to disable HuggingFace fallback
FALLBACK_MODEL_NAME = "google/gemma-3-27b-it"  # HuggingFace model to use as fallback
HF_API_TOKEN = st.secrets["HF_API_TOKEN"] if "HF_API_TOKEN" in st.secrets else ""  # Get from Streamlit secrets
FORCE_HF_FOR_TESTING = False  # Set to True to force HuggingFace for testing (bypass Gemini)

# Debug: Print token status
print(f"DEBUG: HF_API_TOKEN is set: {bool(HF_API_TOKEN)}")
if HF_API_TOKEN:
    print(f"DEBUG: HF_API_TOKEN starts with: {HF_API_TOKEN[:10]}...")
print(f"DEBUG: FORCE_HF_FOR_TESTING: {FORCE_HF_FOR_TESTING}")
print(f"DEBUG: USE_FALLBACK_MODEL: {USE_FALLBACK_MODEL}")

###############################################
######### COMPATIBILITY & PAGE CONFIG #########
###############################################

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

# Detect if running on Streamlit Cloud (not localhost)
def is_streamlit_cloud():
    """Check if the app is running on Streamlit Cloud vs localhost."""
    # Multiple ways to detect Streamlit Cloud environment
    hostname = os.getenv("HOSTNAME", "")
    # Streamlit Cloud containers have hostnames starting with "streamlit"
    # Also check for common cloud environment indicators
    return (
        hostname.startswith("streamlit") or
        os.getenv("STREAMLIT_SERVER_HEADLESS") == "true" or
        os.path.exists("/mount/src")  # Streamlit Cloud mounts repos here
    )

# CSS to hide Streamlit branding (only applied on Streamlit Cloud)
# Hides menu/profile but keeps GitHub icon visible
CLOUD_HIDE_CSS = """
        /* Hide main menu and Fork button */
        .stMainMenu.st-emotion-cache-czk5ss.e7d0y4c8,
        [data-testid="stToolbarActionButton"]:has([data-testid="stToolbarActionButtonLabel"]) {{
            display: none !important;
            visibility: hidden !important;
        }}

        /* Show GitHub button via its unique icon class */
        [data-testid="stToolbarActionButton"] button:has(.ekuhni81) {{
            display: inline-flex !important;
            visibility: visible !important;
        }}

        /* Hide bottom-right Streamlit branding */
        ._container_gzau3_1, 
        ._profileContainer_gzau3_53,
        footer {{
            display: none !important;
            visibility: hidden !important;
        }}
""" if is_streamlit_cloud() else ""

# Combined CSS, logo, and header HTML with embedded base64 image
st.markdown(
    f"""
    <style>
        /* Base layout */
        body, .stApp {{
            background-color: #ffffff;
        }}
        .stApp {{
            max-width: 100vw;
            margin: 0;
        }}
        .st-bw {{
            color: #1a3d2b;
        }}
        {CLOUD_HIDE_CSS}

        /* Buttons */
        .stForm button, .stButton>button {{
            background-color: #d6f13a;
            color: #1a3d2b;
            border-radius: 8px;
            font-weight: bold;
            margin-left: auto;
        }}
        .stForm button:hover, .stButton>button:hover {{
            background-color: #b8d82a;
        }}

        /* Form container */
        .stForm {{
            background: #1a3d2b;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(214, 241, 58, 0.08);
            margin-bottom: 1.5rem;
        }}
        .stForm textarea {{
            min-height: 38px !important;
            height: 38px !important;
            max-height: 200px !important;
            padding: 8px 12px !important;
            overflow-y: auto !important;
        }}
        .stForm .stTextArea {{
            margin-bottom: 0 !important;
        }}
        .stForm .stTextInput {{
            margin-bottom: 0 !important;
        }}
        .stForm [data-testid="stFormSubmitButton"] {{
            margin-top: 0 !important;
        }}
        .stForm [data-testid="stFormSubmitButton"] button {{
            height: 38px !important;
            min-height: 38px !important;
        }}

        /* Text input (legacy) */
        .stTextInput>div>div>input {{
            background-color: #eaf5e1;
            color: #1a3d2b;
        }}

        /* Header */
        .logo263-header {{
            width: 100%;
            display: flex;
            justify-content: left;
            margin: 1.5rem 0 0.5rem;
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
            margin: 0 0 1.5rem;
        }}

        /* Chat bubbles */
        .chat-bubble {{
            margin-bottom: 4px;
            padding: 8px;
            border-radius: 6px;
            word-break: break-word;
            color: #1a3d2b;
        }}
        .chat-bubble.user {{
            background: #eaf5e1;
            text-align: right;
        }}
        .chat-bubble.bot {{
            background: #d6f13a;
            text-align: left;
        }}
        .custom-hr {{
            border: none;
            margin: 2rem 0;
            width: 100%;
        }}

        /* Animated thinking dots */
        .thinking-dots::after {{
            content: '.';
            animation: dots 1.5s steps(4, end) infinite;
        }}
        @keyframes dots {{
            0%, 20% {{ content: '.'; }}
            40%, 60% {{ content: '..'; }}
            80%, 100% {{ content: '...'; }}
        }}

        /* Footer disclaimer */
        .ai-disclaimer {{
            text-align: center;
            color: #888;
            font-size: 0.85rem;
            padding: 1rem 0;
            margin-top: 2rem;
            # border-top: 1px solid #eee;
        }}

        /* Reduce bottom padding on main container */
        .stMainBlockContainer.block-container {{
            padding-bottom: 2rem !important;
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

###############################################
########## GEMINI API KEY SETUP ###############
###############################################

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    st.warning("Please set your Gemini API key in Streamlit secrets or as the GEMINI_API_KEY environment variable.")

###############################################
######## SIDEBAR: FILE UPLOAD & TOOLS ########
###############################################

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

###############################################
######## FILE EXTRACTION HELPERS ##############
###############################################

def extract_text_from_file(uploaded_file):
    """Extract text from PDF, DOCX, or TXT file."""
    if uploaded_file.type == "application/pdf":
        reader = pypdf.PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() or '' for page in reader.pages)
        return text
    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    else:
        return ""

###############################################
####### CONTEXT ASSEMBLY FROM SECRETS & FILES #
###############################################

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

###############################################
######### GEMINI TOOLS CONFIGURATION ##########
###############################################

# Native Gemini Tools (built-in, no custom code needed)
# These are passed directly to the API config - Gemini handles everything automatically
# 
# Available native tools:
#   1. google_search  - Web search for current information
#   2. code_execution - Python code execution for calculations/analysis
#   3. url_context    - Fetch and understand content from URLs in the prompt
#   4. file_search    - Search through uploaded files (requires FileSearchStore)
#
# See: https://ai.google.dev/gemini-api/docs/pricing#pricing-for-tools

def get_gemini_tools(include_file_search=False, file_search_store_name=None):
    """
    Get the list of native Gemini tools to use.
    
    Args:
        include_file_search: Whether to include file search (requires a FileSearchStore)
        file_search_store_name: The name of the FileSearchStore to use for file search
    
    Returns:
        List of tool configurations for Gemini API
    """
    from google.genai import types
    
    # Basic tools that work out of the box
    tools = [
        {"google_search": {}},      # Web search for current information
        {"code_execution": {}},     # Python code execution
        {"url_context": {}},        # Fetch content from URLs in the prompt
    ]
    
    # File Search requires a FileSearchStore to be created first
    # Only add if configured
    if include_file_search and file_search_store_name:
        tools.append(
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[file_search_store_name]
                )
            )
        )
        print(f"DEBUG: File Search Tool enabled with store: {file_search_store_name}")
    
    print(f"DEBUG: Gemini tools configured: google_search, code_execution, url_context")
    return tools

# Default tools (without file search - that requires setup)
GEMINI_TOOLS = get_gemini_tools()

###############################################
########### GEMINI CHAT FUNCTIONS #############
###############################################

def extract_gemini_text(response):
    """
    Robustly extract text from a Gemini response or chunk.
    Handles both streaming chunks and full responses from google-genai library.
    Returns empty string if no text found.
    """
    try:
        # Handle streaming chunks - they are GenerateContentResponse objects
        if hasattr(response, 'candidates') and response.candidates:
            text_parts = []
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
            return "".join(text_parts)
        # Fallback for direct .text attribute (shouldn't happen with google-genai)
        elif hasattr(response, 'text'):
            return response.text
        else:
            return ""
    except Exception as e:
        print(f"DEBUG: extract_gemini_text error: {e}, response type: {type(response)}")
        return ""

def gemini_chat(messages, context_texts=None):
    """Send chat history and context to Gemini LLM and return the response."""
    if not GEMINI_API_KEY:
        return "[Gemini API key not set]"
    client = genai.Client(api_key=GEMINI_API_KEY)
    model = 'gemini-2.5-flash'
    
    # Add context from files as first message if available
    if context_texts:
        context_prompt = "\n\n".join(context_texts)
        messages = [f"[Context for LLM]\n{context_prompt}"] + messages
    
    response = client.models.generate_content(
        model=model,
        contents=messages
    )
    return extract_gemini_text(response) if response else "[No response received from Gemini.]"

def gemini_chat_stream(messages, context_texts=None, chunk_type="word", enable_tools=True):
    """
    Stream chat history and context to Gemini LLM with tool support and yield the response as it arrives.
    Applies word-level or character-level streaming to simulate smoother token-by-token output
    and counteract API-side buffering.
    
    Supports native Gemini tools:
    - Google Search Retrieval: Web search for augmenting knowledge with current information
    - Code Execution: Python code execution for calculations/analysis
    
    Args:
        messages: List of message strings to send to the model
        context_texts: Optional list of context strings to prepend
        chunk_type: "word" for word-level streaming (faster), "char" for character-level (slower, smoother)
        enable_tools: Enable Gemini tools for this request (default: True)
    """
    if not GEMINI_API_KEY:
        yield "[Gemini API key not set]"
        return
    
    from google.genai import types
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    model = 'gemini-2.5-flash'
    
    # Add context from files as first message if available
    if context_texts:
        context_prompt = "\n\n".join(context_texts)
        messages = [f"[Context for LLM]\n{context_prompt}"] + messages
    
    try:
        # Build configuration with tools if enabled
        config_kwargs = {}
        if enable_tools and GEMINI_TOOLS:
            config_kwargs['tools'] = GEMINI_TOOLS
            print(f"DEBUG: Gemini tools enabled (Google Search + Code Execution)")
        
        # Generate content with streaming and tools support
        response = client.models.generate_content_stream(
            model=model,
            contents=messages,
            config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        )
        
        # Process streaming chunks
        for chunk in response:
            # Handle regular text output from the model
            text = extract_gemini_text(chunk)
            if text:
                # Stream word-by-word or character-by-character for smoother visual effect
                # This helps counteract API-side buffering and provides better UX
                if chunk_type == "word":
                    # Split on whitespace to get words (including punctuation)
                    tokens = re.split(r'(\s+)', text)  # Split on whitespace but keep whitespace
                    for token in tokens:
                        if token:  # Skip empty strings
                            yield token
                            time.sleep(0.02)  # Small delay between words
                else:  # character-level
                    for char in text:
                        yield char
                        time.sleep(0.01)  # Small delay between characters
    except Exception as e:
        error_str = str(e).lower()
        print(f"DEBUG: gemini_chat_stream error: {e}")
        # Check if it's a rate limit error - if so, re-raise to trigger fallback
        is_rate_limit = (
            "429" in error_str or
            "rate_limit" in error_str or
            "resource_exhausted" in error_str or
            "resourceexhausted" in error_str or
            "quota" in error_str
        )
        if is_rate_limit:
            raise  # Re-raise to let fallback wrapper handle it
        yield f"[Error during streaming: {str(e)}]"

###############################################
####### HUGGINGFACE INFERENCE API FALLBACK ####
###############################################

def huggingface_chat_stream(messages, context_texts=None, chunk_type="word"):
    """
    Stream chat history and context using HuggingFace Inference API (fallback).
    Uses the InferenceClient for chat_completion with proper streaming.
    
    Args:
        messages: List of message strings to send to the model
        context_texts: Optional list of context strings to prepend
        chunk_type: "word" for word-level streaming (faster), "char" for character-level (slower, smoother)
    """
    if not HF_API_TOKEN:
        yield "[HuggingFace API token not set. Please set HF_API_TOKEN in Streamlit secrets.]"
        return
    
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=HF_API_TOKEN)
        
        # Build the chat messages list in proper format
        chat_messages = []
        
        # Add context as system message if available
        if context_texts:
            context_prompt = "\n\n".join(context_texts)
            chat_messages.append({
                "role": "system",
                "content": f"You are a helpful assistant. Here is some context:\n\n{context_prompt}"
            })
        
        # Add message history (alternate user/assistant)
        for i, msg in enumerate(messages):
            if i == len(messages) - 1:
                # Last message is from user
                chat_messages.append({"role": "user", "content": msg})
            else:
                # Alternate between user and assistant for history
                role = "user" if i % 2 == 0 else "assistant"
                chat_messages.append({"role": role, "content": msg})
        
        # Generate response
        print(f"DEBUG: Generating response with HuggingFace chat_completion...")
        response = client.chat_completion(
            messages=chat_messages,
            model=FALLBACK_MODEL_NAME,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            stream=True  # Enable streaming
        )
        
        # Stream the response
        for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and choice.delta:
                    text = choice.delta.get('content', '') or choice.delta.content or ''
                    if text:
                        # Stream word-by-word or character-by-character
                        if chunk_type == "word":
                            tokens = re.split(r'(\s+)', text)
                            for token in tokens:
                                if token:
                                    yield token
                                    time.sleep(0.02)
                        else:  # character-level
                            for char in text:
                                yield char
                                time.sleep(0.01)
            
    except Exception as e:
        print(f"DEBUG: huggingface_chat_stream error: {type(e).__name__}: {e}")
        yield f"[Error with HuggingFace fallback: {str(e)}]"

###############################################
####### AUTOMATIC FALLBACK WRAPPER ############
###############################################

def chat_stream_with_fallback(messages, context_texts=None, chunk_type="word"):
    """
    Try Gemini first, fall back to HuggingFace on rate limit errors.
    Automatically handles switching between primary and fallback models.
    
    Args:
        messages: List of message strings
        context_texts: Optional list of context strings
        chunk_type: "word" or "char" for streaming animation
    
    Yields:
        Tuples of (chunk, is_fallback_notice) where is_fallback_notice is True only for the final notice
    """
    # Force HuggingFace for testing if enabled
    if FORCE_HF_FOR_TESTING:
        print(f"DEBUG: FORCE_HF_FOR_TESTING enabled, using HuggingFace directly")
        for chunk in huggingface_chat_stream(messages, context_texts, chunk_type):
            yield (chunk, False)
        return
    
    try:
        # Try Gemini first
        for chunk in gemini_chat_stream(messages, context_texts, chunk_type):
            yield (chunk, False)
    except Exception as e:
        error_str = str(e).lower()
        # Check if it's a rate limit error (429, ResourceExhausted, etc.)
        is_rate_limit = (
            "429" in error_str or
            "rate_limit" in error_str or
            "resourceexhausted" in error_str or
            "quota" in error_str or
            "limit" in error_str
        )
        
        if is_rate_limit and USE_FALLBACK_MODEL:
            print(f"DEBUG: Gemini rate limited, switching to HuggingFace fallback")
            # Stream HuggingFace response
            for chunk in huggingface_chat_stream(messages, context_texts, chunk_type):
                yield (chunk, False)
            # Yield fallback notice separately (marked as notice, not to be saved)
            yield (f"<em>Note: Gemini rate limit reached. Response provided by HuggingFace fallback model: {FALLBACK_MODEL_NAME}.</em>", True)
        else:
            # Re-raise the error if it's not a rate limit or fallback is disabled
            raise

###############################################
############# CHATBOT UI & LOGIC ##############
###############################################

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
            placeholder="Type your prompt here...",
            max_chars=None
        )
    with col2:
        send_clicked = st.form_submit_button("Send", use_container_width=True)

if send_clicked and user_input:
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Show the user's prompt as a placeholder bubble above the streaming response
    stream_placeholder.markdown(f"<div class='chat-bubble user'>{user_input}</div>", unsafe_allow_html=True)
    
    # Show animated "Thinking..." indicator while waiting for response
    bot_stream.markdown("<div class='chat-bubble bot'><em>  <span class='thinking-dots'></span></em></div>", unsafe_allow_html=True)

    # Build full message history for Gemini
    gemini_messages = []
    for sender, msg in st.session_state['chat_history']:
        if sender == "user":
            gemini_messages.append(msg)
        elif sender == "bot":
            gemini_messages.append(msg)
    gemini_messages.append(user_input)
    
    # Stream the bot reply above the input form, styled as a green chat bubble
    streamed_text = ""  # What gets saved to history (no notice)
    display_text = ""   # What gets displayed (includes notice)
    for chunk, is_notice in chat_stream_with_fallback(gemini_messages, context_texts):
        if chunk:
            display_text += chunk
            if not is_notice:
                streamed_text += chunk  # Only save actual response, not the notice
            bot_stream.markdown(f"<div class='chat-bubble bot'>{display_text}</div>", unsafe_allow_html=True)
    
    if not streamed_text:
        streamed_text = "[No response received from Gemini. The model may have finished early or returned no content.]"
        bot_stream.markdown(f"<div class='chat-bubble bot'>{streamed_text}</div>", unsafe_allow_html=True)
    st.session_state['chat_history'].append(("user", user_input))
    st.session_state['chat_history'].append(("bot", streamed_text))
    rerun()

# Footer with AI disclaimer
st.markdown(
    "<div class='ai-disclaimer'><em>LLM responses may contain inaccuracies. Please verify important information.</em></div>",
    unsafe_allow_html=True
)
