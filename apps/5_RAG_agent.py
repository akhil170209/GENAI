import base64
import os
from dotenv import load_dotenv
load_dotenv()
 # Ignores missing python-dotenv on Streamlit Cloud
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st


# ============================================================
# HELPER: LOAD LOCAL LOGO IMAGE
# ============================================================

def get_base64_image(image_filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, image_filename)
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

logo_base64 = get_base64_image("logo.png") 
img_src = f"data:image/png;base64,{logo_base64}" if logo_base64 else ""

# SESSION STATE


if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# RAG PROCESS FUNCTION


def process_document(path):

    # Load documents
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents=docs)

    # Embeddings & Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    vector_db = InMemoryVectorStore.from_documents(
        embedding=embeddings,
        documents=docs
    )

    # LLM & Tools setup
    llm = ChatGroq(
        model="openai/gpt-oss-120b"
    )

    @tool
    def retrieve_context(query: str):
        """Retrieve documents relevant to a query from the knowledge base."""

        context = ""

        docs = vector_db.similarity_search(
            query=query,
            k=3
        )

        for doc in docs:
            context += doc.page_content + "\n\n"

        return context

    system_prompt = """You are a helpful assistant that answers questions using retrieved context. 
        My knowledge base consists of the details from the uploaded document. 
        ALWAYS use the `retrieve_context` tool for questions requiring external knowledge."""

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BuddyBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STREAMLIT UI DESIGN & NEON STYLING


st.markdown(
    """
    <style>

    /* 1. LAYOUT & PROMPT BAR WIDTH ADJUSTMENT */
    .main .block-container {
        max-width: 800px !important; /* Gemini-style centered, readable width */
        padding-top: 5rem !important;  /* Space under the top header */
        padding-bottom: 8rem !important; /* Space above the fixed prompt bar */
    }

    /* 2. HEADER & SIDEBAR TOGGLE FIX */
    header[data-testid="stHeader"] {
        background: rgba(9, 10, 15, 0.8) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        z-index: 999999 !important;
    }

    /* style the sidebar button background & border */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarExpandButton"],
    button[data-testid="baseButton-headerNoPadding"],
    header[data-testid="stHeader"] button {
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: #ff007f !important; /* Solid pink background for high visibility */
        border: 1px solid #ff007f !important;
        border-radius: 8px !important;
        padding: 6px !important;
        box-shadow: 0 0 10px rgba(255, 0, 127, 0.5) !important;
        z-index: 9999999 !important;
        transition: all 0.2s ease-in-out !important;
    }

    /* Hover effect */
    [data-testid="stSidebarCollapseButton"]:hover,
    [data-testid="stSidebarExpandButton"]:hover,
    header[data-testid="stHeader"] button:hover {
        background-color: #00f3ff !important; /* Swaps to cyan on hover */
        border-color: #00f3ff !important;
        box-shadow: 0 0 12px rgba(0, 243, 255, 0.7) !important;
        cursor: pointer !important;
    }

    /* Ensure the chevron icon inside remains sharp and bright white */
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebarExpandButton"] svg,
    header[data-testid="stHeader"] button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
        stroke: #ffffff !important;
        width: 18px !important;
        height: 18px !important;
    }

    /* 3. MAIN NEON BACKGROUND */
    .stApp {
        background: radial-gradient(circle at 50% 20%, #1a0b2e 0%, #090a0f 100%) !important;
        color: #ffffff !important;
    }

    /* 4. NEON SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #0d0e15 !important;
        border-right: 2px solid #ff007f !important;
        box-shadow: 5px 0 20px rgba(255, 0, 127, 0.3) !important;
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        color: #00f3ff !important;
        text-shadow: 0 0 8px rgba(0, 243, 255, 0.6);
    }

    .sidebar-subtitle {
        font-size: 13px;
        color: #a7a9be !important;
    }

    .sidebar-section {
        font-size: 12px;
        font-weight: 600;
        color: #ff007f !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    /* 5. MAIN HEADER & WELCOME CARDS */
    .main-header {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 15px 20px;
        background: rgba(18, 19, 28, 0.85) !important;
        border: 1px solid rgba(0, 243, 255, 0.3) !important;
        border-radius: 16px !important;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.15) !important;
        margin-bottom: 25px;
    }

    .main-title {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff !important;
    }

    .main-subtitle {
        font-size: 13px;
        color: #a7a9be !important;
    }

    .welcome-container {
        text-align: center;
        margin-top: 8vh;
        margin-bottom: 30px;
        padding: 40px 20px;
        background: rgba(18, 19, 28, 0.85) !important;
        border: 1px solid rgba(0, 243, 255, 0.3) !important;
        border-radius: 20px !important;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.15) !important;
    }

    .welcome-title {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 10px;
    }

    .welcome-text {
        font-size: 15px;
        color: #a7a9be !important;
    }

    /* 6. CHAT MESSAGES */
    [data-testid="stChatMessage"] {
        background: rgba(18, 18, 26, 0.85) !important;
        border: 1px solid #00f3ff !important;
        border-radius: 12px !important;
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.25) !important;
        margin-bottom: 15px !important;
    }

    [data-testid="stChatMessage"] p, 
    [data-testid="stChatMessage"] span, 
    [data-testid="stChatMessage"] div {
        color: #ffffff !important;
    }

    /* 7. FIXED BOTTOM PROMPT CONTAINER */
    [data-testid="stBottom"] {
        background-color: #090a0f !important; /* Solid dark background matching app theme */
        border-top: 1px solid rgba(0, 243, 255, 0.15) !important;
        padding-top: 12px !important;
        padding-bottom: 16px !important;
        z-index: 9999 !important;
    }

    div[data-testid="stBottom"] > div {
        background-color: transparent !important;
        border: none !important;
        max-width: 700px !important; /* Reduced container width */
        margin: 0 auto !important;
    }

    /* Inner Chat Input Container */
    [data-testid="stChatInput"] {
        background-color: #12131c !important;
        border: 1px solid rgba(0, 243, 255, 0.3) !important;
        border-radius: 16px !important;
        padding: 4px 8px !important;
        box-shadow: 0 0 12px rgba(0, 243, 255, 0.15) !important;
    }

    /* Text Input Field & Typed Text Color Fix */
    [data-testid="stChatInput"] textarea {
        background-color: #12131c !important;
        color: #ffffff !important;
        font-size: 14px !important;
        caret-color: #00f3ff !important;
    }

    /* Placeholder Text ("Message BuddyBot...") */
    [data-testid="stChatInput"] textarea::placeholder {
        color: #a7a9be !important;
        opacity: 1 !important;
    }

    /* Active Focus State */
    [data-testid="stChatInput"]:focus-within {
        border-color: #ff007f !important;
        box-shadow: 0 0 15px rgba(255, 0, 127, 0.4) !important;
    }

    /* 8. BUTTONS & CARDS */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #00f3ff !important;
        background-color: #161722 !important;
        color: #00f3ff !important;
        font-weight: 600;
        box-shadow: 0 0 8px rgba(0, 243, 255, 0.2);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        border-color: #ff007f !important;
        color: #ff007f !important;
        box-shadow: 0 0 12px rgba(255, 0, 127, 0.5);
    }

    .status-card {
        padding: 12px;
        border-radius: 10px;
        background-color: #161722 !important;
        border: 1px solid #00f3ff !important;
        color: #ffffff !important;
        font-size: 13px;
        margin-bottom: 10px;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #22c55e;
        border-radius: 50%;
        margin-right: 7px;
    }

    .chat-footer {
        text-align: center;
        font-size: 11px;
        color: #9ca3af;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# SIDEBAR

with st.sidebar:

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px;">
        <img src="{img_src}" style="width: 45px; height: 45px; border-radius: 12px; object-fit: cover;">
        <div>
            <h3 style="margin: 0; padding: 0;">BuddyBot</h3>
            <p style="margin: 0; padding: 0; font-size: 14px; color: #888;">AI-powered document assistant</p>
        </div>
    </div>
""", 
    unsafe_allow_html=True
    )

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Knowledge Base Upload
    st.markdown('<div class="sidebar-section">Knowledge Base</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Select PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible"
    )

    if uploaded:
        if st.button("📚 Process Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                path = "./doc_files/"
                for file in uploaded:
                    with open(path + file.name, "wb") as f:
                        f.write(file.getvalue())

                process_document(path)
                st.rerun()

    # Knowledge Base Status
    st.markdown('<div class="sidebar-section">Status</div>', unsafe_allow_html=True)

    if st.session_state.document_uploaded:
        st.markdown(
            """
            <div class="status-card">
                <span class="status-dot"></span>
                Knowledge Base Ready
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="status-card">
                📄 No document uploaded
            </div>
            """,
            unsafe_allow_html=True
        )

    # Chat Search History Section
    st.markdown('<div class="sidebar-section">Chat History</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        for item in reversed(st.session_state.chat_history):
            st.markdown(
                f"""
                <div style="color: #ffffff; font-size: 13px; margin-bottom: 8px; background: #161722; padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(0,243,255,0.2);">
                    💬 {item['prompt'][:22]}...
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown('<div style="color: #a7a9be; font-size: 12px;">No search history yet.</div>', unsafe_allow_html=True)

    # About Section
    st.markdown('<div class="sidebar-section">About BuddyBot</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="color:#a7a9be; font-size:13px; line-height:1.6;">
        BuddyBot is a GenAI document assistant.
        Upload your PDF documents and ask questions about their content.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Technology Section
    st.markdown('<div class="sidebar-section">Technology</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="color: #ffffff; font-size: 13px; line-height: 1.8;">
            🧠 Groq LLM<br>
            🔎 Gemini Embeddings<br>
            📚 LangChain<br>
            🗂 In-Memory Vector Store
        </div>
        """,
        unsafe_allow_html=True
    )



# MAIN HEADER


st.markdown(
    f"""
    <div class="main-header">
        <img src="{img_src}" style="width: 45px; height: 45px; border-radius: 12px; object-fit: cover;">
        <div>
            <div class="main-title">BuddyBot</div>
            <div class="main-subtitle">AI-powered document assistant</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



# CHAT UI


if not st.session_state.document_uploaded:

    st.markdown(
        f"""
        <div class="welcome-container">
            <img src="{img_src}" style="width: 80px; height: 80px; border-radius: 20px; object-fit: cover; margin-bottom: 20px;">
            <div class="welcome-title">How can I help you today?</div>
            <div class="welcome-text">
                Upload a PDF document from the sidebar and start chatting with BuddyBot.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("👈 Upload your PDF from the sidebar to start chatting.")

else:

    # Display Previous Messages
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")

        if role == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)

    # Chat Input
    query = st.chat_input("Message BuddyBot...")

    if query:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Save to chat history list
        st.session_state.chat_history.append({"prompt": query})

        # Display user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(query)

        # AI Response Generation
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("BuddyBot is thinking..."):
                response = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    {"configurable": {"thread_id": 2}}
                )

            answer = response["messages"][-1].content
            st.markdown(answer)

        # Save AI response
        st.session_state.messages.append({"role": "ai", "content": answer})


# FOOTER

if st.session_state.document_uploaded:
    st.markdown(
        """
        <div class="chat-footer">
            BuddyBot can make mistakes. Check important information against your original documents.
        </div>
        """,
        unsafe_allow_html=True
    )
