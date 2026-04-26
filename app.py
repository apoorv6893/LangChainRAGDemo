# -------------------------------
# COMPLETE RAG DEMO (UI + MODEL SELECTION)
# -------------------------------

import streamlit as st
import time
import tempfile

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader

# -------------------------------
# UI CONFIG
# -------------------------------

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("📄 RAG Demo with LangChain")

# -------------------------------
# SIDEBAR: API + MODEL CONFIG
# -------------------------------

st.sidebar.header("🔐 API Configuration")

provider = st.sidebar.selectbox(
    "Select Provider",
    ["OpenAI", "Gemini"]
)

api_key = st.sidebar.text_input(
    "Enter API Key",
    type="password"
)

if provider == "OpenAI":
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["gpt-4o-mini", "gpt-4o"]
    )
elif provider == "Gemini":
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["gemini-1.5-flash", "gemini-1.5-pro"]
    )

if not api_key:
    st.warning("Please enter your API key to continue.")
    st.stop()

st.sidebar.success(f"Using {provider} - {model_name}")

# -------------------------------
# SIDEBAR: RAG CONTROLS
# -------------------------------

st.sidebar.header("⚙️ RAG Controls")

chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50)
top_k = st.sidebar.slider("Top-K Results", 1, 5, 3)

# -------------------------------
# INIT MODEL + EMBEDDINGS
# -------------------------------

if provider == "OpenAI":
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key
    )

    embeddings = OpenAIEmbeddings(api_key=api_key)

elif provider == "Gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

# -------------------------------
# FILE UPLOAD
# -------------------------------

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("File uploaded!")

    # -------------------------------
    # LOAD DOCUMENT
    # -------------------------------

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()
    st.write(f"Loaded {len(documents)} document(s)")

    # -------------------------------
    # SPLIT DOCUMENT
    # -------------------------------

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    docs = splitter.split_documents(documents)
    st.write(f"Split into {len(docs)} chunks")

    # -------------------------------
    # EMBEDDINGS + VECTOR STORE
    # -------------------------------

    embed_start = time.time()

    db = FAISS.from_documents(docs, embeddings)

    embed_time = time.time() - embed_start
    st.success(f"Embeddings created in {embed_time:.2f}s")

    # -------------------------------
    # QUERY INPUT
    # -------------------------------

    query = st.text_input("Ask a question")

    if query:

        # -------------------------------
        # RETRIEVAL
        # -------------------------------

        retrieval_start = time.time()

        results = db.similarity_search_with_score(query, k=top_k)

        retrieval_time = time.time() - retrieval_start

        st.subheader("🔍 Retrieved Chunks")

        context_texts = []

        for i, (doc, score) in enumerate(results):
            source = doc.metadata.get("source", "Unknown")

            st.markdown(f"""
            **Chunk {i+1}**
            - Score: `{score:.4f}` (lower = more similar)
            - Source: `{source}`
            """)

            st.write(doc.page_content)
            st.write("---")

            context_texts.append(doc.page_content)

        # -------------------------------
        # RAG ANSWER
        # -------------------------------

        llm_start = time.time()

        context = "\n".join(context_texts)

        rag_prompt = f"""
        Answer ONLY using the context below.
        If answer is not present, say "I don't know".

        Context:
        {context}

        Question:
        {query}
        """

        rag_answer = llm.predict(rag_prompt)

        rag_time = time.time() - llm_start

        # -------------------------------
        # NO-RAG ANSWER
        # -------------------------------

        no_rag_answer = llm.predict(query)

        # -------------------------------
        # OUTPUT
        # -------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ RAG Answer")
            st.write(rag_answer)

        with col2:
            st.subheader("⚠️ No RAG Answer")
            st.write(no_rag_answer)

        # -------------------------------
        # LATENCY
        # -------------------------------

        st.subheader("⏱️ Latency Breakdown")

        st.write(f"""
        - Embedding time: {embed_time:.2f}s  
        - Retrieval time: {retrieval_time:.2f}s  
        - LLM time: {rag_time:.2f}s  
        """)

        # -------------------------------
        # DEBUG
        # -------------------------------

        with st.expander("🧠 Debug Info"):
            st.write("Query:", query)
            st.write("Context used:", context)

