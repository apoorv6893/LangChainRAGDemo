import streamlit as st
import tempfile
import time
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("🔍 RAG Demo (End-to-End)")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Settings")

provider = st.sidebar.selectbox(
    "LLM Provider ℹ️",
    ["OpenAI", "Gemini"],
    help="Choose which AI brain to use to answer your question"
)

api_key = st.sidebar.text_input(
    "API Key 🔐",
    type="password",
    value=os.getenv("OPENAI_API_KEY", ""),
    help="Your private key to access the AI model. It is not stored anywhere."
)

model = st.sidebar.text_input(
    "Model ℹ️",
    value="gpt-4o-mini" if provider == "OpenAI" else "gemini-1.5-flash",
    help="Different models have different quality and speed"
)

chunk_size = st.sidebar.slider(
    "Chunk Size 📦",
    200, 1500, 500,
    help="How big each piece of document is. Smaller = more precise, larger = more context"
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap 🔁",
    0, 300, 50,
    help="How much chunks overlap. Helps avoid missing important info between splits"
)

top_k = st.sidebar.slider(
    "Top-K Results 🎯",
    1, 10, 3,
    help="How many relevant chunks we fetch before answering"
)

debug_mode = st.sidebar.checkbox(
    "Debug Mode 🛠",
    help="Shows internal working like chunks, context, and timings"
)

# -------------------------------
# File Upload
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload documents (PDF or TXT) 📄",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="Upload documents that the AI should use to answer"
)

query = st.text_input(
    "Ask a question ❓",
    help="Ask anything based on your uploaded documents"
)

# -------------------------------
# Helper Functions
# -------------------------------

def load_documents(files):
    docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)

        docs.extend(loader.load())
    return docs


def get_embeddings():
    if provider == "OpenAI":
        return OpenAIEmbeddings(api_key=api_key)
    else:
        return GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model="models/embedding-001"
        )


def get_llm():
    if provider == "OpenAI":
        return ChatOpenAI(api_key=api_key, model=model, temperature=0)
    else:
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=0
        )


def build_prompt(context, query):
    return f"""
You are a strict assistant.

Answer ONLY using the context below.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{query}
"""


# -------------------------------
# Main Execution
# -------------------------------

if uploaded_files and query and api_key:

    # Load docs
    docs = load_documents(uploaded_files)

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    # Embeddings
    start_embed = time.time()
    embeddings = get_embeddings()
    vectorstore = DocArrayInMemorySearch.from_documents(chunks, embeddings)
    embed_time = time.time() - start_embed

    # Retrieval
    start_retrieval = time.time()
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    retrieval_time = time.time() - start_retrieval

    retrieved_docs = [doc for doc, score in results]
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # LLM (RAG)
    start_llm = time.time()
    llm = get_llm()
    rag_response = llm.invoke(build_prompt(context, query))
    llm_time = time.time() - start_llm

    # -------------------------------
    # Output
    # -------------------------------

    st.subheader("✅ Answer (with RAG)")
    st.write(rag_response.content)

    # No-RAG comparison
    st.subheader("❌ Answer (without RAG)")
    no_rag = llm.invoke(query)
    st.write(no_rag.content)

    # Retrieved chunks
    st.subheader("📄 Retrieved Chunks")
    for i, (doc, score) in enumerate(results):
        st.markdown(f"**Chunk {i+1} | Similarity Score: {score:.4f}**")
        st.write(doc.page_content[:500])

    # Debug
    if debug_mode:
        st.subheader("🛠 Debug Info")

        st.markdown("### Context sent to model")
        st.code(context[:2000])

        st.markdown("### Latency")
        st.write(f"Embedding: {embed_time:.2f}s")
        st.write(f"Retrieval: {retrieval_time:.2f}s")
        st.write(f"LLM: {llm_time:.2f}s")

        st.write(f"Total chunks: {len(chunks)}")

# -------------------------------
# Empty state guidance
# -------------------------------
else:
    st.info("Upload documents, enter API key, and ask a question to begin.")
