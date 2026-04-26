import streamlit as st
import tempfile
import time
from typing import List
import os

# -------------------------------
# Robust Imports (Cloud-safe)
# -------------------------------

# Text splitter (fallback)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Loaders (fallback)
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader, TextLoader

# Vector store (fallback)
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

from langchain.schema import Document

# OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Gemini
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

# -------------------------------
# Utility Functions
# -------------------------------

def load_documents(uploaded_files) -> List[Document]:
    docs = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        docs.extend(loader.load())

    return docs


def split_documents(docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def get_embeddings(provider, api_key):
    if provider == "OpenAI":
        return OpenAIEmbeddings(api_key=api_key)
    else:
        return GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model="models/embedding-001"
        )


def get_llm(provider, api_key, model):
    if provider == "OpenAI":
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0
        )
    else:
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=0
        )


def build_vectorstore(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)


def retrieve_chunks(vectorstore, query, k):
    return vectorstore.similarity_search_with_score(query, k=k)


def build_prompt(context, query):
    return f"""
You are a strict question-answering assistant.

Answer ONLY using the provided context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{query}
"""


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("🔍 RAG Demo (LangChain + Streamlit)")

st.sidebar.header("⚙️ Configuration")

provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "Gemini"])

api_key = st.sidebar.text_input(
    "API Key",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password"
)

if provider == "OpenAI":
    model = st.sidebar.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o"]
    )
else:
    model = st.sidebar.selectbox(
        "Model",
        ["gemini-1.5-pro", "gemini-1.5-flash"]
    )

chunk_size = st.sidebar.slider("Chunk Size", 200, 1500, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 300, 50)
top_k = st.sidebar.slider("Top-K Retrieval", 1, 10, 3)

debug_mode = st.sidebar.checkbox("🛠 Debug Mode")

uploaded_files = st.file_uploader(
    "Upload documents (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

query = st.text_input("Ask a question")

# -------------------------------
# Execution
# -------------------------------

if uploaded_files and query and api_key:

    docs = load_documents(uploaded_files)
    chunks = split_documents(docs, chunk_size, chunk_overlap)

    # Embeddings
    start_embed = time.time()
    embeddings = get_embeddings(provider, api_key)
    vectorstore = build_vectorstore(chunks, embeddings)
    embed_time = time.time() - start_embed

    # Retrieval
    start_retrieval = time.time()
    results = retrieve_chunks(vectorstore, query, top_k)
    retrieval_time = time.time() - start_retrieval

    retrieved_docs = [doc for doc, score in results]
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # LLM (RAG)
    start_llm = time.time()
    llm = get_llm(provider, api_key, model)

    rag_prompt = build_prompt(context, query)
    rag_response = llm.invoke(rag_prompt)
    llm_time = time.time() - start_llm

    # -------------------------------
    # Outputs
    # -------------------------------

    st.subheader("✅ RAG Answer")
    st.write(rag_response.content)

    st.subheader("❌ No-RAG Answer")
    no_rag_response = llm.invoke(f"Answer the question: {query}")
    st.write(no_rag_response.content)

    st.subheader("📄 Retrieved Chunks")
    for i, (doc, score) in enumerate(results):
        st.markdown(f"**Chunk {i+1} | Score: {score:.4f}**")
        st.write(doc.page_content[:500])

    if debug_mode:
        st.subheader("🛠 Debug Info")

        st.markdown("### Context sent to LLM")
        st.code(context[:2000])

        st.markdown("### Latency Breakdown")
        st.write(f"Embedding Time: {embed_time:.2f}s")
        st.write(f"Retrieval Time: {retrieval_time:.2f}s")
        st.write(f"LLM Time: {llm_time:.2f}s")

        st.markdown("### Total Chunks")
        st.write(len(chunks))
