import streamlit as st
import tempfile

# Stable imports (older LangChain for Cloud reliability)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

st.set_page_config(page_title="RAG Demo - Step 2", layout="wide")

st.title("🔍 RAG Demo (Step 2: Chunking)")

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.header("⚙️ Configuration")

chunk_size = st.sidebar.slider("Chunk Size", 200, 1500, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 300, 50)

# -------------------------------
# Upload
# -------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
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


def split_docs(docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

# -------------------------------
# Main
# -------------------------------

if uploaded_files:

    st.success(f"{len(uploaded_files)} file(s) uploaded")

    # Load
    docs = load_documents(uploaded_files)

    st.subheader("📄 Raw Documents")
    st.write(f"Total documents loaded: {len(docs)}")

    # Split
    chunks = split_docs(docs, chunk_size, chunk_overlap)

    st.subheader("🧩 Chunks Created")
    st.write(f"Total chunks: {len(chunks)}")

    # Show chunks
    for i, chunk in enumerate(chunks[:5]):
        st.markdown(f"**Chunk {i+1}**")
        st.write(chunk.page_content[:500])
