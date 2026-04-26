import streamlit as st
import tempfile
import time

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import HuggingFaceEmbeddings

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("🔍 RAG Demo (Free + Gemini Stable)")

st.markdown("""
### 🧠 What happens behind the scenes?

1. 📄 Read documents  
2. ✂️ Split into chunks  
3. 🔢 Convert text into embeddings (numbers)  
4. 🎯 Retrieve relevant chunks  
5. 🤖 Answer using ONLY those chunks  

No paid embedding API used.
""")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Settings")

gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

model = st.sidebar.text_input(
    "Preferred Model",
    value="gemini-pro",
    help="If unavailable, system auto-falls back"
)

chunk_size = st.sidebar.slider("Chunk Size", 200, 1500, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 300, 50)
top_k = st.sidebar.slider("Top-K Results", 1, 10, 3)

debug_mode = st.sidebar.checkbox("Debug Mode")

# -------------------------------
# Inputs
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF or TXT",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

query = st.text_input("Ask a question")
run = st.button("🚀 Run RAG Query")

# -------------------------------
# Helpers
# -------------------------------
def load_documents(files):
    docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        loader = PyPDFLoader(path) if file.name.endswith(".pdf") else TextLoader(path)
        docs.extend(loader.load())

    return docs


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_llm(model_name):
    return ChatGoogleGenerativeAI(
        google_api_key=gemini_key,
        model=model_name,
        temperature=0
    )


def safe_invoke(model_name, prompt):
    """Smart invoke with fallback and correct retry handling"""

    try:
        llm = get_llm(model_name)
        return llm.invoke(prompt)

    except Exception as e:
        error = str(e)

        # ❌ Do NOT retry for model-not-found
        if "404" in error:
            st.warning("Model not available → switching to gemini-pro")
            llm = get_llm("gemini-pro")
            return llm.invoke(prompt)

        # ✅ Retry only for transient errors
        delays = [1, 2, 4]
        for d in delays:
            try:
                time.sleep(d)
                llm = get_llm(model_name)
                return llm.invoke(prompt)
            except:
                continue

        # final fallback
        llm = get_llm("gemini-pro")
        return llm.invoke(prompt)


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
if run:

    if not uploaded_files:
        st.error("Upload documents")
        st.stop()

    if not query:
        st.error("Enter a question")
        st.stop()

    if not gemini_key:
        st.error("Enter Gemini API key")
        st.stop()

    with st.spinner("Processing..."):

        # Load
        docs = load_documents(uploaded_files)

        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)

        # Embeddings
        t1 = time.time()
        embeddings = get_embeddings()
        vectorstore = DocArrayInMemorySearch.from_documents(chunks, embeddings)
        embed_time = time.time() - t1

        # Retrieval
        t2 = time.time()
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        retrieval_time = time.time() - t2

        context = "\n\n".join([doc.page_content for doc, _ in results])

        # RAG Answer
        t3 = time.time()
        rag_response = safe_invoke(model, build_prompt(context, query))
        llm_time = time.time() - t3

    # -------------------------------
    # Output
    # -------------------------------
    st.subheader("✅ Answer (with RAG)")
    st.write(rag_response.content)

    st.subheader("❌ Answer (without RAG)")
    st.write(safe_invoke(model, query).content)

    st.subheader("📄 Retrieved Chunks")

    for i, (doc, score) in enumerate(results):
        st.markdown(f"""
**Chunk {i+1}**

📊 Score: {score:.4f}  
👉 Lower = better match
""")
        st.write(doc.page_content[:400])

    # Latency
    col1, col2, col3 = st.columns(3)
    col1.metric("Embedding", f"{embed_time:.2f}s")
    col2.metric("Retrieval", f"{retrieval_time:.2f}s")
    col3.metric("LLM", f"{llm_time:.2f}s")

    # Debug
    if debug_mode:
        st.subheader("🛠 Debug Info")
        st.code(context[:2000])
        st.write(f"Total chunks: {len(chunks)}")
