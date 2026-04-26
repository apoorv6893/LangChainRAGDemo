import streamlit as st
import tempfile
import time
import os

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# Models
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("🔍 RAG Demo (Production-ready Demo)")

st.markdown("""
### 🧠 What happens behind the scenes?

1. 📄 We read your documents  
2. ✂️ Break them into smaller pieces  
3. 🔢 Convert text into numbers (embeddings)  
4. 🎯 Find most relevant pieces  
5. 🤖 Answer using ONLY those pieces  

This prevents guessing and improves accuracy.
""")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Settings")

provider = st.sidebar.selectbox(
    "LLM Provider",
    ["OpenAI", "Gemini"],
    help="Choose which AI model answers your question"
)

api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    help="Your API key (not stored)"
)

model = st.sidebar.text_input(
    "Model",
    value="gpt-4o-mini" if provider == "OpenAI" else "gemini-1.5-flash",
    help="Model to use for answering"
)

chunk_size = st.sidebar.slider(
    "Chunk Size",
    200, 1500, 500,
    help="How big each text piece is"
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    0, 300, 50,
    help="Overlap between chunks to avoid missing context"
)

top_k = st.sidebar.slider(
    "Top-K Results",
    1, 10, 3,
    help="Number of relevant chunks used to answer"
)

debug_mode = st.sidebar.checkbox(
    "Debug Mode",
    help="Show internal working"
)

# -------------------------------
# Inputs
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

query = st.text_input("Ask a question")

run = st.button("🚀 Run RAG Query")

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
    # Always OpenAI for stability
    return OpenAIEmbeddings(api_key=api_key)


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
if run:

    if not uploaded_files:
        st.error("Upload at least one document")
    elif not query:
        st.error("Enter a question")
    elif not api_key:
        st.error("Enter API key")
    else:

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
            start_embed = time.time()
            embeddings = get_embeddings()
            vectorstore = DocArrayInMemorySearch.from_documents(chunks, embeddings)
            embed_time = time.time() - start_embed

            # Retrieval
            start_retrieval = time.time()
            results = vectorstore.similarity_search_with_score(query, k=top_k)
            retrieval_time = time.time() - start_retrieval

            retrieved_docs = [doc for doc, _ in results]
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # LLM (RAG)
            start_llm = time.time()
            llm = get_llm()
            rag_response = llm.invoke(build_prompt(context, query))
            llm_time = time.time() - start_llm

        # -------------------------------
        # Outputs
        # -------------------------------

        st.subheader("✅ Answer (with RAG)")
        st.write(rag_response.content)

        st.subheader("❌ Answer (without RAG)")
        st.write(llm.invoke(query).content)

        # Retrieved chunks
        st.subheader("📄 Retrieved Chunks")

        for i, (doc, score) in enumerate(results):
            st.markdown(f"""
**Chunk {i+1}**

📊 Relevance Score: {score:.4f}  
👉 Lower score = better match
""")
            st.write(doc.page_content[:500])

        # Latency
        col1, col2, col3 = st.columns(3)
        col1.metric("Embedding Time", f"{embed_time:.2f}s")
        col2.metric("Retrieval Time", f"{retrieval_time:.2f}s")
        col3.metric("LLM Time", f"{llm_time:.2f}s")

        # Debug
        if debug_mode:
            st.subheader("🛠 Debug Info")
            st.code(context[:2000])
            st.write(f"Total chunks: {len(chunks)}")
