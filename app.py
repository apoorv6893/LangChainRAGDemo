import streamlit as st
import tempfile
import time

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# Models
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("🔍 RAG Demo (End-to-End)")

st.markdown("""
### 🧠 What happens behind the scenes?

1. 📄 Read documents  
2. ✂️ Split into chunks  
3. 🔢 Convert text into numbers (embeddings)  
4. 🎯 Find relevant chunks  
5. 🤖 Answer using ONLY those chunks  

This reduces hallucination.
""")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Settings")

provider = st.sidebar.selectbox(
    "LLM Provider",
    ["OpenAI", "Gemini"],
    help="Which model generates the final answer"
)

# 🔥 Separate keys (IMPORTANT)
openai_key = st.sidebar.text_input(
    "OpenAI API Key (for embeddings)",
    type="password"
)

gemini_key = st.sidebar.text_input(
    "Gemini API Key (for LLM)",
    type="password"
)

model = st.sidebar.text_input(
    "Model",
    value="gpt-4o-mini" if provider == "OpenAI" else "gemini-1.5-flash",
    help="Example: gemini-1.5-flash or gemini-2.5-flash"
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

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)

        docs.extend(loader.load())
    return docs


def get_embeddings():
    return OpenAIEmbeddings(api_key=openai_key)


def get_llm():
    if provider == "OpenAI":
        return ChatOpenAI(api_key=openai_key, model=model, temperature=0)
    else:
        return ChatGoogleGenerativeAI(
            google_api_key=gemini_key,
            model=model,
            temperature=0
        )


def build_prompt(context, query):
    return f"""
You are a strict assistant.

Answer ONLY from the context below.
If answer is not present, say "I don't know."

Context:
{context}

Question:
{query}
"""


# -------------------------------
# Main
# -------------------------------
if run:

    # Validation
    if not uploaded_files:
        st.error("Upload documents")
        st.stop()

    if not query:
        st.error("Enter a question")
        st.stop()

    if not openai_key:
        st.error("OpenAI key required for embeddings")
        st.stop()

    if provider == "Gemini" and not gemini_key:
        st.error("Gemini key required")
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

        # LLM (RAG)
        t3 = time.time()
        llm = get_llm()
        rag_response = llm.invoke(build_prompt(context, query))
        llm_time = time.time() - t3

    # -------------------------------
    # Output
    # -------------------------------

    st.subheader("✅ Answer (with RAG)")
    st.write(rag_response.content)

    st.subheader("❌ Answer (without RAG)")
    st.write(llm.invoke(query).content)

    st.subheader("📄 Retrieved Chunks")

    for i, (doc, score) in enumerate(results):
        st.markdown(f"""
**Chunk {i+1}**

📊 Score: {score:.4f}  
👉 Lower = more relevant
""")
        st.write(doc.page_content[:400])

    col1, col2, col3 = st.columns(3)
    col1.metric("Embedding", f"{embed_time:.2f}s")
    col2.metric("Retrieval", f"{retrieval_time:.2f}s")
    col3.metric("LLM", f"{llm_time:.2f}s")

    if debug_mode:
        st.subheader("🛠 Debug")
        st.code(context[:2000])
        st.write(f"Chunks: {len(chunks)}")
