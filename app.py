import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="LangChain Teaching Demo", layout="wide")
st.title("🧠 LangChain Teaching Demo (Stable Version)")

st.markdown("""
This demo shows:
1. Prompt Templates  
2. LLM Calls  
3. Chains  
4. Chunking  
5. RAG vs No-RAG  
""")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")

hf_key = st.sidebar.text_input("HuggingFace API Key", type="password")

model_repo = st.sidebar.text_input(
    "Model Repo",
    value="google/flan-t5-large",
    help="Free and reliable model"
)

chunk_size = st.sidebar.slider("Chunk Size", 200, 1000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50)

# ---------------- LLM ----------------
def get_llm():
    return HuggingFaceHub(
        repo_id=model_repo,
        huggingfacehub_api_token=hf_key,
        model_kwargs={"temperature": 0}
    )

# ---------------- SECTION 1 ----------------
st.header("1️⃣ Prompt Template → LLM")

user_input = st.text_input("Enter a question")

prompt_template = PromptTemplate.from_template(
    "Explain simply:\n{question}"
)

if st.button("Run Prompt"):
    if not hf_key:
        st.error("Enter HuggingFace API key")
    else:
        llm = get_llm()
        prompt = prompt_template.format(question=user_input)

        st.subheader("📜 Prompt")
        st.code(prompt)

        response = llm.invoke(prompt)

        st.subheader("🤖 Output")
        st.write(response)

# ---------------- SECTION 2 ----------------
st.header("2️⃣ Chain")

chain_input = st.text_input("Enter text")

if st.button("Run Chain"):
    if not hf_key:
        st.error("Enter API key")
    else:
        llm = get_llm()

        step1 = llm.invoke(f"Extract key points:\n{chain_input}")
        step2 = llm.invoke(f"Summarize:\n{step1}")

        st.write("Step 1:", step1)
        st.write("Step 2:", step2)

# ---------------- SECTION 3 ----------------
st.header("3️⃣ Chunking")

text = st.text_area("Paste text")

if text:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_text(text)

    st.write(f"Chunks: {len(chunks)}")

    for i, c in enumerate(chunks[:5]):
        st.markdown(f"**Chunk {i+1}**")
        st.write(c)

# ---------------- SECTION 4 ----------------
st.header("4️⃣ RAG vs No-RAG")

rag_query = st.text_input("Ask a question")

if st.button("Compare"):
    if not hf_key:
        st.error("Enter API key")
    elif not text:
        st.error("Paste text first")
    else:
        llm = get_llm()
        chunks = splitter.split_text(text)
        context = chunks[0]

        no_rag = llm.invoke(rag_query)

        rag_prompt = f"""
Answer ONLY using this:
{context}

Question:
{rag_query}
"""

        rag = llm.invoke(rag_prompt)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("❌ Without Context")
            st.write(no_rag)

        with col2:
            st.subheader("✅ With Context")
            st.write(rag)
