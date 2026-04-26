import streamlit as st
import tempfile

# LangChain core concepts
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="LangChain Teaching Demo", layout="wide")

st.title("🧠 LangChain Teaching Demo")

st.markdown("""
This app demonstrates core LangChain concepts:

1. Prompt Templates  
2. LLM (Model)  
3. Chains (multi-step reasoning)  
4. Chunking (document preparation)  
5. RAG vs No-RAG  
""")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Settings")

gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

model = st.sidebar.text_input(
    "Model",
    value="models/gemini-1.0-pro",
    help="Use a model your key supports"
)

chunk_size = st.sidebar.slider("Chunk Size", 200, 1000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50)

# -------------------------------
# Initialize LLM
# -------------------------------
def get_llm():
    return ChatGoogleGenerativeAI(
        google_api_key=gemini_key,
        model=model,
        temperature=0
    )

# -------------------------------
# SECTION 1: Prompt Templates
# -------------------------------
st.header("1️⃣ Prompt Template → LLM")

user_input = st.text_input("Enter a question")

prompt_template = PromptTemplate.from_template(
    "Explain the following question in simple terms:\n{question}"
)

if st.button("Run Prompt"):
    if not gemini_key:
        st.error("Enter API key")
    else:
        llm = get_llm()

        formatted_prompt = prompt_template.format(question=user_input)

        st.subheader("📜 Final Prompt Sent to Model")
        st.code(formatted_prompt)

        response = llm.invoke(formatted_prompt)

        st.subheader("🤖 LLM Output")
        st.write(response.content)

# -------------------------------
# SECTION 2: Chains
# -------------------------------
st.header("2️⃣ Chain (Multi-step reasoning)")

chain_input = st.text_input("Enter text to process")

if st.button("Run Chain"):
    if not gemini_key:
        st.error("Enter API key")
    else:
        llm = get_llm()

        # Step 1: Extract keywords
        step1_prompt = f"Extract keywords from: {chain_input}"
        step1 = llm.invoke(step1_prompt).content

        # Step 2: Summarize
        step2_prompt = f"Summarize this: {step1}"
        step2 = llm.invoke(step2_prompt).content

        st.subheader("Step 1: Keywords")
        st.write(step1)

        st.subheader("Step 2: Summary")
        st.write(step2)

        st.success("👉 This is a simple chain: multiple LLM calls")

# -------------------------------
# SECTION 3: Chunking
# -------------------------------
st.header("3️⃣ Document Chunking")

uploaded_file = st.file_uploader("Upload TXT file for chunking", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_text(text)

    st.write(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:5]):
        st.markdown(f"**Chunk {i+1}**")
        st.write(chunk)

# -------------------------------
# SECTION 4: RAG vs No-RAG
# -------------------------------
st.header("4️⃣ RAG vs No-RAG")

rag_query = st.text_input("Ask a question based on document")

if st.button("Compare RAG vs No-RAG"):
    if not gemini_key:
        st.error("Enter API key")
    elif not uploaded_file:
        st.error("Upload a document first")
    else:
        llm = get_llm()

        # No RAG
        no_rag = llm.invoke(rag_query).content

        # Fake RAG (use first chunk as context)
        context = chunks[0]

        rag_prompt = f"""
Answer ONLY from this context:
{context}

Question:
{rag_query}
"""

        rag = llm.invoke(rag_prompt).content

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("❌ Without Context")
            st.write(no_rag)

        with col2:
            st.subheader("✅ With Context (RAG)")
            st.write(rag)

        st.info("👉 Notice how context improves accuracy")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("🚀 This demo focuses on understanding LangChain, not infra complexity.")
