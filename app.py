import streamlit as st
import time

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="LangChain Teaching Demo", layout="wide")
st.title("🧠 LangChain Teaching Demo (Stable Version)")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Settings")

gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

chunk_size = st.sidebar.slider("Chunk Size", 200, 1000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50)

# -------------------------------
# Gemini Model Detection
# -------------------------------
CANDIDATE_MODELS = [
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
    "models/gemini-pro",
    "models/gemini-1.0-pro"
]

def get_llm(model):
    return ChatGoogleGenerativeAI(
        google_api_key=gemini_key,
        model=model,
        temperature=0
    )

def find_working_model():
    for m in CANDIDATE_MODELS:
        try:
            llm = get_llm(m)
            llm.invoke("Say OK")
            return m
        except Exception:
            continue
    return None

# -------------------------------
# Initialize Model
# -------------------------------
if gemini_key:
    working_model = find_working_model()
    if working_model:
        st.success(f"✅ Using model: {working_model}")
    else:
        st.error("❌ No Gemini model available for this API key")
        st.stop()
else:
    st.info("Enter Gemini API key to begin")
    st.stop()

llm = get_llm(working_model)

# -------------------------------
# SECTION 1: Prompt Template
# -------------------------------
st.header("1️⃣ Prompt Template → LLM")

user_input = st.text_input("Enter a question")

prompt_template = PromptTemplate.from_template(
    "Explain this simply:\n{question}"
)

if st.button("Run Prompt"):
    final_prompt = prompt_template.format(question=user_input)

    st.subheader("📜 Prompt Sent")
    st.code(final_prompt)

    response = llm.invoke(final_prompt)

    st.subheader("🤖 Output")
    st.write(response.content)

# -------------------------------
# SECTION 2: Chains
# -------------------------------
st.header("2️⃣ Chain (Multi-step reasoning)")

chain_input = st.text_input("Enter text")

if st.button("Run Chain"):
    step1 = llm.invoke(f"Extract key points:\n{chain_input}").content
    step2 = llm.invoke(f"Summarize:\n{step1}").content

    st.write("Step 1 (Key Points):", step1)
    st.write("Step 2 (Summary):", step2)

# -------------------------------
# SECTION 3: Chunking
# -------------------------------
st.header("3️⃣ Chunking")

text = st.text_area("Paste text here")

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

# -------------------------------
# SECTION 4: RAG vs No-RAG
# -------------------------------
st.header("4️⃣ RAG vs No-RAG")

rag_query = st.text_input("Ask a question")

if st.button("Compare"):
    if not text:
        st.error("Paste text first")
    else:
        chunks = splitter.split_text(text)
        context = chunks[0]

        no_rag = llm.invoke(rag_query).content

        rag_prompt = f"""
Answer ONLY using this:
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
            st.subheader("✅ With Context")
            st.write(rag)

        st.info("👉 Context improves grounding")
