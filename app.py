import streamlit as st
import tempfile

st.set_page_config(page_title="RAG Demo - Step 1", layout="wide")

st.title("🔍 RAG Demo (Step 1: Inputs Only)")

# -------------------------------
# Sidebar (API + Model)
# -------------------------------

st.sidebar.header("⚙️ Configuration")

provider = st.sidebar.selectbox(
    "LLM Provider",
    ["OpenAI", "Gemini"]
)

api_key = st.sidebar.text_input(
    "API Key",
    type="password"
)

model = st.sidebar.text_input(
    "Model (optional)",
    placeholder="e.g. gpt-4o-mini / gemini-1.5-flash"
)

# -------------------------------
# File Upload
# -------------------------------

st.subheader("📄 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded")

    for file in uploaded_files:
        st.write(f"• {file.name}")

# -------------------------------
# Query Input
# -------------------------------

st.subheader("❓ Ask a Question")

query = st.text_input("Enter your question")

# -------------------------------
# Run Button
# -------------------------------

if st.button("Run Query"):

    if not api_key:
        st.error("Please enter API key")
    elif not uploaded_files:
        st.error("Please upload at least one document")
    elif not query:
        st.error("Please enter a question")
    else:
        st.success("Inputs captured successfully")

        st.write("### Debug Info")
        st.write(f"Provider: {provider}")
        st.write(f"Model: {model}")
        st.write(f"Query: {query}")
        st.write(f"Files: {[f.name for f in uploaded_files]}")

        # Preview first file content (basic check)
        first_file = uploaded_files[0]

        if first_file.name.endswith(".txt"):
            content = first_file.read().decode("utf-8")
            st.write("### Preview (First 500 chars)")
            st.write(content[:500])
        else:
            st.info("PDF uploaded (preview not shown yet)")
