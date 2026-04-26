import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Hierarchical Agents Demo", layout="wide")

st.title("🧠 Hierarchical Agents Demo")
st.markdown("""
**Pattern: Orchestrator + Specialized Agents**

Flow:
1. Orchestrator breaks task  
2. Delegates to:
   - Research Agent  
   - Writing Agent  
   - Review Agent  
3. Combines outputs → Final answer
""")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Settings")

api_key = st.sidebar.text_input("Gemini API Key", type="password")

model = st.sidebar.text_input(
    "Gemini Model (exact name)",
    value="models/gemini-1.5-flash",
    help="Example: models/gemini-1.5-flash"
)

temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.2)

# -------------------------------
# LLM
# -------------------------------
def get_llm():
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=model,
        temperature=temperature
    )

# -------------------------------
# AGENTS
# -------------------------------

def research_agent(llm, query):
    prompt = f"""
You are a RESEARCH agent.

Task:
- Gather facts
- Provide bullet points
- Be concise

Query:
{query}
"""
    return llm.invoke(prompt).content


def writing_agent(llm, research_output):
    prompt = f"""
You are a WRITING agent.

Task:
- Convert research into structured answer
- Make it readable
- Add flow

Input:
{research_output}
"""
    return llm.invoke(prompt).content


def review_agent(llm, draft):
    prompt = f"""
You are a REVIEW agent.

Task:
- Check factual consistency
- Improve clarity
- Fix errors
- Output final polished answer

Draft:
{draft}
"""
    return llm.invoke(prompt).content


# -------------------------------
# ORCHESTRATOR
# -------------------------------
def orchestrator(llm, user_query):
    steps = {}

    # Step 1: Research
    steps["research"] = research_agent(llm, user_query)

    # Step 2: Writing
    steps["writing"] = writing_agent(llm, steps["research"])

    # Step 3: Review
    steps["review"] = review_agent(llm, steps["writing"])

    return steps


# -------------------------------
# UI INPUT
# -------------------------------
user_query = st.text_area("Enter your task")

run = st.button("🚀 Run Agent System")

# -------------------------------
# MAIN
# -------------------------------
if run:

    if not api_key:
        st.error("Enter Gemini API key")
        st.stop()

    if not user_query:
        st.error("Enter a task")
        st.stop()

    try:
        llm = get_llm()

        with st.spinner("Orchestrator running..."):
            results = orchestrator(llm, user_query)

        # -------------------------------
        # OUTPUT
        # -------------------------------
        st.subheader("🧠 Orchestrator Output")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🔍 Research Agent")
            st.write(results["research"])

        with col2:
            st.markdown("### ✍️ Writing Agent")
            st.write(results["writing"])

        with col3:
            st.markdown("### ✅ Review Agent")
            st.write(results["review"])

        st.markdown("---")
        st.subheader("🎯 Final Answer")
        st.success(results["review"])

    except Exception as e:
        st.error(f"Error: {e}")
