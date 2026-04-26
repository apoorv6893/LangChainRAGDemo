import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------
# Page
# -------------------------------
st.set_page_config(page_title="Prompt Quality Agent", layout="wide")
st.title("🧠 Prompt Quality Agent")

st.markdown("""
This app:
1. Evaluates your prompt
2. Scores it (0–10)
3. Explains what's missing
4. Decides whether to fix it
5. Improves it if needed
""")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Settings")

api_key = st.sidebar.text_input("Gemini API Key", type="password")

model = st.sidebar.selectbox(
    "Select Gemini Model",
    [
        "models/gemini-2.5-flash",
        "models/gemini-1.5-pro"
    ]
)

temperature = st.sidebar.slider("Creativity", 0.0, 1.0, 0.2)

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

def evaluate_prompt(llm, prompt):
    eval_prompt = f"""
You are a PROMPT QUALITY EVALUATOR.

Evaluate the prompt based on:
- clarity
- specificity
- context
- structure

Return STRICT JSON:

{{
  "score": (0-10),
  "reason": "...",
  "missing": "...",
  "verdict": "good" or "needs_improvement"
}}

Prompt:
{prompt}
"""
    return llm.invoke(eval_prompt).content


def improve_prompt(llm, prompt, missing):
    fix_prompt = f"""
You are a PROMPT IMPROVEMENT AGENT.

Improve the prompt by fixing:
{missing}

Keep intent same but make it clearer and more effective.

Original Prompt:
{prompt}
"""
    return llm.invoke(fix_prompt).content


# -------------------------------
# UI
# -------------------------------
user_prompt = st.text_area("Enter a prompt")

run = st.button("🚀 Analyze Prompt")

# -------------------------------
# MAIN
# -------------------------------
if run:

    if not api_key:
        st.error("Enter API key")
        st.stop()

    if not user_prompt:
        st.error("Enter a prompt")
        st.stop()

    llm = get_llm()

    with st.spinner("Evaluating prompt..."):
        evaluation = evaluate_prompt(llm, user_prompt)

    st.subheader("📊 Evaluation Output")
    st.code(evaluation)

    # crude parsing (simple approach)
    eval_lower = evaluation.lower()

    if "needs_improvement" in eval_lower or '"score":' in eval_lower:

        if "needs_improvement" in eval_lower:
            st.warning("⚠️ Prompt needs improvement")

            with st.spinner("Improving prompt..."):
                improved = improve_prompt(llm, user_prompt, evaluation)

            st.subheader("✨ Improved Prompt")
            st.success(improved)

        else:
            st.success("✅ Prompt looks good")

    else:
        st.info("Could not parse evaluation clearly. Showing raw output.")
