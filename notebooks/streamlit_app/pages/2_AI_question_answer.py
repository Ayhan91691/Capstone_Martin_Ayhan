import streamlit as st
import pandas as pd

from ml_functions.ml_rag_qa_groq import (
    ask_bmw_rag_question_groq,
    load_prompt_configs,
)


st.set_page_config(
    page_title="Feedback Q&A",
    page_icon="💬",
    layout="wide",
)

# =========================
# LOAD PROMPTS
# =========================
prompt_configs = load_prompt_configs()
prompt_keys = list(prompt_configs.keys())

if "selected_prompt_key" not in st.session_state:
    st.session_state["selected_prompt_key"] = "strict" if "strict" in prompt_configs else prompt_keys[0]


# =========================
# STYLE
# =========================
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #f8fafc !important;
    }

    .hero-box {
        padding: 1.8rem 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(168,85,247,0.18));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
        margin-bottom: 1.5rem;
    }

    .glass-card {
        padding: 1.2rem 1.3rem;
        border-radius: 22px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        backdrop-filter: blur(8px);
        margin-bottom: 1rem;
    }

    .kpi-card {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .answer-box {
        padding: 1.2rem;
        border-radius: 18px;
        background: rgba(59,130,246,0.12);
        border: 1px solid rgba(255,255,255,0.08);
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="hero-box">
        <h1>💬 Ask Your Feedback Data</h1>
        <p>Ask questions about your app reviews using RAG + LLM</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Settings")

    selected_k = st.slider("Top Reviews", 5, 25, 10)
    negative_only = st.toggle("Negative only", True)

    model_name = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Controls randomness. 0 = deterministic, 1 = more creative",
    )

    selected_prompt_key = st.selectbox(
        "Prompt Style",
        options=prompt_keys,
        index=prompt_keys.index(st.session_state["selected_prompt_key"]),
        format_func=lambda key: prompt_configs[key]["label"],
    )

    st.session_state["selected_prompt_key"] = selected_prompt_key

    st.caption(prompt_configs[selected_prompt_key]["description"])

# =========================
# INPUT
# =========================
col1, col2, col3 = st.columns([6, 1.5, 1.5], vertical_alignment="bottom")

with col1:
    question = st.text_input(
        "Ask something",
        placeholder="e.g. What are the main login problems?",
        label_visibility="collapsed",
    )

with col2:
    ask_clicked = st.button("Ask", width="stretch")

with col3:
    clear_clicked = st.button("Clear", width="stretch")

# =========================
# CLEAR
# =========================
if clear_clicked:
    st.session_state.pop("qa_result", None)
    st.rerun()

# =========================
# ASK
# =========================
if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            result = ask_bmw_rag_question_groq(
                question=question,
                k=selected_k,
                negative_only=negative_only,
                model_name=model_name,
                temperature=temperature,
                prompt_key=selected_prompt_key,
                max_context_chars=500,
            )
            st.session_state["qa_result"] = result

# =========================
# RESULT
# =========================
result = st.session_state.get("qa_result")

if result:
    colA, colB, colC, colD, colE = st.columns(5)

    with colA:
        st.markdown(
            f"<div class='kpi-card'><b>Model</b><br>{result['model_name']}</div>",
            unsafe_allow_html=True,
        )

    with colB:
        st.markdown(
            f"<div class='kpi-card'><b>Used in Answer</b><br>{len(result['sources'])}</div>",
            unsafe_allow_html=True,
        )

    with colC:
        st.markdown(
            f"<div class='kpi-card'><b>Mode</b><br>{'Negative' if negative_only else 'All'}</div>",
            unsafe_allow_html=True,
        )

    with colD:
        st.markdown(
            f"<div class='kpi-card'><b>Temperature</b><br>{result['temperature']}</div>",
            unsafe_allow_html=True,
        )

    with colE:
        st.markdown(
            f"<div class='kpi-card'><b>Prompt</b><br>{result['prompt_label']}</div>",
            unsafe_allow_html=True,
        )

    colF, colG, colH = st.columns(3)

    with colF:
        st.markdown(
            f"<div class='kpi-card'><b>Dataset Size</b><br>{result.get('dataset_size', '-')}</div>",
            unsafe_allow_html=True,
        )

    with colG:
        st.markdown(
            f"<div class='kpi-card'><b>Candidate Pool</b><br>{result.get('candidate_pool', '-')}</div>",
            unsafe_allow_html=True,
        )

    with colH:
        st.markdown(
            f"<div class='kpi-card'><b>Retrieved</b><br>{result.get('retrieved_count', '-')}</div>",
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3 = st.tabs(["💡 Answer", "📚 Sources", "🧾 Context"])

    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Answer")
        st.markdown(f"<div class='answer-box'>{result['answer']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        df = result["sources"].copy()
        cols = [c for c in ["content", "score", "distance"] if c in df.columns]
        st.dataframe(df[cols], width="stretch")

    with tab3:
        for i, txt in enumerate(result["used_reviews"], 1):
            st.markdown(f"**Review {i}**")
            st.write(txt)
            st.divider()