import streamlit as st
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

st.set_page_config(page_title="Forecast Sentiment", layout="centered")

st.title("Forecast Sentiment")

# 👉 Elemente nebeneinander
col1, col2, col3 = st.columns([4, 2, 1])

with col1:
    text_input = st.text_input(
        "Text to forecast sentiment",
        label_visibility="collapsed",
        placeholder="Enter text here..."
    )

with col2:
    selected_class = st.selectbox(
        "Class",
        options=["2class", "3class"],
        label_visibility="collapsed"
    )

with col3:
    forecast_clicked = st.button("Forecast", width="stretch")


# =========================
# MODEL LOADING (FIXED)
# =========================
@st.cache_resource
def load_model(model_path):
    # ✅ FIX für Mistral Tokenizer Bug
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        fix_mistral_regex=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )


MODEL_PATHS = {
    "2class": "data/best_model_2class_auto",
    "3class": "data/best_model_3class_auto",
}


# =========================
# LABEL NORMALIZATION
# =========================
def normalize_label(raw_label, mode):
    label = str(raw_label).lower().strip()

    # Star-basierte Labels
    if "star" in label:
        if "1" in label or "2" in label:
            return "negative"
        if "3" in label:
            return "neutral" if mode == "3class" else "positive"
        if "4" in label or "5" in label:
            return "positive"
        return label

    # HuggingFace Labels
    if mode == "2class":
        if "label_0" in label:
            return "negative"
        if "label_1" in label:
            return "positive"

    if mode == "3class":
        if "label_0" in label:
            return "negative"
        if "label_1" in label:
            return "neutral"
        if "label_2" in label:
            return "positive"

    # Falls Modell schon Klartext liefert
    if "positive" in label:
        return "positive"
    if "negative" in label:
        return "negative"
    if "neutral" in label:
        return "neutral"

    return label


# =========================
# RESULT
# =========================
if forecast_clicked:
    if not text_input.strip():
        st.warning("Please enter text.")
    else:
        with st.spinner("Running prediction..."):
            model = load_model(MODEL_PATHS[selected_class])
            result = model(text_input)[0]
            normalized = normalize_label(result["label"], selected_class)

        st.subheader("Result")

        col4, col5 = st.columns(2)

        with col4:
            st.metric("Sentiment", normalized)

        with col5:
            st.metric("Confidence", f"{result['score']:.2f}")