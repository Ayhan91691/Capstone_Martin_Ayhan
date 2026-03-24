import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_sentiment_model()

def predict_sentiment_batch(texts):
    results = sentiment_model(texts)

    return [
        "Positive" if r["label"] == "POSITIVE" else "Negative"
        for r in results
    ]



