import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import pickle
import numpy as np
import streamlit as st
import json

from src.autocorrect import autocorrect_text
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🤖",
    layout="wide"
)
st.markdown(
    """
    <style>

    .main{
        padding-top:2rem;
    }

    h1{
        color:#0E76A8;
        text-align:center;
    }

    h3{
        color:#1E88E5;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_prediction_model():
    return load_model("models/next_word_model.keras")


@st.cache_resource
def load_tokenizer():
    with open("models/tokenizer.pkl", "rb") as file:
        return pickle.load(file)


model = load_prediction_model()
tokenizer = load_tokenizer()


# -----------------------------
# Load Configuration
# -----------------------------
with open("models/config.json", "r") as file:
    config = json.load(file)

MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
VOCAB_SIZE = config["vocab_size"]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📊 Project Information")

st.sidebar.markdown("---")

st.sidebar.metric(
    "Vocabulary",
    f"{VOCAB_SIZE:,}"
)

st.sidebar.metric(
    "Dataset",
    "Tiny Shakespeare"
)

st.sidebar.metric(
    "Model",
    "LSTM"
)

st.sidebar.metric(
    "Framework",
    "TensorFlow"
)

st.sidebar.markdown("---")

st.sidebar.success(
    """
Developer:
Bhuvanesh Gupta

Built with:
• TensorFlow
• Streamlit
• Python
"""
)
# -----------------------------
# Prediction Function
# -----------------------------
def predict_next_words(text, top_k=3):

    token_list = tokenizer.texts_to_sequences([text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=MAX_SEQUENCE_LENGTH - 1,
        padding="pre"
    )

    prediction = model.predict(
        token_list,
        verbose=0
    )[0]

    top_indices = np.argsort(prediction)[-top_k:][::-1]

    results = []

    for index in top_indices:

        word = tokenizer.index_word.get(index)

        if word:
            confidence = prediction[index] * 100
            results.append((word, confidence))

    return results


# -----------------------------
# User Interface
# -----------------------------
st.title("⌨ AI Keyboard Assistant")

st.markdown(
"""
### LSTM Based Next Word Prediction

Predict the next word using a Deep Learning model trained on the Tiny Shakespeare dataset.
"""
)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Vocabulary", f"{VOCAB_SIZE:,}")

with col2:
    st.metric("Dataset", "Shakespeare")

with col3:
    st.metric("Predictions", "Top 3")

sentence = st.text_input(
    "✍ Enter your sentence",
    placeholder="Example: to be"
)
enable_autocorrect = st.checkbox(
    "Enable Autocorrect",
    value=False
)

if st.button("🚀 Predict"):

    if sentence.strip() == "":
        st.warning("Please enter a sentence.")
    else:

        processed_sentence = sentence

        if enable_autocorrect:
            processed_sentence = autocorrect_text(sentence)

        predictions = predict_next_words(processed_sentence)
        if enable_autocorrect:
            st.markdown("## ✍ Input Processing")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Original Input**")
                st.info(sentence)

            with col2:
                st.write("**Corrected Input**")
                st.success(processed_sentence)

        st.success("Prediction Generated Successfully!")

        st.markdown("## 🏆 Top Predictions")

        medals = ["🥇", "🥈", "🥉"]

        for medal, (word, confidence) in zip(medals, predictions):

            st.markdown(f"## {medal} **{word.title()}**")

            progress_value = float(confidence) / 100
            st.progress(progress_value)

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

            st.divider()
            st.markdown("---")

st.markdown("---")

st.caption(
    "Developed by Bhuvanesh Gupta | LSTM-Based Next Word Prediction System | TensorFlow • Streamlit • Python"
)