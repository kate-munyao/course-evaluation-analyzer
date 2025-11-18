import streamlit as st
import joblib
import json
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------------------------
# FIX: NLTK SETUP FOR STREAMLIT CLOUD
# -------------------------------------------------
NLTK_PATH = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_PATH, exist_ok=True)
nltk.data.path = [NLTK_PATH]

for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_PATH)

# Load stopwords AFTER download
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# -------------------------------------------------
# PAGE SETUP
# -------------------------------------------------
st.set_page_config(
    page_title="Course Evaluation Analyzer",
    page_icon="🎓",
    layout="wide"
)

st.title("Student Course Evaluation Analyzer")
st.write("Analyze topics and sentiment from student course evaluations for BBT 4106 & BBT 4206.")

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
try:
    lda_model = joblib.load('./model/topic_model_lda.pkl')
    topic_vectorizer = joblib.load('./model/topic_vectorizer.pkl')

    # Correct Sentiment Model Files
    sentiment_model = joblib.load('./model/sentiment_classifier.pkl')
    sentiment_vectorizer = joblib.load('./model/sentiment_vectorizer.pkl')

    with open('./model/topic_labels.json', 'r', encoding='utf-8') as f:
        topic_labels = json.load(f)
        topic_labels = {int(k): v for k, v in topic_labels.items()}

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# -------------------------------------------------
# CLEANING FUNCTIONS
# -------------------------------------------------
def clean_topic_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -------------------------------------------------
# FINAL FIXED SENTIMENT CLEANING
# -------------------------------------------------
def clean_sentiment_text(text):
    """
    Final simplified cleaning to match LogisticRegression training.
    - No stopword removal
    - No stemming
    - Keep words as-is except basic cleaning
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------------------------
# PREDICTION FUNCTIONS
# -------------------------------------------------
def get_topic_prediction(text):
    try:
        cleaned = clean_topic_text(text)
        vec = topic_vectorizer.transform([cleaned])
        probs = lda_model.transform(vec)[0]
        topic_id = int(np.argmax(probs))

        feature_names = topic_vectorizer.get_feature_names_out()
        comp = lda_model.components_[topic_id]
        top_idx = comp.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_idx]

        return {
            'topic_id': topic_id,
            'topic_label': topic_labels.get(topic_id, f"Topic {topic_id}"),
            'confidence': float(probs[topic_id]),
            'top_words': top_words,
            'all_probabilities': {
                topic_labels.get(i, f"Topic {i}"): float(p)
                for i, p in enumerate(probs)
            }
        }
    except Exception as e:
        return {'error': str(e)}



# -------------------------------------------------
# FINAL FIXED SENTIMENT PREDICTION
# -------------------------------------------------
def get_sentiment_prediction(text):
    try:
        cleaned = clean_sentiment_text(text)

        X = sentiment_vectorizer.transform([cleaned])
        pred = sentiment_model.predict(X)[0]
        proba = sentiment_model.predict_proba(X)[0]

        return {
            "sentiment": pred,
            "confidence": float(max(proba)),
            "all_probabilities": {
                cls: float(p) for cls, p in zip(sentiment_model.classes_, proba)
            }
        }

    except Exception as e:
        return {"error": str(e)}


# -------------------------------------------------
# UI START
# -------------------------------------------------

# Session
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

st.text_area(
    "Student Feedback:",
    value=st.session_state["input_text"],
    key="input_text",
    height=150
)

# Sample Inputs
st.markdown("### Sample Inputs")
col1, col2, col3 = st.columns(3)

with col1:
    st.button(
        "Positive Example",
        on_click=lambda: st.session_state.update({
            "input_text": "The practical sessions were very helpful and the lecturer explained things well."
        })
    )
with col2:
    st.button(
        "Neutral Example",
        on_click=lambda: st.session_state.update({
            "input_text": "The course was fine, but some topics moved too fast."
        })
    )
with col3:
    st.button(
        "Negative Example",
        on_click=lambda: st.session_state.update({
            "input_text": "The course was confusing and the instructions for assignments were unclear."
        })
    )

# Analyze
input_text = st.session_state["input_text"]

if st.button("Analyze Feedback"):
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            topic_result = get_topic_prediction(input_text)
            sentiment_result = get_sentiment_prediction(input_text)

            if 'error' in topic_result:
                st.error(f"Topic Error: {topic_result['error']}")
            elif 'error' in sentiment_result:
                st.error(f"Sentiment Error: {sentiment_result['error']}")
            else:
                st.success("Analysis Complete")
                emoji_map = {'positive': "😊", 'neutral': "😐", 'negative': "😞"}

                st.markdown("## Summary")
                st.markdown(f"""
**Topic:** {topic_result['topic_label']}  
**Sentiment:** {sentiment_result['sentiment'].capitalize()} {emoji_map.get(sentiment_result['sentiment'], '')}  
**Topic Confidence:** {topic_result['confidence']:.1%}  
**Sentiment Confidence:** {sentiment_result['confidence']:.1%}
                """)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🔍 Topic Details")
                    st.write("Top Words:", ", ".join(topic_result['top_words']))
                    st.markdown("#### Topic Probabilities")
                    for label, prob in topic_result['all_probabilities'].items():
                        st.progress(prob, text=f"{label}: {prob:.1%}")

                with col2:
                    st.subheader("📊 Sentiment Details")
                    for s, prob in sentiment_result['all_probabilities'].items():
                        st.progress(prob, text=f"{s.capitalize()}: {prob:.1%}")

# -------------------------------------------------
# OVERALL CHARTS
# -------------------------------------------------
st.subheader("📊 Overall Topic & Sentiment Summary")

try:
    df_full = pd.read_csv("./data/course_evals_with_topics_and_sentiments.csv")

    # Topic distribution
    topic_counts = df_full['topic_label'].value_counts().sort_index()

    fig1, ax1 = plt.subplots(figsize=(5, 3.2))
    sns.barplot(
        x=topic_counts.values,
        y=topic_counts.index,
        palette="viridis",
        ax=ax1
    )
    ax1.set_xlabel("Count", fontsize=8)
    ax1.set_ylabel("Topic", fontsize=8)
    ax1.set_title("Topic Distribution", fontsize=10)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig1, use_container_width=True)

    st.markdown("###")

    # Sentiment by topic
    sentiment_order = ["positive", "neutral", "negative"]
    colors = ["green", "orange", "red"]

    sentiment_by_topic = (
        df_full.groupby(["topic_label", "predicted_sentiment"])
        .size()
        .unstack(fill_value=0)[sentiment_order]
    )

    fig2, ax2 = plt.subplots(figsize=(5, 3.2))
    sentiment_by_topic.plot(
        kind="bar",
        stacked=True,
        color=colors,
        ax=ax2,
        width=0.7
    )

    ax2.set_ylabel("Count", fontsize=8)
    ax2.set_title("Sentiment by Topic", fontsize=10)
    ax2.legend(title="Sentiment", prop={"size": 7})
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig2, use_container_width=True)

except Exception as e:
    st.warning(f"Unable to show charts: {e}")

# Footer
st.markdown("---")
st.markdown("### About This Project")
st.write("This app was created for coursework to analyze student evaluations using NLP.")
