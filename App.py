import streamlit as st
import joblib
import json
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Basic Streamlit page setup
st.set_page_config(
    page_title="Course Evaluation Analyzer",
    page_icon="🎓",
    layout="wide"
)

# Download NLTK data if missing
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

download_nltk_data()

# Load saved models and vectorizers
@st.cache_resource
def load_models():
    try:
        lda_model = joblib.load('./model/topic_model_lda.pkl')
        topic_vectorizer = joblib.load('./model/topic_vectorizer.pkl')
        sentiment_model = joblib.load('./model/sentiment_classifier.pkl')
        sentiment_vectorizer = joblib.load('./model/topic_vectorizer_using_tfidf.pkl')

        with open('./model/topic_labels.json', 'r', encoding='utf-8') as f:
            topic_labels = json.load(f)
            topic_labels = {int(k): v for k, v in topic_labels.items()}

        return lda_model, topic_vectorizer, sentiment_model, sentiment_vectorizer, topic_labels

    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

lda_model, topic_vectorizer, sentiment_model, sentiment_vectorizer, topic_labels = load_models()

# NLP setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Clean text for topic model
def clean_text_for_topic(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean text for sentiment model
def clean_text_for_sentiment(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# Predict topic
def get_topic_prediction(text):
    try:
        cleaned = clean_text_for_topic(text)
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

# Predict sentiment
def get_sentiment_prediction(text):
    try:
        cleaned = clean_text_for_sentiment(text)
        vec = sentiment_vectorizer.transform([cleaned])
        pred = sentiment_model.predict(vec)[0]
        proba = sentiment_model.predict_proba(vec)[0]

        return {
            'sentiment': pred,
            'confidence': float(max(proba)),
            'all_probabilities': {
                s: float(p) for s, p in zip(sentiment_model.classes_, proba)
            }
        }

    except Exception as e:
        return {'error': str(e)}

# Streamlit UI Starts Here

st.title(" Student Course Evaluation Analyzer")

st.markdown("""
This tool helps analyze student feedback for **BBT 4106** and **BBT 4206**.
It identifies the topic of the feedback and the overall sentiment (positive, neutral, or negative).
""")

st.subheader("Enter Feedback Below")

input_text = st.text_area(
    "Student Feedback:",
    placeholder="Example: The labs were helpful and the instructor explained concepts clearly.",
    height=150
)

# Example feedback buttons
st.markdown("### Sample Inputs")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Positive Example"):
        input_text = "The practical sessions were very helpful and the lecturer explained things well."
        st.rerun()

with col2:
    if st.button("Neutral Example"):
        input_text = "The course was fine, but some topics moved too fast."
        st.rerun()

with col3:
    if st.button("Negative Example"):
        input_text = "The course was confusing and the instructions for assignments were unclear."
        st.rerun()


# Analyze button
if st.button("Analyze Feedback", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text before running the analysis.")
    else:
        with st.spinner("Processing..."):
            topic_result = get_topic_prediction(input_text)
            sentiment_result = get_sentiment_prediction(input_text)

            if 'error' in topic_result:
                st.error(f"Topic Error: {topic_result['error']}")
            elif 'error' in sentiment_result:
                st.error(f"Sentiment Error: {sentiment_result['error']}")
            else:
                st.success("Analysis Complete")

                emoji_map = {
                    'positive': "😊",
                    'neutral': "😐",
                    'negative': "😞"
                }

                # Summary
                st.markdown("## Summary")
                st.markdown(f"""
                **Topic:** {topic_result['topic_label']}  
                **Sentiment:** {sentiment_result['sentiment'].capitalize()} {emoji_map.get(sentiment_result['sentiment'], '')}  
                **Topic Confidence:** {topic_result['confidence']:.1%}  
                **Sentiment Confidence:** {sentiment_result['confidence']:.1%}
                """)

                # Detailed section
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Topic Details")
                    st.write("Top Words:", ", ".join(topic_result['top_words']))

                    st.markdown("#### Topic Probabilities:")
                    for label, prob in sorted(topic_result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                        st.progress(prob, text=f"{label}: {prob:.1%}")

                with col2:
                    st.markdown("### Sentiment Details")
                    for s, prob in sentiment_result['all_probabilities'].items():
                        st.progress(prob, text=f"{s.capitalize()}: {prob:.1%}")

# Footer
st.markdown("""
---
### About This Project
This app was created for coursework to help analyze student evaluations using basic NLP.
""")
