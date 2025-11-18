"""
Course Evaluation Analyzer - Streamlit App
BBT 4206 - Business Intelligence II
"""

import streamlit as st
import joblib
import json
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Course Evaluation Analyzer",
    page_icon="🎓",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        lda = joblib.load('./model/topic_model_lda.pkl')
        topic_vec = joblib.load('./model/topic_vectorizer.pkl')
        sentiment_model = joblib.load('./model/sentiment_classifier.pkl')
        sentiment_vec = joblib.load('./model/sentiment_vectorizer.pkl')
        
        with open('./model/topic_labels.json', 'r') as f:
            labels = json.load(f)
            labels = {int(k): v for k, v in labels.items()}
        
        return lda, topic_vec, sentiment_model, sentiment_vec, labels
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

lda, topic_vec, sentiment_model, sentiment_vec, topic_labels = load_models()

# Load results data for charts
@st.cache_data
def load_results():
    try:
        df = pd.read_csv('./data/course_evals_with_topics_and_sentiments.csv')
        return df
    except:
        return None

results_df = load_results()

# Text cleaning functions
def clean_for_topic(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_for_sentiment(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    # Simple stopwords
    stop = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
    words = [w for w in words if w not in stop and len(w) > 2]
    return ' '.join(words)

# Prediction functions
def predict_topic(text):
    cleaned = clean_for_topic(text)
    X = topic_vec.transform([cleaned])
    probs = lda.transform(X)[0]
    topic_id = int(np.argmax(probs))
    
    feature_names = topic_vec.get_feature_names_out()
    topic = lda.components_[topic_id]
    top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
    
    return {
        'topic_id': topic_id,
        'topic_label': topic_labels.get(topic_id, f'Topic {topic_id}'),
        'confidence': float(probs[topic_id]),
        'top_words': top_words,
        'all_probs': {topic_labels.get(i, f'Topic {i}'): float(p) for i, p in enumerate(probs)}
    }

def predict_sentiment(text):
    cleaned = clean_for_sentiment(text)
    X = sentiment_vec.transform([cleaned])
    pred = sentiment_model.predict(X)[0]
    probs = sentiment_model.predict_proba(X)[0]
    
    return {
        'sentiment': pred,
        'confidence': float(probs.max()),
        'all_probs': {s: float(p) for s, p in zip(sentiment_model.classes_, probs)}
    }

# Title and intro
st.title("🎓 Course Evaluation Analyzer")
st.markdown("""
### NLP-Powered Analysis for Business Intelligence Courses

Automatically analyze student course evaluations using **Topic Modeling** and **Sentiment Analysis**.
""")

# Sidebar with charts
if results_df is not None:
    with st.sidebar:
        st.header("📊 Analysis Results")
        
        # Topic distribution
        st.subheader("Topic Distribution")
        topic_counts = results_df['topic_label'].value_counts()
        fig1 = px.bar(
            x=topic_counts.index,
            y=topic_counts.values,
            labels={'x': 'Topic', 'y': 'Count'},
            color=topic_counts.values,
            color_continuous_scale='Blues'
        )
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Sentiment by topic
        st.subheader("Sentiment by Topic")
        sentiment_by_topic = results_df.groupby(['topic_label', 'predicted_sentiment']).size().unstack(fill_value=0)
        
        # Ensure all sentiment columns exist
        for sent in ['positive', 'neutral', 'negative']:
            if sent not in sentiment_by_topic.columns:
                sentiment_by_topic[sent] = 0
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Positive', x=sentiment_by_topic.index, 
                              y=sentiment_by_topic['positive'], marker_color='green'))
        fig2.add_trace(go.Bar(name='Neutral', x=sentiment_by_topic.index, 
                              y=sentiment_by_topic['neutral'], marker_color='orange'))
        fig2.add_trace(go.Bar(name='Negative', x=sentiment_by_topic.index, 
                              y=sentiment_by_topic['negative'], marker_color='red'))
        
        fig2.update_layout(barmode='group', height=400, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

# Main content
st.markdown("---")
st.subheader("📝 Analyze Student Feedback")

# Input area
input_text = st.text_area(
    "Enter course evaluation text:",
    height=120,
    placeholder="Example: The hands-on labs were excellent and helped me understand the concepts..."
)

# Example buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("✅ Positive Example"):
        input_text = "The hands-on labs were excellent and really helped me understand the concepts. The instructor explained things clearly."
        st.rerun()

with col2:
    if st.button("😐 Neutral Example"):
        input_text = "The course covered the required topics. Some parts were interesting."
        st.rerun()

with col3:
    if st.button("❌ Negative Example"):
        input_text = "The lecture materials were confusing and poorly organized. Not enough examples provided."
        st.rerun()

with col4:
    if st.button("🔄 Clear"):
        input_text = ""
        st.rerun()

# Analyze button
if st.button("🔍 Analyze Feedback", type="primary"):
    if not input_text or input_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Get predictions
            topic_result = predict_topic(input_text)
            sentiment_result = predict_sentiment(input_text)
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Summary
            sentiment_emoji = {
                'positive': '😊',
                'negative': '😞',
                'neutral': '😐'
            }
            
            st.markdown("## 📋 Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Topic",
                    value=topic_result['topic_label'],
                    delta=f"{topic_result['confidence']:.1%} confidence"
                )
            
            with col2:
                st.metric(
                    label="Sentiment",
                    value=f"{sentiment_result['sentiment'].capitalize()} {sentiment_emoji[sentiment_result['sentiment']]}",
                    delta=f"{sentiment_result['confidence']:.1%} confidence"
                )
            
            st.markdown("---")
            
            # Detailed results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Topic Analysis")
                st.markdown(f"**Identified Topic:** {topic_result['topic_label']}")
                st.markdown(f"**Confidence:** {topic_result['confidence']:.1%}")
                st.markdown("**Key Words:**")
                st.write(", ".join(topic_result['top_words'][:8]))
                
                st.markdown("**Topic Probabilities:**")
                for topic, prob in sorted(topic_result['all_probs'].items(), key=lambda x: x[1], reverse=True):
                    st.progress(prob, text=f"{topic}: {prob:.1%}")
            
            with col2:
                st.markdown("### 💭 Sentiment Analysis")
                st.markdown(f"**Detected Sentiment:** {sentiment_result['sentiment'].capitalize()} {sentiment_emoji[sentiment_result['sentiment']]}")
                st.markdown(f"**Confidence:** {sentiment_result['confidence']:.1%}")
                
                st.markdown("**Sentiment Probabilities:**")
                for sent in ['positive', 'neutral', 'negative']:
                    if sent in sentiment_result['all_probs']:
                        prob = sentiment_result['all_probs'][sent]
                        emoji = sentiment_emoji[sent]
                        st.progress(prob, text=f"{sent.capitalize()} {emoji}: {prob:.1%}")

# Footer
st.markdown("---")
st.markdown("""
## 📖 About

This tool uses:
- **Topic Modeling (LDA)** to identify themes in course evaluations
- **Machine Learning Classification** to determine sentiment

**Developed for:** BBT 4106 & BBT 4206 - Business Intelligence  
**Institution:** Strathmore University  
**Academic Year:** 2024/2025
""")