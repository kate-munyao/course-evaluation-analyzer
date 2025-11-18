"""
Simple Sentiment Analysis for Course Evaluations
BBT 4206 - Business Intelligence II
"""

import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK data
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

print("="*70)
print("SENTIMENT ANALYSIS - COURSE EVALUATIONS")
print("="*70)

# Load data
print("\n[1/6] Loading data...")
df = pd.read_csv('./data/course_evals_with_topics.csv')
print(f"✅ Loaded {len(df)} evaluations")
print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}\n")

# Clean text for sentiment
print("\n[2/6] Cleaning text for sentiment analysis...")
stop_words = set(stopwords.words('english'))

def clean_for_sentiment(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

df['clean_text_sa'] = df['clean_text'].apply(clean_for_sentiment)
print("✅ Text cleaned")

# Create TF-IDF features
print("\n[3/6] Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2)
)

X = tfidf.fit_transform(df['clean_text_sa'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Features created: {X.shape}")
print(f"Training: {X_train.shape[0]} | Testing: {X_test.shape[0]}")

# Train models
print("\n[4/6] Training models...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": MultinomialNB()
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"  {name}: {scores.mean():.3f} accuracy")

# Pick best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

print(f"\n✅ Best model: {best_model_name} ({results[best_model_name]:.3f})")

# Evaluate
y_pred = best_model.predict(X_test)
print("\n" + "="*70)
print("CLASSIFICATION REPORT:")
print("="*70)
print(classification_report(y_test, y_pred))

# Predict for all data
print("\n[5/6] Predicting sentiments for all evaluations...")
df['predicted_sentiment'] = best_model.predict(X)
df['prediction_confidence'] = best_model.predict_proba(X).max(axis=1)

# Create sentiment by topic chart
print("\n[6/6] Creating charts...")
sentiment_by_topic = df.groupby(['topic_label', 'predicted_sentiment']).size().unstack(fill_value=0)

# Make sure we have all sentiment columns
for sent in ['positive', 'neutral', 'negative']:
    if sent not in sentiment_by_topic.columns:
        sentiment_by_topic[sent] = 0

sentiment_by_topic = sentiment_by_topic[['positive', 'neutral', 'negative']]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(sentiment_by_topic.index))
width = 0.25

ax.bar(x - width, sentiment_by_topic['positive'], width, label='Positive', color='green')
ax.bar(x, sentiment_by_topic['neutral'], width, label='Neutral', color='orange')
ax.bar(x + width, sentiment_by_topic['negative'], width, label='Negative', color='red')

ax.set_xlabel('Topic', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Sentiment Distribution by Topic', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sentiment_by_topic.index, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('./model/sentiment_by_topic.png', dpi=150, bbox_inches='tight')
print("✅ Chart saved: ./model/sentiment_by_topic.png")
plt.close()

# Save everything
print("\n[SAVING] Saving models and data...")
df.to_csv('./data/course_evals_with_topics_and_sentiments.csv', index=False)
joblib.dump(best_model, './model/sentiment_classifier.pkl')
joblib.dump(tfidf, './model/sentiment_vectorizer.pkl')

print("✅ All files saved!")
print("\n" + "="*70)
print("SENTIMENT ANALYSIS COMPLETE!")
print("="*70)
print(f"\nSummary:")
print(f"  Positive: {(df['predicted_sentiment'] == 'positive').sum()}")
print(f"  Neutral: {(df['predicted_sentiment'] == 'neutral').sum()}")
print(f"  Negative: {(df['predicted_sentiment'] == 'negative').sum()}")
