# Sentiment Analysis for Course Evaluations
# BBT 4206: Business Intelligence II

import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud

# Download NLTK data
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

print("="*80)
print("SENTIMENT ANALYSIS FOR COURSE EVALUATIONS")
print("="*80)

# Step 1: Load data with topics
print("\n[1/8] Loading data...")
df = pd.read_csv('./data/course_evals_with_topics.csv')
print(f"✅ Loaded {len(df)} course evaluations")
print(f"   Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")

# Step 2: Text preprocessing for sentiment analysis
print("\n[2/8] Preprocessing text for sentiment analysis...")
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text_for_sentiment(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    return " ".join(stemmed)

df['clean_text_sa'] = df['clean_text'].apply(clean_text_for_sentiment)
print("✅ Text preprocessed")

# Step 3: Feature engineering with TF-IDF
print("\n[3/8] Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2)  # unigrams and bigrams
)

X = tfidf.fit_transform(df['clean_text_sa'])
y = df['sentiment']

print(f"✅ Feature matrix created: {X.shape}")
print(f"   Vocabulary size: {len(tfidf.get_feature_names_out())}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")

# Step 4: Train multiple models with cross-validation
print("\n[4/8] Training and evaluating models...")
models = {
    "Logistic Regression": LogisticRegression(multi_class='multinomial', 
                                               solver='lbfgs', max_iter=1000, 
                                               random_state=53),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=53),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, 
                                             random_state=53, n_jobs=-1)
}

scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=53)
cv_results = {}

for name, model in models.items():
    print(f"   Cross-validating {name}...")
    scores = cross_validate(
        model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
    )
    cv_results[name] = {
        'Accuracy': scores['test_accuracy'].mean(),
        'Precision': scores['test_precision_weighted'].mean(),
        'Recall': scores['test_recall_weighted'].mean(),
        'F1-Score': scores['test_f1_weighted'].mean()
    }

# Display results
results_df = pd.DataFrame(cv_results).T.sort_values('F1-Score', ascending=False)
print("\n✅ Cross-validation results:")
print(results_df.round(4))

# Step 5: Select and train best model
print("\n[5/8] Training best model...")
best_model_name = results_df.index[0]
best_model = models[best_model_name]
best_model.fit(X_train, y_train)
print(f"✅ Best model: {best_model_name}")
print(f"   F1-Score: {results_df.loc[best_model_name, 'F1-Score']:.4f}")

# Evaluate on test set
y_pred = best_model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=['positive', 'neutral', 'negative'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Positive', 'Neutral', 'Negative'],
            yticklabels=['Positive', 'Neutral', 'Negative'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('./model/confusion_matrix.png', dpi=150)
print("✅ Saved: ./model/confusion_matrix.png")
plt.close()

# Step 6: Predict sentiment for all evaluations
print("\n[6/8] Predicting sentiment for all evaluations...")
df['predicted_sentiment'] = best_model.predict(X)
df['prediction_confidence'] = best_model.predict_proba(X).max(axis=1)
print("✅ Predictions complete")

# Step 7: Analyze sentiment by topic
print("\n[7/8] Analyzing sentiment distribution by topic...")
sentiment_by_topic = df.groupby(['topic_label', 'predicted_sentiment']).size().unstack(fill_value=0)

sentiment_colors = {'positive': 'green', 'neutral': 'orange', 'negative': 'red'}
sentiment_order = ['positive', 'neutral', 'negative']
colors = [sentiment_colors[s] for s in sentiment_order if s in sentiment_by_topic.columns]

plt.figure(figsize=(12, 6))
ax = sentiment_by_topic[sentiment_order].plot(kind='bar', stacked=False, 
                                                color=colors, edgecolor='black')

# Wrap long labels
wrapped_labels = ['\n'.join(textwrap.wrap(label, width=15)) 
                  for label in sentiment_by_topic.index]
ax.set_xticklabels(wrapped_labels, rotation=45, ha='right')

plt.title('Sentiment Distribution by Topic', fontsize=14)
plt.xlabel('Topic', fontsize=12)
plt.ylabel('Number of Evaluations', fontsize=12)
plt.legend(title='Sentiment', loc='upper right')
plt.tight_layout()

# Add count labels
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=9)

plt.savefig('./model/sentiment_by_topic.png', dpi=150)
print("✅ Saved: ./model/sentiment_by_topic.png")
plt.close()

# Print summary
print("\n📊 Sentiment by Topic:")
print(sentiment_by_topic)

# Step 8: Save models and results
print("\n[8/8] Saving models and results...")

# Save final data
output_path = './data/course_evals_with_topics_and_sentiments.csv'
df.to_csv(output_path, index=False)
print(f"✅ Data saved to {output_path}")

# Save models
joblib.dump(best_model, './model/sentiment_classifier.pkl')
joblib.dump(tfidf, './model/topic_vectorizer_using_tfidf.pkl')
print("✅ Models saved:")
print("   - ./model/sentiment_classifier.pkl")
print("   - ./model/topic_vectorizer_using_tfidf.pkl")

# Generate word clouds by sentiment
print("\n📊 Generating word clouds...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, sentiment in enumerate(['positive', 'neutral', 'negative']):
    text = " ".join(df[df['predicted_sentiment'] == sentiment]['text'])
    if text.strip():
        wordcloud = WordCloud(background_color='white', max_words=50).generate(text)
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f"{sentiment.capitalize()} Evaluations", fontsize=14)
        axes[i].axis('off')

plt.tight_layout()
plt.savefig('./model/sentiment_wordclouds.png', dpi=150)
print("✅ Saved: ./model/sentiment_wordclouds.png")
plt.close()

print("\n" + "="*80)
print("SENTIMENT ANALYSIS COMPLETE!")
print("="*80)
print(f"\nSummary:")
print(f"  Total evaluations: {len(df)}")
print(f"  Best model: {best_model_name}")
print(f"  Test F1-Score: {results_df.loc[best_model_name, 'F1-Score']:.4f}")
print(f"\nSentiment distribution:")
for sentiment in ['positive', 'neutral', 'negative']:
    count = (df['predicted_sentiment'] == sentiment).sum()
    pct = count / len(df) * 100
    print(f"  {sentiment.capitalize()}: {count} ({pct:.1f}%)")
print(f"\nNext step: Run the Gradio app with: python app.py")