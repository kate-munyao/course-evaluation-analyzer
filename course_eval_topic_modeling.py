"""
Simple Topic Modeling for Course Evaluations
BBT 4206 - Business Intelligence II
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

print("="*70)
print("TOPIC MODELING - COURSE EVALUATIONS")
print("="*70)

# Load data
print("\n[1/6] Loading course evaluation data...")
df = pd.read_csv('./data/202511-ft_bi1_bi2_course_evaluation.csv')
print(f"✅ Loaded {len(df)} evaluations")

# Combine text columns
print("\n[2/6] Preparing text data...")
df['text'] = (
    df['f_3_Write_at_least_two_things_you_liked_about_the_teaching_and_learning_in_this_course'].fillna('') + ' ' +
    df['f_4_Write_at_least_one_recommendation_to_improve_the_teaching_and_learning_in_this_course_(for_future_classes)'].fillna('')
)

# Remove empty rows
df = df[df['text'].str.strip() != ''].reset_index(drop=True)
print(f"✅ Prepared {len(df)} evaluations")

# Clean text
print("\n[3/6] Cleaning text...")
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)
print("✅ Text cleaned")

# Create document-term matrix
print("\n[4/6] Creating document-term matrix...")
vectorizer = CountVectorizer(
    max_df=0.90,
    min_df=2,
    max_features=500,
    stop_words='english'
)

dtm = vectorizer.fit_transform(df['clean_text'])
print(f"✅ Matrix created: {dtm.shape[0]} documents x {dtm.shape[1]} words")

# Train LDA model
print("\n[5/6] Training topic model...")
lda = LatentDirichletAllocation(
    n_components=5,
    random_state=42,
    max_iter=20,
    n_jobs=-1
)

lda.fit(dtm)
print(f"✅ Model trained (Perplexity: {lda.perplexity(dtm):.2f})")

# Show topics
print("\n" + "="*70)
print("DISCOVERED TOPICS:")
print("="*70)
feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
    print(f"\nTopic {topic_idx + 1}: {' | '.join(top_words)}")

# Assign topics to documents
topic_results = lda.transform(dtm)
df['dominant_topic'] = topic_results.argmax(axis=1)
df['topic_probability'] = topic_results.max(axis=1)

# Label topics
topic_labels = {
    0: "Assessment Methods and Feedback",
    1: "Course Content and Structure",
    2: "Learning Resources and Materials",
    3: "Practical Labs and Hands-on Learning",
    4: "Teaching Quality and Engagement"
}

df['topic_label'] = df['dominant_topic'].map(topic_labels)

# Visualize
print("\n[6/6] Creating visualization...")
plt.figure(figsize=(10, 6))
topic_counts = df['dominant_topic'].value_counts().sort_index()

plt.bar(range(len(topic_counts)), topic_counts.values, color='steelblue', edgecolor='black')
plt.xlabel('Topic ID', fontsize=12)
plt.ylabel('Number of Evaluations', fontsize=12)
plt.title('Topic Distribution', fontsize=14, fontweight='bold')
plt.xticks(range(len(topic_counts)), [f"Topic {i+1}" for i in range(len(topic_counts))])

for i, count in enumerate(topic_counts.values):
    plt.text(i, count + 1, str(count), ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('./model/topic_distribution.png', dpi=150, bbox_inches='tight')
print("✅ Chart saved: ./model/topic_distribution.png")
plt.close()

# Save everything
print("\n[SAVING] Saving models and data...")
df[['text', 'clean_text', 'sentiment', 'average_course_evaluation_rating', 
    'dominant_topic', 'topic_label', 'topic_probability']].to_csv(
    './data/course_evals_with_topics.csv', index=False
)

joblib.dump(lda, './model/topic_model_lda.pkl')
joblib.dump(vectorizer, './model/topic_vectorizer.pkl')

with open('./model/topic_labels.json', 'w') as f:
    json.dump(topic_labels, f, indent=2)

print("✅ All files saved!")
print("\n" + "="*70)
print("TOPIC MODELING COMPLETE!")
print("="*70)
print("\nNext step: Run sentiment analysis script")