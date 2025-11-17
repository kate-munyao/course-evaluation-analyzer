# Topic Modeling for Course Evaluations
# BBT 4206: Business Intelligence II

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import textwrap
import math
import joblib
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

print("="*80)
print("TOPIC MODELING FOR COURSE EVALUATIONS")
print("="*80)

# Step 1: Load the course evaluation data
print("\n[1/9] Loading data...")
df = pd.read_csv('./data/202511-ft_bi1_bi2_course_evaluation.csv')
print(f"✅ Loaded {len(df)} course evaluations")

# Step 2: Combine text columns for topic modeling
print("\n[2/9] Combining text fields...")
# Combine f_3 (what they liked) and f_4 (recommendations)
df['text'] = df['f_3_Write_at_least_two_things_you_liked_about_the_teaching_and_learning_in_this_course'].fillna('') + ' ' + \
             df['f_4_Write_at_least_one_recommendation_to_improve_the_teaching_and_learning_in_this_course_(for_future_classes)'].fillna('')

# Remove rows with empty text
df = df[df['text'].str.strip() != ''].reset_index(drop=True)
print(f"✅ Combined text from {len(df)} evaluations")

# Step 3: Clean text
print("\n[3/9] Cleaning text...")
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)
print("✅ Text cleaned")
print(f"Sample: {df['clean_text'].iloc[0][:100]}...")

# Step 4: Create Document-Term Matrix
print("\n[4/9] Creating Document-Term Matrix...")
vectorizer = CountVectorizer(
    max_df=0.90,  # Ignore words appearing in >90% of docs
    min_df=2,     # Ignore words appearing in <2 docs
    max_features=500,  # Limit vocabulary
    stop_words='english'
)

doc_term_matrix = vectorizer.fit_transform(df['clean_text'])
print(f"✅ DTM created: {doc_term_matrix.shape} (documents x features)")
print(f"   Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# Step 5: Train LDA model
print("\n[5/9] Training LDA topic model...")
n_topics = 5  # You can adjust this

lda = LatentDirichletAllocation(
    n_components=n_topics,
    learning_method='online',
    random_state=53,
    max_iter=15,
    n_jobs=-1
)

lda.fit(doc_term_matrix)
print("✅ Model training complete")
print(f"   Perplexity: {lda.perplexity(doc_term_matrix):.2f}")

# Step 6: Display discovered topics
print("\n[6/9] Discovered Topics:")
print("="*80)
feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
    top_idx = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_idx]
    print(f"\nTopic #{topic_idx+1}:")
    print("   " + " | ".join(top_words))

# Step 7: Assign topics to documents
print("\n[7/9] Assigning topics to documents...")
topic_results = lda.transform(doc_term_matrix)
df['dominant_topic'] = topic_results.argmax(axis=1)
df['topic_probability'] = topic_results.max(axis=1)
print("✅ Topics assigned")

# Visualize topic distribution
plt.figure(figsize=(10, 5))
topic_counts = df['dominant_topic'].value_counts().sort_index()
topic_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Distribution of Evaluations Across Topics", fontsize=14)
plt.xlabel("Topic ID", fontsize=12)
plt.ylabel("Number of Evaluations", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, count in enumerate(topic_counts):
    plt.text(i, count+0.5, str(count), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('./model/topic_distribution.png', dpi=150)
print("✅ Saved visualization: ./model/topic_distribution.png")
plt.close()

# Step 8: Show representative evaluations for each topic
print("\n[8/9] Representative evaluations per topic:")
print("="*80)

for topic_id in range(n_topics):
    topic_docs = df[df['dominant_topic'] == topic_id]
    if len(topic_docs) == 0:
        continue
    
    # Get top 2 most representative documents
    top_docs = topic_docs.nlargest(2, 'topic_probability')
    
    print(f"\n📌 Topic #{topic_id+1} ({len(topic_docs)} evaluations)")
    print(f"Key words: {' | '.join([feature_names[i] for i in lda.components_[topic_id].argsort()[:-6:-1]])}")
    print("-"*80)
    
    for idx, (_, row) in enumerate(top_docs.iterrows(), 1):
        print(f"\nExample {idx} (confidence: {row['topic_probability']:.2%}):")
        print(textwrap.fill(row['text'][:300], width=80))

# Step 9: Label topics (Human-in-the-loop)
print("\n[9/9] Topic Labeling:")
print("="*80)
print("Based on the words and sample evaluations, suggest labels for each topic:")
print("(You should update these labels based on your interpretation)")

# Default labels - UPDATE THESE based on what you see above
topic_labels = {
    0: "Course Content and Structure",
    1: "Teaching Quality and Engagement",
    2: "Practical Labs and Hands-on Learning",
    3: "Learning Resources and Materials",
    4: "Assessment Methods and Feedback"
}

print("\n📝 Suggested Topic Labels:")
for idx, label in topic_labels.items():
    print(f"   Topic {idx+1}: {label}")

# Add labels to dataframe
df['topic_label'] = df['dominant_topic'].map(topic_labels)

# Save results
print("\n[SAVING] Saving models and data...")

# Save CSV with topics
output_path = './data/course_evals_with_topics.csv'
df[['a_3_class_group', 'text', 'clean_text', 'sentiment', 
    'average_course_evaluation_rating', 'dominant_topic', 
    'topic_label', 'topic_probability']].to_csv(output_path, index=False)
print(f"✅ Data saved to {output_path}")

# Save models
joblib.dump(lda, './model/topic_model_lda.pkl')
joblib.dump(vectorizer, './model/topic_vectorizer.pkl')

with open('./model/topic_labels.json', 'w', encoding='utf-8') as f:
    json.dump(topic_labels, f, ensure_ascii=False, indent=2)

print("✅ Models saved:")
print("   - ./model/topic_model_lda.pkl")
print("   - ./model/topic_vectorizer.pkl")
print("   - ./model/topic_labels.json")

print("\n" + "="*80)
print("TOPIC MODELING COMPLETE!")
print("="*80)
print(f"\nSummary:")
print(f"  Total evaluations analyzed: {len(df)}")
print(f"  Number of topics discovered: {n_topics}")
print(f"  Vocabulary size: {len(feature_names)}")
print(f"\nNext step: Run 2_sentiment_analysis_course_evals.py")