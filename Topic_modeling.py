# Topic Modeling for Course Evaluations
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import textwrap
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
print("Loaded", len(df), "course evaluations")

# Step 2: Combine text fields
print("\n[2/9] Combining text fields...")
# Combine liked and recommendation columns
df['text'] = (
    df['f_3_Write_at_least_two_things_you_liked_about_the_teaching_and_learning_in_this_course'].fillna('') 
    + " " 
    + df['f_4_Write_at_least_one_recommendation_to_improve_the_teaching_and_learning_in_this_course_(for_future_classes)'].fillna('')
)
# remove empty rows
df = df[df['text'].str.strip() != ""].reset_index(drop=True)
print("Combined text for", len(df), "evaluations")

# Step 3: Clean text
print("\n[3/9] Cleaning text...")
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r'[^a-zA-Z\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

df['clean_text'] = df['text'].apply(clean_text)
print("Text cleaned")
print("Sample:", df['clean_text'].iloc[0][:100], "...")

# Step 4: Document-Term Matrix
print("\n[4/9] Creating Document-Term Matrix...")
vectorizer = CountVectorizer(
    max_df=0.9,
    min_df=2,
    max_features=500,
    stop_words='english'
)
doc_term_matrix = vectorizer.fit_transform(df['clean_text'])
print("DTM shape:", doc_term_matrix.shape)
print("Vocabulary size:", len(vectorizer.get_feature_names_out()))

# Step 5: Train LDA topic model
print("\n[5/9] Training LDA topic model...")
n_topics = 5
lda = LatentDirichletAllocation(
    n_components=n_topics,
    learning_method='online',
    random_state=53,
    max_iter=15,
    n_jobs=1 # safer for Windows
)
lda.fit(doc_term_matrix)
print("Model trained")
print("Perplexity:", lda.perplexity(doc_term_matrix))

# Step 6: Show topics
print("\n[6/9] Discovered Topics")
print("="*80)
feature_names = vectorizer.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    top_words = [feature_names[j] for j in topic.argsort()[:-11:-1]]
    print("\nTopic", i+1)
    print(" ", " | ".join(top_words))

# Step 7: Assign topic to each evaluation
print("\n[7/9] Assigning topics...")
topic_probs = lda.transform(doc_term_matrix)
df['dominant_topic'] = topic_probs.argmax(axis=1)
df['topic_probability'] = topic_probs.max(axis=1)
print("Done assigning topics")

# Plot distribution
plt.figure(figsize=(10, 5))
topic_counts = df['dominant_topic'].value_counts().sort_index()
topic_counts.plot(kind='bar')
plt.title("Topic Distribution")
plt.xlabel("Topic ID")
plt.ylabel("Count")
for i, c in enumerate(topic_counts):
    plt.text(i, c + 0.3, str(c), ha='center')
plt.tight_layout()
plt.savefig('./model/topic_distribution.png')
plt.close()

# Step 8: Representative examples
print("\n[8/9] Representative evaluations")
print("="*80)
for t in range(n_topics):
    group = df[df['dominant_topic'] == t]
    if len(group) == 0:
        continue
    top_examples = group.nlargest(2, 'topic_probability')
    print("\nTopic", t+1, "(", len(group), "evaluations )")
    print("-" * 70)
    for idx, row in top_examples.iterrows():
        print("\nExample (confidence =", round(row['topic_probability'], 2), ")")
        print(textwrap.fill(row['text'][:300], width=80))

# Step 9: Suggested topic labels
print("\n[9/9] Topic labeling")
print("="*80)
topic_labels = {
    0: "Course Content and Structure",
    1: "Teaching Quality and Engagement",
    2: "Practical Labs and Hands-on Learning",
    3: "Learning Resources and Materials",
    4: "Assessment Methods and Feedback"
}
print("\nSuggested labels:")
for k, v in topic_labels.items():
    print("Topic", k+1, ":", v)

df['topic_label'] = df['dominant_topic'].map(topic_labels)

# Saving
print("\nSaving results...")
df.to_csv('./data/course_evals_with_topics.csv', index=False)
joblib.dump(lda, './model/topic_model_lda.pkl')
joblib.dump(vectorizer, './model/topic_vectorizer.pkl')
with open('./model/topic_labels.json', "w") as f:
    json.dump(topic_labels, f, indent=2)
print("Saved everything.")
print("="*80)
print("TOPIC MODELING COMPLETE")
print("="*80)
