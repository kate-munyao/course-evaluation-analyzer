# ============================================================
# TOPIC MODELING FOR COURSE EVALUATIONS
# ============================================================
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import textwrap
import joblib
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


print("=" * 90)
print(" TOPIC MODELING FOR COURSE EVALUATIONS")
print("=" * 90)


# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------
print("\n[1/9] Loading dataset...")

df = pd.read_csv("./data/202511-ft_bi1_bi2_course_evaluation.csv")
print(f"Loaded {len(df)} raw evaluation entries.")


# ------------------------------------------------------------
# 2. Combine Text Fields
# ------------------------------------------------------------
print("\n[2/9] Combining 'liked' and 'recommendation' fields...")

df["text"] = (
    df["f_3_Write_at_least_two_things_you_liked_about_the_teaching_and_learning_in_this_course"].fillna("") + " " +
    df["f_4_Write_at_least_one_recommendation_to_improve_the_teaching_and_learning_in_this_course_(for_future_classes)"].fillna("")
)

# Remove empty evaluations
df = df[df["text"].str.strip() != ""].reset_index(drop=True)
print(f"Remaining valid evaluations: {len(df)}")


# ------------------------------------------------------------
# 3. Clean Text
# ------------------------------------------------------------
print("\n[3/9] Cleaning text...")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

print("Sample cleaned text:")
print(" →", df["clean_text"].iloc[0][:120], "...")


# ------------------------------------------------------------
# 4. Document-Term Matrix
# ------------------------------------------------------------
print("\n[4/9] Building Document-Term Matrix...")

vectorizer = CountVectorizer(
    stop_words="english",
    max_df=0.90,
    min_df=2,
    max_features=500
)

doc_term_matrix = vectorizer.fit_transform(df["clean_text"])
vocab = vectorizer.get_feature_names_out()

print("DTM shape:", doc_term_matrix.shape)
print("Vocabulary size:", len(vocab))


# ------------------------------------------------------------
# 5. Train LDA Model
# ------------------------------------------------------------
print("\n[5/9] Training LDA topic model...")

n_topics = 5

lda = LatentDirichletAllocation(
    n_components=n_topics,
    learning_method="online",
    random_state=53,
    max_iter=15
)

lda.fit(doc_term_matrix)

print("LDA model trained.")
print("Perplexity:", round(lda.perplexity(doc_term_matrix), 2))


# ------------------------------------------------------------
# 6. Display Topics
# ------------------------------------------------------------
print("\n[6/9] Discovered Topics")
print("=" * 90)

for i, topic in enumerate(lda.components_):
    top_words = [vocab[j] for j in topic.argsort()[:-11:-1]]
    print(f"\nTopic {i+1}:")
    print("   ", " | ".join(top_words))


# ------------------------------------------------------------
# 7. Assign Topic to Each Evaluation
# ------------------------------------------------------------
print("\n[7/9] Assigning topics to each evaluation...")

topic_dist = lda.transform(doc_term_matrix)
df["dominant_topic"] = topic_dist.argmax(axis=1)
df["topic_probability"] = topic_dist.max(axis=1)

print("Topic assignment complete.")

# Plot distribution
plt.figure(figsize=(10, 5))
counts = df["dominant_topic"].value_counts().sort_index()

counts.plot(kind="bar")
plt.title("Topic Distribution Across Feedback")
plt.xlabel("Topic ID")
plt.ylabel("Number of Evaluations")

for i, val in enumerate(counts):
    plt.text(i, val + 0.3, str(val), ha="center")

plt.tight_layout()
plt.savefig("./model/topic_distribution.png")
plt.close()


# ------------------------------------------------------------
# 8. Representative Evaluations
# ------------------------------------------------------------
print("\n[8/9] Representative evaluations from each topic")
print("=" * 90)

for t in range(n_topics):
    subset = df[df["dominant_topic"] == t]
    if len(subset) == 0:
        continue

    top_examples = subset.nlargest(2, "topic_probability")

    print(f"\nTopic {t+1} ({len(subset)} evaluations)")
    print("-" * 70)

    for _, row in top_examples.iterrows():
        print(f"\nExample (confidence = {round(row['topic_probability'], 2)})")
        print(textwrap.fill(row["text"][:300], width=80))


# ------------------------------------------------------------
# 9. Suggested Topic Labels
# ------------------------------------------------------------
print("\n[9/9] Suggested labels for discovered topics")
print("=" * 90)

topic_labels = {
    0: "Course Content & Structure",
    1: "Teaching Quality & Engagement",
    2: "Practical Labs & Hands-on Learning",
    3: "Learning Resources & Materials",
    4: "Assessments & Feedback"
}

for t, label in topic_labels.items():
    print(f"Topic {t+1}: {label}")

df["topic_label"] = df["dominant_topic"].map(topic_labels)


# ------------------------------------------------------------
# Save Outputs
# ------------------------------------------------------------
print("\nSaving processed data and models...")

df.to_csv("./data/course_evals_with_topics.csv", index=False)
joblib.dump(lda, "./model/topic_model_lda.pkl")
joblib.dump(vectorizer, "./model/topic_vectorizer.pkl")

with open("./model/topic_labels.json", "w") as f:
    json.dump(topic_labels, f, indent=2)

print("All topic modeling files saved successfully.")
print("=" * 90)
print(" TOPIC MODELING COMPLETE")
print("=" * 90)
