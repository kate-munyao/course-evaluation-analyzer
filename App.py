import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------------------------------
# NLTK Setup
# -----------------------------------------------------
try:
    stopwords.words("english")
except:
    nltk.download("stopwords")
    nltk.download("punkt")


print("="*70)
print(" FIXED SENTIMENT MODEL TRAINING SCRIPT")
print("="*70)


# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------
print("\nStep 1: Loading dataset...")
df = pd.read_csv("./data/course_evals_with_topics.csv")
print(f"Loaded {len(df)} rows")
print("Sentiment distribution:", df["sentiment"].value_counts().to_dict())



# -----------------------------------------------------
# TWO CLEANING FUNCTIONS
# -----------------------------------------------------

# (A) Aggressive cleaner for Topic Modeling
def clean_for_topic(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# (B) Light cleaner for Sentiment (keeps NOT / NO / NEVER)
sent_stopwords = set(stopwords.words("english")) - {"not", "no", "never"}

def clean_for_sentiment(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z'\s]", " ", text)

    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in sent_stopwords]

    return " ".join(tokens)



# -----------------------------------------------------
# Apply Sentiment Cleaning Only
# -----------------------------------------------------
print("\nStep 2: Cleaning text (sentiment only)...")
df["clean_text"] = df["text"].apply(clean_for_sentiment)



# -----------------------------------------------------
# TF-IDF Vectorizer (sentiment only)
# -----------------------------------------------------
print("\nStep 3: Building TF-IDF features...")

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.98
)

X = tfidf.fit_transform(df["clean_text"])
y = df["sentiment"]

print("Feature matrix:", X.shape)
print("Vocabulary size:", len(tfidf.get_feature_names_out()))



# -----------------------------------------------------
# Train/Test Split
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)



# -----------------------------------------------------
# Train Models + Cross Validation
# -----------------------------------------------------
print("\nStep 4: Training models...")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",     # FIXED FOR NEUTRAL BIAS
        random_state=42
    ),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        n_jobs=-1,
        random_state=42
    )
}

scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

results = {}

for name, model in models.items():
    print(f"Running CV for: {name}")

    scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)

    results[name] = {
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision_weighted"].mean(),
        "Recall": scores["test_recall_weighted"].mean(),
        "F1": scores["test_f1_weighted"].mean()
    }

results_df = pd.DataFrame(results).T.sort_values("F1", ascending=False)
print("\nCross-validation results:")
print(results_df.round(4))



# -----------------------------------------------------
# Select Best Model
# -----------------------------------------------------
print("\nStep 5: Selecting best model...")

best_name = results_df.index[0]
best_model = models[best_name]

best_model.fit(X_train, y_train)

print(f"Best Model: {best_name}")



# -----------------------------------------------------
# Evaluate Performance
# -----------------------------------------------------
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"])
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Positive", "Neutral", "Negative"],
    yticklabels=["Positive", "Neutral", "Negative"]
)
plt.tight_layout()
plt.savefig("./model/confusion_matrix.png")
plt.close()



# -----------------------------------------------------
# Save Model Outputs
# -----------------------------------------------------
print("\nStep 6: Saving model & vectorizer...")

joblib.dump(best_model, "./model/sentiment_classifier.pkl")
joblib.dump(tfidf, "./model/sentiment_vectorizer.pkl")

df.to_csv("./data/course_evals_with_topics_and_sentiments.csv", index=False)

print("Saved:")
print(" - sentiment_classifier.pkl")
print(" - sentiment_vectorizer.pkl")

print("\nAll tasks completed successfully!")
print("="*70)
