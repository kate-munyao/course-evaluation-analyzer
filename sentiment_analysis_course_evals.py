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


# =====================================================
# SETUP NLTK
# =====================================================
try:
    stopwords.words("english")
except:
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("punkt_tab")


print("="*70)
print(" SENTIMENT ANALYSIS FOR COURSE EVALUATIONS")
print("="*70)


# =====================================================
# LOAD DATA
# =====================================================
print("\nStep 1: Loading dataset...")
df = pd.read_csv("./data/course_evals_with_topics.csv")
print(f"Loaded {len(df)} rows")

print("Sentiment breakdown:", df["sentiment"].value_counts().to_dict())


# =====================================================
# CLEAN TEXT (UNIFIED FUNCTION)
# =====================================================
print("\nStep 2: Cleaning text...")

stop_words = set(stopwords.words("english")) - {"not", "no", "never"}

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)


# =====================================================
# TF-IDF VECTORIZATION
# =====================================================
print("\nStep 3: Building TF-IDF features...")

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X = tfidf.fit_transform(df["clean_text"])
y = df["sentiment"]

print("Feature matrix:", X.shape)
print("Vocabulary size:", len(tfidf.get_feature_names_out()))


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# =====================================================
# TRAIN MODELS WITH CROSS-VALIDATION
# =====================================================
print("\nStep 4: Training models...")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=53
    ),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=53),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=53,
        n_jobs=-1
    )
}

scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=53)

cv_results = {}

for name, model in models.items():
    print(f"Running CV for: {name}")
    scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)

    cv_results[name] = {
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision_weighted"].mean(),
        "Recall": scores["test_recall_weighted"].mean(),
        "F1": scores["test_f1_weighted"].mean(),
    }

results_df = pd.DataFrame(cv_results).T.sort_values("F1", ascending=False)
print("\nCross-validation results:")
print(results_df.round(4))


# =====================================================
# SELECT BEST MODEL
# =====================================================
print("\nStep 5: Selecting best model...")

best_model_name = results_df.index[0]
best_model = models[best_model_name]

best_model.fit(X_train, y_train)

print("Best model:", best_model_name)


# Evaluate
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Positive", "Neutral", "Negative"],
            yticklabels=["Positive", "Neutral", "Negative"])
plt.tight_layout()
plt.savefig("./model/confusion_matrix.png")
plt.close()


# =====================================================
# PREDICT ALL DATA
# =====================================================
print("\nStep 6: Predicting all data...")

df["predicted_sentiment"] = best_model.predict(X)
df["confidence"] = best_model.predict_proba(X).max(axis=1)


# =====================================================
# SAVE OUTPUT
# =====================================================
print("\nStep 7: Saving model + vectorizer...")

df.to_csv("./data/course_evals_with_topics_and_sentiments.csv", index=False)

joblib.dump(best_model, "./model/sentiment_classifier.pkl")
joblib.dump(tfidf, "./model/sentiment_vectorizer.pkl")

print("Saved:")
print("  - sentiment_classifier.pkl")
print("  - sentiment_vectorizer.pkl")

print("\nAll tasks complete!")
print("="*70)
