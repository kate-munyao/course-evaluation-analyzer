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

# -----------------------------------------------------------
# Setup NLTK resources
# -----------------------------------------------------------
try:
    stopwords.words("english")
except:
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("punkt_tab")

print("="*70)
print("SENTIMENT ANALYSIS FOR COURSE EVALUATIONS")
print("="*70)

# 1. Load dataset

print("\nStep 1: Loading data...")
df = pd.read_csv("./data/course_evals_with_topics.csv")
print(f"Loaded {len(df)} rows")
print("Sentiment breakdown:", df["sentiment"].value_counts().to_dict())


# 2. Preprocess text 

print("\nStep 2: Cleaning text...")

stop_words = set(stopwords.words("english")) - {"not", "no", "never"}

def clean_text_for_sentiment(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)


# 3. TF-IDF vectorizer

print("\nStep 3: Building TF-IDF features...")

tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95
)
X = tfidf.fit_transform(df["clean_text"])
y = df["sentiment"]

print("Feature matrix:", X.shape)
print("TF-IDF vocabulary size:", len(tfidf.get_feature_names_out()))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Train models + cross-validation

print("\nStep 4: Training models...")

models = {
    "Logistic Regression": LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
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
        "F1-Score": scores["test_f1_weighted"].mean(),
    }

results_df = pd.DataFrame(cv_results).T.sort_values("F1-Score", ascending=False)
print("\nCross-validation results:")
print(results_df.round(4))

# 5. Select best model

print("\nStep 5: Selecting best model...")

best_model_name = results_df.index[0]
best_model = models[best_model_name]

best_model.fit(X_train, y_train)

print("Best model:", best_model_name)
print("F1-Score:", results_df.loc[best_model_name, "F1-Score"])

# Evaluate
y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"])
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Positive", "Neutral", "Negative"],
    yticklabels=["Positive", "Neutral", "Negative"]
)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig("./model/confusion_matrix.png", dpi=150)
plt.close()

# 6. Predict full dataset

print("\nStep 6: Predicting sentiment for full dataset...")

df["predicted_sentiment"] = best_model.predict(X)
df["prediction_confidence"] = best_model.predict_proba(X).max(axis=1)

# 7. Sentiment per topic

print("\nStep 7: Analysing sentiment per topic...")

sentiment_by_topic = df.groupby(
    ["topic_label", "predicted_sentiment"]
).size().unstack(fill_value=0)

sentiment_order = ["positive", "neutral", "negative"]
colors = ["green", "orange", "red"]

ax = sentiment_by_topic[sentiment_order].plot(
    kind="bar",
    figsize=(12, 6),
    color=colors,
    edgecolor="black"
)

ax.set_xticklabels(
    ['\n'.join(textwrap.wrap(label, 15)) for label in sentiment_by_topic.index],
    rotation=45,
    ha="right"
)

plt.title("Sentiment Distribution by Topic")
plt.tight_layout()
plt.savefig("./model/sentiment_by_topic.png", dpi=150)
plt.close()

print(sentiment_by_topic)

# 8. Save everything

print("\nStep 8: Saving output...")

df.to_csv("./data/course_evals_with_topics_and_sentiments.csv", index=False)
joblib.dump(best_model, "./model/sentiment_classifier.pkl")
joblib.dump(tfidf, "./model/topic_vectorizer_using_tfidf.pkl")

print("Saved:")
print(" - sentiment_classifier.pkl")
print(" - topic_vectorizer_using_tfidf.pkl")

# Word clouds

print("\nGenerating word clouds...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, sentiment in enumerate(["positive", "neutral", "negative"]):
    text = " ".join(df[df["predicted_sentiment"] == sentiment]["text"])
    if text.strip():
        wc = WordCloud(background_color="white", max_words=50).generate(text)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"{sentiment.capitalize()} Reviews")
        axes[i].axis("off")

plt.tight_layout()
plt.savefig("./model/sentiment_wordclouds.png", dpi=150)
plt.close()

print("\nAll tasks completed successfully.")
print("="*70)
