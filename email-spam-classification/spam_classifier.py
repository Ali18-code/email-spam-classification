"""
Task 1: Email/SMS Spam Classification
Dataset: SMS Spam Collection (labelled ham/spam dataset, same one distributed on Kaggle
as "SMS Spam Collection Dataset")
"""
import os
import re
import string
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

SS = "screenshots"
os.makedirs(SS, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv("sms_spam.tsv", sep="\t", header=None, names=["label", "message"])
print("Dataset shape:", df.shape)
print(df["label"].value_counts())

# ---------------------------------------------------------------------------
# 2. EXPLORATORY VISUAL: class distribution
# ---------------------------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.countplot(x="label", data=df, palette=["#4C72B0", "#DD8452"])
plt.title("Class Distribution: Ham vs Spam")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{SS}/01_class_distribution.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. TEXT PREPROCESSING
# ---------------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"\d+", " ", text)                      # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()               # normalize whitespace
    return text

df["clean_message"] = df["message"].apply(clean_text)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

print(df[["message", "clean_message"]].head())

# message length feature (before/after) for a quick visual
df["msg_len"] = df["message"].apply(len)
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="msg_len", hue="label", bins=40, kde=False, palette=["#4C72B0", "#DD8452"])
plt.title("Message Length Distribution by Class")
plt.xlabel("Message length (characters)")
plt.tight_layout()
plt.savefig(f"{SS}/02_message_length.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_message"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
)

# ---------------------------------------------------------------------------
# 5. FEATURE EXTRACTION: TF-IDF
# ---------------------------------------------------------------------------
tfidf = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("TF-IDF feature matrix shape:", X_train_tfidf.shape)

# ---------------------------------------------------------------------------
# 6. TRAIN MULTIPLE MODELS
# ---------------------------------------------------------------------------
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    results[name] = {
        "model": model,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "preds": preds,
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

# Pick the best model by F1 score
best_name = max(results, key=lambda k: results[k]["f1"])
best = results[best_name]
print(f"\nBest model: {best_name}")

# ---------------------------------------------------------------------------
# 7. COMPARISON CHART
# ---------------------------------------------------------------------------
comp_df = pd.DataFrame({
    name: {"Accuracy": r["accuracy"], "Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"]}
    for name, r in results.items()
}).T

plt.figure(figsize=(8, 5))
comp_df.plot(kind="bar", ax=plt.gca())
plt.title("Model Comparison")
plt.ylabel("Score")
plt.ylim(0.8, 1.01)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{SS}/03_model_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 8. CONFUSION MATRIX (best model)
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, best["preds"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{SS}/04_confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 9. ROC CURVE (best model, if it supports probabilities/decision function)
# ---------------------------------------------------------------------------
plt.figure(figsize=(5, 4))
for name, r in results.items():
    model = r["model"]
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test_tfidf)[:, 1]
    else:
        scores = model.decision_function(X_test_tfidf)
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{SS}/05_roc_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 10. TOP SPAM-INDICATIVE WORDS (from Logistic Regression coefficients)
# ---------------------------------------------------------------------------
lr = results["Logistic Regression"]["model"]
feature_names = np.array(tfidf.get_feature_names_out())
top_spam_idx = np.argsort(lr.coef_[0])[-15:][::-1]
top_spam_words = feature_names[top_spam_idx]
top_spam_scores = lr.coef_[0][top_spam_idx]

plt.figure(figsize=(7, 5))
sns.barplot(x=top_spam_scores, y=top_spam_words, color="#C44E52")
plt.title("Top 15 Spam-Indicative Terms (Logistic Regression Weights)")
plt.xlabel("Coefficient weight (higher = more spammy)")
plt.tight_layout()
plt.savefig(f"{SS}/06_top_spam_words.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 11. SAMPLE PREDICTIONS TABLE (screenshot)
# ---------------------------------------------------------------------------
sample = X_test.sample(8, random_state=1)
sample_preds = best["model"].predict(tfidf.transform(sample))
sample_true = y_test.loc[sample.index]
label_map = {0: "ham", 1: "spam"}

fig, ax = plt.subplots(figsize=(11, 4))
ax.axis("off")
table_data = [["Message (cleaned, truncated)", "Actual", "Predicted"]]
for msg, t, p in zip(sample, sample_true, sample_preds):
    table_data.append([msg[:60] + ("..." if len(msg) > 60 else ""), label_map[t], label_map[p]])
tbl = ax.table(cellText=table_data, loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.8)
for i in range(len(table_data[0])):
    tbl[0, i].set_facecolor("#4C72B0")
    tbl[0, i].set_text_props(color="white", weight="bold")
plt.title("Sample Predictions on Test Set", pad=20)
plt.tight_layout()
plt.savefig(f"{SS}/07_sample_predictions.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 12. SAVE SUMMARY METRICS TO TEXT
# ---------------------------------------------------------------------------
with open("results_summary.txt", "w") as f:
    f.write(f"Dataset size: {df.shape[0]} messages ({df['label'].value_counts().to_dict()})\n")
    f.write(f"Train/Test split: {len(X_train)} / {len(X_test)}\n")
    f.write(f"TF-IDF vocabulary size: {X_train_tfidf.shape[1]}\n\n")
    for name, r in results.items():
        f.write(f"{name}:\n")
        f.write(f"  Accuracy:  {r['accuracy']:.4f}\n")
        f.write(f"  Precision: {r['precision']:.4f}\n")
        f.write(f"  Recall:    {r['recall']:.4f}\n")
        f.write(f"  F1 Score:  {r['f1']:.4f}\n\n")
    f.write(f"Best model: {best_name}\n")

print("\nDone. Screenshots saved to", SS)
