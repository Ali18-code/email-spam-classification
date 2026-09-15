"""Train and evaluate the Email/SMS spam classification pipeline."""

from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from text_utils import clean_text

DATA_PATH = Path("sms_spam.tsv")
SCREENSHOT_DIR = Path("screenshots")
MODEL_PATH = Path("spam_classifier_model.joblib")
VECTORIZER_PATH = Path("tfidf_vectorizer.joblib")
RESULTS_PATH = Path("results_summary.txt")
RANDOM_STATE = 42


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python download_data.py` first."
        )

    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    df = df.dropna(subset=["label", "message"]).copy()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["message"] = df["message"].astype(str)
    df["clean_message"] = df["message"].map(clean_text)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_exploration(df: pd.DataFrame) -> None:
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="label")
    plt.title("Class Distribution: Ham vs Spam")
    plt.xlabel("Label")
    plt.ylabel("Count")
    save_figure(SCREENSHOT_DIR / "01_class_distribution.png")

    plot_df = df.assign(msg_len=df["message"].str.len())
    plt.figure(figsize=(6, 4))
    sns.histplot(data=plot_df, x="msg_len", hue="label", bins=40, kde=False)
    plt.title("Message Length Distribution by Class")
    plt.xlabel("Message length (characters)")
    save_figure(SCREENSHOT_DIR / "02_message_length.png")


def train_models(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=3000,
        ngram_range=(1, 2),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "preds": preds,
        }
        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds, target_names=["ham", "spam"], zero_division=0))

    return vectorizer, X_train_tfidf, X_test_tfidf, results


def plot_evaluation(vectorizer, X_test_tfidf, X_test, y_test, results) -> None:
    comparison = pd.DataFrame(
        {
            name: {
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
            }
            for name, result in results.items()
        }
    ).T
    comparison.plot(kind="bar", figsize=(8, 5))
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=15)
    save_figure(SCREENSHOT_DIR / "03_model_comparison.png")

    best_name = max(results, key=lambda name: results[name]["f1"])
    best = results[best_name]

    cm = confusion_matrix(y_test, best["preds"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_figure(SCREENSHOT_DIR / "04_confusion_matrix.png")

    plt.figure(figsize=(6, 5))
    for name, result in results.items():
        model = result["model"]
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test_tfidf)[:, 1]
        else:
            scores = model.decision_function(X_test_tfidf)
        fpr, tpr, _ = roc_curve(y_test, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    save_figure(SCREENSHOT_DIR / "05_roc_curves.png")

    logistic = results["Logistic Regression"]["model"]
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    top_idx = np.argsort(logistic.coef_[0])[-15:][::-1]
    plt.figure(figsize=(7, 5))
    sns.barplot(x=logistic.coef_[0][top_idx], y=feature_names[top_idx])
    plt.title("Top Spam-Indicative Terms")
    plt.xlabel("Logistic Regression coefficient")
    save_figure(SCREENSHOT_DIR / "06_top_spam_words.png")

    sample = X_test.sample(min(8, len(X_test)), random_state=1)
    sample_preds = best["model"].predict(vectorizer.transform(sample))
    sample_true = y_test.loc[sample.index]
    label_map = {0: "ham", 1: "spam"}

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    rows = [["Message (cleaned, truncated)", "Actual", "Predicted"]]
    for message, actual, predicted in zip(sample, sample_true, sample_preds):
        short = message[:60] + ("..." if len(message) > 60 else "")
        rows.append([short, label_map[int(actual)], label_map[int(predicted)]])
    table = ax.table(cellText=rows, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    plt.title("Sample Predictions on Test Set", pad=20)
    save_figure(SCREENSHOT_DIR / "07_sample_predictions.png")


def write_summary(df, X_train, X_test, X_train_tfidf, results, best_name) -> None:
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        handle.write(f"Dataset size: {len(df)} messages ({df['label'].value_counts().to_dict()})\n")
        handle.write(f"Train/Test split: {len(X_train)} / {len(X_test)}\n")
        handle.write(f"TF-IDF vocabulary size: {X_train_tfidf.shape[1]}\n\n")
        for name, result in results.items():
            handle.write(f"{name}:\n")
            handle.write(f"  Accuracy:  {result['accuracy']:.4f}\n")
            handle.write(f"  Precision: {result['precision']:.4f}\n")
            handle.write(f"  Recall:    {result['recall']:.4f}\n")
            handle.write(f"  F1 Score:  {result['f1']:.4f}\n\n")
        handle.write(f"Best model by F1 score: {best_name}\n")


def main() -> None:
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    print("Dataset shape:", df.shape)
    print(df["label"].value_counts())

    plot_exploration(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label_num"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label_num"],
    )

    vectorizer, X_train_tfidf, X_test_tfidf, results = train_models(
        X_train, X_test, y_train, y_test
    )
    best_name = max(results, key=lambda name: results[name]["f1"])
    best_model = results[best_name]["model"]
    print(f"\nBest model by F1 score: {best_name}")

    plot_evaluation(vectorizer, X_test_tfidf, X_test, y_test, results)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(best_model, MODEL_PATH)
    write_summary(df, X_train, X_test, X_train_tfidf, results, best_name)

    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved evaluation plots to {SCREENSHOT_DIR}/")


if __name__ == "__main__":
    main()
