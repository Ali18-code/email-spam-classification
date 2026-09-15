# Email & SMS Spam Classification

A complete NLP classification project that trains and compares multiple machine-learning models for spam detection, then serves the best model through an interactive Streamlit application.

## Why this project matters

This repository demonstrates an end-to-end classical machine-learning workflow rather than only a notebook: data acquisition, text preprocessing, feature engineering, model comparison, evaluation, artifact persistence, and interactive inference.

## Highlights

- **Dataset:** SMS Spam Collection — 5,572 labelled messages
- **Preprocessing:** lowercasing, URL/number/punctuation removal, whitespace normalization
- **Feature extraction:** TF-IDF with unigrams + bigrams and a 3,000-feature vocabulary
- **Models compared:** Multinomial Naive Bayes, Logistic Regression, Linear SVM
- **Best recorded result:** Linear SVM — **98.39% accuracy**, **0.937 F1**
- **App:** Streamlit interface for real-time message classification
- **Reproducible training:** dataset downloader + training script + saved metrics/plots

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 97.22% | 100.00% | 79.19% | 0.884 |
| Logistic Regression | 96.95% | 100.00% | 77.18% | 0.871 |
| **Linear SVM** | **98.39%** | **97.81%** | **89.93%** | **0.937** |

> Metrics above are the recorded results produced by the current training pipeline and split configuration. Re-training can produce slightly different results if the pipeline is changed.

## Project structure

```text
email-spam-classification/
├── app.py                       # Streamlit inference application
├── spam_classifier.py           # Training + evaluation pipeline
├── download_data.py             # Dataset downloader
├── requirements.txt             # Python dependencies
├── spam_classifier_model.joblib # Saved best classifier
├── tfidf_vectorizer.joblib      # Saved TF-IDF vectorizer
├── results_summary.txt          # Generated evaluation summary
└── screenshots/                 # Generated evaluation visualizations
```

## How it works

```text
Raw message
   ↓
Text cleaning
   ↓
TF-IDF vectorization
   ↓
Model inference
   ↓
Spam / Ham prediction
```

The training pipeline evaluates three classifiers and automatically saves the model with the highest F1 score.

## Run locally

### 1. Clone

```bash
git clone https://github.com/Ali18-code/email-spam-classification.git
cd email-spam-classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the app

The repository already contains trained artifacts:

```bash
streamlit run app.py
```

## Retrain from scratch

```bash
python download_data.py
python spam_classifier.py
```

Training regenerates the model artifacts, evaluation summary, and charts in `screenshots/`.

## Evaluation visualizations

| Class distribution | Confusion matrix | Top spam-indicative terms |
|---|---|---|
| ![](screenshots/01_class_distribution.png) | ![](screenshots/04_confusion_matrix.png) | ![](screenshots/06_top_spam_words.png) |

## Important model note

The selected Linear SVM exposes a **decision margin**, not a calibrated probability. The Streamlit app intentionally reports the margin instead of converting it into a fake confidence percentage. This keeps the UI technically honest.

## Tech stack

- Python
- pandas / NumPy
- scikit-learn
- TF-IDF
- Streamlit
- Matplotlib / Seaborn
- joblib

## What I learned

- Building a repeatable NLP preprocessing pipeline
- Working with imbalanced binary classification metrics
- Comparing baseline and margin-based classifiers
- Persisting ML artifacts for inference
- Separating model training from the user-facing application
- Presenting model outputs without overstating confidence

## Possible next improvements

- Add cross-validation and hyperparameter tuning
- Add calibrated SVM probabilities when probability output is required
- Add automated tests for preprocessing and inference
- Containerize the Streamlit application
- Add CI checks for formatting and tests

## License

This project is intended for educational and portfolio use. Check the source dataset's terms before redistributing dataset files.
