# Email & SMS Spam Classification

A complete NLP classification project that trains and compares multiple machine-learning models for spam detection, then serves the best model through an interactive Streamlit application.

## Why this project matters

This repository demonstrates an end-to-end classical machine-learning workflow rather than only a notebook: data acquisition, reusable preprocessing, feature engineering, model comparison, evaluation, artifact persistence, automated tests, CI, and interactive inference.

## Highlights

- **Dataset:** SMS Spam Collection — 5,572 labelled messages
- **Preprocessing:** lowercasing, URL/number/punctuation removal, whitespace normalization
- **Feature extraction:** TF-IDF with unigrams + bigrams and a 3,000-feature vocabulary
- **Models compared:** Multinomial Naive Bayes, Logistic Regression, Linear SVM
- **Best recorded result:** Linear SVM — **98.39% accuracy**, **0.937 F1**
- **App:** Streamlit interface for real-time message classification
- **Reproducible training:** dataset downloader + training script + saved metrics/plots
- **Quality checks:** pytest unit tests + GitHub Actions CI

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 97.22% | 100.00% | 79.19% | 0.884 |
| Logistic Regression | 96.95% | 100.00% | 77.18% | 0.871 |
| **Linear SVM** | **98.39%** | **97.81%** | **89.93%** | **0.937** |

> These are recorded results from the repository's current experiment configuration. If the data, dependencies, preprocessing, or model settings change, retraining may produce different metrics.

## Project structure

```text
email-spam-classification/
├── .github/workflows/ci.yml      # Automated tests and syntax checks
├── tests/
│   └── test_text_utils.py        # Preprocessing unit tests
├── app.py                        # Streamlit inference application
├── spam_classifier.py            # Training + evaluation pipeline
├── text_utils.py                 # Shared training/inference preprocessing
├── download_data.py              # Dataset downloader
├── requirements.txt              # Runtime dependencies
├── requirements-dev.txt          # Development/test dependencies
├── spam_classifier_model.joblib  # Saved best classifier
├── tfidf_vectorizer.joblib       # Saved TF-IDF vectorizer
├── results_summary.txt           # Generated evaluation summary
└── screenshots/                  # Generated evaluation visualizations
```

## How it works

```text
Raw message
   ↓
Shared text preprocessing
   ↓
TF-IDF vectorization
   ↓
Model inference
   ↓
Spam / Ham prediction
```

Training and inference import the same `clean_text()` function, reducing the risk of preprocessing drift between experiments and the deployed app.

## Run locally

```bash
git clone https://github.com/Ali18-code/email-spam-classification.git
cd email-spam-classification
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and launch the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain from scratch

```bash
python download_data.py
python spam_classifier.py
```

Training regenerates the saved model/vectorizer, `results_summary.txt`, and evaluation charts.

## Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The GitHub Actions workflow runs the tests and compiles the Python sources on every push to `main` and on pull requests.

## Evaluation visualizations

| Class distribution | Confusion matrix | Top spam-indicative terms |
|---|---|---|
| ![](screenshots/01_class_distribution.png) | ![](screenshots/04_confusion_matrix.png) | ![](screenshots/06_top_spam_words.png) |

## Important model note

The selected Linear SVM exposes a **decision margin**, not a calibrated probability. The Streamlit app reports that margin directly instead of converting it into a fabricated confidence percentage. If probability estimates are required, the classifier should be calibrated explicitly.

## Tech stack

Python, pandas, NumPy, scikit-learn, TF-IDF, Streamlit, Matplotlib, Seaborn, joblib, pytest, GitHub Actions.

## Engineering choices

- Shared preprocessing between training and inference
- Stratified train/test split for the imbalanced target
- F1-based best-model selection instead of accuracy alone
- Fixed random state for repeatable experiments
- Saved model/vectorizer artifacts for reproducible inference
- Unit-tested text normalization
- CI checks for regressions and syntax errors

## Possible next improvements

- Add cross-validation and hyperparameter search
- Add calibrated probability estimates when needed
- Add inference-level tests around saved artifacts
- Containerize the Streamlit application
- Track experiments and model metadata explicitly

## License / data note

This repository is intended for educational and portfolio use. Check the source dataset's terms before redistributing dataset files.
