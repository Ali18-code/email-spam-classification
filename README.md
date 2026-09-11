# Email/SMS Spam Classification

An interactive machine learning pipeline that classifies text messages as **spam** or **ham** (not spam) using NLP preprocessing, TF-IDF feature extraction, and multiple classifiers, complete with a live Streamlit web interface.

## Overview

- **Dataset:** [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (Kaggle) — 5,572 labelled messages (4,825 ham / 747 spam)
- **Preprocessing:** lowercasing, URL/number/punctuation removal, whitespace normalization
- **Features:** TF-IDF (unigrams + bigrams, English stop words removed, 3,000-feature vocabulary)
- **Models compared:** Multinomial Naive Bayes, Logistic Regression, Linear SVM
- **Best result:** Linear SVM — **98.39% accuracy**, F1 score 0.937
- **Interactive UI:** Built with **Streamlit** to test custom messages in real-time.

## Architecture

1. **`spam_classifier.py`**: The training pipeline. Normalizes the data, trains the models, generates evaluation charts, and saves the TF-IDF vectorizer and best model using `joblib`.
2. **`app.py`**: The live inference web application. Takes custom user input, cleans it, vectorizes it, and returns a real-time spam probability.

## Project Structure

```
email-spam-classification/
├── app.py                 # Live Interactive Streamlit Web App
├── spam_classifier.py     # Training script: preprocessing, training, evaluation
├── download_data.py       # Fetches the dataset (sms_spam.tsv)
├── tfidf_vectorizer.joblib# Saved TF-IDF vocabulary 
├── spam_classifier_model.joblib # Saved best model (Linear SVC)
├── requirements.txt       # Python dependencies
├── results_summary.txt    # Generated metrics summary
└── screenshots/           # Generated charts and plots
```

## Setup & Usage

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone <your-repo-url>
cd email-spam-classification
pip install -r requirements.txt
```

### 2. Run the Live Web App! 🚀
If the models are already generated, you can launch the interactive spam checker immediately:
```bash
streamlit run app.py
```

### 3. Retrain the Models (Optional)
If you want to train the models from scratch and generate new evaluation metrics:
```bash
# Download the dataset
python download_data.py

# Run the training pipeline
python spam_classifier.py
```

## Results

| Model                   | Accuracy | Precision | Recall | F1 Score |
|--------------------------|----------|-----------|--------|----------|
| Multinomial Naive Bayes  | 97.22%   | 100.00%   | 79.19% | 0.884    |
| Logistic Regression      | 96.95%   | 100.00%   | 77.18% | 0.871    |
| **Linear SVM (best)**    | **98.39%** | 97.81%  | 89.93% | **0.937** |

### Sample Outputs

| Class Distribution | Confusion Matrix | Top Spam Words |
|---|---|---|
| ![](screenshots/01_class_distribution.png) | ![](screenshots/04_confusion_matrix.png) | ![](screenshots/06_top_spam_words.png) |

## Tech Stack

Python, Streamlit, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

## License

This project is for educational/internship purposes.
