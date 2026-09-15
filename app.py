import streamlit as st
import joblib
import re
import string
import os

st.set_page_config(page_title="Email Spam Classifier", page_icon="✉️", layout="centered")

st.title("✉️ Email & SMS Spam Classifier")
st.write("Enter an email or text message below to check if it's **Spam** or **Ham** (Safe).")

# Load models
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
MODEL_PATH = "spam_classifier_model.joblib"

@st.cache_resource
def load_models():
    if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
        return None, None
    tfidf = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return tfidf, model

tfidf, model = load_models()

if tfidf is None or model is None:
    st.error("⚠️ Models not found. Please run `python spam_classifier.py` first to train the models.")
    st.stop()

# Text cleaning function (matches the one in training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"\d+", " ", text)                      # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()               # normalize whitespace
    return text

user_input = st.text_area("Message Content:", height=150, placeholder="Congratulations! You've won a $1,000 Walmart gift card. Click here to claim now...")

if st.button("Analyze Message", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            cleaned_input = clean_text(user_input)
            
            if cleaned_input == "":
                st.warning("Message only contained ignored characters (numbers, punctuation, links).")
            else:
                # Transform and Predict
                features = tfidf.transform([cleaned_input])
                prediction = model.predict(features)[0]
                
                # Check for probabilities
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    confidence = max(probabilities) * 100
                else:
                    # For LinearSVC which uses decision_function
                    dist = model.decision_function(features)[0]
                    # Simple pseudo-probability scaling for visual purposes
                    confidence = min(100, max(50, (abs(dist) * 20) + 50))
                
                st.markdown("---")
                
                if prediction == 1:
                    st.error("### 🚨 This looks like SPAM!")
                else:
                    st.success("### ✅ This looks like HAM (Safe).")
                
                st.progress(int(confidence) / 100)
                st.caption(f"Confidence: {confidence:.2f}%")
                
st.markdown("---")
st.markdown("*Built with [Streamlit](https://streamlit.io/) and Scikit-Learn. Powered by TF-IDF and Linear SVM / Naive Bayes / Logistic Regression.*")
