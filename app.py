import os

import joblib
import streamlit as st

from text_utils import clean_text


st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱", layout="centered")
st.title("📱 SMS Spam Classifier")
st.write("Enter an SMS-style message to classify it as **Spam** or **Ham** (not spam).")

VECTORIZER_PATH = "tfidf_vectorizer.joblib"
MODEL_PATH = "spam_classifier_model.joblib"


@st.cache_resource
def load_models():
    if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
        return None, None
    return joblib.load(VECTORIZER_PATH), joblib.load(MODEL_PATH)


tfidf, model = load_models()

if tfidf is None or model is None:
    st.error("Models not found. Run `python spam_classifier.py` to generate them.")
    st.stop()

user_input = st.text_area(
    "Message content:",
    height=150,
    placeholder="Congratulations! You have won a prize. Claim it now...",
)

if st.button("Analyze Message", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        cleaned_input = clean_text(user_input)

        if not cleaned_input:
            st.warning("The message contained only ignored characters, numbers, or links.")
        else:
            features = tfidf.transform([cleaned_input])
            prediction = model.predict(features)[0]

            st.markdown("---")
            if prediction == 1:
                st.error("### 🚨 This looks like spam")
            else:
                st.success("### ✅ This looks like ham (not spam)")

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(features)[0][prediction] * 100
                st.caption(f"Estimated class probability: {probability:.2f}%")
            else:
                st.caption(
                    "The Linear SVM returns a decision score, not a calibrated "
                    "probability, so no confidence percentage is shown."
                )

st.markdown("---")
st.caption(
    "Educational SMS classifier built with Streamlit, TF-IDF, and scikit-learn. "
    "Predictions may be wrong; do not use this as a security filter."
)
