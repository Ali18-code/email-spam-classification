import os

import joblib
import streamlit as st

from text_utils import clean_text

st.set_page_config(page_title="Email Spam Classifier", page_icon="✉️", layout="centered")

VECTORIZER_PATH = "tfidf_vectorizer.joblib"
MODEL_PATH = "spam_classifier_model.joblib"


@st.cache_resource
def load_models():
    if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
        return None, None
    return joblib.load(VECTORIZER_PATH), joblib.load(MODEL_PATH)


st.title("✉️ Email & SMS Spam Classifier")
st.write(
    "Enter an email or text message to classify it as **spam** or **ham** "
    "using a TF-IDF text pipeline and a trained scikit-learn classifier."
)

tfidf, model = load_models()

if tfidf is None or model is None:
    st.error(
        "Model artifacts were not found. Run `python spam_classifier.py` first "
        "to train the classifier and create the `.joblib` files."
    )
    st.stop()

user_input = st.text_area(
    "Message content",
    height=150,
    placeholder="Congratulations! You've won a prize. Click here to claim it...",
)

if st.button("Analyze message", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter a message before running the classifier.")
    else:
        cleaned_input = clean_text(user_input)

        if not cleaned_input:
            st.warning("The message contains no usable text after preprocessing.")
        else:
            features = tfidf.transform([cleaned_input])
            prediction = int(model.predict(features)[0])

            st.divider()
            if prediction == 1:
                st.error("### 🚨 Prediction: Spam")
            else:
                st.success("### ✅ Prediction: Ham")

            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(features)[0][prediction])
                st.metric("Model probability", f"{probability:.1%}")
                st.caption("Model probabilities are estimates, not guarantees.")
            elif hasattr(model, "decision_function"):
                margin = float(model.decision_function(features)[0])
                st.metric("SVM decision margin", f"{margin:.3f}")
                st.caption(
                    "Linear SVM does not output calibrated probabilities by default. "
                    "The margin measures distance from the decision boundary."
                )

            with st.expander("Preprocessed text"):
                st.code(cleaned_input)

st.divider()
st.caption("Built with Streamlit, TF-IDF, and scikit-learn.")
