import joblib


def test_saved_model_artifacts_can_predict():
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    model = joblib.load("spam_classifier_model.joblib")

    features = vectorizer.transform(["free prize winner claim now"])
    prediction = model.predict(features)

    assert prediction.shape == (1,)
    assert int(prediction[0]) in {0, 1}
