import joblib
import streamlit as st


@st.cache_resource
def load_artifacts():

    model = joblib.load(
        "models/sentiment_model.pkl"
    )

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    return model, vectorizer


def predict_sentiment(
    review,
    model,
    vectorizer
):

    review_vector = vectorizer.transform(
        [review]
    )

    prediction = model.predict(
        review_vector
    )[0]

    probability = model.predict_proba(
        review_vector
    )[0]

    if prediction == 1:

        sentiment = "Positive Review"

        confidence = (
            probability[1] * 100
        )

    else:

        sentiment = "Negative Review"

        confidence = (
            probability[0] * 100
        )

    return sentiment, confidence