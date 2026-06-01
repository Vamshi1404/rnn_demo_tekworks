import streamlit as st

from sentiment_analysis import (
    load_artifacts,
    predict_sentiment
)

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    layout="wide"
)

st.title("IMDB Sentiment Analysis")

review = st.text_area(
    "Enter Movie Review"
)

if st.button("Predict"):

    if review.strip() == "":

        st.warning(
            "Please enter a review"
        )

    else:

        model, vectorizer = load_artifacts()

        sentiment, confidence = predict_sentiment(
            review,
            model,
            vectorizer
        )

        st.header("Prediction")

        st.success(sentiment)

        st.write(
            f"Confidence: {confidence:.2f}%"
        )