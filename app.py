import streamlit as st

from rnn_analysis import (
    load_model,
    predict_sentiment
)

st.set_page_config(page_title="IMDB Sentiment Analysis", layout="wide")

st.title("RNN IMDB Sentiment Analysis")

review = st.text_area(
    "Enter Movie Review"
)

if st.button("Predict"):

    if review.strip() == "":

        st.warning(
            "Please enter a review"
        )

    else:

        model = load_model()

        sentiment, confidence = predict_sentiment(
            review,
            model
        )

        st.header("Prediction")

        st.success(sentiment)

        st.write(
            f"Confidence: {confidence:.2f}%"
        )