import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense
)

VOCAB_SIZE = 10000

MAX_LEN = 200


def train_model():

    (X_train, y_train), (X_test, y_test) = imdb.load_data(
        num_words=VOCAB_SIZE
    )

    X_train = pad_sequences(
        X_train,
        maxlen=MAX_LEN
    )

    X_test = pad_sequences(
        X_test,
        maxlen=MAX_LEN
    )

    model = Sequential([
        Embedding(
            VOCAB_SIZE,
            32,
            input_length=MAX_LEN
        ),

        SimpleRNN(64),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=5,
        batch_size=64
    )

    model.save(
        "rnn_imdb.keras"
    )

    return model


def load_model():

    try:

        model = tf.keras.models.load_model(
            "rnn_imdb.keras"
        )

    except:

        model = train_model()

    return model


def predict_sentiment(text, model):

    word_index = imdb.get_word_index()

    words = text.lower().split()

    sequence = []

    for word in words:

        if word in word_index:

            sequence.append(
                word_index[word] + 3
            )

        else:

            sequence.append(2)

    sequence = pad_sequences(
        [sequence],
        maxlen=MAX_LEN
    )

    prediction = model.predict(
        sequence,
        verbose=0
    )[0][0]

    if prediction > 0.5:

        sentiment = "Positive Review"

        confidence = prediction * 100

    else:

        sentiment = "Negative Review"

        confidence = (1 - prediction) * 100

    return sentiment, confidence