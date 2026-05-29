
import numpy as np
import tensorflow as tf
import streamlit as st

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #dcdcdc;
    margin-bottom: 30px;
}

.stTextArea textarea {
    background-color: #1f2937;
    color: white;
    border-radius: 10px;
    border: 2px solid #4b5563;
    font-size: 16px;
}

.stButton>button {
    width: 100%;
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #ff1e1e;
    color: white;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}

.positive {
    background-color: rgba(0, 255, 127, 0.2);
    border: 2px solid #00ff7f;
    color: #00ff7f;
}

.negative {
    background-color: rgba(255, 99, 71, 0.2);
    border: 2px solid tomato;
    color: tomato;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# LOAD DATA AND MODEL
# -----------------------------------

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')


# -----------------------------------
# HELPER FUNCTIONS
# -----------------------------------

def decode_review(encoded_review):
    return ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in encoded_review]
    )


def preprocess_text(text):
    words = text.lower().split()

    encoded_review = [
        word_index.get(word, 2) + 3 for word in words
    ]

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review


def predict_sentiment(review):

    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)

    sentiment = (
        'Positive 😊'
        if prediction[0][0] > 0.5
        else 'Negative 😔'
    )

    return sentiment, prediction[0][0]


# -----------------------------------
# UI
# -----------------------------------

st.markdown(
    '<div class="title">🎬 IMDB Movie Review Sentiment Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Analyze whether a movie review is Positive or Negative using Deep Learning</div>',
    unsafe_allow_html=True
)


user_input = st.text_area(
    "✍ Enter Your Movie Review",
    height=200,
    placeholder="Type your movie review here..."
)


if st.button("🔍 Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        sentiment, score = predict_sentiment(user_input)

        confidence = float(score) * 100

        if "Positive" in sentiment:

            st.markdown(
                f"""
                <div class="result-box positive">
                    {sentiment}
                    <br><br>
                    Confidence Score: {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-box negative">
                    {sentiment}
                    <br><br>
                    Confidence Score: {(100-confidence):.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(float(score))


# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")
st.caption("Built using TensorFlow, Keras and Streamlit")

