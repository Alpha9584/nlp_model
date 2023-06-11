import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from cleaning import clean

loaded_model = tf.keras.models.load_model("sentiment_analysis_model.h5")

with open("tokenizer.pickle", "rb") as handle:
    loaded_tokenizer = pickle.load(handle)

MAX_LEN = 100

def predict_sentiment(user_input):
    cleaned_input = clean(user_input)
    print(f"Cleaned input: {cleaned_input}")
    input_sequence = loaded_tokenizer.texts_to_sequences([cleaned_input])
    print(f"Input sequence: {input_sequence}")
    input_padded = pad_sequences(input_sequence, maxlen=MAX_LEN, padding="post", truncating="post")
    return loaded_model.predict(input_padded)[0][0]


while True:
    user_input = input("Enter a sentence to analyze sentiment (type 'quit' to exit): ")

    if user_input.strip().lower() == "quit":
        break

    sentiment = predict_sentiment(user_input)
    print(f"Sentiment score: {sentiment:.4f} (0-negative, 1-positive)")
