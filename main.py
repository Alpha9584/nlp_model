import pandas as pd
import numpy as np
import tensorflow as tf
from cleaning import clean
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import pickle

data = pd.read_csv("clean.csv")
data["cleaned_text"] = data["text"].apply(lambda x: clean(x))

MAX_WORDS = 30000

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(data["cleaned_text"])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(data["cleaned_text"])

MAX_LEN = 100

padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

X = np.array(padded)
y = np.array(data["sentiment"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(MAX_WORDS, 32, input_length=MAX_LEN),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=7000, validation_split=0.15, verbose=2, batch_size=32, callbacks=[early_stopping])

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

model.save("sentiment_analysis_model.h5")

with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
