"""
predict.py

Predict the next word using the trained LSTM model.
"""

import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model
model = load_model("models/next_word_model.keras")


# Load tokenizer
with open("models/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)


# Maximum sequence length
MAX_SEQUENCE_LENGTH = 16


def predict_next_words(text, top_k=3):
    """
    Predict the top-k next words with confidence scores.
    """

    token_list = tokenizer.texts_to_sequences([text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=MAX_SEQUENCE_LENGTH - 1,
        padding="pre"
    )

    prediction = model.predict(
        token_list,
        verbose=0
    )[0]

    # Get indices of top-k predictions
    top_indices = np.argsort(prediction)[-top_k:][::-1]

    results = []

    for index in top_indices:

        word = None

        word = tokenizer.index_word.get(index)

        if word:
            confidence = prediction[index] * 100
            results.append((word, confidence))

    return results

if __name__ == "__main__":

    while True:

        sentence = input("\nEnter text (type 'exit' to quit): ")

        if sentence.lower() == "exit":
            break

        predictions = predict_next_words(sentence)

        print("\nTop Predictions:\n")

        for i, (word, confidence) in enumerate(predictions, start=1):
          print(f"{i}. {word:<15} {confidence:.2f}%")