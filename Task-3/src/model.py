"""
model.py

Build the LSTM model.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout
)


def build_model(vocab_size, max_sequence_length):
    """
    Build and compile the LSTM model.

    Args:
        vocab_size (int): Number of words in vocabulary.
        max_sequence_length (int): Maximum sequence length.

    Returns:
        Compiled TensorFlow model.
    """

    model = Sequential()

    # Embedding Layer
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=100,
            input_length=max_sequence_length - 1
        )
    )

    # LSTM Layer
    model.add(
        LSTM(128)
    )

    # Dropout Layer
    model.add(
        Dropout(0.2)
    )

    # Hidden Dense Layer
    model.add(
        Dense(
            128,
            activation="relu"
        )
    )

    # Output Layer
    model.add(
        Dense(
            vocab_size,
            activation="softmax"
        )
    )

    # Compile
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model