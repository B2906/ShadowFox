"""
data_preparation.py

Prepare training data for the LSTM model.
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np


def get_max_sequence_length(sequences):
    """
    Find the maximum sequence length.

    Args:
        sequences (list): List of token sequences.

    Returns:
        int: Maximum sequence length.
    """
    return max(len(sequence) for sequence in sequences)
def pad_training_sequences(sequences, max_length):
    """
    Pad all sequences to the same length.

    Args:
        sequences (list): Token sequences.
        max_length (int): Maximum sequence length.

    Returns:
        numpy.ndarray
    """
    return pad_sequences(
        sequences,
        maxlen=max_length,
        padding="pre"
    )
def split_features_labels(padded_sequences):
    """
    Split padded sequences into X and y.

    Returns:
        X, y
    """

    X = padded_sequences[:, :-1]
    y = padded_sequences[:, -1]

    return X, y
def encode_labels(y, vocab_size):
    """
    Convert labels into one-hot vectors.
    """

    return to_categorical(
        y,
        num_classes=vocab_size
    )