"""
tokenizer_utils.py

Utility functions for creating, saving and loading
the TensorFlow tokenizer.
"""

import pickle
from pathlib import Path

from tensorflow.keras.preprocessing.text import Tokenizer


def create_tokenizer(num_words=None, oov_token="<OOV>"):
    """
    Create and return a TensorFlow tokenizer.

    Args:
        num_words (int, optional):
            Maximum vocabulary size.
            None means use every word.

        oov_token (str):
            Token used for unknown words.

    Returns:
        Tokenizer
    """
    return Tokenizer(
        num_words=num_words,
        oov_token=oov_token
    )


def fit_tokenizer(tokenizer, text):
    """
    Learn the vocabulary from text.
    """
    tokenizer.fit_on_texts([text])


def text_to_sequence(tokenizer, text):
    """
    Convert text into a sequence of integers.
    """
    return tokenizer.texts_to_sequences([text])[0]


def save_tokenizer(tokenizer, filepath):
    """
    Save tokenizer to disk.
    """
    filepath = Path(filepath)

    with open(filepath, "wb") as file:
        pickle.dump(tokenizer, file)


def load_tokenizer(filepath):
    """
    Load tokenizer from disk.
    """
    filepath = Path(filepath)

    with open(filepath, "rb") as file:
        return pickle.load(file)