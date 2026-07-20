"""
preprocessing.py

This module handles text preprocessing.
"""

import re


def preprocess_text(text: str) -> str:
    """
    Preprocess raw text.

    Steps:
    1. Convert to lowercase
    2. Remove extra spaces
    3. Remove tabs
    4. Remove multiple newlines

    Args:
        text (str): Raw text.

    Returns:
        str: Cleaned text.
    """

    # Convert to lowercase
    text = text.lower()

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Replace multiple newlines with a single newline
    text = re.sub(r"\n+", "\n", text)

    # Replace multiple spaces with a single space
    text = re.sub(r" +", " ", text)

    return text.strip()
def generate_sequences(tokenizer, text):
    """
    Generate incremental token sequences for next-word prediction.

    Args:
        tokenizer: Fitted TensorFlow tokenizer
        text (str): Preprocessed text

    Returns:
        list: List of token sequences
    """

    sequences = []

    lines = text.split("\n")

    for line in lines:

        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(token_list)):
            sequences.append(token_list[:i+1])

    return sequences