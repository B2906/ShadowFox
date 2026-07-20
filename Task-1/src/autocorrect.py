"""
autocorrect.py

Utility functions for correcting spelling.
"""

from textblob import TextBlob


def autocorrect_text(text):
    """
    Correct spelling mistakes.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Corrected sentence.
    """

    corrected = TextBlob(text).correct()

    return str(corrected)