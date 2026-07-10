"""
dataset_loader.py

This module is responsible for loading the raw text corpus.
No preprocessing is performed here.
"""

from pathlib import Path


def load_dataset(file_path=None):
    """
    Load the text corpus from the dataset file.

    Args:
        file_path (str | Path, optional):
            Path to the corpus file.
            If None, the default path is used.

    Returns:
        str:
            Complete contents of the text file.

    Raises:
        FileNotFoundError:
            If the dataset file does not exist.
    """

    if file_path is None:
        project_root = Path(__file__).resolve().parent.parent
        file_path = project_root / "data" / "raw" / "corpus.txt"

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text