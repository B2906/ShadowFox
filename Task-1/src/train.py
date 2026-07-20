"""
train.py

Train the Next Word Prediction LSTM model.
"""

from pathlib import Path
import pickle

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.dataset_loader import load_dataset
from src.preprocessing import preprocess_text, generate_sequences
from src.tokenizer_utils import create_tokenizer, fit_tokenizer
from src.data_preparation import (
    get_max_sequence_length,
    pad_training_sequences,
    split_features_labels,
    encode_labels
)
from src.utils import save_training_plots, save_config
from src.model import build_model
def prepare_training_data():
    """
    Load the dataset and prepare it for model training.

    Returns:
        X: Input features
        y: One-hot encoded labels
        tokenizer: Fitted tokenizer
        vocab_size: Vocabulary size
        max_length: Maximum sequence length
    """

    # Load dataset
    text = load_dataset()

    # Preprocess text
    text = preprocess_text(text)

    # Create tokenizer
    tokenizer = create_tokenizer()

    # Fit tokenizer
    fit_tokenizer(tokenizer, text)

    # Generate sequences
    sequences = generate_sequences(tokenizer, text)

    # Maximum sequence length
    max_length = get_max_sequence_length(sequences)

    # Pad sequences
    padded_sequences = pad_training_sequences(
        sequences,
        max_length
    )

    # Split into X and y
    X, y = split_features_labels(
        padded_sequences
    )

    # Vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    # One-hot encode labels
    y = encode_labels(
        y,
        vocab_size
    )

    return X, y, tokenizer, vocab_size, max_length

def train_model(X, y, vocab_size, max_length):
    """
    Build and train the LSTM model.

    Args:
        X: Input features
        y: One-hot encoded labels
        vocab_size: Vocabulary size
        max_length: Maximum sequence length

    Returns:
        model, history
    """

    # Build model
    model = build_model(
        vocab_size,
        max_length
    )

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    # Save best model
    checkpoint = ModelCheckpoint(
        filepath="models/next_word_model.keras",
        monitor="val_loss",
        save_best_only=True
    )

    # Train model
    history = model.fit(
        X,
        y,
        epochs=20,
        batch_size=128,
        validation_split=0.2,
        callbacks=[
            early_stopping,
            checkpoint
        ],
        verbose=1
    )

    return model, history
def save_tokenizer(tokenizer):
    """
    Save tokenizer for future predictions.
    """

    tokenizer_path = Path("models/tokenizer.pkl")

    with open(tokenizer_path, "wb") as file:
        pickle.dump(tokenizer, file)
if __name__ == "__main__":

    print("=" * 60)
    print("Preparing Training Data...")
    print("=" * 60)

    X, y, tokenizer, vocab_size, max_length = prepare_training_data()

    print("Training Started...\n")

    model, history = train_model(
        X,
        y,
        vocab_size,
        max_length
    )

    save_tokenizer(tokenizer)
    save_training_plots(history)

    save_config(
        vocab_size,
        max_length
    )

    print("\n")
    print("=" * 60)
    print("Training Completed Successfully!")
    print("=" * 60)

    print("Model Saved To : models/next_word_model.keras")
    print("Tokenizer Saved To : models/tokenizer.pkl")