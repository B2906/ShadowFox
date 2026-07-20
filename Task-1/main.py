"""
main.py

Main file to test the complete data preprocessing pipeline.
"""

from src.dataset_loader import load_dataset
from src.preprocessing import (
    preprocess_text,
    generate_sequences
)
from src.tokenizer_utils import (
    create_tokenizer,
    fit_tokenizer
)
from src.data_preparation import (
    get_max_sequence_length,
    pad_training_sequences,
    split_features_labels,
    encode_labels
)
from src.model import build_model

def main():
    # Step 1: Load dataset
    text = load_dataset()

    # Step 2: Preprocess text
    text = preprocess_text(text)

    # Step 3: Create tokenizer
    tokenizer = create_tokenizer()

    # Step 4: Fit tokenizer
    fit_tokenizer(tokenizer, text)

    # Step 5: Generate training sequences
    sequences = generate_sequences(tokenizer, text)

    print("=" * 60)
    print("Total Training Sequences")
    print("=" * 60)
    print(len(sequences))

    print("\nFirst Five Sequences:\n")
    for seq in sequences[:5]:
        print(seq)

    # Step 6: Find maximum sequence length
    max_length = get_max_sequence_length(sequences)

    print("\nMaximum Sequence Length:", max_length)

    # Step 7: Pad sequences
    padded_sequences = pad_training_sequences(
        sequences,
        max_length
    )

    # Step 8: Split into features and labels
    X, y = split_features_labels(padded_sequences)

    # Step 9: Vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    # Step 10: One-hot encode labels
    y = encode_labels(y, vocab_size)
    model = build_model(
        vocab_size,
        max_length
    )
    model.build(input_shape=(None, max_length - 1))
    print("\n")
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    model.summary()
    print("\nShape of X:", X.shape)
    print("Shape of y:", y.shape)


if __name__ == "__main__":
    main()