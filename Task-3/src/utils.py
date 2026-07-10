# Shared utility functions placeholder.
# Placeholder file.
"""
utils.py

Utility functions for saving plots and configurations.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt


def save_training_plots(history):

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "accuracy.png")
    plt.close()

    # Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss.png")
    plt.close()


def save_config(vocab_size, max_sequence_length):

    config = {
        "vocab_size": vocab_size,
        "max_sequence_length": max_sequence_length
    }

    with open("models/config.json", "w") as file:
        json.dump(config, file, indent=4)