"""
Phase 2.3: Sequence Padding and Final Dataset Creation

Converts variable-length token sequences into fixed-length
sequences of MAX_SEQUENCE_LENGTH.

Padding:
    post

Truncation:
    post
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

import config


# ============================================================================
# PATHS
# ============================================================================

TRAIN_CLEAN_PATH = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "train_clean.csv"
)

VALIDATION_CLEAN_PATH = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "validation_clean.csv"
)

TEST_CLEAN_PATH = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "test_clean.csv"
)

TOKENIZER_PATH = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "tokenizer.json"
)


# ============================================================================
# LOAD TOKENIZER
# ============================================================================

def load_tokenizer():

    with open(
        TOKENIZER_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        tokenizer_json = file.read()

    tokenizer = (
        tf.keras.preprocessing.text.tokenizer_from_json(
            tokenizer_json
        )
    )

    return tokenizer


# ============================================================================
# LOAD DATA
# ============================================================================

def load_dataframe(path):

    return pd.read_csv(path)


# ============================================================================
# CONVERT TEXT TO SEQUENCES
# ============================================================================

def convert_to_sequences(
    tokenizer,
    dataframe
):

    return tokenizer.texts_to_sequences(
        dataframe["clean_text"].tolist()
    )


# ============================================================================
# PAD SEQUENCES
# ============================================================================

def pad_sequences(
    sequences
):

    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=config.MAX_SEQUENCE_LENGTH,
        padding=config.PADDING,
        truncating=config.TRUNCATING,
        value=0
    )

    return padded


# ============================================================================
# SAVE NPZ DATASET
# ============================================================================

def save_dataset(
    path,
    sequences,
    labels
):

    np.savez_compressed(
        path,
        X=sequences.astype(
            np.int32
        ),
        y=labels.astype(
            np.int32
        )
    )

    print(
        f"Saved: {path}"
    )

    print(
        f"X shape: {sequences.shape}"
    )

    print(
        f"y shape: {labels.shape}"
    )


# ============================================================================
# DISPLAY PADDING INFORMATION
# ============================================================================

def display_padding_information(
    sequences,
    name
):

    zero_counts = (
        sequences == 0
    ).sum(axis=1)

    print(
        f"\n{name} padding statistics:"
    )

    print(
        f"Minimum padding tokens : "
        f"{zero_counts.min():,}"
    )

    print(
        f"Maximum padding tokens : "
        f"{zero_counts.max():,}"
    )

    print(
        f"Average padding tokens : "
        f"{zero_counts.mean():.2f}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("\n" + "=" * 70)
    print(
        "PHASE 2.3 - PADDING AND ENCODING"
    )
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------------

    tokenizer = load_tokenizer()

    print(
        f"Vocabulary size configured: "
        f"{config.VOCAB_SIZE:,}"
    )

    print(
        f"Maximum sequence length: "
        f"{config.MAX_SEQUENCE_LENGTH}"
    )

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    train_df = load_dataframe(
        TRAIN_CLEAN_PATH
    )

    validation_df = load_dataframe(
        VALIDATION_CLEAN_PATH
    )

    test_df = load_dataframe(
        TEST_CLEAN_PATH
    )

    # ------------------------------------------------------------------------
    # Convert to integer sequences
    # ------------------------------------------------------------------------

    train_sequences = convert_to_sequences(
        tokenizer,
        train_df
    )

    validation_sequences = convert_to_sequences(
        tokenizer,
        validation_df
    )

    test_sequences = convert_to_sequences(
        tokenizer,
        test_df
    )

    # ------------------------------------------------------------------------
    # Pad
    # ------------------------------------------------------------------------

    X_train = pad_sequences(
        train_sequences
    )

    X_validation = pad_sequences(
        validation_sequences
    )

    X_test = pad_sequences(
        test_sequences
    )

    # ------------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------------

    y_train = train_df[
        "label"
    ].to_numpy(
        dtype=np.int32
    )

    y_validation = validation_df[
        "label"
    ].to_numpy(
        dtype=np.int32
    )

    y_test = test_df[
        "label"
    ].to_numpy(
        dtype=np.int32
    )

    # ------------------------------------------------------------------------
    # Display shapes
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL DATA SHAPES")
    print("=" * 70)

    print(
        f"X_train       : {X_train.shape}"
    )

    print(
        f"y_train       : {y_train.shape}"
    )

    print(
        f"X_validation  : {X_validation.shape}"
    )

    print(
        f"y_validation  : {y_validation.shape}"
    )

    print(
        f"X_test        : {X_test.shape}"
    )

    print(
        f"y_test        : {y_test.shape}"
    )

    # ------------------------------------------------------------------------
    # Padding information
    # ------------------------------------------------------------------------

    display_padding_information(
        X_train,
        "Train"
    )

    display_padding_information(
        X_validation,
        "Validation"
    )

    display_padding_information(
        X_test,
        "Test"
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    save_dataset(
        config.TRAIN_DATA_PATH,
        X_train,
        y_train
    )

    save_dataset(
        config.VALIDATION_DATA_PATH,
        X_validation,
        y_validation
    )

    save_dataset(
        config.TEST_DATA_PATH,
        X_test,
        y_test
    )

    # ------------------------------------------------------------------------
    # Show example
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PADDED SEQUENCE EXAMPLE")
    print("=" * 70)

    print(
        X_train[0][:100]
    )

    # ------------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "PHASE 2.3 PADDING AND ENCODING COMPLETED"
    )
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()