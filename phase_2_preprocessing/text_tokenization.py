"""
Phase 2.2: Tokenization

The tokenizer is fitted ONLY on the training dataset.

Validation and test datasets are transformed using the
same training vocabulary.

This prevents data leakage.
"""

from pathlib import Path
import sys
import json

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
# LOAD DATA
# ============================================================================

def load_data():

    train_df = pd.read_csv(
        TRAIN_CLEAN_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_CLEAN_PATH
    )

    test_df = pd.read_csv(
        TEST_CLEAN_PATH
    )

    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================================
# CREATE TOKENIZER
# ============================================================================

def create_tokenizer():

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=config.VOCAB_SIZE,
        oov_token=config.OOV_TOKEN,
        filters="",
        lower=False
    )

    return tokenizer


# ============================================================================
# FIT TOKENIZER
# ============================================================================

def fit_tokenizer(
    tokenizer,
    train_text
):

    print(
        "\nFitting tokenizer on training data..."
    )

    tokenizer.fit_on_texts(
        train_text
    )

    print(
        f"Total discovered words: "
        f"{len(tokenizer.word_index):,}"
    )

    print(
        f"Vocabulary used by model: "
        f"{config.VOCAB_SIZE:,}"
    )


# ============================================================================
# SAVE TOKENIZER
# ============================================================================

def save_tokenizer(tokenizer):

    tokenizer_json = tokenizer.to_json()

    with open(
        TOKENIZER_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            tokenizer_json
        )

    print(
        f"Saved tokenizer: "
        f"{TOKENIZER_PATH}"
    )


# ============================================================================
# SAVE VOCABULARY INFORMATION
# ============================================================================

def save_vocabulary(tokenizer):

    vocabulary = {}

    for word, index in tokenizer.word_index.items():

        if index < config.VOCAB_SIZE:

            vocabulary[word] = index

    vocabulary_path = (
        PROJECT_ROOT /
        "data" /
        "processed" /
        "tokenizer_vocabulary.json"
    )

    with open(
        vocabulary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            vocabulary,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Saved vocabulary: "
        f"{vocabulary_path}"
    )


# ============================================================================
# TOKENIZE DATASET
# ============================================================================

def tokenize_dataset(
    tokenizer,
    dataframe,
    dataset_name
):

    sequences = tokenizer.texts_to_sequences(
        dataframe["clean_text"].tolist()
    )

    print(
        f"{dataset_name} reviews tokenized: "
        f"{len(sequences):,}"
    )

    return sequences


# ============================================================================
# SAVE TOKENIZATION STATISTICS
# ============================================================================

def save_tokenization_statistics(
    train_sequences,
    validation_sequences,
    test_sequences
):

    rows = []

    datasets = [
        ("Train", train_sequences),
        ("Validation", validation_sequences),
        ("Test", test_sequences)
    ]

    for name, sequences in datasets:

        lengths = [
            len(sequence)
            for sequence in sequences
        ]

        rows.append({
            "dataset": name,
            "samples": len(lengths),
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "mean_tokens": sum(lengths) / len(lengths),
            "median_tokens": float(
                pd.Series(lengths).median()
            )
        })

    statistics_df = pd.DataFrame(
        rows
    )

    output_path = (
        PROJECT_ROOT /
        "phase_2_preprocessing" /
        "results" /
        "tokenization_statistics.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    statistics_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("\n" + "=" * 70)
    print(
        "PHASE 2.2 - TOKENIZATION"
    )
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df
    ) = load_data()

    print(
        f"\nTrain      : {len(train_df):,}"
    )

    print(
        f"Validation : {len(validation_df):,}"
    )

    print(
        f"Test       : {len(test_df):,}"
    )

    # ------------------------------------------------------------------------
    # Create tokenizer
    # ------------------------------------------------------------------------

    tokenizer = create_tokenizer()

    # ------------------------------------------------------------------------
    # Fit ONLY on training data
    # ------------------------------------------------------------------------

    fit_tokenizer(
        tokenizer,
        train_df["clean_text"]
    )

    # ------------------------------------------------------------------------
    # Save tokenizer
    # ------------------------------------------------------------------------

    save_tokenizer(
        tokenizer
    )

    save_vocabulary(
        tokenizer
    )

    # ------------------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------------------

    train_sequences = tokenize_dataset(
        tokenizer,
        train_df,
        "Train"
    )

    validation_sequences = tokenize_dataset(
        tokenizer,
        validation_df,
        "Validation"
    )

    test_sequences = tokenize_dataset(
        tokenizer,
        test_df,
        "Test"
    )

    # ------------------------------------------------------------------------
    # Save statistics
    # ------------------------------------------------------------------------

    save_tokenization_statistics(
        train_sequences,
        validation_sequences,
        test_sequences
    )

    # ------------------------------------------------------------------------
    # Display example
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOKENIZATION EXAMPLE")
    print("=" * 70)

    example_text = train_df.iloc[0][
        "clean_text"
    ]

    example_sequence = train_sequences[0]

    print("\nOriginal cleaned review:")
    print(
        example_text[:500]
    )

    print("\nToken sequence:")
    print(
        example_sequence[:50]
    )

    print("\n" + "=" * 70)
    print(
        "PHASE 2.2 TOKENIZATION COMPLETED"
    )
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()