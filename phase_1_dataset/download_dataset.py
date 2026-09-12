"""
Phase 1: Download and prepare the IMDB dataset.

The IMDB dataset is used for binary sentiment classification:
    0 = Negative
    1 = Positive

This script:
1. Downloads the IMDB dataset using TensorFlow/Keras.
2. Creates the required project directories.
3. Saves the raw dataset.
4. Creates train, validation, and test datasets.
5. Saves the datasets for later phases.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

import sys

# Allow importing config.py from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import config


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR = config.RAW_DATA_DIR
PROCESSED_DIR = config.PROCESSED_DATA_DIR

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load IMDB dataset
# ---------------------------------------------------------------------------

def load_imdb_dataset():
    """Load the IMDB dataset from TensorFlow/Keras."""

    print("=" * 70)
    print("LOADING IMDB DATASET")
    print("=" * 70)

    dataset = tf.keras.datasets.imdb

    (x_train, y_train), (x_test, y_test) = dataset.load_data(
        num_words=config.VOCAB_SIZE
    )

    print(f"Training samples : {len(x_train)}")
    print(f"Testing samples  : {len(x_test)}")

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Load word index
# ---------------------------------------------------------------------------

def load_word_index():
    """Load IMDB word-to-index mapping."""

    word_index = tf.keras.datasets.imdb.get_word_index()

    # Keras IMDB uses special indices internally.
    # Shift normal words so that special tokens can be reserved.
    word_index = {
        word: index + 3
        for word, index in word_index.items()
    }

    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2

    return word_index


# ---------------------------------------------------------------------------
# Decode reviews
# ---------------------------------------------------------------------------

def decode_review(encoded_review, reverse_word_index):
    """Convert an encoded IMDB review back to text."""

    words = []

    for index in encoded_review:
        if index in reverse_word_index:
            words.append(reverse_word_index[index])
        else:
            words.append("<UNK>")

    return " ".join(words)


# ---------------------------------------------------------------------------
# Convert encoded data into DataFrame
# ---------------------------------------------------------------------------

def create_dataframe(x, y, reverse_word_index):
    """Create a DataFrame containing text and labels."""

    texts = [
        decode_review(review, reverse_word_index)
        for review in x
    ]

    dataframe = pd.DataFrame({
        "text": texts,
        "label": y.astype(int)
    })

    dataframe["class_name"] = dataframe["label"].map({
        0: "negative",
        1: "positive"
    })

    return dataframe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("\nQuantum-Inspired Attention Project")
    print("Phase 1 - Dataset Preparation\n")

    # Load data
    x_train, y_train, x_test, y_test = load_imdb_dataset()

    # Load vocabulary
    word_index = load_word_index()

    reverse_word_index = {
        index: word
        for word, index in word_index.items()
    }

    print(f"Vocabulary entries: {len(word_index):,}")

    # Convert to DataFrames
    train_df = create_dataframe(
        x_train,
        y_train,
        reverse_word_index
    )

    test_df = create_dataframe(
        x_test,
        y_test,
        reverse_word_index
    )

    # ---------------------------------------------------------------
    # Validation split
    # ---------------------------------------------------------------

    validation_size = int(
        len(train_df) * config.VALIDATION_SPLIT
    )

    # Shuffle reproducibly
    shuffled_train = train_df.sample(
        frac=1,
        random_state=config.RANDOM_SEED
    ).reset_index(drop=True)

    validation_df = shuffled_train.iloc[:validation_size].copy()

    final_train_df = shuffled_train.iloc[validation_size:].copy()

    # Reset indices
    final_train_df.reset_index(drop=True, inplace=True)
    validation_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # ---------------------------------------------------------------
    # Save raw datasets
    # ---------------------------------------------------------------

    raw_train_path = RAW_DIR / "imdb_train_raw.csv"
    raw_test_path = RAW_DIR / "imdb_test_raw.csv"

    train_df.to_csv(
        raw_train_path,
        index=False
    )

    test_df.to_csv(
        raw_test_path,
        index=False
    )

    # ---------------------------------------------------------------
    # Save processed datasets
    # ---------------------------------------------------------------

    final_train_df.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    validation_df.to_csv(
        PROCESSED_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    # ---------------------------------------------------------------
    # Save vocabulary
    # ---------------------------------------------------------------

    import json

    vocabulary_path = PROCESSED_DIR / "vocabulary.json"

    with open(
        vocabulary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            word_index,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ---------------------------------------------------------------
    # Display final information
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 70)

    print("\nDataset sizes:")
    print(f"Train      : {len(final_train_df):,}")
    print(f"Validation : {len(validation_df):,}")
    print(f"Test       : {len(test_df):,}")

    print("\nClass distribution - Train:")
    print(final_train_df["class_name"].value_counts())

    print("\nClass distribution - Validation:")
    print(validation_df["class_name"].value_counts())

    print("\nClass distribution - Test:")
    print(test_df["class_name"].value_counts())

    print("\nFiles created:")

    print(f"  {raw_train_path}")
    print(f"  {raw_test_path}")
    print(f"  {PROCESSED_DIR / 'train.csv'}")
    print(f"  {PROCESSED_DIR / 'validation.csv'}")
    print(f"  {PROCESSED_DIR / 'test.csv'}")
    print(f"  {vocabulary_path}")

    print("\nPhase 1 dataset download completed successfully.")


if __name__ == "__main__":"""
Phase 1: Download and prepare the IMDB dataset.

Dataset:
    IMDB Movie Reviews

Task:
    Binary sentiment classification

Labels:
    0 = Negative
    1 = Positive

This script:
1. Downloads the official IMDB dataset.
2. Decodes the reviews into text.
3. Creates a stratified train/validation split.
4. Keeps the official IMDB test set unchanged.
5. Saves raw and processed CSV files.
6. Saves the IMDB vocabulary.
"""

from pathlib import Path
import sys
import random
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

import config


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)


# ============================================================================
# DIRECTORIES
# ============================================================================

config.create_project_directories()


# ============================================================================
# LOAD IMDB DATASET
# ============================================================================

def load_imdb_dataset():

    print("=" * 70)
    print("LOADING IMDB DATASET")
    print("=" * 70)

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.imdb.load_data(
            num_words=config.VOCAB_SIZE
        )
    )

    print(
        f"Original training samples : {len(x_train):,}"
    )

    print(
        f"Original testing samples  : {len(x_test):,}"
    )

    return x_train, y_train, x_test, y_test


# ============================================================================
# LOAD IMDB WORD INDEX
# ============================================================================

def load_word_index():

    word_index = (
        tf.keras.datasets.imdb.get_word_index()
    )

    # Reserve:
    #
    # 0 -> PAD
    # 1 -> START
    # 2 -> UNKNOWN
    #
    word_index = {
        word: index + 3
        for word, index in word_index.items()
    }

    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2

    return word_index


# ============================================================================
# CREATE REVERSE WORD INDEX
# ============================================================================

def create_reverse_word_index(word_index):

    return {
        index: word
        for word, index in word_index.items()
    }


# ============================================================================
# DECODE REVIEW
# ============================================================================

def decode_review(encoded_review, reverse_word_index):

    words = []

    for index in encoded_review:

        word = reverse_word_index.get(
            index,
            "<UNK>"
        )

        words.append(word)

    return " ".join(words)


# ============================================================================
# CREATE DATAFRAME
# ============================================================================

def create_dataframe(
    x,
    y,
    reverse_word_index
):

    texts = [
        decode_review(
            review,
            reverse_word_index
        )
        for review in x
    ]

    dataframe = pd.DataFrame({
        "text": texts,
        "label": y.astype(int)
    })

    dataframe["class_name"] = (
        dataframe["label"]
        .map(config.CLASS_NAMES)
    )

    return dataframe


# ============================================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================================

def create_stratified_split(train_df):

    print("\n" + "=" * 70)
    print("CREATING STRATIFIED TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    train_df, validation_df = train_test_split(
        train_df,
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=train_df["label"]
    )

    train_df = train_df.reset_index(drop=True)

    validation_df = validation_df.reset_index(drop=True)

    print(
        f"Training samples   : {len(train_df):,}"
    )

    print(
        f"Validation samples : {len(validation_df):,}"
    )

    return train_df, validation_df


# ============================================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================================

def display_distribution(
    name,
    dataframe
):

    print(f"\n{name} class distribution:")

    counts = (
        dataframe["label"]
        .value_counts()
        .sort_index()
    )

    for label, count in counts.items():

        percentage = (
            count / len(dataframe)
        ) * 100

        class_name = config.CLASS_NAMES[
            int(label)
        ]

        print(
            f"  {label} ({class_name}): "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================================
# SAVE DATASETS
# ============================================================================

def save_datasets(
    original_train_df,
    train_df,
    validation_df,
    test_df,
    word_index
):

    print("\n" + "=" * 70)
    print("SAVING DATASETS")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Raw datasets
    # ------------------------------------------------------------------------

    original_train_df.to_csv(
        config.RAW_TRAIN_PATH,
        index=False
    )

    test_df.to_csv(
        config.RAW_TEST_PATH,
        index=False
    )

    # ------------------------------------------------------------------------
    # Processed datasets
    # ------------------------------------------------------------------------

    train_df.to_csv(
        config.TRAIN_TEXT_PATH,
        index=False
    )

    validation_df.to_csv(
        config.VALIDATION_TEXT_PATH,
        index=False
    )

    test_df.to_csv(
        config.TEST_TEXT_PATH,
        index=False
    )

    # ------------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------------

    with open(
        config.VOCAB_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            word_index,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\nFiles saved:")

    print(
        f"  {config.RAW_TRAIN_PATH}"
    )

    print(
        f"  {config.RAW_TEST_PATH}"
    )

    print(
        f"  {config.TRAIN_TEXT_PATH}"
    )

    print(
        f"  {config.VALIDATION_TEXT_PATH}"
    )

    print(
        f"  {config.TEST_TEXT_PATH}"
    )

    print(
        f"  {config.VOCAB_PATH}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("\n")
    print("=" * 70)
    print("QUANTUM-INSPIRED ATTENTION PROJECT")
    print("PHASE 1 - DATASET PREPARATION")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------------

    x_train, y_train, x_test, y_test = (
        load_imdb_dataset()
    )

    # ------------------------------------------------------------------------
    # Load vocabulary
    # ------------------------------------------------------------------------

    word_index = load_word_index()

    reverse_word_index = (
        create_reverse_word_index(word_index)
    )

    print(
        f"\nVocabulary entries : "
        f"{len(word_index):,}"
    )

    # ------------------------------------------------------------------------
    # Convert encoded reviews to text
    # ------------------------------------------------------------------------

    print("\nDecoding training reviews...")

    complete_train_df = create_dataframe(
        x_train,
        y_train,
        reverse_word_index
    )

    print("Decoding test reviews...")

    test_df = create_dataframe(
        x_test,
        y_test,
        reverse_word_index
    )

    # ------------------------------------------------------------------------
    # Preserve original training dataset
    # ------------------------------------------------------------------------

    original_train_df = complete_train_df.copy()

    # ------------------------------------------------------------------------
    # Stratified split
    # ------------------------------------------------------------------------

    train_df, validation_df = (
        create_stratified_split(
            complete_train_df
        )
    )

    # ------------------------------------------------------------------------
    # Display distributions
    # ------------------------------------------------------------------------

    display_distribution(
        "Train",
        train_df
    )

    display_distribution(
        "Validation",
        validation_df
    )

    display_distribution(
        "Test",
        test_df
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    save_datasets(
        original_train_df,
        train_df,
        validation_df,
        test_df,
        word_index
    )

    # ------------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    print(
        f"{'Dataset':<15}"
        f"{'Samples':>10}"
        f"{'Positive':>12}"
        f"{'Negative':>12}"
    )

    print("-" * 50)

    print(
        f"{'Train':<15}"
        f"{len(train_df):>10,}"
        f"{(train_df['label'] == 1).sum():>12,}"
        f"{(train_df['label'] == 0).sum():>12,}"
    )

    print(
        f"{'Validation':<15}"
        f"{len(validation_df):>10,}"
        f"{(validation_df['label'] == 1).sum():>12,}"
        f"{(validation_df['label'] == 0).sum():>12,}"
    )

    print(
        f"{'Test':<15}"
        f"{len(test_df):>10,}"
        f"{(test_df['label'] == 1).sum():>12,}"
        f"{(test_df['label'] == 0).sum():>12,}"
    )

    print("\n" + "=" * 70)
    print("PHASE 1 DATASET PREPARATION COMPLETED")
    print("=" * 70)


# ============================================================================
# PROGRAM ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
    main()