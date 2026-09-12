"""
Phase 2.1: Text Cleaning

Project:
    Quantum-Inspired Attention Mechanism for Text Classification

Dataset:
    IMDB Movie Reviews

This script performs lightweight text normalization.

Important:
    We intentionally do NOT remove sentiment-bearing words such as:
    "not", "never", "bad", "excellent", etc.

Removing stopwords aggressively can damage sentiment information.
"""

from pathlib import Path
import sys
import re

import pandas as pd


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

import config


# ============================================================================
# DIRECTORIES
# ============================================================================

RESULTS_DIR = (
    PROJECT_ROOT /
    "phase_2_preprocessing" /
    "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================================
# TEXT CLEANING FUNCTION
# ============================================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # ------------------------------------------------------------------------
    # Convert HTML line breaks to spaces
    # ------------------------------------------------------------------------

    text = re.sub(
        r"<br\s*/?>",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # ------------------------------------------------------------------------
    # Convert text to lowercase
    # ------------------------------------------------------------------------

    text = text.lower()

    # ------------------------------------------------------------------------
    # Remove HTML tags
    # ------------------------------------------------------------------------

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # ------------------------------------------------------------------------
    # Keep alphabetic characters and apostrophes
    #
    # Example:
    #       don't -> don't
    #       wasn't -> wasn't
    #
    # This helps preserve sentiment information.
    # ------------------------------------------------------------------------

    text = re.sub(
        r"[^a-zA-Z'\s]",
        " ",
        text
    )

    # ------------------------------------------------------------------------
    # Normalize repeated apostrophes
    # ------------------------------------------------------------------------

    text = re.sub(
        r"'+",
        "'",
        text
    )

    # ------------------------------------------------------------------------
    # Normalize whitespace
    # ------------------------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = text.strip()

    return text


# ============================================================================
# PROCESS DATASET
# ============================================================================

def process_dataset(
    input_path,
    output_path,
    dataset_name
):

    print(
        f"\nProcessing {dataset_name}..."
    )

    dataframe = pd.read_csv(
        input_path
    )

    print(
        f"Original reviews: {len(dataframe):,}"
    )

    dataframe["clean_text"] = (
        dataframe["text"]
        .apply(clean_text)
    )

    # Remove completely empty reviews
    dataframe = dataframe[
        dataframe["clean_text"].str.len() > 0
    ].copy()

    dataframe.reset_index(
        drop=True,
        inplace=True
    )

    dataframe.to_csv(
        output_path,
        index=False
    )

    print(
        f"Processed reviews: {len(dataframe):,}"
    )

    print(
        f"Saved: {output_path}"
    )

    return dataframe


# ============================================================================
# SAVE CLEANING EXAMPLES
# ============================================================================

def save_cleaning_examples(
    original_df,
    cleaned_df
):

    number_of_examples = min(
        10,
        len(original_df)
    )

    examples = []

    for index in range(
        number_of_examples
    ):

        examples.append({
            "example": index + 1,
            "original_text": original_df.iloc[
                index
            ]["text"],
            "cleaned_text": cleaned_df.iloc[
                index
            ]["clean_text"]
        })

    examples_df = pd.DataFrame(
        examples
    )

    output_path = (
        RESULTS_DIR /
        "cleaning_examples.csv"
    )

    examples_df.to_csv(
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
        "PHASE 2.1 - TEXT CLEANING"
    )
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Input/output paths
    # ------------------------------------------------------------------------

    train_output = (
        PROJECT_ROOT /
        "data" /
        "processed" /
        "train_clean.csv"
    )

    validation_output = (
        PROJECT_ROOT /
        "data" /
        "processed" /
        "validation_clean.csv"
    )

    test_output = (
        PROJECT_ROOT /
        "data" /
        "processed" /
        "test_clean.csv"
    )

    # ------------------------------------------------------------------------
    # Load original training data for examples
    # ------------------------------------------------------------------------

    original_train_df = pd.read_csv(
        config.TRAIN_TEXT_PATH
    )

    # ------------------------------------------------------------------------
    # Process datasets
    # ------------------------------------------------------------------------

    train_df = process_dataset(
        config.TRAIN_TEXT_PATH,
        train_output,
        "TRAIN"
    )

    validation_df = process_dataset(
        config.VALIDATION_TEXT_PATH,
        validation_output,
        "VALIDATION"
    )

    test_df = process_dataset(
        config.TEST_TEXT_PATH,
        test_output,
        "TEST"
    )

    # ------------------------------------------------------------------------
    # Save examples
    # ------------------------------------------------------------------------

    save_cleaning_examples(
        original_train_df,
        train_df
    )

    # ------------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLEANING SUMMARY")
    print("=" * 70)

    print(
        f"Train      : {len(train_df):,}"
    )

    print(
        f"Validation : {len(validation_df):,}"
    )

    print(
        f"Test       : {len(test_df):,}"
    )

    print("\n" + "=" * 70)
    print(
        "PHASE 2.1 TEXT CLEANING COMPLETED"
    )
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()