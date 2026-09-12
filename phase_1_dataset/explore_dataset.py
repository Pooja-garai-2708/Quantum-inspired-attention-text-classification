"""
Phase 1: Explore the IMDB Dataset

This script performs basic exploratory data analysis (EDA):

1. Loads train, validation, and test datasets.
2. Displays dataset dimensions.
3. Checks missing values.
4. Checks duplicate reviews.
5. Displays class distribution.
6. Calculates review length statistics.
7. Displays sample reviews.
8. Saves exploration results.
"""

from pathlib import Path
import sys

import pandas as pd
import numpy as np


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

import config


# ============================================================================
# CREATE RESULT DIRECTORY
# ============================================================================

RESULTS_DIR = PROJECT_ROOT / "phase_1_dataset" / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================================
# LOAD DATASET
# ============================================================================

def load_datasets():

    print("=" * 70)
    print("LOADING IMDB DATASETS")
    print("=" * 70)

    train_df = pd.read_csv(config.TRAIN_TEXT_PATH)

    validation_df = pd.read_csv(
        config.VALIDATION_TEXT_PATH
    )

    test_df = pd.read_csv(
        config.TEST_TEXT_PATH
    )

    print("\nDatasets loaded successfully.")

    return train_df, validation_df, test_df


# ============================================================================
# DISPLAY DATASET INFORMATION
# ============================================================================

def display_dataset_information(
    train_df,
    validation_df,
    test_df
):

    print("\n" + "=" * 70)
    print("DATASET DIMENSIONS")
    print("=" * 70)

    print(
        f"Training set     : "
        f"{train_df.shape[0]:,} rows × {train_df.shape[1]} columns"
    )

    print(
        f"Validation set   : "
        f"{validation_df.shape[0]:,} rows × {validation_df.shape[1]} columns"
    )

    print(
        f"Test set         : "
        f"{test_df.shape[0]:,} rows × {test_df.shape[1]} columns"
    )

    print("\nColumns:")

    print(list(train_df.columns))


# ============================================================================
# CHECK MISSING VALUES
# ============================================================================

def check_missing_values(
    train_df,
    validation_df,
    test_df
):

    print("\n" + "=" * 70)
    print("MISSING VALUE ANALYSIS")
    print("=" * 70)

    datasets = {
        "Train": train_df,
        "Validation": validation_df,
        "Test": test_df
    }

    all_results = []

    for name, df in datasets.items():

        print(f"\n{name} dataset:")

        missing = df.isnull().sum()

        print(missing)

        for column, count in missing.items():

            all_results.append({
                "dataset": name,
                "column": column,
                "missing_values": int(count)
            })

    missing_df = pd.DataFrame(all_results)

    missing_df.to_csv(
        RESULTS_DIR / "missing_values.csv",
        index=False
    )

    return missing_df


# ============================================================================
# CHECK DUPLICATES
# ============================================================================

def check_duplicates(
    train_df,
    validation_df,
    test_df
):

    print("\n" + "=" * 70)
    print("DUPLICATE ANALYSIS")
    print("=" * 70)

    datasets = {
        "Train": train_df,
        "Validation": validation_df,
        "Test": test_df
    }

    duplicate_results = []

    for name, df in datasets.items():

        duplicate_count = df["text"].duplicated().sum()

        print(
            f"{name} duplicate reviews : "
            f"{duplicate_count:,}"
        )

        duplicate_results.append({
            "dataset": name,
            "duplicate_reviews": int(duplicate_count)
        })

    duplicate_df = pd.DataFrame(
        duplicate_results
    )

    duplicate_df.to_csv(
        RESULTS_DIR / "duplicate_analysis.csv",
        index=False
    )

    return duplicate_df


# ============================================================================
# CLASS DISTRIBUTION
# ============================================================================

def analyze_class_distribution(
    train_df,
    validation_df,
    test_df
):

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    datasets = {
        "Train": train_df,
        "Validation": validation_df,
        "Test": test_df
    }

    distribution_results = []

    for name, df in datasets.items():

        print(f"\n{name}:")

        counts = df["label"].value_counts().sort_index()

        total = len(df)

        for label, count in counts.items():

            percentage = (
                count / total
            ) * 100

            class_name = config.CLASS_NAMES.get(
                int(label),
                str(label)
            )

            print(
                f"  {label} ({class_name}): "
                f"{count:,} "
                f"({percentage:.2f}%)"
            )

            distribution_results.append({
                "dataset": name,
                "label": int(label),
                "class_name": class_name,
                "count": int(count),
                "percentage": percentage
            })

    distribution_df = pd.DataFrame(
        distribution_results
    )

    distribution_df.to_csv(
        RESULTS_DIR / "class_distribution.csv",
        index=False
    )

    return distribution_df


# ============================================================================
# REVIEW LENGTH ANALYSIS
# ============================================================================

def calculate_review_lengths(
    train_df,
    validation_df,
    test_df
):

    print("\n" + "=" * 70)
    print("REVIEW LENGTH ANALYSIS")
    print("=" * 70)

    datasets = {
        "Train": train_df,
        "Validation": validation_df,
        "Test": test_df
    }

    results = []

    for name, df in datasets.items():

        # Number of whitespace-separated words
        lengths = df["text"].fillna("").apply(
            lambda x: len(str(x).split())
        )

        print(f"\n{name}:")

        print(
            f"  Minimum words : {lengths.min():.0f}"
        )

        print(
            f"  Maximum words : {lengths.max():.0f}"
        )

        print(
            f"  Mean words    : {lengths.mean():.2f}"
        )

        print(
            f"  Median words  : {lengths.median():.2f}"
        )

        print(
            f"  Std deviation : {lengths.std():.2f}"
        )

        print(
            f"  95th percentile: "
            f"{np.percentile(lengths, 95):.0f}"
        )

        for length in lengths:

            results.append({
                "dataset": name,
                "word_count": int(length)
            })

    length_df = pd.DataFrame(results)

    length_df.to_csv(
        RESULTS_DIR / "review_lengths.csv",
        index=False
    )

    return length_df


# ============================================================================
# DISPLAY SAMPLE REVIEWS
# ============================================================================

def display_sample_reviews(
    train_df
):

    print("\n" + "=" * 70)
    print("SAMPLE REVIEWS")
    print("=" * 70)

    sample = train_df.sample(
        n=5,
        random_state=config.RANDOM_SEED
    )

    for index, row in sample.iterrows():

        class_name = config.CLASS_NAMES[
            int(row["label"])
        ]

        print("\n" + "-" * 70)

        print(
            f"Label: {row['label']} "
            f"({class_name})"
        )

        print("\nReview:")

        print(
            str(row["text"])[:1000]
        )

        print()


# ============================================================================
# DATASET SUMMARY
# ============================================================================

def create_dataset_summary(
    train_df,
    validation_df,
    test_df
):

    summary = pd.DataFrame({

        "dataset": [
            "Train",
            "Validation",
            "Test"
        ],

        "samples": [
            len(train_df),
            len(validation_df),
            len(test_df)
        ],

        "positive": [
            (train_df["label"] == 1).sum(),
            (validation_df["label"] == 1).sum(),
            (test_df["label"] == 1).sum()
        ],

        "negative": [
            (train_df["label"] == 0).sum(),
            (validation_df["label"] == 0).sum(),
            (test_df["label"] == 0).sum()
        ]
    })

    summary.to_csv(
        RESULTS_DIR / "dataset_summary.csv",
        index=False
    )

    return summary


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():

    print("\n")
    print("=" * 70)
    print("QUANTUM-INSPIRED ATTENTION PROJECT")
    print("PHASE 1 - DATASET EXPLORATION")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    train_df, validation_df, test_df = load_datasets()

    # ------------------------------------------------------------------------
    # Dataset information
    # ------------------------------------------------------------------------

    display_dataset_information(
        train_df,
        validation_df,
        test_df
    )

    # ------------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------------

    check_missing_values(
        train_df,
        validation_df,
        test_df
    )

    # ------------------------------------------------------------------------
    # Duplicate reviews
    # ------------------------------------------------------------------------

    check_duplicates(
        train_df,
        validation_df,
        test_df
    )

    # ------------------------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------------------------

    analyze_class_distribution(
        train_df,
        validation_df,
        test_df
    )

    # ------------------------------------------------------------------------
    # Review length
    # ------------------------------------------------------------------------

    calculate_review_lengths(
        train_df,
        validation_df,
        test_df
    )

    # ------------------------------------------------------------------------
    # Sample reviews
    # ------------------------------------------------------------------------

    display_sample_reviews(
        train_df
    )

    # ------------------------------------------------------------------------
    # Dataset summary
    # ------------------------------------------------------------------------

    summary = create_dataset_summary(
        train_df,
        validation_df,
        test_df
    )

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(summary.to_string(index=False))

    # ------------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1 DATASET EXPLORATION COMPLETED")
    print("=" * 70)

    print(
        f"\nResults saved in:\n"
        f"{RESULTS_DIR}"
    )


# ============================================================================
# PROGRAM ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()