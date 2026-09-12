"""
Phase 1: Dataset Statistics and Visualization

Project:
    Quantum-Inspired Attention Mechanism for Text Classification

Dataset:
    IMDB Movie Reviews

This script:
1. Loads train, validation and test datasets.
2. Calculates dataset statistics.
3. Calculates class distributions.
4. Calculates review/document length statistics.
5. Generates visualization graphs.
6. Saves all results for later analysis and reporting.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

import config


# ============================================================================
# DIRECTORIES
# ============================================================================

RESULTS_DIR = PROJECT_ROOT / "phase_1_dataset" / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================================
# LOAD DATA
# ============================================================================

def load_datasets():

    print("=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    train_df = pd.read_csv(
        config.TRAIN_TEXT_PATH
    )

    validation_df = pd.read_csv(
        config.VALIDATION_TEXT_PATH
    )

    test_df = pd.read_csv(
        config.TEST_TEXT_PATH
    )

    print(
        f"Train      : {len(train_df):,} samples"
    )

    print(
        f"Validation : {len(validation_df):,} samples"
    )

    print(
        f"Test       : {len(test_df):,} samples"
    )

    return train_df, validation_df, test_df


# ============================================================================
# CALCULATE TEXT LENGTH
# ============================================================================

def calculate_text_lengths(dataframe):

    dataframe = dataframe.copy()

    dataframe["word_count"] = (
        dataframe["text"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    dataframe["character_count"] = (
        dataframe["text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    return dataframe


# ============================================================================
# TEXT LENGTH STATISTICS
# ============================================================================

def calculate_length_statistics(
    name,
    dataframe
):

    word_lengths = dataframe["word_count"]

    character_lengths = dataframe["character_count"]

    statistics = {
        "dataset": name,
        "samples": len(dataframe),

        "min_words": word_lengths.min(),
        "max_words": word_lengths.max(),
        "mean_words": word_lengths.mean(),
        "median_words": word_lengths.median(),
        "std_words": word_lengths.std(),

        "min_characters": character_lengths.min(),
        "max_characters": character_lengths.max(),
        "mean_characters": character_lengths.mean(),
        "median_characters": character_lengths.median(),
        "std_characters": character_lengths.std(),
    }

    return statistics


# ============================================================================
# CLASS DISTRIBUTION
# ============================================================================

def calculate_class_distribution(
    name,
    dataframe
):

    distribution = (
        dataframe["label"]
        .value_counts()
        .sort_index()
    )

    rows = []

    for label, count in distribution.items():

        percentage = (
            count / len(dataframe)
        ) * 100

        rows.append({
            "dataset": name,
            "label": int(label),
            "class_name": config.CLASS_NAMES[
                int(label)
            ],
            "count": int(count),
            "percentage": percentage
        })

    return rows


# ============================================================================
# PRINT STATISTICS
# ============================================================================

def print_statistics(
    name,
    dataframe
):

    print("\n" + "=" * 70)

    print(
        f"{name.upper()} TEXT STATISTICS"
    )

    print("=" * 70)

    print(
        f"Number of reviews       : "
        f"{len(dataframe):,}"
    )

    print(
        f"Minimum words           : "
        f"{dataframe['word_count'].min():,}"
    )

    print(
        f"Maximum words           : "
        f"{dataframe['word_count'].max():,}"
    )

    print(
        f"Average words           : "
        f"{dataframe['word_count'].mean():.2f}"
    )

    print(
        f"Median words            : "
        f"{dataframe['word_count'].median():.2f}"
    )

    print(
        f"Standard deviation      : "
        f"{dataframe['word_count'].std():.2f}"
    )

    print(
        f"Minimum characters     : "
        f"{dataframe['character_count'].min():,}"
    )

    print(
        f"Maximum characters     : "
        f"{dataframe['character_count'].max():,}"
    )

    print(
        f"Average characters     : "
        f"{dataframe['character_count'].mean():.2f}"
    )


# ============================================================================
# GRAPH 1 — CLASS DISTRIBUTION
# ============================================================================

def plot_class_distribution(
    train_df,
    validation_df,
    test_df
):

    datasets = [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df)
    ]

    labels = [
        "Negative",
        "Positive"
    ]

    negative_counts = []
    positive_counts = []

    for _, dataframe in datasets:

        negative_counts.append(
            (dataframe["label"] == 0).sum()
        )

        positive_counts.append(
            (dataframe["label"] == 1).sum()
        )

    x = np.arange(len(datasets))

    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        negative_counts,
        width,
        label="Negative"
    )

    plt.bar(
        x + width / 2,
        positive_counts,
        width,
        label="Positive"
    )

    plt.xticks(
        x,
        ["Train", "Validation", "Test"]
    )

    plt.ylabel("Number of Reviews")

    plt.xlabel("Dataset")

    plt.title(
        "IMDB Dataset Class Distribution"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        RESULTS_DIR /
        "class_distribution.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================================
# GRAPH 2 — DOCUMENT LENGTH DISTRIBUTION
# ============================================================================

def plot_document_length_distribution(
    train_df
):

    plt.figure(figsize=(10, 6))

    plt.hist(
        train_df["word_count"],
        bins=50
    )

    plt.xlabel(
        "Number of Words per Review"
    )

    plt.ylabel(
        "Number of Reviews"
    )

    plt.title(
        "Distribution of IMDB Review Lengths"
    )

    plt.axvline(
        config.MAX_SEQUENCE_LENGTH,
        linestyle="--",
        label=(
            f"Maximum Sequence Length "
            f"({config.MAX_SEQUENCE_LENGTH})"
        )
    )

    plt.legend()

    plt.tight_layout()

    path = (
        RESULTS_DIR /
        "document_length_distribution.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================================
# GRAPH 3 — LENGTH BY SENTIMENT
# ============================================================================

def plot_length_by_class(
    train_df
):

    negative_lengths = train_df[
        train_df["label"] == 0
    ]["word_count"]

    positive_lengths = train_df[
        train_df["label"] == 1
    ]["word_count"]

    plt.figure(figsize=(10, 6))

    plt.boxplot(
        [
            negative_lengths,
            positive_lengths
        ],
        labels=[
            "Negative",
            "Positive"
        ],
        showfliers=False
    )

    plt.ylabel(
        "Number of Words"
    )

    plt.xlabel(
        "Sentiment Class"
    )

    plt.title(
        "Review Length by Sentiment Class"
    )

    plt.tight_layout()

    path = (
        RESULTS_DIR /
        "review_length_by_class.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================================
# GRAPH 4 — DATASET SIZE
# ============================================================================

def plot_dataset_sizes(
    train_df,
    validation_df,
    test_df
):

    names = [
        "Train",
        "Validation",
        "Test"
    ]

    sizes = [
        len(train_df),
        len(validation_df),
        len(test_df)
    ]

    plt.figure(figsize=(9, 6))

    plt.bar(
        names,
        sizes
    )

    plt.ylabel(
        "Number of Reviews"
    )

    plt.xlabel(
        "Dataset"
    )

    plt.title(
        "IMDB Dataset Split Sizes"
    )

    for index, value in enumerate(sizes):

        plt.text(
            index,
            value,
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    path = (
        RESULTS_DIR /
        "dataset_split_sizes.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================================
# SAVE SUMMARY TABLE
# ============================================================================

def save_summary_statistics(
    train_df,
    validation_df,
    test_df
):

    datasets = [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df)
    ]

    rows = []

    for name, dataframe in datasets:

        rows.append(
            calculate_length_statistics(
                name,
                dataframe
            )
        )

    summary_df = pd.DataFrame(rows)

    path = (
        RESULTS_DIR /
        "text_length_statistics.csv"
    )

    summary_df.to_csv(
        path,
        index=False
    )

    print(
        f"Saved: {path}"
    )

    return summary_df


# ============================================================================
# SAVE CLASS DISTRIBUTION TABLE
# ============================================================================

def save_class_distribution(
    train_df,
    validation_df,
    test_df
):

    datasets = [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df)
    ]

    rows = []

    for name, dataframe in datasets:

        rows.extend(
            calculate_class_distribution(
                name,
                dataframe
            )
        )

    distribution_df = pd.DataFrame(
        rows
    )

    path = (
        RESULTS_DIR /
        "class_distribution_statistics.csv"
    )

    distribution_df.to_csv(
        path,
        index=False
    )

    print(
        f"Saved: {path}"
    )

    return distribution_df


# ============================================================================
# SAVE OVERALL DATASET SUMMARY
# ============================================================================

def save_dataset_summary(
    train_df,
    validation_df,
    test_df
):

    rows = []

    for name, dataframe in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df)
    ]:

        rows.append({
            "dataset": name,
            "samples": len(dataframe),
            "positive": int(
                (dataframe["label"] == 1).sum()
            ),
            "negative": int(
                (dataframe["label"] == 0).sum()
            ),
            "positive_percentage": (
                (dataframe["label"] == 1).sum()
                / len(dataframe)
                * 100
            ),
            "negative_percentage": (
                (dataframe["label"] == 0).sum()
                / len(dataframe)
                * 100
            ),
            "average_words": dataframe[
                "word_count"
            ].mean(),
            "median_words": dataframe[
                "word_count"
            ].median()
        })

    summary_df = pd.DataFrame(rows)

    path = (
        RESULTS_DIR /
        "dataset_statistics_summary.csv"
    )

    summary_df.to_csv(
        path,
        index=False
    )

    print(
        f"Saved: {path}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("\n")

    print("=" * 70)
    print(
        "QUANTUM-INSPIRED ATTENTION"
    )
    print(
        "PHASE 1 - DATASET STATISTICS"
    )
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    train_df, validation_df, test_df = (
        load_datasets()
    )

    # ------------------------------------------------------------------------
    # Calculate lengths
    # ------------------------------------------------------------------------

    train_df = calculate_text_lengths(
        train_df
    )

    validation_df = calculate_text_lengths(
        validation_df
    )

    test_df = calculate_text_lengths(
        test_df
    )

    # ------------------------------------------------------------------------
    # Print statistics
    # ------------------------------------------------------------------------

    print_statistics(
        "Train",
        train_df
    )

    print_statistics(
        "Validation",
        validation_df
    )

    print_statistics(
        "Test",
        test_df
    )

    # ------------------------------------------------------------------------
    # Save CSV statistics
    # ------------------------------------------------------------------------

    save_summary_statistics(
        train_df,
        validation_df,
        test_df
    )

    save_class_distribution(
        train_df,
        validation_df,
        test_df
    )

    save_dataset_summary(
        train_df,
        validation_df,
        test_df
    )

    # ------------------------------------------------------------------------
    # Generate graphs
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    plot_class_distribution(
        train_df,
        validation_df,
        test_df
    )

    plot_document_length_distribution(
        train_df
    )

    plot_length_by_class(
        train_df
    )

    plot_dataset_sizes(
        train_df,
        validation_df,
        test_df
    )

    # ------------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1 DATASET STATISTICS COMPLETED")
    print("=" * 70)

    print("\nGenerated result files:")

    for file in sorted(
        RESULTS_DIR.iterdir()
    ):

        if file.is_file():

            print(
                f"  ✓ {file.name}"
            )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()