"""
PHASE 6 — ATTENTION VISUALIZATION

Creates visual comparisons between:

1. Standard Additive Attention
2. Quantum-Inspired Attention

Outputs:
- Attention heatmaps
- Top-attended words
- Attention comparison CSV
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

RESULTS_DIR = (
    PROJECT_ROOT
    / "phase_6_attention_visualization"
    / "results"
)

STANDARD_PATH = (
    RESULTS_DIR
    / "standard_attention_samples.csv"
)

QUANTUM_PATH = (
    RESULTS_DIR
    / "quantum_attention_samples.csv"
)


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

def load_data():

    if not STANDARD_PATH.exists():

        raise FileNotFoundError(
            f"\nMissing:\n{STANDARD_PATH}\n"
            "Run extract_standard_attention.py first."
        )

    if not QUANTUM_PATH.exists():

        raise FileNotFoundError(
            f"\nMissing:\n{QUANTUM_PATH}\n"
            "Run extract_quantum_attention.py first."
        )

    standard = pd.read_csv(
        STANDARD_PATH
    )

    quantum = pd.read_csv(
        QUANTUM_PATH
    )

    return standard, quantum


# ---------------------------------------------------------------------
# TOP ATTENTION WORDS
# ---------------------------------------------------------------------

def get_top_words(
    df,
    model_name,
    top_n=20
):

    grouped = (
        df.groupby("token")[
            "attention_weight"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(top_n)
    )

    result = pd.DataFrame(
        {
            "model":
                model_name,

            "token":
                grouped.index,

            "mean_attention":
                grouped.values
        }
    )

    return result


# ---------------------------------------------------------------------
# CREATE HEATMAP
# ---------------------------------------------------------------------

def create_heatmap(
    df,
    sample_id,
    model_name,
    output_path
):

    sample = df[
        df["sample_id"] == sample_id
    ].copy()

    if sample.empty:
        print(
            f"No data for sample {sample_id}"
        )
        return

    tokens = sample[
        "token"
    ].tolist()

    weights = sample[
        "attention_weight"
    ].to_numpy()

    # Limit display for readability
    max_tokens = 50

    tokens = tokens[:max_tokens]
    weights = weights[:max_tokens]

    plt.figure(
        figsize=(16, 5)
    )

    plt.bar(
        range(len(tokens)),
        weights
    )

    plt.xticks(
        range(len(tokens)),
        tokens,
        rotation=70,
        ha="right"
    )

    plt.xlabel(
        "Input Tokens"
    )

    plt.ylabel(
        "Attention Weight"
    )

    plt.title(
        f"{model_name} — Sample {sample_id}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ---------------------------------------------------------------------
# COMPARE ATTENTION
# ---------------------------------------------------------------------

def compare_attention(
    standard,
    quantum
):

    merged = pd.merge(
        standard[
            [
                "sample_id",
                "position",
                "token",
                "attention_weight"
            ]
        ],
        quantum[
            [
                "sample_id",
                "position",
                "token",
                "attention_weight"
            ]
        ],
        on=[
            "sample_id",
            "position",
            "token"
        ],
        suffixes=(
            "_standard",
            "_quantum"
        )
    )

    merged[
        "attention_difference"
    ] = (
        merged["attention_weight_quantum"]
        -
        merged["attention_weight_standard"]
    )

    return merged


# ---------------------------------------------------------------------
# COMPARISON PLOT
# ---------------------------------------------------------------------

def create_comparison_plot(
    comparison,
    sample_id,
    output_path
):

    sample = comparison[
        comparison["sample_id"] == sample_id
    ].copy()

    if sample.empty:
        return

    sample = sample.head(40)

    positions = np.arange(
        len(sample)
    )

    width = 0.38

    plt.figure(
        figsize=(16, 6)
    )

    plt.bar(
        positions - width / 2,
        sample[
            "attention_weight_standard"
        ],
        width,
        label="Standard Attention"
    )

    plt.bar(
        positions + width / 2,
        sample[
            "attention_weight_quantum"
        ],
        width,
        label="Quantum-Inspired Attention"
    )

    plt.xticks(
        positions,
        sample["token"],
        rotation=70,
        ha="right"
    )

    plt.xlabel(
        "Input Tokens"
    )

    plt.ylabel(
        "Attention Weight"
    )

    plt.title(
        f"Attention Comparison — Sample {sample_id}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("\n")
    print("=" * 70)
    print(
        "PHASE 6 — ATTENTION VISUALIZATION"
    )
    print("=" * 70)

    standard, quantum = load_data()

    print(
        f"\nStandard records: "
        f"{len(standard):,}"
    )

    print(
        f"Quantum records: "
        f"{len(quantum):,}"
    )

    # -------------------------------------------------------------
    # TOP WORDS
    # -------------------------------------------------------------

    standard_top = get_top_words(
        standard,
        "Standard Attention",
        top_n=20
    )

    quantum_top = get_top_words(
        quantum,
        "Quantum-Inspired Attention",
        top_n=20
    )

    top_words = pd.concat(
        [
            standard_top,
            quantum_top
        ],
        ignore_index=True
    )

    top_words_path = (
        RESULTS_DIR
        / "top_attention_words.csv"
    )

    top_words.to_csv(
        top_words_path,
        index=False
    )

    print(
        f"\nTop attention words saved to:"
    )

    print(top_words_path)

    # -------------------------------------------------------------
    # ATTENTION COMPARISON
    # -------------------------------------------------------------

    comparison = compare_attention(
        standard,
        quantum
    )

    comparison_path = (
        RESULTS_DIR
        / "attention_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print(
        f"Attention comparison saved to:"
    )

    print(comparison_path)

    # -------------------------------------------------------------
    # VISUALIZE FIRST SAMPLE
    # -------------------------------------------------------------

    sample_id = int(
        standard["sample_id"].iloc[0]
    )

    standard_plot = (
        RESULTS_DIR
        / "standard_attention_heatmap.png"
    )

    quantum_plot = (
        RESULTS_DIR
        / "quantum_attention_heatmap.png"
    )

    comparison_plot = (
        RESULTS_DIR
        / "attention_comparison.png"
    )

    create_heatmap(
        standard,
        sample_id,
        "Standard Attention",
        standard_plot
    )

    create_heatmap(
        quantum,
        sample_id,
        "Quantum-Inspired Attention",
        quantum_plot
    )

    create_comparison_plot(
        comparison,
        sample_id,
        comparison_plot
    )

    # -------------------------------------------------------------
    # DISPLAY TOP WORDS
    # -------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOP STANDARD ATTENTION WORDS")
    print("=" * 70)

    print(
        standard_top.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 70)
    print("TOP QUANTUM-INSPIRED ATTENTION WORDS")
    print("=" * 70)

    print(
        quantum_top.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print(
        "ATTENTION VISUALIZATION COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()