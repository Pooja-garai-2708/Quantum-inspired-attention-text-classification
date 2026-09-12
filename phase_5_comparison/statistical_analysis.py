"""
PHASE 5 — STATISTICAL ANALYSIS

Statistical comparison of:

    Standard Attention
    vs
    Quantum-Inspired Attention

Tests:
    1. McNemar's Test
    2. Bootstrap Confidence Interval
    3. Prediction Agreement
    4. Accuracy Difference

No external statistical package is required for McNemar's
exact binomial calculation.
"""

import sys
from pathlib import Path
import os
import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_DIR = (
    PROJECT_ROOT
    / "phase_5_comparison"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------------------
# LOAD PREDICTIONS
# ---------------------------------------------------------------------

def load_predictions():

    path = (
        RESULTS_DIR
        / "model_predictions.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            "\nPrediction file not found.\n"
            "Run compare_models.py first.\n\n"
            f"Expected:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "actual",
        "standard_prediction",
        "quantum_prediction"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise KeyError(
                f"Missing column: {column}"
            )

    return df


# ---------------------------------------------------------------------
# MCNEMAR'S TEST
# ---------------------------------------------------------------------
def mcnemar_exact(y_true, pred_standard, pred_quantum):

    standard_correct = (
        pred_standard == y_true
    )

    quantum_correct = (
        pred_quantum == y_true
    )

    b = int(np.sum(
        standard_correct &
        ~quantum_correct
    ))

    c = int(np.sum(
        ~standard_correct &
        quantum_correct
    ))

    discordant = b + c

    if discordant == 0:
        p_value = 1.0

    else:
        k = min(b, c)

        log_terms = []

        for i in range(k + 1):

            log_combination = (
                math.lgamma(discordant + 1)
                - math.lgamma(i + 1)
                - math.lgamma(discordant - i + 1)
            )

            log_probability = (
                log_combination
                - discordant * math.log(2.0)
            )

            log_terms.append(log_probability)

        max_log = max(log_terms)

        lower_tail = (
            math.exp(max_log)
            * sum(
                math.exp(value - max_log)
                for value in log_terms
            )
        )

        p_value = min(
            1.0,
            2.0 * lower_tail
        )

    return {
        "Standard_correct_Quantum_wrong": b,
        "Standard_wrong_Quantum_correct": c,
        "Discordant_pairs": discordant,
        "McNemar_p_value": p_value
    }
# ---------------------------------------------------------------------
# BOOTSTRAP
# ---------------------------------------------------------------------

def bootstrap_accuracy_difference(
    y_true,
    pred_standard,
    pred_quantum,
    n_bootstrap=2000,
    confidence=0.95,
    seed=42
):

    rng = np.random.default_rng(seed)

    n = len(y_true)

    differences = np.zeros(
        n_bootstrap
    )

    standard_correct = (
        pred_standard == y_true
    )

    quantum_correct = (
        pred_quantum == y_true
    )

    for i in range(n_bootstrap):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        standard_accuracy = np.mean(
            standard_correct[indices]
        )

        quantum_accuracy = np.mean(
            quantum_correct[indices]
        )

        differences[i] = (
            quantum_accuracy
            -
            standard_accuracy
        )

    alpha = 1.0 - confidence

    lower = np.percentile(
        differences,
        100 * (alpha / 2)
    )

    upper = np.percentile(
        differences,
        100 * (1 - alpha / 2)
    )

    observed_difference = (
        np.mean(quantum_correct)
        -
        np.mean(standard_correct)
    )

    return {
        "Observed_Accuracy_Difference":
            observed_difference,
        "Bootstrap_CI_Lower":
            lower,
        "Bootstrap_CI_Upper":
            upper,
        "Bootstrap_Confidence":
            confidence,
        "Bootstrap_Samples":
            n_bootstrap
    }


# ---------------------------------------------------------------------
# AGREEMENT
# ---------------------------------------------------------------------

def prediction_agreement(
    y_true,
    pred_standard,
    pred_quantum
):

    same_predictions = (
        pred_standard == pred_quantum
    )

    agreement = np.mean(
        same_predictions
    )

    both_correct = np.mean(
        (pred_standard == y_true)
        &
        (pred_quantum == y_true)
    )

    both_wrong = np.mean(
        (pred_standard != y_true)
        &
        (pred_quantum != y_true)
    )

    return {
        "Prediction_Agreement":
            agreement,
        "Both_Correct_Rate":
            both_correct,
        "Both_Wrong_Rate":
            both_wrong
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("\n")
    print("=" * 70)
    print("PHASE 5 — STATISTICAL ANALYSIS")
    print("=" * 70)

    df = load_predictions()

    y_true = df["actual"].to_numpy()

    pred_standard = (
        df["standard_prediction"].to_numpy()
    )

    pred_quantum = (
        df["quantum_prediction"].to_numpy()
    )

    print(
        f"\nTotal test samples: {len(y_true):,}"
    )

    # -------------------------------------------------------------
    # ACCURACY
    # -------------------------------------------------------------

    standard_accuracy = np.mean(
        pred_standard == y_true
    )

    quantum_accuracy = np.mean(
        pred_quantum == y_true
    )

    difference = (
        quantum_accuracy
        -
        standard_accuracy
    )

    print("\nAccuracy:")
    print(
        f"Standard Attention: "
        f"{standard_accuracy:.4f}"
    )

    print(
        f"Quantum-Inspired Attention: "
        f"{quantum_accuracy:.4f}"
    )

    print(
        f"Difference: "
        f"{difference:+.4f}"
    )

    # -------------------------------------------------------------
    # MCNEMAR
    # -------------------------------------------------------------

    mcnemar = mcnemar_exact(
        y_true,
        pred_standard,
        pred_quantum
    )

    print("\n")
    print("=" * 70)
    print("McNEMAR'S TEST")
    print("=" * 70)

    print(
        "Standard correct / Quantum wrong:",
        mcnemar[
            "Standard_correct_Quantum_wrong"
        ]
    )

    print(
        "Standard wrong / Quantum correct:",
        mcnemar[
            "Standard_wrong_Quantum_correct"
        ]
    )

    print(
        "Discordant pairs:",
        mcnemar[
            "Discordant_pairs"
        ]
    )

    print(
        "p-value:",
        f"{mcnemar['McNemar_p_value']:.6f}"
    )

    if mcnemar["McNemar_p_value"] < 0.05:

        significance = (
            "Statistically significant "
            "difference at alpha = 0.05"
        )

    else:

        significance = (
            "No statistically significant "
            "difference at alpha = 0.05"
        )

    print(significance)

    # -------------------------------------------------------------
    # BOOTSTRAP
    # -------------------------------------------------------------

    bootstrap = bootstrap_accuracy_difference(
        y_true,
        pred_standard,
        pred_quantum,
        n_bootstrap=2000,
        confidence=0.95,
        seed=42
    )

    print("\n")
    print("=" * 70)
    print("BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 70)

    print(
        "Observed accuracy difference:",
        f"{bootstrap['Observed_Accuracy_Difference']:+.4f}"
    )

    print(
        "95% CI:",
        f"[{bootstrap['Bootstrap_CI_Lower']:.4f}, "
        f"{bootstrap['Bootstrap_CI_Upper']:.4f}]"
    )

    # -------------------------------------------------------------
    # AGREEMENT
    # -------------------------------------------------------------

    agreement = prediction_agreement(
        y_true,
        pred_standard,
        pred_quantum
    )

    print("\n")
    print("=" * 70)
    print("PREDICTION AGREEMENT")
    print("=" * 70)

    print(
        "Prediction agreement:",
        f"{agreement['Prediction_Agreement']:.4f}"
    )

    print(
        "Both correct:",
        f"{agreement['Both_Correct_Rate']:.4f}"
    )

    print(
        "Both wrong:",
        f"{agreement['Both_Wrong_Rate']:.4f}"
    )

    # -------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------

    statistical_results = {
        **mcnemar,
        **bootstrap,
        **agreement,
        "Standard_Accuracy":
            standard_accuracy,
        "Quantum_Accuracy":
            quantum_accuracy,
        "Accuracy_Difference":
            difference,
        "Significance":
            significance
    }

    results_df = pd.DataFrame(
        [statistical_results]
    )

    output_path = (
        RESULTS_DIR
        / "statistical_analysis.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nStatistical results saved to:")
    print(output_path)

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()