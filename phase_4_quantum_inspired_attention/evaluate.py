"""
Phase 4 — Evaluate Quantum-Inspired Attention.

Run from project root:

    python phase_4_quantum_inspired_attention\\evaluate.py
"""

import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# IMPORTS
# ================================================================

import config

from model import QuantumInspiredAttentionModel


# ================================================================
# PATHS
# ================================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "phase_4_quantum_inspired_attention"
    / "results"
)

CHECKPOINT_PATH = (
    config.CHECKPOINT_DIR
    / "quantum_inspired_attention_best.weights.h5"
)


# ================================================================
# RANDOM SEED
# ================================================================

def set_random_seed(seed: int):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    tf.keras.utils.set_random_seed(seed)


# ================================================================
# LOAD TEST DATA
# ================================================================

def load_test_dataset():

    if hasattr(
        config,
        "TEST_DATA_PATH"
    ):

        test_path = Path(
            config.TEST_DATA_PATH
        )

    elif hasattr(
        config,
        "PROCESSED_TEST_NPZ"
    ):

        test_path = Path(
            config.PROCESSED_TEST_NPZ
        )

    else:

        test_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "test.npz"
        )

    print("\n")

    print("=" * 70)

    print("LOADING TEST DATASET")

    print("=" * 70)

    print(
        f"\nTest dataset path:"
        f"\n{test_path}"
    )

    if not test_path.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n"
            f"{test_path}"
        )

    with np.load(
        test_path
    ) as data:

        X_test = data["X"]

        y_test = data["y"]

    print(
        "\nTest dataset loaded successfully."
    )

    print(
        f"Test samples    : "
        f"{len(X_test):,}"
    )

    print(
        f"Sequence length : "
        f"{X_test.shape[1]}"
    )

    print(
        f"Feature dtype   : "
        f"{X_test.dtype}"
    )

    print(
        f"Label dtype     : "
        f"{y_test.dtype}"
    )

    return X_test, y_test


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    print("\n")

    print("=" * 70)

    print(
        "LOADING QUANTUM-INSPIRED MODEL"
    )

    print("=" * 70)

    print(
        f"\nCheckpoint:"
        f"\n{CHECKPOINT_PATH}"
    )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            f"Run Phase 4 training first."
        )

    model = QuantumInspiredAttentionModel()

    # Build model

    model(
        tf.zeros(
            (
                1,
                config.MAX_SEQUENCE_LENGTH
            ),
            dtype=tf.int32
        )
    )

    # Load best weights

    model.load_weights(
        str(
            CHECKPOINT_PATH
        )
    )

    print(
        "\nBest checkpoint loaded successfully."
    )

    print(
        f"Total parameters : "
        f"{model.count_params():,}"
    )

    return model


# ================================================================
# GENERATE PREDICTIONS
# ================================================================

def generate_predictions(
    model,
    X_test
):

    print("\n")

    print("=" * 70)

    print(
        "GENERATING TEST PREDICTIONS"
    )

    print("=" * 70)

    probabilities = model.predict(
        X_test,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    print(
        "\nPredictions generated successfully."
    )

    return probabilities, predictions


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(
    y_test,
    predictions
):

    print("\n")

    print("=" * 70)

    print(
        "CLASSIFICATION METRICS"
    )

    print("=" * 70)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n")

    print(
        f"Accuracy  : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision : {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall    : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1-score  : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    return {
        "model":
            "Quantum-Inspired Attention",

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1_score":
            f1
    }


# ================================================================
# CONFUSION MATRIX
# ================================================================

def save_confusion_matrix(
    y_test,
    predictions
):

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n")

    print("=" * 70)

    print(
        "CONFUSION MATRIX"
    )

    print("=" * 70)

    print("\n")

    print(cm)

    # ------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------

    negative_name = config.CLASS_NAMES.get(
        0,
        "negative"
    )

    positive_name = config.CLASS_NAMES.get(
        1,
        "positive"
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual {negative_name}",
            f"Actual {positive_name}"
        ],
        columns=[
            f"Predicted {negative_name}",
            f"Predicted {positive_name}"
        ]
    )

    csv_path = (
        RESULTS_DIR
        / "quantum_confusion_matrix.csv"
    )

    cm_df.to_csv(
        csv_path
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(cm)

    plt.title(
        "Confusion Matrix — Quantum-Inspired Attention"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.xticks(
        [0, 1],
        [
            negative_name,
            positive_name
        ]
    )

    plt.yticks(
        [0, 1],
        [
            negative_name,
            positive_name
        ]
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    png_path = (
        RESULTS_DIR
        / "quantum_confusion_matrix.png"
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nCSV saved to:"
        f"\n{csv_path}"
    )

    print(
        f"\nPlot saved to:"
        f"\n{png_path}"
    )


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

def save_classification_report(
    y_test,
    predictions
):

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            config.CLASS_NAMES.get(
                0,
                "negative"
            ),
            config.CLASS_NAMES.get(
                1,
                "positive"
            )
        ],
        digits=4,
        zero_division=0
    )

    print("\n")

    print("=" * 70)

    print(
        "CLASSIFICATION REPORT"
    )

    print("=" * 70)

    print("\n")

    print(report)

    report_path = (
        RESULTS_DIR
        / "quantum_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "PHASE 4 — QUANTUM-INSPIRED ATTENTION\n\n"
        )

        f.write(report)

    print(
        f"\nReport saved to:"
        f"\n{report_path}"
    )


# ================================================================
# SAVE PREDICTIONS
# ================================================================

def save_predictions(
    y_test,
    predictions,
    probabilities
):

    df = pd.DataFrame({

        "actual_label":
            y_test.astype(int),

        "predicted_label":
            predictions.astype(int),

        "negative_probability":
            probabilities[:, 0],

        "positive_probability":
            probabilities[:, 1]
    })

    df["actual_sentiment"] = (
        df["actual_label"]
        .map(config.CLASS_NAMES)
    )

    df["predicted_sentiment"] = (
        df["predicted_label"]
        .map(config.CLASS_NAMES)
    )

    df["correct"] = (
        df["actual_label"]
        ==
        df["predicted_label"]
    )

    path = (
        RESULTS_DIR
        / "quantum_test_predictions.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"\nPredictions saved to:"
        f"\n{path}"
    )


# ================================================================
# SAVE METRICS
# ================================================================

def save_metrics(
    metrics,
    model
):

    df = pd.DataFrame([{

        "model":
            metrics["model"],

        "accuracy":
            metrics["accuracy"],

        "precision":
            metrics["precision"],

        "recall":
            metrics["recall"],

        "f1_score":
            metrics["f1_score"],

        "total_parameters":
            model.count_params()

    }])

    path = (
        RESULTS_DIR
        / "quantum_test_metrics.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"\nMetrics saved to:"
        f"\n{path}"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    config.create_project_directories()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    set_random_seed(
        config.RANDOM_SEED
    )

    print("\n")

    print("=" * 70)

    print(
        "PHASE 4 — QUANTUM-INSPIRED "
        "ATTENTION EVALUATION"
    )

    print("=" * 70)

    print(
        "\nAttention mechanism : "
        "Quantum-Inspired Probability Attention"
    )

    print(
        "Probability formula  : "
        "|tanh(e)|² / Σ|tanh(e)|²"
    )

    # ------------------------------------------------------------
    # Test data
    # ------------------------------------------------------------

    X_test, y_test = (
        load_test_dataset()
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = load_model()

    # ------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    # ------------------------------------------------------------
    # Keras evaluation
    # ------------------------------------------------------------

    print("\n")

    print("=" * 70)

    print(
        "KERAS TEST EVALUATION"
    )

    print("=" * 70)

    test_results = model.evaluate(
        X_test,
        y_test,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )

    test_loss = float(
        test_results[0]
    )

    test_accuracy = float(
        test_results[1]
    )

    print(
        f"\nTest loss     : "
        f"{test_loss:.4f}"
    )

    print(
        f"Test accuracy : "
        f"{test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)"
    )

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    probabilities, predictions = (
        generate_predictions(
            model,
            X_test
        )
    )

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    save_classification_report(
        y_test,
        predictions
    )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    save_confusion_matrix(
        y_test,
        predictions
    )

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    save_predictions(
        y_test,
        predictions,
        probabilities
    )

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    save_metrics(
        metrics,
        model
    )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print("\n")

    print("=" * 70)

    print(
        "PHASE 4 EVALUATION COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

    print("\nFINAL RESULTS")

    print("-" * 70)

    print(
        f"Test Loss       : "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy   : "
        f"{metrics['accuracy']:.4f} "
        f"({metrics['accuracy'] * 100:.2f}%)"
    )

    print(
        f"Precision       : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall          : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score        : "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"Parameters      : "
        f"{model.count_params():,}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()