"""
Phase 3 — Evaluate Standard Attention Baseline.

This script evaluates the best saved Phase 3 baseline model
on the IMDB test dataset.

Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score

Additional outputs:
    - Classification report
    - Confusion matrix
    - Confusion matrix plot
    - Test predictions
    - Probability predictions
    - Evaluation summary

Run from the project root:

    python phase_3_baseline_model\\evaluate.py
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
    classification_report,
)


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# PROJECT IMPORTS
# ================================================================

import config

from model import BaselineAttentionModel


# ================================================================
# PATHS
# ================================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "phase_3_baseline_model"
    / "results"
)

CHECKPOINT_PATH = (
    config.CHECKPOINT_DIR
    / "baseline_attention_best.weights.h5"
)


# ================================================================
# RANDOM SEED
# ================================================================

def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    tf.keras.utils.set_random_seed(seed)


# ================================================================
# DATASET LOADING
# ================================================================

def load_test_dataset():
    """
    Load the Phase 2 processed test dataset.

    Expected NPZ contents:

        X -> padded review sequences
        y -> binary sentiment labels
    """

    # ------------------------------------------------------------
    # Find the test dataset path.
    #
    # The current project config normally provides TEST_DATA_PATH.
    # If not, fall back to the standard processed/test.npz path.
    # ------------------------------------------------------------

    if hasattr(config, "TEST_DATA_PATH"):

        test_path = Path(config.TEST_DATA_PATH)

    elif hasattr(config, "PROCESSED_TEST_NPZ"):

        test_path = Path(config.PROCESSED_TEST_NPZ)

    else:

        test_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "test.npz"
        )

    print("\n" + "=" * 70)
    print("LOADING TEST DATASET")
    print("=" * 70)

    print(f"\nTest dataset path:")
    print(test_path)

    if not test_path.exists():

        raise FileNotFoundError(
            f"\nTest dataset was not found at:\n"
            f"{test_path}\n\n"
            f"Please run the Phase 2 preprocessing first."
        )

    with np.load(test_path) as data:

        if "X" not in data:

            raise KeyError(
                f"The test dataset does not contain an 'X' array:\n"
                f"{test_path}"
            )

        if "y" not in data:

            raise KeyError(
                f"The test dataset does not contain a 'y' array:\n"
                f"{test_path}"
            )

        x_test = data["X"]
        y_test = data["y"]

    print("\nTest dataset loaded successfully.")

    print(f"Test samples          : {len(x_test)}")
    print(f"Sequence length       : {x_test.shape[1]}")
    print(f"Feature dtype         : {x_test.dtype}")
    print(f"Label dtype           : {y_test.dtype}")

    print("\nClass distribution:")

    unique_labels, counts = np.unique(
        y_test,
        return_counts=True
    )

    for label, count in zip(unique_labels, counts):

        class_name = config.CLASS_NAMES.get(
            int(label),
            str(label)
        )

        print(
            f"  {int(label)} ({class_name}) : "
            f"{int(count)}"
        )

    return x_test, y_test


# ================================================================
# LOAD MODEL
# ================================================================

def load_baseline_model():
    """
    Recreate the Phase 3 baseline architecture and load
    the best saved weights.
    """

    print("\n" + "=" * 70)
    print("LOADING BASELINE MODEL")
    print("=" * 70)

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"\nBest baseline checkpoint was not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            f"Run Phase 3 training first."
        )

    print("\nCheckpoint:")
    print(CHECKPOINT_PATH)

    # ------------------------------------------------------------
    # Recreate architecture
    # ------------------------------------------------------------

    model = BaselineAttentionModel()

    # Build model by giving it one dummy batch.
    model(
        tf.zeros(
            (
                1,
                config.MAX_SEQUENCE_LENGTH
            ),
            dtype=tf.int32
        )
    )

    # ------------------------------------------------------------
    # Load best checkpoint
    # ------------------------------------------------------------

    model.load_weights(
        str(CHECKPOINT_PATH)
    )

    print("\nBest checkpoint loaded successfully.")

    print(
        f"Total parameters : "
        f"{model.count_params():,}"
    )

    return model


# ================================================================
# EVALUATE LOSS / ACCURACY
# ================================================================

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model using Keras.
    """

    print("\n" + "=" * 70)
    print("KERAS TEST EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Compile model for evaluation
    # ------------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------

    test_results = model.evaluate(
        x_test,
        y_test,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )

    test_loss = float(test_results[0])
    test_accuracy = float(test_results[1])

    print("\n" + "-" * 70)
    print("TEST RESULTS")
    print("-" * 70)

    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_accuracy:.4f}")
    print(f"Test accuracy : {test_accuracy * 100:.2f}%")

    return test_loss, test_accuracy


# ================================================================
# PREDICTIONS
# ================================================================

def generate_predictions(model, x_test):
    """
    Generate class probabilities and predicted labels.
    """

    print("\n" + "=" * 70)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 70)

    probabilities = model.predict(
        x_test,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )

    # ------------------------------------------------------------
    # Expected output:
    #
    # probabilities[:, 0] = negative probability
    # probabilities[:, 1] = positive probability
    # ------------------------------------------------------------

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    print("\nPredictions generated successfully.")

    print(
        f"Prediction shape       : "
        f"{predictions.shape}"
    )

    print(
        f"Probability shape      : "
        f"{probabilities.shape}"
    )

    return probabilities, predictions


# ================================================================
# CLASSIFICATION METRICS
# ================================================================

def calculate_metrics(y_test, predictions):
    """
    Calculate Accuracy, Precision, Recall and F1-score.
    """

    print("\n" + "=" * 70)
    print("CLASSIFICATION METRICS")
    print("=" * 70)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="binary",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="binary",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="binary",
        zero_division=0
    )

    print("\n" + "-" * 70)

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

    metrics = {
        "model": "Standard Attention Baseline",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return metrics


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

def save_classification_report(
    y_test,
    predictions
):
    """
    Generate and save the detailed classification report.
    """

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    target_names = [
        config.CLASS_NAMES.get(
            0,
            "negative"
        ),
        config.CLASS_NAMES.get(
            1,
            "positive"
        )
    ]

    report_text = classification_report(
        y_test,
        predictions,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    print("\n")
    print(report_text)

    report_path = (
        RESULTS_DIR
        / "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PHASE 3 — STANDARD ATTENTION BASELINE\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(report_text)

    print(
        f"Classification report saved to:\n"
        f"{report_path}"
    )

    return report_text


# ================================================================
# CONFUSION MATRIX
# ================================================================

def save_confusion_matrix(
    y_test,
    predictions
):
    """
    Generate and save the confusion matrix as CSV and PNG.
    """

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nConfusion Matrix:")

    print(cm)

    # ------------------------------------------------------------
    # Save CSV
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

    cm_csv_path = (
        RESULTS_DIR
        / "confusion_matrix.csv"
    )

    cm_df.to_csv(cm_csv_path)

    # ------------------------------------------------------------
    # Create plot
    # ------------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(cm)

    plt.title(
        "Confusion Matrix — Standard Attention"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.xticks(
        [0, 1],
        [negative_name, positive_name]
    )

    plt.yticks(
        [0, 1],
        [negative_name, positive_name]
    )

    # ------------------------------------------------------------
    # Add values inside matrix
    # ------------------------------------------------------------

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    cm_png_path = (
        RESULTS_DIR
        / "confusion_matrix.png"
    )

    plt.savefig(
        cm_png_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nConfusion matrix CSV saved to:\n"
        f"{cm_csv_path}"
    )

    print(
        f"Confusion matrix plot saved to:\n"
        f"{cm_png_path}"
    )

    return cm


# ================================================================
# SAVE PREDICTIONS
# ================================================================

def save_predictions(
    y_test,
    predictions,
    probabilities
):
    """
    Save detailed test predictions.
    """

    print("\n" + "=" * 70)
    print("SAVING TEST PREDICTIONS")
    print("=" * 70)

    prediction_df = pd.DataFrame({
        "actual_label": y_test.astype(int),

        "predicted_label": predictions.astype(int),

        "negative_probability":
            probabilities[:, 0],

        "positive_probability":
            probabilities[:, 1]
    })

    # Add readable sentiment names

    prediction_df["actual_sentiment"] = (
        prediction_df["actual_label"]
        .map(config.CLASS_NAMES)
    )

    prediction_df["predicted_sentiment"] = (
        prediction_df["predicted_label"]
        .map(config.CLASS_NAMES)
    )

    # ------------------------------------------------------------
    # Correct / Incorrect
    # ------------------------------------------------------------

    prediction_df["correct"] = (
        prediction_df["actual_label"]
        ==
        prediction_df["predicted_label"]
    )

    predictions_path = (
        RESULTS_DIR
        / "test_predictions.csv"
    )

    prediction_df.to_csv(
        predictions_path,
        index=False
    )

    print(
        f"\nPredictions saved to:\n"
        f"{predictions_path}"
    )

    print(
        f"Total predictions : "
        f"{len(prediction_df):,}"
    )

    print(
        f"Correct predictions: "
        f"{prediction_df['correct'].sum():,}"
    )

    print(
        f"Incorrect predictions: "
        f"{(~prediction_df['correct']).sum():,}"
    )

    return prediction_df


# ================================================================
# SAVE METRICS
# ================================================================

def save_metrics(
    metrics,
    test_loss,
    model
):
    """
    Save final Phase 3 evaluation metrics.
    """

    metrics_to_save = {
        "model": metrics["model"],
        "test_loss": test_loss,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "total_parameters": model.count_params()
    }

    metrics_df = pd.DataFrame(
        [metrics_to_save]
    )

    metrics_path = (
        RESULTS_DIR
        / "baseline_test_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print(
        f"\nEvaluation metrics saved to:\n"
        f"{metrics_path}"
    )

    return metrics_df


# ================================================================
# MAIN
# ================================================================

def main():

    # ------------------------------------------------------------
    # Create directories
    # ------------------------------------------------------------

    config.create_project_directories()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    set_random_seed(
        config.RANDOM_SEED
    )

    print("\n")
    print("=" * 70)
    print("PHASE 3 — STANDARD ATTENTION BASELINE EVALUATION")
    print("=" * 70)

    print(
        "\nAttention mechanism : "
        "Standard Additive Attention"
    )

    print(
        "Dataset             : IMDB"
    )

    print(
        "Number of classes   : "
        f"{config.NUM_CLASSES}"
    )

    print(
        "Random seed         : "
        f"{config.RANDOM_SEED}"
    )

    # ------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------

    x_test, y_test = load_test_dataset()

    # ------------------------------------------------------------
    # Load best trained model
    # ------------------------------------------------------------

    model = load_baseline_model()

    # ------------------------------------------------------------
    # Evaluate model
    # ------------------------------------------------------------

    test_loss, test_accuracy = evaluate_model(
        model,
        x_test,
        y_test
    )

    # ------------------------------------------------------------
    # Generate predictions
    # ------------------------------------------------------------

    probabilities, predictions = generate_predictions(
        model,
        x_test
    )

    # ------------------------------------------------------------
    # Calculate metrics
    # ------------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    # ------------------------------------------------------------
    # Classification report
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
    # Save predictions
    # ------------------------------------------------------------

    save_predictions(
        y_test,
        predictions,
        probabilities
    )

    # ------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------

    save_metrics(
        metrics,
        test_loss,
        model
    )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PHASE 3 EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nFINAL TEST RESULTS")
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

    print("\nGenerated evaluation files:")

    print(
        f"\n✓ Test metrics:"
        f"\n  {RESULTS_DIR / 'baseline_test_metrics.csv'}"
    )

    print(
        f"\n✓ Classification report:"
        f"\n  {RESULTS_DIR / 'classification_report.txt'}"
    )

    print(
        f"\n✓ Confusion matrix:"
        f"\n  {RESULTS_DIR / 'confusion_matrix.csv'}"
    )

    print(
        f"\n✓ Confusion matrix plot:"
        f"\n  {RESULTS_DIR / 'confusion_matrix.png'}"
    )

    print(
        f"\n✓ Test predictions:"
        f"\n  {RESULTS_DIR / 'test_predictions.csv'}"
    )

    print("\n" + "=" * 70)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()