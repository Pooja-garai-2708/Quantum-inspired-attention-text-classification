"""
PHASE 5 — MODEL COMPARISON

Compares:
    Phase 3: Standard Additive Attention
    Phase 4: Quantum-Inspired Probability Attention

Evaluation:
    Accuracy
    Precision
    Recall
    F1-score
    Confusion Matrix
    Parameter Count
    Best Validation Accuracy
    Training Accuracy
    Training Loss
    Validation Loss

All evaluation is performed on the same IMDB test dataset.
"""

import os
import sys
import time
from pathlib import Path

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

# ---------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

from phase_3_baseline_model.model import BaselineAttentionModel
from phase_4_quantum_inspired_attention.model import (
    QuantumInspiredAttentionModel
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "phase_5_comparison" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "baseline_attention_best.weights.h5"
)

QUANTUM_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "quantum_inspired_attention_best.weights.h5"
)

# ---------------------------------------------------------------------
# RANDOM SEED
# ---------------------------------------------------------------------

np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def find_test_npz():
    """
    Locate the processed test.npz file.
    """

    candidates = [
        PROJECT_ROOT / "data" / "processed" / "test.npz",
        PROJECT_ROOT / "data" / "processed" / "test_data.npz",
        PROJECT_ROOT / "data" / "processed" / "test_dataset.npz",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find test.npz.\n"
        "Expected location:\n"
        f"{PROJECT_ROOT / 'data' / 'processed' / 'test.npz'}"
    )


def load_test_data():
    """
    Load X_test and y_test from processed NPZ file.
    """

    test_path = find_test_npz()

    print("\nLoading test dataset...")
    print(f"Test file: {test_path}")

    data = np.load(test_path)

    print("Available keys:", list(data.keys()))

    # Common naming possibilities
    x_candidates = [
        "X_test",
        "x_test",
        "X",
        "x",
        "features"
    ]

    y_candidates = [
        "y_test",
        "Y_test",
        "y",
        "Y",
        "labels"
    ]

    X_test = None
    y_test = None

    for key in x_candidates:
        if key in data:
            X_test = data[key]
            break

    for key in y_candidates:
        if key in data:
            y_test = data[key]
            break

    if X_test is None or y_test is None:
        raise KeyError(
            "Could not identify X_test/y_test in test.npz. "
            f"Available keys: {list(data.keys())}"
        )

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).astype(int)

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_test, y_test


# ---------------------------------------------------------------------
# MODEL BUILDING
# ---------------------------------------------------------------------

def build_and_load_model(model_class, checkpoint_path):
    """
    Build a model and load its weights.
    """

    print("\nBuilding model...")
    print(f"Model class: {model_class.__name__}")
    print(f"Checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    model = model_class()

    # Build model before loading weights.
    dummy_input = tf.zeros(
        (1, config.MAX_SEQUENCE_LENGTH),
        dtype=tf.int32
    )

    model(dummy_input, training=False)

    model.load_weights(str(checkpoint_path))

    print("Weights loaded successfully.")

    return model


# ---------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate one model on the test dataset.
    """

    print("\n" + "=" * 70)
    print(f"EVALUATING {model_name.upper()}")
    print("=" * 70)

    # Warm-up
    warmup_size = min(8, len(X_test))

    model(
        tf.convert_to_tensor(
            X_test[:warmup_size],
            dtype=tf.int32
        ),
        training=False
    )

    # Measure inference time
    start_time = time.perf_counter()

    probabilities = model.predict(
        X_test,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )

    end_time = time.perf_counter()

    inference_time = end_time - start_time

    predictions = np.argmax(
        probabilities,
        axis=1
    )

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

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nResults:")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nInference time: {inference_time:.4f} seconds")
    print(
        f"Average/sample: "
        f"{(inference_time / len(X_test)) * 1000:.4f} ms"
    )

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "predictions": predictions,
        "probabilities": probabilities,
        "inference_time_seconds": inference_time,
        "inference_time_per_sample_ms":
            (inference_time / len(X_test)) * 1000,
        "parameters": model.count_params()
    }


# ---------------------------------------------------------------------
# TRAINING HISTORY
# ---------------------------------------------------------------------

def load_training_history(model_type):
    """
    Load training history CSV generated during Phase 3 or Phase 4.
    """

    if model_type == "standard":
        path = (
            PROJECT_ROOT
            / "phase_3_baseline_model"
            / "results"
            / "training_history.csv"
        )
    else:
        path = (
            PROJECT_ROOT
            / "phase_4_quantum_inspired_attention"
            / "results"
            / "training_history.csv"
        )

    if not path.exists():
        print(f"Warning: history file not found: {path}")
        return None

    return pd.read_csv(path)


def get_history_summary(model_type):
    """
    Extract best training/validation results.
    """

    history = load_training_history(model_type)

    if history is None:
        return {
            "best_epoch": np.nan,
            "training_accuracy": np.nan,
            "validation_accuracy": np.nan,
            "training_loss": np.nan,
            "validation_loss": np.nan
        }

    # Handle possible column naming differences.
    columns = {
        str(c).lower().strip(): c
        for c in history.columns
    }

    def find_column(names):
        for name in names:
            if name.lower() in columns:
                return columns[name.lower()]
        return None

    acc_col = find_column([
        "accuracy",
        "train_accuracy"
    ])

    val_acc_col = find_column([
        "val_accuracy"
    ])

    loss_col = find_column([
        "loss",
        "train_loss"
    ])

    val_loss_col = find_column([
        "val_loss"
    ])

    if val_acc_col is not None:
        best_index = history[val_acc_col].idxmax()
    else:
        best_index = history.index[-1]

    row = history.loc[best_index]

    return {
        "best_epoch": (
            int(row["epoch"])
            if "epoch" in history.columns
            else int(best_index + 1)
        ),
        "training_accuracy":
            float(row[acc_col]) if acc_col else np.nan,
        "validation_accuracy":
            float(row[val_acc_col]) if val_acc_col else np.nan,
        "training_loss":
            float(row[loss_col]) if loss_col else np.nan,
        "validation_loss":
            float(row[val_loss_col]) if val_loss_col else np.nan
    }


# ---------------------------------------------------------------------
# SAVE CONFUSION MATRIX
# ---------------------------------------------------------------------

def save_confusion_matrix(cm, filename):
    """
    Save confusion matrix as CSV.
    """

    df = pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"]
    )

    df.to_csv(
        RESULTS_DIR / filename
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("\n")
    print("=" * 70)
    print("PHASE 5 — MODEL COMPARISON")
    print("=" * 70)

    print("\nExperiment:")
    print("Phase 3 → Standard Additive Attention")
    print("Phase 4 → Quantum-Inspired Probability Attention")

    print("\nExperimental control:")
    print("Same dataset")
    print("Same vocabulary")
    print("Same sequence length")
    print("Same embedding")
    print("Same BiLSTM")
    print("Same classifier")
    print("Same optimizer")
    print("Same learning rate")
    print("Same batch size")
    print("Same random seed")
    print("Only attention mechanism changes.")

    # -------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------

    X_test, y_test = load_test_data()

    # -------------------------------------------------------------
    # STANDARD MODEL
    # -------------------------------------------------------------

    baseline_model = build_and_load_model(
        BaselineAttentionModel,
        BASELINE_CHECKPOINT
    )

    baseline_results = evaluate_model(
        baseline_model,
        X_test,
        y_test,
        "Standard Attention"
    )

    # -------------------------------------------------------------
    # QUANTUM MODEL
    # -------------------------------------------------------------

    quantum_model = build_and_load_model(
        QuantumInspiredAttentionModel,
        QUANTUM_CHECKPOINT
    )

    quantum_results = evaluate_model(
        quantum_model,
        X_test,
        y_test,
        "Quantum-Inspired Attention"
    )

    # -------------------------------------------------------------
    # TRAINING HISTORY
    # -------------------------------------------------------------

    baseline_history = get_history_summary("standard")
    quantum_history = get_history_summary("quantum")

    # -------------------------------------------------------------
    # COMPARISON TABLE
    # -------------------------------------------------------------

    comparison = pd.DataFrame([
        {
            "Model": "Standard Attention",
            "Accuracy": baseline_results["accuracy"],
            "Precision": baseline_results["precision"],
            "Recall": baseline_results["recall"],
            "F1_Score": baseline_results["f1_score"],
            "Parameters": baseline_results["parameters"],
            "Best_Epoch":
                baseline_history["best_epoch"],
            "Training_Accuracy":
                baseline_history["training_accuracy"],
            "Validation_Accuracy":
                baseline_history["validation_accuracy"],
            "Training_Loss":
                baseline_history["training_loss"],
            "Validation_Loss":
                baseline_history["validation_loss"],
            "Inference_Time_Seconds":
                baseline_results["inference_time_seconds"],
            "Inference_Time_Per_Sample_ms":
                baseline_results["inference_time_per_sample_ms"]
        },
        {
            "Model": "Quantum-Inspired Attention",
            "Accuracy": quantum_results["accuracy"],
            "Precision": quantum_results["precision"],
            "Recall": quantum_results["recall"],
            "F1_Score": quantum_results["f1_score"],
            "Parameters": quantum_results["parameters"],
            "Best_Epoch":
                quantum_history["best_epoch"],
            "Training_Accuracy":
                quantum_history["training_accuracy"],
            "Validation_Accuracy":
                quantum_history["validation_accuracy"],
            "Training_Loss":
                quantum_history["training_loss"],
            "Validation_Loss":
                quantum_history["validation_loss"],
            "Inference_Time_Seconds":
                quantum_results["inference_time_seconds"],
            "Inference_Time_Per_Sample_ms":
                quantum_results["inference_time_per_sample_ms"]
        }
    ])

    # -------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------

    comparison_path = (
        RESULTS_DIR / "model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    save_confusion_matrix(
        baseline_results["confusion_matrix"],
        "confusion_matrix_standard.csv"
    )

    save_confusion_matrix(
        quantum_results["confusion_matrix"],
        "confusion_matrix_quantum.csv"
    )

    # Save predictions for statistical analysis.
    predictions_df = pd.DataFrame({
        "actual": y_test,
        "standard_prediction":
            baseline_results["predictions"],
        "quantum_prediction":
            quantum_results["predictions"]
    })

    predictions_df.to_csv(
        RESULTS_DIR / "model_predictions.csv",
        index=False
    )

    # -------------------------------------------------------------
    # PRINT FINAL COMPARISON
    # -------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score",
                "Parameters"
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------
    # DETERMINE BEST MODEL
    # -------------------------------------------------------------

    if (
        quantum_results["accuracy"]
        >
        baseline_results["accuracy"]
    ):
        best_model = "Quantum-Inspired Attention"
    elif (
        quantum_results["accuracy"]
        <
        baseline_results["accuracy"]
    ):
        best_model = "Standard Attention"
    else:
        best_model = "Tie"

    print("\nBest test-accuracy model:")
    print(best_model)

    print("\nResults saved to:")
    print(RESULTS_DIR)

    print("\n" + "=" * 70)
    print("PHASE 5 MODEL COMPARISON COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()