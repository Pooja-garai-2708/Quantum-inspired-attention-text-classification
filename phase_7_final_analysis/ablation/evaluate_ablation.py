"""
PHASE 7 — ABLATION STUDY
Mean Pooling Test Evaluation
"""

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
    confusion_matrix
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import config

from phase_7_final_analysis.ablation.train_ablation import (
    MeanPoolingAblationModel
)


# ============================================================
# PATHS
# ============================================================

TEST_DATA_PATH = Path(
    config.TEST_DATA_PATH
)

MODEL_PATH = (
    PROJECT_ROOT
    / "phase_7_final_analysis"
    / "ablation"
    / "results"
    / "mean_pooling_best.weights.h5"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "phase_7_final_analysis"
    / "ablation"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    print(
        "\nLoading test data..."
    )

    data = np.load(
        TEST_DATA_PATH
    )

    print(
        "Available arrays:",
        data.files
    )

    X_test = data["X"]
    y_test = data["y"]

    print(
        f"X_test shape: {X_test.shape}"
    )

    print(
        f"y_test shape: {y_test.shape}"
    )

    return X_test, y_test


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\nLoading Mean Pooling model..."
    )

    model = MeanPoolingAblationModel()

    dummy_input = tf.zeros(
        (
            1,
            config.MAX_SEQUENCE_LENGTH
        ),
        dtype=tf.int32
    )

    _ = model(
        dummy_input,
        training=False
    )

    model.load_weights(
        MODEL_PATH
    )

    print(
        "Mean Pooling model loaded successfully."
    )

    print(
        f"Parameters: {model.count_params():,}"
    )

    return model


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print(
        "\nEvaluating Mean Pooling model..."
    )

    start_time = time.perf_counter()

    probabilities = model.predict(
        X_test,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )

    end_time = time.perf_counter()

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    inference_time = (
        end_time - start_time
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

    average_time_ms = (
        inference_time
        / len(X_test)
    ) * 1000


    print("\n" + "=" * 70)
    print("MEAN POOLING TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy:   {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision:  {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall:     {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score:   {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"Parameters: {model.count_params():,}"
    )

    print(
        f"Inference time: "
        f"{inference_time:.4f} seconds"
    )

    print(
        f"Average/sample: "
        f"{average_time_ms:.4f} ms"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "Parameters",
                "Inference Time (seconds)",
                "Average Time per Sample (ms)"
            ],

            "Mean Pooling": [
                accuracy,
                precision,
                recall,
                f1,
                model.count_params(),
                inference_time,
                average_time_ms
            ]
        }
    )

    metrics_path = (
        RESULTS_DIR
        / "mean_pooling_test_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False
    )


    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Negative",
            "Actual Positive"
        ],
        columns=[
            "Predicted Negative",
            "Predicted Positive"
        ]
    )

    cm_path = (
        RESULTS_DIR
        / "mean_pooling_confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_path
    )


    print(
        f"\nMetrics saved to:\n{metrics_path}"
    )

    print(
        f"Confusion matrix saved to:\n{cm_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("PHASE 7 — MEAN POOLING ABLATION TEST EVALUATION")
    print("=" * 70)

    X_test, y_test = load_test_data()

    model = load_model()

    evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\n")
    print("=" * 70)
    print("MEAN POOLING TEST EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()