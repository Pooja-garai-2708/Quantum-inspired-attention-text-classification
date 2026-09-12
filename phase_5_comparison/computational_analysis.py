"""
PHASE 5 — COMPUTATIONAL ANALYSIS

Compares computational characteristics of:

    1. Standard Additive Attention
    2. Quantum-Inspired Probability Attention

Measures:
    - Parameter count
    - Model size
    - Inference time
    - Average inference time/sample
    - Throughput
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

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
# DATA
# ---------------------------------------------------------------------

def load_test_data():

    test_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "test.npz"
    )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{test_path}"
        )

    data = np.load(test_path)

    x_key = None
    y_key = None

    for key in ["X_test", "x_test", "X", "x", "features"]:
        if key in data:
            x_key = key
            break

    for key in ["y_test", "Y_test", "y", "Y", "labels"]:
        if key in data:
            y_key = key
            break

    if x_key is None or y_key is None:
        raise KeyError(
            f"Could not identify test arrays. "
            f"Available keys: {list(data.keys())}"
        )

    return (
        np.asarray(data[x_key]),
        np.asarray(data[y_key]).astype(int)
    )


# ---------------------------------------------------------------------
# MODEL BENCHMARK
# ---------------------------------------------------------------------

def benchmark_model(model, X_test, model_name):

    print("\n" + "=" * 70)
    print(f"COMPUTATIONAL ANALYSIS — {model_name}")
    print("=" * 70)

    # Build model.
    model(
        tf.zeros(
            (1, config.MAX_SEQUENCE_LENGTH),
            dtype=tf.int32
        ),
        training=False
    )

    parameter_count = model.count_params()

    # Approximate float32 model size.
    model_size_mb = (
        parameter_count * 4
    ) / (
        1024 * 1024
    )

    # Warm-up.
    warmup = tf.convert_to_tensor(
        X_test[:min(32, len(X_test))],
        dtype=tf.int32
    )

    model(
        warmup,
        training=False
    )

    # Benchmark several batches.
    num_samples = min(
        2048,
        len(X_test)
    )

    benchmark_data = tf.convert_to_tensor(
        X_test[:num_samples],
        dtype=tf.int32
    )

    # Multiple repetitions reduce timing noise.
    repetitions = 3

    times = []

    for _ in range(repetitions):

        start = time.perf_counter()

        model(
            benchmark_data,
            training=False
        )

        # Ensure TensorFlow execution is complete.
        tf.constant(0).numpy()

        end = time.perf_counter()

        times.append(
            end - start
        )

    average_time = float(
        np.mean(times)
    )

    std_time = float(
        np.std(times)
    )

    time_per_sample_ms = (
        average_time
        / num_samples
        * 1000
    )

    throughput = (
        num_samples
        / average_time
    )

    results = {
        "Model": model_name,
        "Parameters": parameter_count,
        "Model_Size_MB": model_size_mb,
        "Benchmark_Samples": num_samples,
        "Average_Inference_Time_Seconds":
            average_time,
        "Std_Inference_Time_Seconds":
            std_time,
        "Time_Per_Sample_ms":
            time_per_sample_ms,
        "Throughput_Samples_Per_Second":
            throughput
    }

    print(
        f"Parameters: {parameter_count:,}"
    )

    print(
        f"Approx. model size: "
        f"{model_size_mb:.2f} MB"
    )

    print(
        f"Average inference time: "
        f"{average_time:.4f} seconds"
    )

    print(
        f"Time/sample: "
        f"{time_per_sample_ms:.4f} ms"
    )

    print(
        f"Throughput: "
        f"{throughput:.2f} samples/sec"
    )

    return results


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("\n")
    print("=" * 70)
    print("PHASE 5 — COMPUTATIONAL ANALYSIS")
    print("=" * 70)

    X_test, y_test = load_test_data()

    print(
        f"\nTest samples available: {len(X_test):,}"
    )

    # -------------------------------------------------------------
    # STANDARD
    # -------------------------------------------------------------

    standard_model = BaselineAttentionModel()

    standard_model(
        tf.zeros(
            (1, config.MAX_SEQUENCE_LENGTH),
            dtype=tf.int32
        )
    )

    standard_model.load_weights(
        str(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / "baseline_attention_best.weights.h5"
        )
    )

    standard_results = benchmark_model(
        standard_model,
        X_test,
        "Standard Attention"
    )

    # -------------------------------------------------------------
    # QUANTUM
    # -------------------------------------------------------------

    quantum_model = QuantumInspiredAttentionModel()

    quantum_model(
        tf.zeros(
            (1, config.MAX_SEQUENCE_LENGTH),
            dtype=tf.int32
        )
    )

    quantum_model.load_weights(
        str(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / "quantum_inspired_attention_best.weights.h5"
        )
    )

    quantum_results = benchmark_model(
        quantum_model,
        X_test,
        "Quantum-Inspired Attention"
    )

    # -------------------------------------------------------------
    # COMPARISON
    # -------------------------------------------------------------

    results = pd.DataFrame([
        standard_results,
        quantum_results
    ])

    output_path = (
        RESULTS_DIR
        / "computational_analysis.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("COMPUTATIONAL COMPARISON")
    print("=" * 70)

    print(
        results.to_string(index=False)
    )

    # Parameter difference
    p1 = standard_results["Parameters"]
    p2 = quantum_results["Parameters"]

    print("\nParameter difference:")
    print(f"{abs(p1 - p2):,}")

    if p1 == p2:
        print(
            "Both models have identical parameter counts."
        )

    print("\nSaved:")
    print(output_path)

    print("\n" + "=" * 70)
    print("COMPUTATIONAL ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()