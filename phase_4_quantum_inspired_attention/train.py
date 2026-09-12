"""
Phase 4 — Train Quantum-Inspired Attention Model.

Run from project root:

    python phase_4_quantum_inspired_attention\\train.py
"""

import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


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

CHECKPOINT_DIR = (
    config.CHECKPOINT_DIR
)

MODEL_DIR = (
    config.MODEL_DIR
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "quantum_inspired_attention_best.weights.h5"
)

FINAL_WEIGHTS_PATH = (
    MODEL_DIR
    / "quantum_inspired_attention.weights.h5"
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
# LOAD DATA
# ================================================================

def load_dataset(
    path: Path,
    dataset_name: str
):

    print(
        f"\nLoading {dataset_name} dataset..."
    )

    if not path.exists():

        raise FileNotFoundError(
            f"{dataset_name} dataset not found:\n"
            f"{path}"
        )

    with np.load(path) as data:

        if "X" not in data or "y" not in data:

            raise KeyError(
                f"{path} must contain X and y arrays."
            )

        X = data["X"]
        y = data["y"]

    return X, y


# ================================================================
# TRAINING PLOTS
# ================================================================

def save_training_plots(
    history_df
):

    plots = [

        (
            "accuracy",
            "val_accuracy",
            "Accuracy",
            "quantum_attention_accuracy.png"
        ),

        (
            "loss",
            "val_loss",
            "Loss",
            "quantum_attention_loss.png"
        )
    ]

    for (
        train_column,
        validation_column,
        label,
        filename
    ) in plots:

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            history_df[train_column],
            label=f"Training {label}"
        )

        plt.plot(
            history_df[validation_column],
            label=f"Validation {label}"
        )

        plt.xlabel("Epoch")

        plt.ylabel(label)

        plt.title(
            f"Quantum-Inspired Attention — {label}"
        )

        plt.legend()

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / filename,
            dpi=300
        )

        plt.close()


# ================================================================
# MAIN
# ================================================================

def main():

    # ------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------

    config.create_project_directories()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------

    set_random_seed(
        config.RANDOM_SEED
    )

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------

    print("\n")

    print("=" * 70)

    print(
        "PHASE 4 — QUANTUM-INSPIRED "
        "ATTENTION TRAINING"
    )

    print("=" * 70)

    # ------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------

    x_train, y_train = load_dataset(
        config.TRAIN_DATA_PATH,
        "Training"
    )

    x_validation, y_validation = load_dataset(
        config.VALIDATION_DATA_PATH,
        "Validation"
    )

    # ------------------------------------------------------------
    # Dataset information
    # ------------------------------------------------------------

    print("\n")

    print("=" * 70)

    print("DATASET INFORMATION")

    print("=" * 70)

    print(
        f"Training samples       : "
        f"{len(x_train):,}"
    )

    print(
        f"Validation samples     : "
        f"{len(x_validation):,}"
    )

    print(
        f"Training sequence size : "
        f"{x_train.shape[1]}"
    )

    print(
        f"Vocabulary size        : "
        f"{config.VOCAB_SIZE:,}"
    )

    print(
        f"Embedding dimension    : "
        f"{config.EMBEDDING_DIM}"
    )

    print(
        f"BiLSTM units           : "
        f"{config.LSTM_UNITS}"
    )

    print(
        f"Quantum attention dim  : "
        f"{config.QUANTUM_ATTENTION_DIM}"
    )

    print(
        f"Batch size             : "
        f"{config.BATCH_SIZE}"
    )

    print(
        f"Maximum epochs         : "
        f"{config.EPOCHS}"
    )

    print(
        f"Learning rate          : "
        f"{config.LEARNING_RATE}"
    )

    print(
        f"Random seed            : "
        f"{config.RANDOM_SEED}"
    )

    print(
        f"Epsilon               : "
        f"{config.EPSILON}"
    )

    print("=" * 70)

    # ------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------

    print("\n")

    print("=" * 70)

    print("CREATING QUANTUM-INSPIRED MODEL")

    print("=" * 70)

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

    # ------------------------------------------------------------
    # Model summary
    # ------------------------------------------------------------

    print("\n")

    print("=" * 70)

    print("MODEL ARCHITECTURE")

    print("=" * 70)

    model.summary()

    print(
        f"\nTotal parameters: "
        f"{model.count_params():,}"
    )

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
    # Callbacks
    # ------------------------------------------------------------

    print("\n")

    print("=" * 70)

    print("CONFIGURING TRAINING CALLBACKS")

    print("=" * 70)

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            filepath=str(
                CHECKPOINT_PATH
            ),

            monitor="val_accuracy",

            mode="max",

            save_best_only=True,

            save_weights_only=True,

            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(

            monitor="val_accuracy",

            mode="max",

            patience=config.EARLY_STOPPING_PATIENCE,

            restore_best_weights=True,

            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=config.LR_REDUCTION_FACTOR,

            patience=config.LR_REDUCTION_PATIENCE,

            min_lr=1e-7,

            verbose=1
        )
    ]

    # ------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------

    print("\n")

    print("=" * 70)

    print("STARTING QUANTUM-INSPIRED ATTENTION TRAINING")

    print("=" * 70)

    print(
        "\nAttention mechanism : "
        "Quantum-Inspired Probability Attention"
    )

    print(
        "Probability formula  : "
        "|tanh(e)|² / Σ|tanh(e)|²"
    )

    print(
        "Optimizer            : Adam"
    )

    print(
        f"Learning rate        : "
        f"{config.LEARNING_RATE}"
    )

    print(
        f"Batch size           : "
        f"{config.BATCH_SIZE}"
    )

    print(
        f"Maximum epochs       : "
        f"{config.EPOCHS}"
    )

    print(
        f"Early stopping       : "
        f"{config.EARLY_STOPPING_PATIENCE} epochs"
    )

    # ------------------------------------------------------------
    # Fit model
    # ------------------------------------------------------------

    history = model.fit(

        x_train,

        y_train,

        validation_data=(
            x_validation,
            y_validation
        ),

        batch_size=config.BATCH_SIZE,

        epochs=config.EPOCHS,

        callbacks=callbacks,

        verbose=1
    )

    # ------------------------------------------------------------
    # Save final weights
    # ------------------------------------------------------------

    model.save_weights(
        str(
            FINAL_WEIGHTS_PATH
        )
    )

    # ------------------------------------------------------------
    # Save training history
    # ------------------------------------------------------------

    history_df = pd.DataFrame(
        history.history
    )

    history_path = (
        RESULTS_DIR
        / "training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    # ------------------------------------------------------------
    # Best epoch
    # ------------------------------------------------------------

    best_epoch = int(
        history_df[
            "val_accuracy"
        ].idxmax()
    )

    best_validation_accuracy = (
        history_df.loc[
            best_epoch,
            "val_accuracy"
        ]
    )

    training_accuracy = (
        history_df.loc[
            best_epoch,
            "accuracy"
        ]
    )

    training_loss = (
        history_df.loc[
            best_epoch,
            "loss"
        ]
    )

    validation_loss = (
        history_df.loc[
            best_epoch,
            "val_loss"
        ]
    )

    # ------------------------------------------------------------
    # Model information
    # ------------------------------------------------------------

    model_info = pd.DataFrame([{

        "model":
            "Quantum-Inspired Attention",

        "attention_mechanism":
            "Quantum Probability Inspired",

        "attention_formula":
            "|tanh(e)|^2 / sum(|tanh(e)|^2)",

        "total_parameters":
            model.count_params(),

        "best_epoch":
            best_epoch + 1,

        "best_validation_accuracy":
            best_validation_accuracy,

        "training_accuracy_at_best_epoch":
            training_accuracy,

        "training_loss_at_best_epoch":
            training_loss,

        "validation_loss_at_best_epoch":
            validation_loss

    }])

    model_info_path = (
        RESULTS_DIR
        / "quantum_attention_model_information.csv"
    )

    model_info.to_csv(
        model_info_path,
        index=False
    )

    # ------------------------------------------------------------
    # Training plots
    # ------------------------------------------------------------

    print("\n")

    print("=" * 70)

    print("GENERATING TRAINING PLOTS")

    print("=" * 70)

    save_training_plots(
        history_df
    )

    # ------------------------------------------------------------
    # Final results
    # ------------------------------------------------------------

    print("\n")

    print("=" * 70)

    print("BEST TRAINING RESULTS")

    print("=" * 70)

    print(
        f"Best epoch               : "
        f"{best_epoch + 1}"
    )

    print(
        f"Training accuracy        : "
        f"{training_accuracy:.4f}"
    )

    print(
        f"Validation accuracy      : "
        f"{best_validation_accuracy:.4f}"
    )

    print(
        f"Training loss            : "
        f"{training_loss:.4f}"
    )

    print(
        f"Validation loss          : "
        f"{validation_loss:.4f}"
    )

    # ------------------------------------------------------------
    # Files
    # ------------------------------------------------------------

    print("\n")

    print("=" * 70)

    print(
        "PHASE 4 TRAINING COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

    print("\nGenerated files:")

    print(
        f"\n✓ Best checkpoint:"
        f"\n  {CHECKPOINT_PATH}"
    )

    print(
        f"\n✓ Final model weights:"
        f"\n  {FINAL_WEIGHTS_PATH}"
    )

    print(
        f"\n✓ Training history:"
        f"\n  {history_path}"
    )

    print(
        f"\n✓ Accuracy plot:"
        f"\n  "
        f"{RESULTS_DIR / 'quantum_attention_accuracy.png'}"
    )

    print(
        f"\n✓ Loss plot:"
        f"\n  "
        f"{RESULTS_DIR / 'quantum_attention_loss.png'}"
    )

    print(
        f"\n✓ Model information:"
        f"\n  {model_info_path}"
    )

    print("\n")

    print(
        "The Quantum-Inspired Attention model "
        "is ready for evaluation."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()