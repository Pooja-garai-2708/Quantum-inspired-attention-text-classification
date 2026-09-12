"""
Phase 3 - Baseline Model Training
=================================

Model:
    Input
      ↓
    Embedding
      ↓
    BiLSTM
      ↓
    Standard Additive Attention
      ↓
    Dense Classifier
      ↓
    Softmax
      ↓
    Negative / Positive

Run from the project root:

    python phase_3_baseline_model/train.py
"""

# ============================================================
# 1. IMPORTS AND PROJECT PATH
# ============================================================

import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# Import project configuration
# ------------------------------------------------------------

import config

from model import BaselineAttentionModel


# ============================================================
# 2. PATHS
# ============================================================

# Phase 3 results directory
RESULTS_DIR = (
    PROJECT_ROOT
    / "phase_3_baseline_model"
    / "results"
)


# Best model checkpoint
CHECKPOINT_PATH = (
    config.CHECKPOINT_DIR
    / "baseline_attention_best.weights.h5"
)


# Final model weights
FINAL_WEIGHTS_PATH = (
    config.MODEL_DIR
    / "baseline_attention.weights.h5"
)


# ============================================================
# 3. CREATE REQUIRED DIRECTORIES
# ============================================================

config.create_project_directories()

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_WEIGHTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. RANDOM SEED
# ============================================================

def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducible experiments.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    tf.keras.utils.set_random_seed(seed)


# ============================================================
# 5. LOAD DATASET
# ============================================================

def load_dataset(
    path: Path,
    dataset_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a processed .npz dataset.

    Expected arrays:
        X -> token sequences
        y -> labels
    """

    if not path.exists():

        raise FileNotFoundError(
            f"\n{dataset_name} dataset was not found.\n\n"
            f"Expected location:\n{path}\n\n"
            "Please run the Phase 2 preprocessing scripts first."
        )

    print(
        f"\nLoading {dataset_name} dataset..."
    )

    with np.load(path) as data:

        if "X" not in data:

            raise KeyError(
                f"{path} does not contain an 'X' array."
            )

        if "y" not in data:

            raise KeyError(
                f"{path} does not contain a 'y' array."
            )

        X = data["X"]

        y = data["y"]

    return X, y


# ============================================================
# 6. DISPLAY DATASET INFORMATION
# ============================================================

def print_dataset_information(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray
) -> None:

    print("\n")
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(
        f"Training samples       : {len(x_train):,}"
    )

    print(
        f"Validation samples     : {len(x_validation):,}"
    )

    print(
        f"Training sequence size : {x_train.shape[1]}"
    )

    print(
        f"Vocabulary size        : {config.VOCAB_SIZE:,}"
    )

    print(
        f"Embedding dimension    : {config.EMBEDDING_DIM}"
    )

    print(
        f"BiLSTM units           : {config.LSTM_UNITS}"
    )

    print(
        f"Attention dimension    : "
        f"{config.STANDARD_ATTENTION_DIM}"
    )

    print(
        f"Batch size             : {config.BATCH_SIZE}"
    )

    print(
        f"Maximum epochs         : {config.EPOCHS}"
    )

    print(
        f"Learning rate          : {config.LEARNING_RATE}"
    )

    print(
        f"Random seed            : {config.RANDOM_SEED}"
    )

    print("=" * 70)


# ============================================================
# 7. SAVE TRAINING PLOTS
# ============================================================

def save_training_plots(
    history: pd.DataFrame
) -> None:
    """
    Save training/validation accuracy and loss plots.
    """

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy_plot = RESULTS_DIR / "baseline_accuracy.png"

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Baseline Standard Attention — Accuracy"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        accuracy_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_plot = RESULTS_DIR / "baseline_loss.png"

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Baseline Standard Attention — Loss"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        loss_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"\nAccuracy plot saved to:\n{accuracy_plot}"
    )

    print(
        f"Loss plot saved to:\n{loss_plot}"
    )


# ============================================================
# 8. SAVE MODEL INFORMATION
# ============================================================

def save_model_information(
    model: tf.keras.Model,
    history_df: pd.DataFrame
) -> None:
    """
    Save baseline model configuration and
    best validation performance.
    """

    best_epoch_index = int(
        history_df["val_accuracy"].idxmax()
    )

    best_epoch = (
        best_epoch_index + 1
    )

    best_validation_accuracy = (
        history_df.loc[
            best_epoch_index,
            "val_accuracy"
        ]
    )

    training_accuracy = (
        history_df.loc[
            best_epoch_index,
            "accuracy"
        ]
    )

    information = {

        "model":
            "Standard Attention Baseline",

        "dataset":
            config.DATASET_NAME,

        "vocab_size":
            config.VOCAB_SIZE,

        "max_sequence_length":
            config.MAX_SEQUENCE_LENGTH,

        "embedding_dimension":
            config.EMBEDDING_DIM,

        "lstm_units":
            config.LSTM_UNITS,

        "attention_dimension":
            config.STANDARD_ATTENTION_DIM,

        "dense_units":
            config.DENSE_UNITS,

        "dropout_rate":
            config.DROPOUT_RATE,

        "recurrent_dropout_rate":
            config.RECURRENT_DROPOUT_RATE,

        "batch_size":
            config.BATCH_SIZE,

        "maximum_epochs":
            config.EPOCHS,

        "learning_rate":
            config.LEARNING_RATE,

        "random_seed":
            config.RANDOM_SEED,

        "total_parameters":
            model.count_params(),

        "best_epoch":
            best_epoch,

        "best_validation_accuracy":
            float(best_validation_accuracy),

        "training_accuracy_at_best_epoch":
            float(training_accuracy)
    }

    model_info_df = pd.DataFrame(
        [information]
    )

    output_path = (
        RESULTS_DIR
        / "baseline_model_information.csv"
    )

    model_info_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nModel information saved to:\n{output_path}"
    )


# ============================================================
# 9. MAIN TRAINING FUNCTION
# ============================================================

def main() -> None:

    print("\n")

    print("=" * 70)
    print(
        "PHASE 3 — STANDARD ATTENTION BASELINE TRAINING"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Set random seed
    # --------------------------------------------------------

    set_random_seed(
        config.RANDOM_SEED
    )


    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    x_train, y_train = load_dataset(
        config.TRAIN_DATA_PATH,
        "Training"
    )


    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    x_validation, y_validation = load_dataset(
        config.VALIDATION_DATA_PATH,
        "Validation"
    )


    # --------------------------------------------------------
    # Display dataset information
    # --------------------------------------------------------

    print_dataset_information(
        x_train,
        y_train,
        x_validation,
        y_validation
    )


    # ========================================================
    # 10. CREATE BASELINE MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CREATING BASELINE MODEL")
    print("=" * 70)


    model = BaselineAttentionModel()


    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    dummy_input = tf.zeros(
        (
            1,
            config.MAX_SEQUENCE_LENGTH
        ),
        dtype=tf.int32
    )

    model(
        dummy_input,
        training=False
    )


    # ========================================================
    # 11. COMPILE MODEL
    # ========================================================

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )


    # ========================================================
    # 12. MODEL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)

    model.summary()


    total_parameters = (
        model.count_params()
    )

    print(
        f"\nTotal parameters: "
        f"{total_parameters:,}"
    )


    # ========================================================
    # 13. CALLBACKS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CONFIGURING TRAINING CALLBACKS")
    print("=" * 70)


    # --------------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------------

    checkpoint_callback = (
        tf.keras.callbacks.ModelCheckpoint(

            filepath=str(
                CHECKPOINT_PATH
            ),

            monitor="val_accuracy",

            mode="max",

            save_best_only=True,

            save_weights_only=True,

            verbose=1
        )
    )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping_callback = (
        tf.keras.callbacks.EarlyStopping(

            monitor="val_accuracy",

            mode="max",

            patience=config.EARLY_STOPPING_PATIENCE,

            restore_best_weights=True,

            verbose=1
        )
    )


    # --------------------------------------------------------
    # Learning-rate reduction
    # --------------------------------------------------------

    reduce_lr_callback = (
        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=config.LR_REDUCTION_FACTOR,

            patience=config.LR_REDUCTION_PATIENCE,

            min_lr=1e-7,

            verbose=1
        )
    )


    callbacks = [

        checkpoint_callback,

        early_stopping_callback,

        reduce_lr_callback
    ]


    # ========================================================
    # 14. START TRAINING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STARTING BASELINE MODEL TRAINING")
    print("=" * 70)

    print(
        "\nAttention mechanism : Standard Additive Attention"
    )

    print(
        "Optimizer           : Adam"
    )

    print(
        f"Learning rate       : {config.LEARNING_RATE}"
    )

    print(
        f"Batch size          : {config.BATCH_SIZE}"
    )

    print(
        f"Maximum epochs      : {config.EPOCHS}"
    )

    print(
        f"Early stopping      : "
        f"{config.EARLY_STOPPING_PATIENCE} epochs"
    )

    print("\n")


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


    # ========================================================
    # 15. SAVE FINAL MODEL WEIGHTS
    # ========================================================

    model.save_weights(
        str(FINAL_WEIGHTS_PATH)
    )

    print(
        "\nFinal model weights saved to:"
    )

    print(
        FINAL_WEIGHTS_PATH
    )


    # ========================================================
    # 16. SAVE TRAINING HISTORY
    # ========================================================

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

    print(
        f"\nTraining history saved to:\n"
        f"{history_path}"
    )


    # ========================================================
    # 17. BEST EPOCH
    # ========================================================

    best_epoch_index = int(
        history_df["val_accuracy"].idxmax()
    )

    best_epoch = (
        best_epoch_index + 1
    )

    best_validation_accuracy = (
        history_df.loc[
            best_epoch_index,
            "val_accuracy"
        ]
    )

    best_training_accuracy = (
        history_df.loc[
            best_epoch_index,
            "accuracy"
        ]
    )

    best_validation_loss = (
        history_df.loc[
            best_epoch_index,
            "val_loss"
        ]
    )

    best_training_loss = (
        history_df.loc[
            best_epoch_index,
            "loss"
        ]
    )


    # ========================================================
    # 18. DISPLAY BEST RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BEST TRAINING RESULTS")
    print("=" * 70)

    print(
        f"Best epoch               : {best_epoch}"
    )

    print(
        f"Training accuracy        : "
        f"{best_training_accuracy:.4f}"
    )

    print(
        f"Validation accuracy      : "
        f"{best_validation_accuracy:.4f}"
    )

    print(
        f"Training loss            : "
        f"{best_training_loss:.4f}"
    )

    print(
        f"Validation loss          : "
        f"{best_validation_loss:.4f}"
    )


    # ========================================================
    # 19. SAVE TRAINING PLOTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("GENERATING TRAINING PLOTS")
    print("=" * 70)

    save_training_plots(
        history_df
    )


    # ========================================================
    # 20. SAVE MODEL INFORMATION
    # ========================================================

    save_model_information(
        model,
        history_df
    )


    # ========================================================
    # 21. FINAL MESSAGE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 3 TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated files:")
    print()

    print(
        f"✓ Best checkpoint:"
    )

    print(
        f"  {CHECKPOINT_PATH}"
    )

    print()

    print(
        f"✓ Final model weights:"
    )

    print(
        f"  {FINAL_WEIGHTS_PATH}"
    )

    print()

    print(
        f"✓ Training history:"
    )

    print(
        f"  {history_path}"
    )

    print()

    print(
        "✓ Accuracy plot:"
    )

    print(
        f"  {RESULTS_DIR / 'baseline_accuracy.png'}"
    )

    print()

    print(
        "✓ Loss plot:"
    )

    print(
        f"  {RESULTS_DIR / 'baseline_loss.png'}"
    )

    print()

    print(
        "✓ Model information:"
    )

    print(
        f"  {RESULTS_DIR / 'baseline_model_information.csv'}"
    )

    print("\n")
    print(
        "The Standard Attention baseline is ready for evaluation."
    )

    print("=" * 70)


# ============================================================
# 22. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()