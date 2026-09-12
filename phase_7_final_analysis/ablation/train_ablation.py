"""
PHASE 7 — ABLATION STUDY
Mean Pooling Baseline

Architecture:
Embedding → BiLSTM → Mean Pooling → Classifier

Purpose:
Determine whether attention mechanisms provide an advantage
over simple mean pooling of BiLSTM hidden states.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import config

from phase_3_baseline_model.bilstm import BiLSTMEncoder
from phase_3_baseline_model.classifier import SentimentClassifier


# ============================================================
# PATHS
# ============================================================

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


TRAIN_DATA_PATH = Path(
    config.TRAIN_DATA_PATH
)

VALIDATION_DATA_PATH = Path(
    config.VALIDATION_DATA_PATH
)

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "mean_pooling_ablation.weights.h5"
)

CHECKPOINT_PATH = (
    RESULTS_DIR
    / "mean_pooling_best.weights.h5"
)

HISTORY_PATH = (
    RESULTS_DIR
    / "training_history.csv"
)

MODEL_INFO_PATH = (
    RESULTS_DIR
    / "mean_pooling_model_information.csv"
)


# ============================================================
# RANDOM SEEDS
# ============================================================

np.random.seed(
    config.RANDOM_SEED
)

tf.random.set_seed(
    config.RANDOM_SEED
)


# ============================================================
# MEAN POOLING MODEL
# ============================================================

class MeanPoolingAblationModel(tf.keras.Model):

    def __init__(self, **kwargs):

        super().__init__(
            name="mean_pooling_ablation_model",
            **kwargs
        )

        self.encoder = BiLSTMEncoder()

        self.classifier = SentimentClassifier()


    def call(
        self,
        inputs,
        training=False
    ):

        sequence_output = self.encoder(
            inputs,
            training=training
        )

        # ----------------------------------------------------
        # Padding mask
        # ----------------------------------------------------

        mask = tf.not_equal(
            inputs,
            0
        )

        mask = tf.cast(
            mask,
            sequence_output.dtype
        )

        mask = tf.expand_dims(
            mask,
            axis=-1
        )

        # ----------------------------------------------------
        # Masked mean pooling
        # ----------------------------------------------------

        masked_output = (
            sequence_output * mask
        )

        summed_output = tf.reduce_sum(
            masked_output,
            axis=1
        )

        token_count = tf.reduce_sum(
            mask,
            axis=1
        )

        token_count = tf.maximum(
            token_count,
            tf.constant(
                1.0,
                dtype=sequence_output.dtype
            )
        )

        context_vector = (
            summed_output / token_count
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        predictions = self.classifier(
            context_vector,
            training=training
        )

        return predictions


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print(
        "\nLoading training and validation data..."
    )

    if not TRAIN_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Training data not found:\n"
            f"{TRAIN_DATA_PATH}"
        )

    if not VALIDATION_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Validation data not found:\n"
            f"{VALIDATION_DATA_PATH}"
        )


    train_data = np.load(
        TRAIN_DATA_PATH
    )

    validation_data = np.load(
        VALIDATION_DATA_PATH
    )


    print(
        "Training arrays:",
        train_data.files
    )

    print(
        "Validation arrays:",
        validation_data.files
    )


    # The project stores arrays as X and y.
    X_train = train_data["X"]
    y_train = train_data["y"]

    X_validation = validation_data["X"]
    y_validation = validation_data["y"]


    print(
        f"\nX_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"y_train shape: "
        f"{y_train.shape}"
    )

    print(
        f"X_validation shape: "
        f"{X_validation.shape}"
    )

    print(
        f"y_validation shape: "
        f"{y_validation.shape}"
    )


    return (
        X_train,
        y_train,
        X_validation,
        y_validation
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    print(
        "\nBuilding Mean Pooling ablation model..."
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


    print(
        "Mean Pooling model built successfully."
    )


    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation
):

    print(
        "\nCompiling Mean Pooling model..."
    )


    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE
    )


    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy"
        ]
    )


    print(
        "\nStarting Mean Pooling ablation training..."
    )


    print(
        f"Training samples: {len(X_train):,}"
    )

    print(
        f"Validation samples: "
        f"{len(X_validation):,}"
    )

    print(
        f"Batch size: {config.BATCH_SIZE}"
    )

    print(
        f"Epochs: {config.EPOCHS}"
    )

    print(
        f"Learning rate: "
        f"{config.LEARNING_RATE}"
    )


    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_PATH),
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
            verbose=1
        )
    ]


    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_validation,
            y_validation
        ),

        epochs=config.EPOCHS,

        batch_size=config.BATCH_SIZE,

        callbacks=callbacks,

        verbose=1
    )


    history_df = pd.DataFrame(
        history.history
    )


    history_df.insert(
        0,
        "epoch",
        range(
            1,
            len(history_df) + 1
        )
    )


    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )


    print(
        f"\nTraining history saved to:\n"
        f"{HISTORY_PATH}"
    )


    # Save final restored weights
    model.save_weights(
        MODEL_PATH
    )


    print(
        f"Final Mean Pooling weights saved to:\n"
        f"{MODEL_PATH}"
    )


    return history


# ============================================================
# MODEL INFORMATION
# ============================================================

def save_model_information(model):

    total_params = (
        model.count_params()
    )


    information = pd.DataFrame(
        {
            "Property": [
                "Model",
                "Architecture",
                "Vocabulary Size",
                "Maximum Sequence Length",
                "Embedding Dimension",
                "BiLSTM Units",
                "Classifier Dense Units",
                "Batch Size",
                "Epochs",
                "Learning Rate",
                "Total Parameters"
            ],

            "Value": [
                "Mean Pooling Ablation",
                "Embedding → BiLSTM → Mean Pooling → Classifier",
                config.VOCAB_SIZE,
                config.MAX_SEQUENCE_LENGTH,
                config.EMBEDDING_DIM,
                config.LSTM_UNITS,
                config.DENSE_UNITS,
                config.BATCH_SIZE,
                config.EPOCHS,
                config.LEARNING_RATE,
                total_params
            ]
        }
    )


    information.to_csv(
        MODEL_INFO_PATH,
        index=False
    )


    print(
        f"\nModel information saved to:\n"
        f"{MODEL_INFO_PATH}"
    )


    print(
        f"\nTotal parameters: "
        f"{total_params:,}"
    )


# ============================================================
# TRAINING CURVES
# ============================================================

def create_training_plots(history):

    history_df = pd.DataFrame(
        history.history
    )


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt = __import__(
        "matplotlib.pyplot",
        fromlist=["plt"]
    )


    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        history_df["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history_df["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Mean Pooling Ablation Accuracy"
    )

    plt.legend()

    plt.tight_layout()


    accuracy_path = (
        RESULTS_DIR
        / "mean_pooling_accuracy.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        history_df["loss"],
        label="Training Loss"
    )

    plt.plot(
        history_df["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Mean Pooling Ablation Loss"
    )

    plt.legend()

    plt.tight_layout()


    loss_path = (
        RESULTS_DIR
        / "mean_pooling_loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=300
    )

    plt.close()


    print(
        f"\nSaved: {accuracy_path}"
    )

    print(
        f"Saved: {loss_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("PHASE 7 — ABLATION STUDY")
    print("MEAN POOLING BASELINE")
    print("=" * 70)


    (
        X_train,
        y_train,
        X_validation,
        y_validation
    ) = load_data()


    model = build_model()


    print(
        "\nMean Pooling model parameters:",
        f"{model.count_params():,}"
    )


    history = train_model(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation
    )


    save_model_information(
        model
    )


    create_training_plots(
        history
    )


    print("\n")
    print("=" * 70)
    print("MEAN POOLING ABLATION TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()