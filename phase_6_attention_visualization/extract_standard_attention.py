"""
PHASE 6 — STANDARD ATTENTION EXTRACTION

Extracts attention weights from the trained Standard Attention model
and maps the weights back to input words/tokens.
"""

import sys
import json
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


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

RESULTS_DIR = (
    PROJECT_ROOT
    / "phase_6_attention_visualization"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.npz"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "baseline_attention.weights.h5"
)


# ---------------------------------------------------------------------
# TOKENIZER LOADING
# ---------------------------------------------------------------------

def load_tokenizer():
    """
    Locate and load the tokenizer JSON from data/processed.
    """

    possible_paths = []

    # Check config if tokenizer path exists
    if hasattr(config, "TOKENIZER_PATH"):
        possible_paths.append(
            Path(config.TOKENIZER_PATH)
        )

    # Search processed directory
    processed_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    possible_paths.extend(
        sorted(
            processed_dir.glob("*tokenizer*.json")
        )
    )

    for path in possible_paths:

        if path.exists():

            print(
                f"Tokenizer found: {path}"
            )

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                tokenizer_json = json.load(f)

            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
                json.dumps(tokenizer_json)
            )

            return tokenizer

    raise FileNotFoundError(
        "\nTokenizer JSON not found.\n"
        "Expected a file containing 'tokenizer' "
        "in its filename under:\n"
        f"{processed_dir}"
    )


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
def load_test_data():

    if not TEST_DATA_PATH.exists():

        raise FileNotFoundError(
            f"\nTest dataset not found:\n"
            f"{TEST_DATA_PATH}"
        )

    data = np.load(
        TEST_DATA_PATH
    )

    print(
        f"Test data loaded: {TEST_DATA_PATH}"
    )

    print(
        "Available arrays:",
        data.files
    )

    # test.npz uses X and y
    X_test = data["X"]
    y_test = data["y"]

    return X_test, y_test
# ---------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------

def load_model():

    print("\nLoading Standard Attention model...")

    model = BaselineAttentionModel()

    # Build model
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

    # Load trained weights
    model.load_weights(
        MODEL_PATH
    )

    print(
        "Standard Attention model loaded successfully."
    )

    return model


# ---------------------------------------------------------------------
# EXTRACT ATTENTION
# ---------------------------------------------------------------------

def extract_attention(
    model,
    X_test,
    y_test,
    tokenizer,
    num_samples=10
):

    num_samples = min(
        num_samples,
        len(X_test)
    )

    records = []

    reverse_word_index = {
        index: word
        for word, index
        in tokenizer.word_index.items()
    }

    print(
        f"\nExtracting attention for "
        f"{num_samples} test samples..."
    )

    for sample_index in range(num_samples):

        sequence = X_test[
            sample_index:
            sample_index + 1
        ]

        label = int(
            y_test[sample_index]
        )

        # -------------------------------------------------------------
        # Encoder
        # -------------------------------------------------------------

        sequence_output = model.encoder(
            sequence,
            training=False
        )

        # -------------------------------------------------------------
        # Padding mask
        # -------------------------------------------------------------

        mask = tf.not_equal(
            sequence,
            0
        )

        # -------------------------------------------------------------
        # Standard Attention
        # -------------------------------------------------------------

        context_vector, attention_weights = (
            model.attention(
                sequence_output,
                mask=mask,
                training=False
            )
        )

        # -------------------------------------------------------------
        # Prediction
        # -------------------------------------------------------------

        prediction = model.classifier(
            context_vector,
            training=False
        )

        predicted_class = int(
            tf.argmax(
                prediction,
                axis=1
            ).numpy()[0]
        )

        attention_weights = (
            attention_weights
            .numpy()[0]
        )

        token_ids = sequence[0]

        # -------------------------------------------------------------
        # Convert token IDs → words
        # -------------------------------------------------------------

        for position, token_id in enumerate(
            token_ids
        ):

            token_id = int(token_id)

            # Skip padding
            if token_id == 0:
                continue

            token = reverse_word_index.get(
                token_id,
                "<UNK>"
            )

            attention_value = float(
                attention_weights[position]
            )

            records.append(
                {
                    "sample_id":
                        sample_index,

                    "position":
                        position,

                    "token_id":
                        token_id,

                    "token":
                        token,

                    "actual_label":
                        label,

                    "predicted_label":
                        predicted_class,

                    "attention_weight":
                        attention_value
                }
            )

    return pd.DataFrame(
        records
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("\n")
    print("=" * 70)
    print("PHASE 6 — STANDARD ATTENTION EXTRACTION")
    print("=" * 70)

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Load test data
    X_test, y_test = load_test_data()

    print(
        f"X_test shape: {X_test.shape}"
    )

    print(
        f"y_test shape: {y_test.shape}"
    )

    # Load model
    model = load_model()

    # Extract
    results = extract_attention(
        model,
        X_test,
        y_test,
        tokenizer,
        num_samples=10
    )

    # Save
    output_path = (
        RESULTS_DIR
        / "standard_attention_samples.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nStandard Attention results saved to:"
    )

    print(output_path)

    print(
        f"Total token records: {len(results):,}"
    )

    print("\n" + "=" * 70)
    print(
        "STANDARD ATTENTION EXTRACTION COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()