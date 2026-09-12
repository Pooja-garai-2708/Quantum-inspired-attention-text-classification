"""
Central configuration for the Quantum-Inspired Attention project.

Project:
    Quantum-Inspired Attention Mechanism for Text Classification

Dataset:
    IMDB Movie Reviews

Models:
    1. BiLSTM + Standard Attention
    2. BiLSTM + Quantum-Inspired Attention
"""

from pathlib import Path


# ============================================================================
# 1. PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"


# ============================================================================
# 2. ARTIFACT DIRECTORIES
# ============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"

MODEL_DIR = ARTIFACTS_DIR / "models"

RESULTS_DIR = ARTIFACTS_DIR / "results"

FIGURES_DIR = ARTIFACTS_DIR / "figures"


# ============================================================================
# 3. DATA FILE PATHS
# ============================================================================

# Raw datasets
RAW_TRAIN_PATH = RAW_DATA_DIR / "imdb_train_raw.csv"

RAW_TEST_PATH = RAW_DATA_DIR / "imdb_test_raw.csv"


# Processed text datasets
TRAIN_TEXT_PATH = PROCESSED_DATA_DIR / "train.csv"

VALIDATION_TEXT_PATH = PROCESSED_DATA_DIR / "validation.csv"

TEST_TEXT_PATH = PROCESSED_DATA_DIR / "test.csv"


# Vocabulary
VOCAB_PATH = PROCESSED_DATA_DIR / "vocabulary.json"


# Encoded and padded datasets
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.npz"

VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "validation.npz"

TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.npz"


# ============================================================================
# 4. MODEL PATHS
# ============================================================================

# Standard attention model
BASELINE_MODEL_PATH = (
    MODEL_DIR / "baseline_attention.keras"
)


# Quantum-inspired attention model
QUANTUM_MODEL_PATH = (
    MODEL_DIR / "quantum_inspired_attention.keras"
)


# ============================================================================
# 5. CHECKPOINT PATHS
# ============================================================================

BASELINE_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "baseline_attention_best.keras"
)

QUANTUM_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "quantum_inspired_attention_best.keras"
)


# ============================================================================
# 6. RESULT PATHS
# ============================================================================

BASELINE_HISTORY_PATH = (
    RESULTS_DIR / "baseline_training_history.csv"
)

QUANTUM_HISTORY_PATH = (
    RESULTS_DIR / "quantum_training_history.csv"
)


BASELINE_CLASSIFICATION_REPORT = (
    RESULTS_DIR / "baseline_classification_report.txt"
)

QUANTUM_CLASSIFICATION_REPORT = (
    RESULTS_DIR / "quantum_classification_report.txt"
)


BASELINE_PREDICTIONS_PATH = (
    RESULTS_DIR / "baseline_predictions.csv"
)

QUANTUM_PREDICTIONS_PATH = (
    RESULTS_DIR / "quantum_predictions.csv"
)


MODEL_COMPARISON_PATH = (
    RESULTS_DIR / "model_comparison.csv"
)


# ============================================================================
# 7. FIGURE PATHS
# ============================================================================

# Dataset figures
CLASS_DISTRIBUTION_FIGURE = (
    FIGURES_DIR / "class_distribution.png"
)

DOCUMENT_LENGTH_FIGURE = (
    FIGURES_DIR / "document_length_distribution.png"
)


# Training curves
BASELINE_TRAINING_FIGURE = (
    FIGURES_DIR / "baseline_training_curves.png"
)

QUANTUM_TRAINING_FIGURE = (
    FIGURES_DIR / "quantum_training_curves.png"
)


# Confusion matrices
BASELINE_CONFUSION_MATRIX = (
    FIGURES_DIR / "baseline_confusion_matrix.png"
)

QUANTUM_CONFUSION_MATRIX = (
    FIGURES_DIR / "quantum_confusion_matrix.png"
)


# Model comparison
MODEL_COMPARISON_FIGURE = (
    FIGURES_DIR / "model_comparison.png"
)


# Attention visualization
STANDARD_ATTENTION_FIGURE = (
    FIGURES_DIR / "standard_attention.png"
)

QUANTUM_ATTENTION_FIGURE = (
    FIGURES_DIR / "quantum_attention.png"
)


# ============================================================================
# 8. DATASET CONFIGURATION
# ============================================================================

DATASET_NAME = "imdb"


# Binary sentiment classification
#
# 0 = Negative
# 1 = Positive
#
NUM_CLASSES = 2


# Official IMDB test set
TEST_SPLIT = 0.20


# 10% of the training data is used for validation
VALIDATION_SPLIT = 0.10


# Random seed
RANDOM_SEED = 42


# ============================================================================
# 9. TEXT PREPROCESSING CONFIGURATION
# ============================================================================

# Maximum vocabulary size
VOCAB_SIZE = 20_000


# Maximum number of tokens in one review
MAX_SEQUENCE_LENGTH = 250


# Token for unknown words
OOV_TOKEN = "<OOV>"


# Padding token
PAD_TOKEN = "<PAD>"


# Padding direction
PADDING = "post"


# Truncation direction
TRUNCATING = "post"


# ============================================================================
# 10. MODEL ARCHITECTURE
# ============================================================================

# Word embedding dimension
EMBEDDING_DIM = 128


# Number of LSTM units in each direction
#
# Forward LSTM  = 128
# Backward LSTM = 128
#
# BiLSTM output = 256
#
LSTM_UNITS = 128


# Dropout after model layers
DROPOUT_RATE = 0.30


# Recurrent dropout inside LSTM
RECURRENT_DROPOUT_RATE = 0.20


# Dense classifier units
DENSE_UNITS = 64


# ============================================================================
# 11. STANDARD ATTENTION CONFIGURATION
# ============================================================================

STANDARD_ATTENTION_DIM = 64


# Standard attention uses:
#
# e_i = v^T tanh(W h_i + b)
#
# alpha_i = softmax(e_i)
#
# context = sum(alpha_i * h_i)


# ============================================================================
# 12. QUANTUM-INSPIRED ATTENTION CONFIGURATION
# ============================================================================

QUANTUM_ATTENTION_DIM = 64


# Optional scaling factor
QUANTUM_PHASE_SCALE = 1.0


# Numerical stability constant
EPSILON = 1e-8


# Quantum-inspired attention:
#
# Step 1:
#     e_i = attention score
#
# Step 2:
#     a_i = tanh(e_i)
#
# Step 3:
#     p_i = |a_i|^2
#
# Step 4:
#     alpha_i = p_i / sum(p_j)
#
#
# Therefore:
#
# alpha_i =
#
#             tanh(e_i)^2
#     -----------------------------
#     sum_j tanh(e_j)^2 + epsilon
#
#
# This is quantum-PROBABILITY-inspired.
# It does NOT require:
#     - Quantum computer
#     - Quantum hardware
#     - Qiskit
#     - Quantum gates
#
# It is implemented using normal TensorFlow operations.


# ============================================================================
# 13. TRAINING CONFIGURATION
# ============================================================================

# Number of samples per training batch
BATCH_SIZE = 64


# Maximum number of epochs
EPOCHS = 2


# Initial learning rate
LEARNING_RATE = 1e-3


# Weight decay
WEIGHT_DECAY = 1e-5


# Early stopping
EARLY_STOPPING_PATIENCE = 3


# Learning-rate reduction
LR_REDUCTION_PATIENCE = 2

LR_REDUCTION_FACTOR = 0.5


# ============================================================================
# 14. EVALUATION METRICS
# ============================================================================

# Primary evaluation metric
PRIMARY_METRIC = "accuracy"


# Metrics used during model training
METRICS = [
    "accuracy",
    "precision",
    "recall",
]


# Metrics used for final comparison
FINAL_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]


# ============================================================================
# 15. CLASS LABELS
# ============================================================================

CLASS_NAMES = {
    0: "negative",
    1: "positive",
}


# ============================================================================
# 16. DEVICE CONFIGURATION
# ============================================================================

# TensorFlow automatically selects GPU if available.
# Otherwise CPU will be used.

DEVICE = "auto"


# ============================================================================
# 17. PROJECT INFORMATION
# ============================================================================

PROJECT_TITLE = (
    "Quantum-Inspired Attention Mechanism "
    "for Text Classification"
)


PROJECT_DESCRIPTION = (
    "A comparative study of standard Softmax-based "
    "attention and quantum-probability-inspired "
    "attention for IMDB sentiment classification."
)


# ============================================================================
# 18. CREATE PROJECT DIRECTORIES
# ============================================================================

def create_project_directories():
    """
    Create all directories required by the project.
    """

    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        ARTIFACTS_DIR,
        CHECKPOINT_DIR,
        MODEL_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================================
# 19. CONFIGURATION TEST
# ============================================================================

if __name__ == "__main__":

    create_project_directories()

    print("=" * 70)
    print("QUANTUM-INSPIRED ATTENTION PROJECT")
    print("CONFIGURATION")
    print("=" * 70)

    print("\nProject Information")
    print("-" * 70)
    print("Project :", PROJECT_TITLE)
    print("Dataset :", DATASET_NAME.upper())
    print("Classes :", NUM_CLASSES)

    print("\nDataset Configuration")
    print("-" * 70)
    print("Vocabulary Size      :", VOCAB_SIZE)
    print("Maximum Sequence     :", MAX_SEQUENCE_LENGTH)
    print("Validation Split     :", VALIDATION_SPLIT)
    print("Test Split           :", TEST_SPLIT)
    print("Random Seed          :", RANDOM_SEED)

    print("\nModel Configuration")
    print("-" * 70)
    print("Embedding Dimension  :", EMBEDDING_DIM)
    print("LSTM Units           :", LSTM_UNITS)
    print("Dense Units          :", DENSE_UNITS)
    print("Dropout              :", DROPOUT_RATE)
    print("Recurrent Dropout    :", RECURRENT_DROPOUT_RATE)

    print("\nAttention Configuration")
    print("-" * 70)
    print("Standard Attention  :", STANDARD_ATTENTION_DIM)
    print("Quantum Attention   :", QUANTUM_ATTENTION_DIM)
    print("Quantum Scale       :", QUANTUM_PHASE_SCALE)
    print("Epsilon             :", EPSILON)

    print("\nTraining Configuration")
    print("-" * 70)
    print("Batch Size          :", BATCH_SIZE)
    print("Epochs              :", EPOCHS)
    print("Learning Rate       :", LEARNING_RATE)
    print("Early Stopping      :", EARLY_STOPPING_PATIENCE)
    print("LR Reduction        :", LR_REDUCTION_FACTOR)

    print("\nClass Mapping")
    print("-" * 70)

    for label, class_name in CLASS_NAMES.items():
        print(f"{label} = {class_name}")

    print("\nDirectory Configuration")
    print("-" * 70)
    print("Project Root :", PROJECT_ROOT)
    print("Data         :", DATA_DIR)
    print("Raw Data     :", RAW_DATA_DIR)
    print("Processed    :", PROCESSED_DATA_DIR)
    print("Artifacts    :", ARTIFACTS_DIR)
    print("Models       :", MODEL_DIR)
    print("Results      :", RESULTS_DIR)
    print("Figures      :", FIGURES_DIR)

    print("\nQuantum-Inspired Attention Formula")
    print("-" * 70)
    print("a_i     = tanh(e_i)")
    print("p_i     = |a_i|^2")
    print("alpha_i = p_i / sum(p_j)")

    print("\nConfiguration loaded successfully!")