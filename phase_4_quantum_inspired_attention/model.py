"""
Phase 4 — Quantum-Inspired Attention Model.

Experimental architecture:

Embedding
    ↓
BiLSTM Encoder
    ↓
Quantum-Inspired Attention
    ↓
Sentiment Classifier

IMPORTANT:
The BiLSTM encoder and classifier are reused directly from Phase 3.
Only the attention mechanism is changed.
"""

import tensorflow as tf

import config

# ---------------------------------------------------------------
# IMPORTANT:
# Reuse the exact Phase 3 BiLSTM and classifier.
# This guarantees a fair comparison between Phase 3 and Phase 4.
# ---------------------------------------------------------------

from phase_3_baseline_model.bilstm import BiLSTMEncoder
from phase_3_baseline_model.classifier import SentimentClassifier

# Quantum attention remains Phase 4-specific.
from phase_4_quantum_inspired_attention.quantum_attention import (
    QuantumInspiredAttention
)


class QuantumInspiredAttentionModel(tf.keras.Model):
    """
    Complete Phase 4 model.
    """

    def __init__(self, **kwargs):

        super().__init__(
            name="quantum_inspired_attention_model",
            **kwargs
        )

        # -------------------------------------------------------
        # SAME BiLSTM encoder used in Phase 3
        # -------------------------------------------------------

        self.encoder = BiLSTMEncoder()

        # -------------------------------------------------------
        # NEW attention mechanism
        # -------------------------------------------------------

        self.attention = QuantumInspiredAttention(
            attention_dim=config.QUANTUM_ATTENTION_DIM,
            epsilon=config.EPSILON,
            name="quantum_inspired_attention"
        )

        # -------------------------------------------------------
        # SAME classifier used in Phase 3
        # -------------------------------------------------------

        self.classifier = SentimentClassifier()

    def call(
        self,
        inputs,
        training=False,
        return_attention=False
    ):
        """
        Forward pass.

        Parameters
        ----------
        inputs:
            Integer padded token sequences.

        training:
            True during training, False during inference.

        return_attention:
            If True, also return attention weights.
        """

        # -------------------------------------------------------
        # 1. BiLSTM sequence representation
        # -------------------------------------------------------

        sequence_output = self.encoder(
            inputs,
            training=training
        )

        # -------------------------------------------------------
        # 2. Padding mask
        #
        # Token ID 0 = padding
        # -------------------------------------------------------

        mask = tf.not_equal(
            inputs,
            0
        )

        # -------------------------------------------------------
        # 3. Quantum-Inspired Attention
        # -------------------------------------------------------

        context_vector, attention_weights = self.attention(
            sequence_output,
            mask=mask,
            training=training
        )

        # -------------------------------------------------------
        # 4. Sentiment classification
        # -------------------------------------------------------

        predictions = self.classifier(
            context_vector,
            training=training
        )

        # -------------------------------------------------------
        # 5. Optionally return attention weights
        # -------------------------------------------------------

        if return_attention:

            return (
                predictions,
                attention_weights
            )

        return predictions