"""
Phase 3 - Baseline Model
Complete Standard Attention Model
"""

import tensorflow as tf

import config

try:
    from .bilstm import BiLSTMEncoder
    from .attention import StandardAttention
    from .classifier import SentimentClassifier
except ImportError:
    from bilstm import BiLSTMEncoder
    from attention import StandardAttention
    from classifier import SentimentClassifier


class BaselineAttentionModel(tf.keras.Model):
    """
    Complete baseline sentiment classification model.

    Architecture:

        Input
          ↓
        Embedding
          ↓
        BiLSTM
          ↓
        Standard Attention
          ↓
        Dense
          ↓
        Softmax
    """

    def __init__(self):
        super().__init__()

        self.encoder = BiLSTMEncoder()

        self.attention = StandardAttention()

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
        inputs : Tensor
            Tokenized input sequences.

        training : bool
            Training mode.

        return_attention : bool
            Whether to return attention weights.

        Returns
        -------
        predictions

        OR

        predictions, attention_weights
        """

        # BiLSTM
        sequence_output = self.encoder(
            inputs,
            training=training
        )

        # Padding mask
        mask = tf.not_equal(
            inputs,
            0
        )

        # Standard attention
        context, attention_weights = self.attention(
            sequence_output,
            mask=mask
        )

        # Classification
        predictions = self.classifier(
            context,
            training=training
        )

        if return_attention:

            return predictions, attention_weights

        return predictions