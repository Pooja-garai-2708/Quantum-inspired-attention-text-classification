"""
Phase 3 - Baseline Model
BiLSTM Encoder

Architecture:
Input Tokens
    ↓
Embedding
    ↓
Bidirectional LSTM
    ↓
Sequence Representation
"""

import tensorflow as tf
import config


class BiLSTMEncoder(tf.keras.Model):
    """
    Bidirectional LSTM encoder for the baseline model.
    """

    def __init__(self):
        super().__init__()

        # Word embedding
        self.embedding = tf.keras.layers.Embedding(
            input_dim=config.VOCAB_SIZE,
            output_dim=config.EMBEDDING_DIM,
            mask_zero=True,
            name="embedding"
        )

        # Bidirectional LSTM
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=config.LSTM_UNITS,
                return_sequences=True,
                dropout=config.DROPOUT_RATE,
                recurrent_dropout=config.RECURRENT_DROPOUT_RATE
            ),
            name="bidirectional_lstm"
        )

    def call(self, inputs, training=False):
        """
        Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Integer token sequences.

        training : bool
            Training mode.

        Returns
        -------
        Tensor
            Sequence representation.
        """

        x = self.embedding(inputs)

        x = self.bilstm(
            x,
            training=training
        )

        return x