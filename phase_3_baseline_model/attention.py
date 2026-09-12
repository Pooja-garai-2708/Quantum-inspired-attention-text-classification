"""
Standard Additive Attention for Phase 3 Baseline Model.

The attention mechanism follows the standard additive attention formulation:

    e_i = v^T tanh(W h_i + b)

    alpha_i = softmax(e_i)

    context = sum(alpha_i * h_i)

Padding positions are masked before the softmax calculation.
"""

import tensorflow as tf

import config


class StandardAttention(tf.keras.layers.Layer):
    """
    Standard additive attention layer.

    Input:
        sequence_output:
            Tensor of shape (batch_size, sequence_length, features)

    Mask:
        Boolean tensor of shape (batch_size, sequence_length)
        where True represents a valid token and False represents padding.

    Output:
        context_vector:
            Tensor of shape (batch_size, features)

        attention_weights:
            Tensor of shape (batch_size, sequence_length)
    """

    def __init__(self, attention_dim=None, **kwargs):
        super().__init__(**kwargs)

        if attention_dim is None:
            attention_dim = config.STANDARD_ATTENTION_DIM

        self.attention_dim = attention_dim

        # Intermediate projection:
        # h_i -> tanh(W h_i + b)
        self.feature_projection = tf.keras.layers.Dense(
            self.attention_dim,
            activation="tanh",
            name="attention_feature_projection"
        )

        # Converts projected feature vector into one scalar score e_i
        self.score_projection = tf.keras.layers.Dense(
            1,
            use_bias=True,
            name="attention_score_projection"
        )

        # Tell Keras that this layer can receive a mask.
        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        """
        Calculate standard additive attention.

        Parameters
        ----------
        inputs : tf.Tensor
            Shape:
            (batch_size, sequence_length, features)

        mask : tf.Tensor or None
            Shape:
            (batch_size, sequence_length)

        training : bool
            Included for Keras compatibility.
        """

        # ---------------------------------------------------------
        # 1. Calculate projected hidden representation
        # ---------------------------------------------------------
        projected = self.feature_projection(inputs)

        # ---------------------------------------------------------
        # 2. Calculate attention scores
        # ---------------------------------------------------------
        #
        # e_i = v^T tanh(W h_i + b)
        #
        # Dense(1) gives:
        # (batch, sequence_length, 1)
        #
        scores = self.score_projection(projected)

        # Remove the final dimension:
        #
        # (batch, sequence_length, 1)
        #            ↓
        # (batch, sequence_length)
        #
        scores = tf.squeeze(scores, axis=-1)

        # ---------------------------------------------------------
        # 3. Apply padding mask
        # ---------------------------------------------------------
        #
        # Padding tokens must NOT receive attention probability.
        #
        # True  -> valid token
        # False -> padding
        #
        if mask is not None:

            # Convert mask to boolean tensor
            mask = tf.cast(mask, tf.bool)

            # Give padding positions a very large negative score.
            #
            # softmax(-1e9) ≈ 0
            #
            masked_scores = tf.where(
                mask,
                scores,
                tf.constant(-1e9, dtype=scores.dtype)
            )

        else:
            masked_scores = scores

        # ---------------------------------------------------------
        # 4. Standard Softmax attention
        # ---------------------------------------------------------
        #
        # alpha_i =
        #       exp(e_i)
        #       ----------------
        #       sum_j exp(e_j)
        #
        attention_weights = tf.nn.softmax(
            masked_scores,
            axis=1
        )

        # ---------------------------------------------------------
        # 5. Weighted sum of BiLSTM hidden states
        # ---------------------------------------------------------
        #
        # context =
        #       sum_i alpha_i h_i
        #
        context_vector = tf.reduce_sum(
            inputs * tf.expand_dims(attention_weights, axis=-1),
            axis=1
        )

        return context_vector, attention_weights

    def compute_mask(self, inputs, mask=None):
        """
        Attention produces a single context vector rather than a
        sequence, so no mask should be propagated to the next layer.
        """

        return None

    def get_config(self):
        """
        Make the layer serializable.
        """

        config_dict = super().get_config()

        config_dict.update({
            "attention_dim": self.attention_dim
        })

        return config_dict