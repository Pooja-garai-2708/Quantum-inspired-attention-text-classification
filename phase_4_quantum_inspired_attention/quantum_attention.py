"""
Phase 4 — Quantum-Inspired Attention.

Quantum-probability-inspired attention replaces the conventional
Softmax attention weighting with a probability derived from the
squared attention amplitude.

Standard attention:

    e_i = v^T tanh(W h_i + b)

    alpha_i = softmax(e_i)

Quantum-inspired attention:

    e_i = v^T tanh(W h_i + b)

    a_i = tanh(e_i)

    p_i = |a_i|^2

    alpha_i = p_i / sum(p_i)

    context = sum(alpha_i h_i)

Important:
This is a quantum-inspired mathematical formulation.
It does NOT require a quantum computer or Qiskit.
"""

import tensorflow as tf

import config


class QuantumInspiredAttention(tf.keras.layers.Layer):
    """
    Quantum-probability-inspired attention layer.

    Input:
        sequence_output:
            (batch_size, sequence_length, features)

    Mask:
        (batch_size, sequence_length)

    Output:
        context_vector:
            (batch_size, features)

        attention_weights:
            (batch_size, sequence_length)
    """

    def __init__(self, attention_dim=None, epsilon=None, **kwargs):
        super().__init__(**kwargs)

        if attention_dim is None:
            attention_dim = config.QUANTUM_ATTENTION_DIM

        if epsilon is None:
            epsilon = config.EPSILON

        self.attention_dim = attention_dim
        self.epsilon = epsilon

        # ---------------------------------------------------------
        # Learnable attention transformation
        # ---------------------------------------------------------

        self.feature_projection = tf.keras.layers.Dense(
            self.attention_dim,
            activation="tanh",
            name="quantum_attention_feature_projection"
        )

        # Converts transformed feature vector into scalar score
        self.score_projection = tf.keras.layers.Dense(
            1,
            use_bias=True,
            name="quantum_attention_score_projection"
        )

        # Explicitly support Keras masking
        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):

        # ---------------------------------------------------------
        # Step 1: Transform BiLSTM hidden states
        # ---------------------------------------------------------

        projected = self.feature_projection(inputs)

        # ---------------------------------------------------------
        # Step 2: Calculate conventional attention energy
        #
        # e_i = v^T tanh(W h_i + b)
        # ---------------------------------------------------------

        scores = self.score_projection(projected)

        scores = tf.squeeze(
            scores,
            axis=-1
        )

        # ---------------------------------------------------------
        # Step 3: Convert attention score into amplitude
        #
        # a_i = tanh(e_i)
        # ---------------------------------------------------------

        amplitudes = tf.tanh(scores)

        # ---------------------------------------------------------
        # Step 4: Calculate quantum-inspired probability
        #
        # p_i = |a_i|^2
        #
        # Since amplitudes are real-valued:
        #
        # |a_i|^2 = a_i^2
        # ---------------------------------------------------------

        probabilities = tf.square(
            tf.abs(amplitudes)
        )

        # ---------------------------------------------------------
        # Step 5: Apply padding mask
        #
        # Padding tokens must have zero probability.
        # ---------------------------------------------------------

        if mask is not None:

            mask = tf.cast(
                mask,
                probabilities.dtype
            )

            probabilities = (
                probabilities * mask
            )

        # ---------------------------------------------------------
        # Step 6: Normalize probabilities
        #
        # alpha_i =
        #       p_i
        #       ----------------
        #       sum_j p_j + epsilon
        # ---------------------------------------------------------

        probability_sum = tf.reduce_sum(
            probabilities,
            axis=1,
            keepdims=True
        )

        attention_weights = (
            probabilities
            /
            (
                probability_sum
                + self.epsilon
            )
        )

        # ---------------------------------------------------------
        # Step 7: Weighted context vector
        #
        # c = sum(alpha_i h_i)
        # ---------------------------------------------------------

        context_vector = tf.reduce_sum(
            inputs
            * tf.expand_dims(
                attention_weights,
                axis=-1
            ),
            axis=1
        )

        return context_vector, attention_weights

    def compute_mask(
        self,
        inputs,
        mask=None
    ):
        """
        The attention layer converts a sequence into a single
        context vector, so no sequence mask should be propagated.
        """

        return None

    def get_config(self):

        config_dict = super().get_config()

        config_dict.update({
            "attention_dim": self.attention_dim,
            "epsilon": self.epsilon
        })

        return config_dict