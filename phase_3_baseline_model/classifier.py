"""
Phase 3 - Baseline Model
Sentiment Classifier
"""

import tensorflow as tf
import config


class SentimentClassifier(tf.keras.Model):
    """
    Dense classifier for binary sentiment classification.
    """

    def __init__(self):
        super().__init__()

        self.dense = tf.keras.layers.Dense(
            config.DENSE_UNITS,
            activation="relu",
            name="classifier_dense"
        )

        self.dropout = tf.keras.layers.Dropout(
            config.DROPOUT_RATE,
            name="classifier_dropout"
        )

        self.output_layer = tf.keras.layers.Dense(
            config.NUM_CLASSES,
            activation="softmax",
            name="classification_output"
        )

    def call(self, inputs, training=False):

        x = self.dense(inputs)

        x = self.dropout(
            x,
            training=training
        )

        outputs = self.output_layer(x)

        return outputs