# src/elp_gunshot/cnn.py
"""CNN architecture for ELP Gunshot Detector training."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="elp_gunshot")
class CNN(tf.keras.Model):
    """Compact CNN for gunshot spectrogram classification."""

    def __init__(self, input_shape=(124, 129, 1), dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self._input_shape_arg = tuple(input_shape)
        self._dropout_rate = float(dropout_rate)

        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D(2)

        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D(2)

        self.conv3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPooling2D(2)

        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(256, activation="relu")
        self.dropout = tf.keras.layers.Dropout(self._dropout_rate)
        # Force float32 output so metric/loss numerics stay stable under mixed precision.
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)

        x = self.gap(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "input_shape": self._input_shape_arg,
                "dropout_rate": self._dropout_rate,
            }
        )
        return base
