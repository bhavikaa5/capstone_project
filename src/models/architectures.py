"""Step 4 (part 2) — the three competing architectures.

All three take `(batch, lookback, n_features)` and emit one sigmoid probability
that the next bar closes higher. They are deliberately similar in capacity
(~50-120k parameters) so the comparison is between *architectures* rather than
between a large model and a small one.
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf
from keras import layers

__all__ = [
    "ARCHITECTURES",
    "build_cnn_bilstm_attention",
    "build_lstm",
    "build_transformer",
    "set_seeds",
]


def set_seeds(seed: int = 42) -> None:
    """Seed Python, NumPy and TensorFlow so a rerun reproduces the run."""
    keras.utils.set_random_seed(seed)


@keras.saving.register_keras_serializable(package="capstone")
class AttentionPooling(layers.Layer):
    """Additive (Bahdanau-style) attention pooling over the time axis.

    Collapses `(batch, time, features)` to `(batch, features)` by learning a
    scalar score per timestep and taking a softmax-weighted sum. This is the
    "Attention" in CNN-BiLSTM-Attention: instead of taking only the final LSTM
    state, the model learns *which bars in the window mattered* — which is the
    whole argument for the architecture on data where a single spike several
    bars back can dominate.

    The attention weights are exposed via `last_attention` for interpretation.
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.score_dense = layers.Dense(units, activation="tanh")
        self.score_out = layers.Dense(1)

    def build(self, input_shape):
        # Keras 3 will not reload a saved model whose child layers were created
        # in __init__ but never built — it finds weights in the file with no
        # variables to load them into. Building them here is what makes
        # `keras.models.load_model` work for this architecture.
        self.score_dense.build(input_shape)
        self.score_out.build((*input_shape[:-1], self.units))
        super().build(input_shape)

    def call(self, inputs):
        scores = self.score_out(self.score_dense(inputs))       # (b, t, 1)
        weights = tf.nn.softmax(scores, axis=1)
        self.last_attention = weights
        return tf.reduce_sum(inputs * weights, axis=1)          # (b, f)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return {**super().get_config(), "units": self.units}


def build_lstm(
    input_shape: tuple[int, int],
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Plain stacked LSTM — the reference point the other two must beat."""
    inp = layers.Input(shape=input_shape, name="window")
    x = layers.LSTM(units, return_sequences=True)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units // 2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="p_up")(x)

    model = keras.Model(inp, out, name="lstm")
    return _compile(model, learning_rate)


def build_cnn_bilstm_attention(
    input_shape: tuple[int, int],
    filters: int = 64,
    kernel_size: int = 3,
    units: int = 48,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Conv1D feature extractor -> bidirectional LSTM -> attention pooling.

    The Conv1D learns short local patterns (a 3-bar shape) before the recurrent
    layer sees them, which shortens the effective sequence the LSTM must model.
    `padding="causal"` matters: 'same' padding would let the convolution at bar t
    read bars t+1 and t+2, putting future information inside the window.
    """
    inp = layers.Input(shape=input_shape, name="window")
    x = layers.Conv1D(filters, kernel_size, padding="causal", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, kernel_size, padding="causal", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    x = AttentionPooling(units=units)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="p_up")(x)

    model = keras.Model(inp, out, name="cnn_bilstm_attention")
    return _compile(model, learning_rate)


def build_transformer(
    input_shape: tuple[int, int],
    d_model: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_blocks: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Encoder-only transformer with learned positional embeddings.

    A causal mask is applied inside the attention. On a fixed window that
    predicts only from the final bar it is not strictly required — the window
    already contains no future — but without it every position attends to every
    other, so the representation at bar t mixes in bars after t. Keeping the mask
    means the per-timestep representations stay interpretable and the model
    degrades gracefully if the head is ever changed to predict at every step.
    """
    lookback, n_features = input_shape
    inp = layers.Input(shape=input_shape, name="window")

    x = layers.Dense(d_model)(inp)
    positions = tf.range(start=0, limit=lookback, delta=1)
    pos_emb = layers.Embedding(input_dim=lookback, output_dim=d_model)(positions)
    x = x + pos_emb

    for _ in range(num_blocks):
        attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
        )(x, x, use_causal_mask=True)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

        ff = layers.Dense(ff_dim, activation="relu")(x)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(d_model)(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    # The last position is the only one that has seen the whole window under a
    # causal mask, so pool there rather than averaging over time.
    x = x[:, -1, :]
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="p_up")(x)

    model = keras.Model(inp, out, name="transformer")
    return _compile(model, learning_rate)


def _compile(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


ARCHITECTURES = {
    "lstm": build_lstm,
    "cnn_bilstm_attention": build_cnn_bilstm_attention,
    "transformer": build_transformer,
}
