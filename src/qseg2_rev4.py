"""Stage-2 additions to qseg2: a 20-parameter trigonometric block for the 4->4 slot.

    z = tanh(A cos((pi/2) x + phi)),  A in R^{4x4} (Glorot uniform),  phi in R^4 (uniform on [0, 2 pi), as the circuit angles)

The angle encoding (pi/2) x is the one the circuit uses; like the circuit outputs, the block output is a trigonometric
function of the angles. Global models apply it to the 4-value code, patch-wise models to every cell with one shared set of
weights (as the patch-wise circuit). Models: g_tr, p_tr. Call patch() once after importing qseg2.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import qseg2


class TrigLayer(layers.Layer):
    def __init__(self, n=4, **kw):
        super().__init__(**kw)
        self.n = n

    def build(self, input_shape):
        self.A = self.add_weight(name="A", shape=(self.n, self.n), initializer="glorot_uniform")
        self.phi = self.add_weight(name="phi", shape=(self.n,), initializer=tf.keras.initializers.RandomUniform(0.0, 2 * np.pi))
        super().build(input_shape)

    def call(self, x):
        c = tf.cos(np.pi / 2 * tf.cast(x, tf.float32) + self.phi)
        return tf.tanh(tf.einsum("...i,ji->...j", c, self.A))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "n": self.n}


_orig_block = qseg2._block


def _block(z, kind, n_qubits, n_layers, patch):
    if kind == "tr":
        return TrigLayer(n_qubits, name="trig_replacement")(z)
    return _orig_block(z, kind, n_qubits, n_layers, patch)


def patch():
    qseg2._block = _block
    qseg2.MODELS2.setdefault("g_tr", "Global bottleneck, trigonometric 4->4 layer (20 parameters)")
    qseg2.MODELS2.setdefault("p_tr", "Patch-wise bottleneck, trigonometric 4->4 layer (20 parameters)")
    return qseg2
