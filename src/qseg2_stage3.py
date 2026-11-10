"""Stage-3 additions to qseg2 (additional experiments, prespecified in protocols/PROTOCOL_stage3_additional_experiments.txt).

E1 - faster-learning slot. The 4->4 block learns with an effective learning rate SLOT_SCALE (100) times that of the rest of the
     network. This is done by reparameterization, so one Adam optimizer and the usual ReduceLROnPlateau schedule still apply:
     every slot weight is stored as v = theta / k and used as theta = k * v. Adam moves v by about the learning rate per step,
     so theta moves about k times as far. The initial theta is exactly that of the standard model of the same seed.
     Models: g_q_fast, p_q_fast (circuit), g_cl_fast, p_cl_fast (Dense(4)), g_tr_fast, p_tr_fast (trigonometric).
E2 - shared initialization. The trigonometric and identity controls start from the initial weights of the circuit model of the
     same placement and seed in every layer they share with it (encoder, pre-slot, post-slot and decoder layers).
     Models: g_tr_shared, p_tr_shared, g_id_shared, p_id_shared. The E1 trigonometric models are initialized the same way.

Layer names are those of the standard models (quantum_circuit, classical_replacement, trig_replacement), so the standard
evaluation code applies; the effective slot weights are SLOT_SCALE times the stored variables of the *_fast models.
Call patch() once after importing qseg2 and qseg2_rev4 (qseg2_rev4.patch() is called here).
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import qseg2
import qseg2_rev4

SLOT_SCALE = 100.0
SLOT_LAYERS = ('quantum_circuit', 'classical_replacement', 'trig_replacement')
FAST = ('g_q_fast', 'p_q_fast', 'g_cl_fast', 'p_cl_fast', 'g_tr_fast', 'p_tr_fast')
SHARED = ('g_tr_shared', 'p_tr_shared', 'g_id_shared', 'p_id_shared')
NEEDS_SHARED_INIT = ('g_tr_fast', 'p_tr_fast') + SHARED
AUDIT_LAYERS = SLOT_LAYERS + ('pre_dense', 'pre_conv', 'post_dense', 'post_conv', 'bottleneck_conv2', 'dec1_conv1')


def _shrink(layer, names):
    for n in names:
        v = getattr(layer, n)
        v.assign(v / SLOT_SCALE)


class FastCircuit(qseg2.TFCircuitLayer):
    def build(self, input_shape):
        super().build(input_shape)
        _shrink(self, ['qweights'])

    def call(self, inputs):
        from qseg.quantum_sim import circuit_expvals, tf_ops
        return circuit_expvals(tf.cast(inputs, tf.float32), SLOT_SCALE * tf.cast(self.qweights, tf.float32), tf_ops(), self.n_qubits)


class FastPatchCircuit(FastCircuit):
    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        shape = tf.shape(x)
        out = super().call(tf.reshape(x, [-1, self.n_qubits]))
        return tf.reshape(out, shape)

    def compute_output_shape(self, input_shape):
        return input_shape


class FastDense(layers.Dense):
    def build(self, input_shape):
        super().build(input_shape)
        _shrink(self, ['kernel', 'bias'])

    def call(self, inputs):
        y = tf.tensordot(tf.cast(inputs, tf.float32), SLOT_SCALE * self.kernel, axes=[[-1], [0]]) + SLOT_SCALE * self.bias
        return tf.tanh(y)


class FastTrig(qseg2_rev4.TrigLayer):
    def build(self, input_shape):
        super().build(input_shape)
        _shrink(self, ['A', 'phi'])

    def call(self, x):
        c = tf.cos(np.pi / 2 * tf.cast(x, tf.float32) + SLOT_SCALE * self.phi)
        return tf.tanh(tf.einsum('...i,ji->...j', c, SLOT_SCALE * self.A))


_prev_block = None


def _block(z, kind, n_qubits, n_layers, patch):
    if kind == 'q_fast':
        return (FastPatchCircuit if patch else FastCircuit)(n_qubits, n_layers, name='quantum_circuit')(z)
    if kind == 'cl_fast':
        return FastDense(n_qubits, activation='tanh', name='classical_replacement')(z)
    if kind == 'tr_fast':
        return FastTrig(n_qubits, name='trig_replacement')(z)
    if kind in ('tr_shared', 'id_shared'):
        return _prev_block(z, kind.split('_')[0], n_qubits, n_layers, patch)
    return _prev_block(z, kind, n_qubits, n_layers, patch)


# ----------------------------------------------------------------------------- shared initialization
_base_build = None
INIT_SEED = None          # set by the runner before run_fold; consumed by the first model build
INIT_RECORD = {}          # filled at that build: copy check and initial weights of the audited layers


def effective(layer, name):
    """Effective weights of a layer as one flat float64 vector (slot of *_fast models: SLOT_SCALE x stored)."""
    w = np.concatenate([v.numpy().astype(np.float64).ravel() for v in layer.trainable_weights]) if layer.trainable_weights else np.zeros(0)
    if name in FAST and layer.name in SLOT_LAYERS:
        w = w * SLOT_SCALE
    return w


def build_p2_model(name, *args, **kwargs):
    global INIT_SEED
    model = _base_build(name, *args, **kwargs)
    if INIT_SEED is None or name not in FAST + SHARED:
        return model
    seed, INIT_SEED = INIT_SEED, None
    ref_name = name[:2] + 'q'
    qseg2.set_seeds(seed)
    ref = _base_build(ref_name, *args, **kwargs)                  # standard circuit model of the same placement and seed
    refw = {l.name: l.get_weights() for l in ref.layers if l.weights}
    copied, before = [], {}
    for l in model.layers:
        if not l.weights or l.name not in refw:
            continue
        if l.name in SLOT_LAYERS:
            if name.startswith(('g_q', 'p_q')):                    # circuit: same initial angles as the standard model
                w = refw[l.name]
                l.set_weights([x / SLOT_SCALE for x in w] if name in FAST else w)
            continue
        a, b = l.get_weights(), refw[l.name]
        if [x.shape for x in a] != [x.shape for x in b]:
            continue
        before[l.name] = float(max(np.abs(x - y).max() for x, y in zip(a, b)))
        if name in NEEDS_SHARED_INIT:
            l.set_weights(b); copied.append(l.name)
    after = {l.name: float(max(np.abs(x - y).max() for x, y in zip(l.get_weights(), refw[l.name])))
             for l in model.layers if l.weights and l.name in refw and l.name not in SLOT_LAYERS and l.name in before}
    del ref
    qseg2.set_seeds(seed)                                          # training randomness starts from the seed, as in all runs
    INIT_RECORD.clear()
    INIT_RECORD.update(reference=ref_name, seed=seed, copied_layers=copied, max_diff_before_copy=before,
                       max_diff_after_copy=after, shared_identical=bool(all(v == 0 for v in after.values())),
                       slot_scale=SLOT_SCALE if name in FAST else 1.0,
                       init={l.name: effective(l, name) for l in model.layers if l.name in AUDIT_LAYERS})
    return model


def weight_change(model, name):
    """Change of the audited layers from INIT_RECORD['init'] to the current (trained) weights."""
    out = {}
    for l in model.layers:
        if l.name not in INIT_RECORD.get('init', {}):
            continue
        a = INIT_RECORD['init'][l.name]; b = effective(l, name); d = b - a
        if l.name == 'quantum_circuit':
            d = (d + np.pi) % (2 * np.pi) - np.pi
        out[l.name] = dict(n=int(a.size), max_abs_change=float(np.abs(d).max()), rms_change=float(np.sqrt((d ** 2).mean())),
                           rel_change=float(np.linalg.norm(d) / max(np.linalg.norm(a), 1e-12)))
    return out


def patch():
    global _prev_block, _base_build
    qseg2_rev4.patch()
    if _prev_block is None:
        _prev_block = qseg2._block
        qseg2._block = _block
    if _base_build is None:
        _base_build = qseg2.build_p2_model
        qseg2.build_p2_model = build_p2_model
    lab = {'q': 'circuit', 'cl': 'Dense(4)', 'tr': 'trigonometric', 'id': 'identity'}
    for n in FAST:
        qseg2.MODELS2.setdefault(n, f"{'Global' if n[0] == 'g' else 'Patch-wise'} bottleneck, {lab[n[2:].split('_')[0]]}, slot learning rate x{SLOT_SCALE:g}")
    for n in SHARED:
        qseg2.MODELS2.setdefault(n, f"{'Global' if n[0] == 'g' else 'Patch-wise'} bottleneck, {lab[n[2:].split('_')[0]]}, circuit-model initialization")
    return qseg2
