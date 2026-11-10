"""Stage-4 additions to qseg2 (additional experiments, prespecified in protocols/PROTOCOL_stage4_additional_experiments.txt).

E3 - block learning rate chosen per block. As E1 (qseg2_stage3), but with effective block learning-rate factors 10 and 30:
     models g_/p_ x q_/cl_/tr_ x fast10/fast30 (e.g. g_cl_fast10). The factor-100 models are the E1 models (*_fast).
E4 - wider encoding range. Circuit and trigonometric blocks whose angle encoding is pi*c instead of (pi/2)*c, with the
     factor-100 block learning rate of E1: g_q_wide, p_q_wide, g_tr_wide, p_tr_wide. The Dense(4) block has no encoding,
     so g_cl_fast / p_cl_fast (E1) are its comparators.
Every model starts from the initial weights of the standard circuit model of the same placement and seed in every layer it
shares with it (as the E1 models; checked before training). Layer names are those of the standard models.
Call patch() once after importing qseg2 (qseg2_stage3.patch() is called here).
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import qseg2
import qseg2_rev4
import qseg2_stage3 as S3

ENC_WIDE = 2.0                                   # angle = (pi/2) * ENC * c, so ENC = 2 gives pi * c
FACTORS = {'fast10': 10.0, 'fast30': 30.0, 'wide': 100.0}
NEW = tuple(f'{p}_{b}_{v}' for p in 'gp' for b in ('q', 'cl', 'tr') for v in ('fast10', 'fast30')) + \
      tuple(f'{p}_{b}_wide' for p in 'gp' for b in ('q', 'tr'))


def scale_of(name):
    """Effective slot learning-rate factor of a model (1 for standard models)."""
    v = name.split('_', 2)[-1] if name.count('_') >= 2 else ''
    if v in FACTORS: return FACTORS[v]
    if v == 'fast': return S3.SLOT_SCALE
    return 1.0


def enc_of(name):
    return ENC_WIDE if name.endswith('_wide') else 1.0


def _shrink(layer, names, k):
    for n in names:
        v = getattr(layer, n); v.assign(v / k)


class ScaledCircuit(qseg2.TFCircuitLayer):
    def __init__(self, *a, k=1.0, enc=1.0, **kw):
        super().__init__(*a, **kw); self.k, self.enc = float(k), float(enc)

    def build(self, input_shape):
        super().build(input_shape); _shrink(self, ['qweights'], self.k)

    def call(self, inputs):
        from qseg.quantum_sim import circuit_expvals, tf_ops
        return circuit_expvals(self.enc * tf.cast(inputs, tf.float32), self.k * tf.cast(self.qweights, tf.float32), tf_ops(), self.n_qubits)


class ScaledPatchCircuit(ScaledCircuit):
    def call(self, inputs):
        x = tf.cast(inputs, tf.float32); shape = tf.shape(x)
        return tf.reshape(super().call(tf.reshape(x, [-1, self.n_qubits])), shape)

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDense(layers.Dense):
    def __init__(self, *a, k=1.0, **kw):
        super().__init__(*a, **kw); self.k = float(k)

    def build(self, input_shape):
        super().build(input_shape); _shrink(self, ['kernel', 'bias'], self.k)

    def call(self, inputs):
        return tf.tanh(tf.tensordot(tf.cast(inputs, tf.float32), self.k * self.kernel, axes=[[-1], [0]]) + self.k * self.bias)


class ScaledTrig(qseg2_rev4.TrigLayer):
    def __init__(self, *a, k=1.0, enc=1.0, **kw):
        super().__init__(*a, **kw); self.k, self.enc = float(k), float(enc)

    def build(self, input_shape):
        super().build(input_shape); _shrink(self, ['A', 'phi'], self.k)

    def call(self, x):
        c = tf.cos(np.pi / 2 * self.enc * tf.cast(x, tf.float32) + self.k * self.phi)
        return tf.tanh(tf.einsum('...i,ji->...j', c, self.k * self.A))


_prev_block = None


def _block(z, kind, n_qubits, n_layers, patch):
    base, _, var = kind.partition('_')
    if var in FACTORS:
        k, enc = FACTORS[var], (ENC_WIDE if var == 'wide' else 1.0)
        if base == 'q':
            return (ScaledPatchCircuit if patch else ScaledCircuit)(n_qubits, n_layers, k=k, enc=enc, name='quantum_circuit')(z)
        if base == 'cl':
            return ScaledDense(n_qubits, activation='tanh', k=k, name='classical_replacement')(z)
        if base == 'tr':
            return ScaledTrig(n_qubits, k=k, enc=enc, name='trig_replacement')(z)
    return _prev_block(z, kind, n_qubits, n_layers, patch)


def effective(layer, name):
    w = np.concatenate([v.numpy().astype(np.float64).ravel() for v in layer.trainable_weights]) if layer.trainable_weights else np.zeros(0)
    if layer.name in S3.SLOT_LAYERS:
        w = w * scale_of(name)
    return w


_s3_build = None


def build_p2_model(name, *args, **kwargs):
    """New models: built, then every layer shared with the standard circuit model of the same placement and seed is copied
    from it, and the block gets the standard initial values (scaled). Other models: unchanged (stage-3 code path)."""
    if name not in NEW or S3.INIT_SEED is None:
        return _s3_build(name, *args, **kwargs)
    seed, S3.INIT_SEED = S3.INIT_SEED, None
    model = S3._base_build(name, *args, **kwargs)
    k = scale_of(name)
    qseg2.set_seeds(seed)
    ref = S3._base_build(name[:2] + 'q', *args, **kwargs)                      # standard circuit model, same seed
    refw = {l.name: l.get_weights() for l in ref.layers if l.weights}
    if not name.startswith(('g_q', 'p_q')):                                      # standard block of this kind, same seed
        qseg2.set_seeds(seed)
        own = S3._base_build(name[:2] + name[2:].split('_')[0], *args, **kwargs)
        ownw = {l.name: l.get_weights() for l in own.layers if l.name in S3.SLOT_LAYERS}
        del own
    else:
        ownw = refw
    copied, before = [], {}
    for l in model.layers:
        if not l.weights:
            continue
        if l.name in S3.SLOT_LAYERS:
            l.set_weights([x / k for x in ownw[l.name]]); continue
        if l.name not in refw:
            continue
        a, b = l.get_weights(), refw[l.name]
        if [x.shape for x in a] != [x.shape for x in b]:
            continue
        before[l.name] = float(max(np.abs(x - y).max() for x, y in zip(a, b)))
        l.set_weights(b); copied.append(l.name)
    after = {l.name: float(max(np.abs(x - y).max() for x, y in zip(l.get_weights(), refw[l.name])))
             for l in model.layers if l.name in before}
    del ref
    qseg2.set_seeds(seed)
    S3.INIT_RECORD.clear()
    S3.INIT_RECORD.update(reference=name[:2] + 'q', seed=seed, copied_layers=copied, max_diff_before_copy=before,
                          max_diff_after_copy=after, shared_identical=bool(all(v == 0 for v in after.values())),
                          slot_scale=k, encoding_factor=enc_of(name),
                          init={l.name: effective(l, name) for l in model.layers if l.name in S3.AUDIT_LAYERS})
    return model


def weight_change(model, name):
    out = {}
    for l in model.layers:
        if l.name not in S3.INIT_RECORD.get('init', {}):
            continue
        a = S3.INIT_RECORD['init'][l.name]; b = effective(l, name); d = b - a
        if l.name == 'quantum_circuit':
            d = (d + np.pi) % (2 * np.pi) - np.pi
        out[l.name] = dict(n=int(a.size), max_abs_change=float(np.abs(d).max()), rms_change=float(np.sqrt((d ** 2).mean())),
                           rel_change=float(np.linalg.norm(d) / max(np.linalg.norm(a), 1e-12)))
    return out


def patch():
    global _prev_block, _s3_build
    S3.patch()
    if _prev_block is None:
        _prev_block = qseg2._block; qseg2._block = _block
    if _s3_build is None:
        _s3_build = qseg2.build_p2_model; qseg2.build_p2_model = build_p2_model
    lab = {'q': 'circuit', 'cl': 'Dense(4)', 'tr': 'trigonometric'}
    for n in NEW:
        b, v = n[2:].split('_')
        desc = f'slot learning rate x{FACTORS[v]:g}' + (', encoding pi*c' if v == 'wide' else '')
        qseg2.MODELS2.setdefault(n, f"{'Global' if n[0] == 'g' else 'Patch-wise'} bottleneck, {lab[b]}, {desc}")
    return qseg2
