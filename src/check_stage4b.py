"""Pre-flight check of the stage-4 models (no data, about a minute on a GPU). Stops with an error if any check fails.

For every new model (seed 1): (1) the effective initial block weights equal those of the standard block of the same kind and
seed; (2) for the circuit and Dense(4) learning-rate variants, the output equals that of the standard model on random input
(the standard trigonometric model has its own decoder initialization, so its output differs by design); (3) every layer shared
with the standard circuit model is identical; (4) one Adam step (learning rate 1e-4) moves the effective block weights about
k times as far as in the standard model (k = 10, 30 or 100).
"""
import json
import numpy as np
import tensorflow as tf
import qseg2
import qseg2_stage3 as S3
import qseg2_stage4 as S4

S4.patch()
SHAPE = (256, 256, 1)
rng = np.random.default_rng(0)
x = rng.random((2,) + SHAPE).astype('float32'); y = (rng.random((2,) + SHAPE) > 0.7).astype('float32')
report = {}


def build(name, seed, copy):
    tf.keras.backend.clear_session()
    qseg2.set_seeds(seed)
    S3.INIT_SEED = seed if copy else None
    return qseg2.build_p2_model(name, SHAPE, 4, 2)


def slot(m, name):
    for l in m.layers:
        if l.name in S3.SLOT_LAYERS:
            return S4.effective(l, name)


def one_step(m):
    opt = tf.keras.optimizers.Adam(1e-4)
    with tf.GradientTape() as t:
        loss = qseg2.bce_dice_loss(y, m(x, training=False))
    g = t.gradient(loss, m.trainable_weights)
    opt.apply_gradients(zip(g, m.trainable_weights))


for name in S4.NEW:
    std = name[:2] + name[2:].split('_')[0]
    k = S4.scale_of(name)
    ms = build(std, 1, False); a0 = slot(ms, std); out_s = ms(x, training=False).numpy(); one_step(ms); da = np.abs(slot(ms, std) - a0).max()
    mf = build(name, 1, True); rec = dict(S3.INIT_RECORD)
    b0 = slot(mf, name); out_f = mf(x, training=False).numpy(); one_step(mf); db = np.abs(slot(mf, name) - b0).max()
    r = dict(k=k, init_max_diff=float(np.abs(a0 - b0).max()), output_max_diff=float(np.abs(out_s - out_f).max()),
             ratio=float(db / max(da, 1e-12)), copied=len(rec.get('copied_layers', [])), shared_identical=rec.get('shared_identical'))
    report[name] = r
    assert r['init_max_diff'] < 1e-5, (name, r)
    if not name.endswith('_wide') and name[2:].split('_')[0] in ('q', 'cl'):
        assert r['output_max_diff'] < 1e-4, (name, r)
    assert k / 3 < r['ratio'] < 3 * k, (name, r)
    assert r['shared_identical'] and r['copied'] > 10, (name, r)

print(json.dumps(report, indent=1))
print('STAGE-4 CHECK PASSED')
