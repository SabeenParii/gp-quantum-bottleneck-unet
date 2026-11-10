"""Pre-flight check of the stage-3 models (no data, a few seconds on a GPU). Stops with an error if any check fails.

1. *_fast models: the effective initial block weights equal those of the standard model of the same seed, and the
   output of a fast model equals that of the standard model on random input (the reparameterization changes nothing at start).
2. Copied initialization (g_tr_fast, p_tr_fast, *_shared): every layer shared with the circuit model is identical.
3. One Adam step (learning rate 1e-4) on the same batch moves the effective block weights of a fast model about 100 times
   as far as those of the standard model.
"""
import json, sys
import numpy as np
import tensorflow as tf
import qseg2
import qseg2_stage3 as S

S.patch()
SHAPE = (256, 256, 1)
rng = np.random.default_rng(0)
x = rng.random((2,) + SHAPE).astype('float32'); y = (rng.random((2,) + SHAPE) > 0.7).astype('float32')
report = {}


def build(name, seed, copy=True):
    tf.keras.backend.clear_session()
    qseg2.set_seeds(seed)
    S.INIT_SEED = seed if copy else None
    return qseg2.build_p2_model(name, SHAPE, 4, 2)


def slot(m, name):
    for l in m.layers:
        if l.name in S.SLOT_LAYERS:
            return S.effective(l, name)


def one_step(m):
    opt = tf.keras.optimizers.Adam(1e-4)
    with tf.GradientTape() as t:
        loss = qseg2.bce_dice_loss(y, m(x, training=False))
    g = t.gradient(loss, m.trainable_weights)
    opt.apply_gradients(zip(g, m.trainable_weights))


for p in ('g', 'p'):
    for blk in ('q', 'cl'):
        std, fast = f'{p}_{blk}', f'{p}_{blk}_fast'
        ms = build(std, 1, copy=False); a0 = slot(ms, std); out_s = ms(x, training=False).numpy()
        one_step(ms); da = np.abs(slot(ms, std) - a0).max()
        mf = build(fast, 1); b0 = slot(mf, fast); out_f = mf(x, training=False).numpy()
        one_step(mf); db = np.abs(slot(mf, fast) - b0).max()
        rec = dict(init_max_diff=float(np.abs(a0 - b0).max()), output_max_diff=float(np.abs(out_s - out_f).max()),
                   step_std=float(da), step_fast=float(db), ratio=float(db / max(da, 1e-12)),
                   shared_identical=S.INIT_RECORD.get('shared_identical'))
        report[fast] = rec
        assert rec['init_max_diff'] < 1e-5, (fast, rec)
        assert rec['output_max_diff'] < 1e-4, (fast, rec)
        assert 30 < rec['ratio'] < 1000, (fast, rec)
    for name in (f'{p}_tr_fast', f'{p}_tr_shared', f'{p}_id_shared'):
        build(name, 1)
        rec = dict(copied=len(S.INIT_RECORD['copied_layers']), shared_identical=S.INIT_RECORD['shared_identical'],
                   differed_before=sum(v > 0 for v in S.INIT_RECORD['max_diff_before_copy'].values()))
        report[name] = rec
        assert rec['shared_identical'] and rec['copied'] > 10, (name, rec)
    tr_std = build(f'{p}_tr', 1, copy=False); t0 = slot(tr_std, f'{p}_tr')
    trf = build(f'{p}_tr_fast', 1); assert np.abs(slot(trf, f'{p}_tr_fast') - t0).max() < 1e-5, 'trig fast init differs'

print(json.dumps(report, indent=1))
print('STAGE-3 CHECK PASSED')
