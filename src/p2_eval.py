"""Evaluate saved runs on external / frozen test sets (no training; resumable).

    python p2_eval.py --runs <results>/p2_jit_runs_train <results>/p2_jit_runs_val \
        --set frozen=<.../cbis_mass_256_jit_frozen> --set dmid=<.../dmid_mass_256_jit_v2/roi> --set csaws=<.../csaws_mass_256_jit>

Each set folder holds images/<roi_id>.png, masks/<roi_id>.png and metadata.csv (column roi_id; patient if available).
For every finished run (summary.json and best.weights.h5) and every set, writes <run>/ext_<set>.csv with per-image Dice, IoU,
HD95 and an empty-prediction flag (threshold 0.5, as in the study). Existing files are skipped.
"""
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import tensorflow as tf
import qseg2
import qseg2_rev4
from qseg.config import Config

qseg2_rev4.patch()


def load_set(d):
    meta = pd.read_csv(os.path.join(d, 'metadata.csv'))
    rd = lambda p, m: (tf.image.decode_png(tf.io.read_file(p), channels=1).numpy() > 127).astype(np.float32) if m else \
        tf.image.decode_png(tf.io.read_file(p), channels=1).numpy().astype(np.float32) / 255.0
    x = np.stack([rd(os.path.join(d, 'images', f'{r}.png'), False) for r in meta.roi_id])
    y = np.stack([rd(os.path.join(d, 'masks', f'{r}.png'), True) for r in meta.roi_id])
    return x, y, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='run roots (p2_jit_runs_train, p2_jit_runs_val, ...)')
    ap.add_argument('--set', action='append', required=True, help='name=folder')
    a = ap.parse_args()
    cfg = Config(work_dir='.', max_epochs=200)
    sets = {k: load_set(v) for k, v in (s.split('=', 1) for s in a.set)}
    print({k: v[0].shape for k, v in sets.items()}, flush=True)
    runs = [d for r in a.runs for d in sorted(glob.glob(os.path.join(r, '*_fold*_seed*')))
            if os.path.exists(os.path.join(d, 'summary.json')) and os.path.exists(os.path.join(d, 'best.weights.h5'))]
    for d in runs:
        todo = [k for k in sets if not os.path.exists(os.path.join(d, f'ext_{k}.csv'))]
        if not todo:
            continue
        s = json.load(open(os.path.join(d, 'summary.json')))
        tf.keras.backend.clear_session()
        m = qseg2.build_p2_model(s['model'], (cfg.image_size, cfg.image_size, cfg.in_channels), cfg.n_qubits, cfg.n_q_layers)
        m.load_weights(os.path.join(d, 'best.weights.h5'))
        for k in todo:
            x, y, meta = sets[k]
            p = m.predict(x, batch_size=cfg.batch_size, verbose=0)
            t = qseg2.per_image_scores(y, p, cfg.threshold)
            t.insert(0, 'roi_id', meta.roi_id.values)
            if 'patient' in meta: t.insert(1, 'patient', meta.patient.values)
            t['empty'] = (p[..., 0] >= cfg.threshold).sum((1, 2)) == 0
            t['model'], t['fold'], t['seed'] = s['model'], int(s['fold']), int(s['seed'])
            t.to_csv(os.path.join(d, f'ext_{k}.csv'), index=False)
        print(os.path.basename(d), 'done:', todo, flush=True)


if __name__ == '__main__':
    main()
