"""One training run from the command line (same code path as notebooks 13 and 27; for batch/SLURM use).

    python p2_run.py --model g_q --fold 0 --seed 3 --schedule train \
        --cbis /path/data/processed/cbis_mass_256_jit --out /path/results

Runs go to <out>/p2_jit_runs_<schedule>/<model>_fold<k>_seed<s>/ (or to --root) exactly as in the notebooks (summary.json, history.csv,
per-image CSV, best.weights.h5 without optimizer state). A finished run (summary.json present) is skipped, so a job array can
simply be resubmitted. Settings: Config defaults of the study with max_epochs=200; learning rate halved after 4 epochs without
improvement of the training loss (schedule 'train') or of the validation loss (schedule 'val'). The software and GPU of every
run are written to <run>/env.json.
"""
import argparse, json, os, platform, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import tensorflow as tf
import qseg2
import qseg2_rev4
from qseg.config import Config

qseg2_rev4.patch()                         # adds g_tr / p_tr


def use_schedule(schedule):
    """As notebook 27, cell 1: ReduceLROnPlateau watches 'loss' (train) or 'val_loss' (val)."""
    base = tf.keras.callbacks.ReduceLROnPlateau
    while getattr(base, '_p2_switch', False):
        base = base.__mro__[1]
    monitor = {'train': 'loss', 'val': 'val_loss'}[schedule]

    class SwitchPlateau(base):
        _p2_switch = True

        def __init__(self, monitor_=None, **kw):
            kw.pop('monitor', None)
            super().__init__(monitor=monitor, **kw)

    tf.keras.callbacks.ReduceLROnPlateau = SwitchPlateau


def shrink_weights(cfg, run_dir, name):
    """As notebook 27, cell 2: weight file without optimizer state, verified weight by weight."""
    big, small = f'{run_dir}/best.weights.h5', f'{run_dir}/best_small.weights.h5'
    if not os.path.exists(big) or os.path.exists(f'{run_dir}/.shrunk'):
        return
    if os.path.getsize(big) < 60e6:
        open(f'{run_dir}/.shrunk', 'w').close(); return
    shape = (cfg.image_size, cfg.image_size, cfg.in_channels)
    tf.keras.backend.clear_session()
    m1 = qseg2.build_p2_model(name, shape, cfg.n_qubits, cfg.n_q_layers); m1.load_weights(big); m1.save_weights(small)
    m2 = qseg2.build_p2_model(name, shape, cfg.n_qubits, cfg.n_q_layers); m2.load_weights(small)
    assert all(np.array_equal(a, b) for a, b in zip(m1.get_weights(), m2.get_weights())), f'{run_dir}: weight check failed'
    os.replace(small, big); open(f'{run_dir}/.shrunk', 'w').close()


def env_info():
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        gpu = 'unknown'
    b = tf.sysconfig.get_build_info()
    return dict(python=platform.python_version(), tensorflow=tf.__version__, keras=tf.keras.__version__ if hasattr(tf.keras, '__version__') else '',
                cuda=b.get('cuda_version'), cudnn=b.get('cudnn_version'), gpu=gpu, host=platform.node())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(qseg2.MODELS2))
    ap.add_argument('--fold', type=int, required=True, choices=range(5))
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--schedule', required=True, choices=('train', 'val'))
    ap.add_argument('--cbis', required=True, help='folder with folds.csv, images/, masks/')
    ap.add_argument('--out', required=True, help='results root')
    ap.add_argument('--root', default=None, help='run folder root (default <out>/p2_jit_runs_<schedule>)')
    a = ap.parse_args()

    root = a.root or os.path.join(a.out, f'p2_jit_runs_{a.schedule}')
    run = f'{a.model}_fold{a.fold}_seed{a.seed}'
    d = os.path.join(root, run)
    if os.path.exists(os.path.join(d, 'summary.json')):
        print('already finished:', run); return
    os.makedirs(d, exist_ok=True)
    json.dump(env_info(), open(os.path.join(d, 'env.json'), 'w'), indent=2)

    use_schedule(a.schedule)
    cfg = Config(work_dir=a.out, max_epochs=200)
    if not os.path.exists(os.path.join(root, 'train_config.json')):    # never overwrite the record of earlier runs
        cfg.to_json(os.path.join(root, 'train_config.json'))
    if not os.path.exists(os.path.join(root, 'schedule.json')):
        json.dump({'schedule': a.schedule}, open(os.path.join(root, 'schedule.json'), 'w'))
    data = qseg2.load_cbis(a.cbis)
    t0 = time.time()
    s = qseg2.run_fold(cfg, data, a.model, a.fold, a.seed, out_root=root, min_epochs=15, verbose=2)
    shrink_weights(cfg, d, a.model)
    print(f'{run} [{a.schedule}]: Dice {s["dice_per_image"]:.4f}, {s["epochs_run"]} epochs, {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
