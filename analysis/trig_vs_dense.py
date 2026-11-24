# Trigonometric block versus Dense(4) when both learn 100 times faster (A1 = E1/E1+, five seeds) and with the doubled angle
# range of the trigonometric block (A4 = E4, seeds 0-2). Descriptive contrasts, not in the protocols (main text, Table 5,
# rows marked as descriptive). Same estimator as analysis/additional_experiments.py; numpy default_rng(1); Holm is not used.
# Usage (from the repository root): python analysis/trig_vs_dense.py
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'five_seed_statistics.py')).read()
exec(src[src.index('B = 10_000'):src.index('def load(name)')])
DIRS = [os.path.join(HERE, '..', 'results', 'additional_experiments', 'per_image')]
OUT = os.path.join(HERE, '..', 'results', 'additional_experiments', 'stats', 'trig_vs_dense.csv')
import glob
def read(prefix, key):
    for d in DIRS:
        f = f'{d}/{prefix}_{key}_per_image.csv.gz'; parts = sorted(glob.glob(f'{d}/{prefix}_{key}_per_image.part*.csv.gz'))
        if os.path.exists(f): return pd.read_csv(f)
        if parts: return pd.concat([pd.read_csv(x) for x in parts], ignore_index=True)
def with_cluster(d):
    d = d[d.schedule == 'train'].copy()
    d['cluster'] = d['patient'].astype(str) if 'patient' in d and d['patient'].notna().all() else d.roi_id.str.split('_').str[0]
    return d
rng = np.random.default_rng(1); rows = []
for key in ('cbis', 'frozen'):
    d = pd.concat([read('stage3', key), read('stage4', key)], ignore_index=True)
    d = with_cluster(d[d.model.isin(['g_tr_fast', 'g_cl_fast', 'g_tr_wide'])])
    M, codes = matrix(d, 'dice')
    for a, b, seeds, exp in (('g_tr_fast', 'g_cl_fast', None, 'A1'), ('g_tr_wide', 'g_cl_fast', [0, 1, 2], 'A4')):
        r = family(M, codes, [(a, b)], rng, exp, seeds=seeds)[0]
        rows.append(dict(set=key, experiment=exp, **{k: v for k, v in r.items() if k != 'boot'}))
t = pd.DataFrame(rows); t.to_csv(OUT, index=False)
print(t[['set', 'experiment', 'contrast', 'n_seeds', 'diff', 'ci_lo', 'ci_hi']].round(3).to_string(index=False))
