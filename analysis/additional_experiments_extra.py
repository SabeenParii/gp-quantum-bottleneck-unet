# Additional analyses of E1 (not prespecified; Online Resource Tables S29-S30), primary schedule, seeds 0-2:
#   S29 - E1 contrasts per seed (point estimates) and leaving out one seed at a time (crossed hierarchical bootstrap over the two
#         remaining seeds and the clusters);
#   S30 - learning-rate effect of the trigonometric layer at a common initialization (tr_fast - tr_shared) and the main-run
#         global circuit - Dense(4) contrast of the E1 seeds.
# Same estimator as analysis/additional_experiments.py. The point estimates equal those of Tables S29-S30; the order of the
# bootstrap draws differs from the script used for the Online Resource, so CI limits can differ by up to 0.002.
# Usage: python analysis/additional_experiments_extra.py results/five_seed/per_image results/additional_experiments/per_image \
#        results/additional_experiments/stats
import os, sys, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'five_seed_statistics.py')).read()
exec(src[src.index('B = 10_000'):src.index('def load(name)')])          # B, EQ, holm, matrix, boot, verdict, family

*DIRS, OUT = sys.argv[1:] if len(sys.argv) > 2 else ['results/five_seed/per_image', 'results/additional_experiments/per_image',
                                                      'results/additional_experiments/stats']
os.makedirs(OUT, exist_ok=True)
STANDARD = ['g_q', 'g_cl', 'g_tr', 'g_id', 'p_q', 'p_cl', 'p_tr', 'p_id']
E1 = [(f'{p}_q_fast', f'{p}_{b}_fast') for p in 'gp' for b in ('cl', 'tr')]
S30 = [('g_tr_fast', 'g_tr_shared'), ('p_tr_fast', 'p_tr_shared'), ('g_q', 'g_cl')]


def read(prefix, key):
    for d in DIRS:
        f = f'{d}/{prefix}_{key}_per_image.csv.gz'; parts = sorted(glob.glob(f'{d}/{prefix}_{key}_per_image.part*.csv.gz'))
        if os.path.exists(f): return pd.read_csv(f)
        if parts: return pd.concat([pd.read_csv(x) for x in parts], ignore_index=True)
    return None


rng = np.random.default_rng(0); rows = []
for key in ['cbis', 'frozen', 'dmid', 'csaws', 'cddcesm']:
    st, s3 = read('rev5', key), read('stage3', key)
    if st is None or s3 is None: print('skipped', key); continue
    d = pd.concat([st[st.model.isin(STANDARD)], s3], ignore_index=True); d = d[d.schedule == 'train'].copy()
    d['cluster'] = d['patient'].astype(str) if 'patient' in d and d['patient'].notna().all() else d.roi_id.str.split('_').str[0]
    M, codes = matrix(d)
    def sub(m, ss): s, X = M[m]; return X[[s.index(x) for x in ss]]
    for a, b in E1:
        for sd in range(3):
            rows.append(dict(table='S29', set=key, contrast=f'{a} - {b}', analysis=f'seed {sd}',
                             diff=float(np.nanmean(sub(a, [sd]) - sub(b, [sd])))))
        for drop in range(3):
            ss = [s for s in range(3) if s != drop]; r = boot(sub(a, ss), sub(b, ss), codes, rng); r.pop('boot')
            rows.append(dict(table='S29', set=key, contrast=f'{a} - {b}', analysis=f'without seed {drop}', **r))
    for a, b in S30:
        r = boot(sub(a, [0, 1, 2]), sub(b, [0, 1, 2]), codes, rng); r.pop('boot')
        rows.append(dict(table='S30', set=key, contrast=f'{a} - {b}', analysis='seeds 0-2', **r))
    print('done', key, flush=True)
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/stage3_extra.csv', index=False)
pd.set_option('display.width', 200)
print(R[R.set == 'cbis'][['table', 'contrast', 'analysis', 'diff', 'ci_lo', 'ci_hi']].round(3).to_string(index=False))
