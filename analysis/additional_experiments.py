# Additional experiments (stages 3 and 4): prespecified contrasts and model levels.
#   Stage 3 (protocols/PROTOCOL_stage3_additional_experiments.txt): E1 faster-learning block (seeds 0-2), E2 shared initialization
#           of the trigonometric and identity controls (seeds 0-4).
#   Stage 4 (protocols/PROTOCOL_stage4_additional_experiments.txt): E3 block learning rate selected per block on the validation loss
#           (results/additional_experiments/E3_selection.json), E1+ faster-learning comparison with five seeds, E4 angle encoding pi*c.
# Same estimator as analysis/five_seed_statistics.py (functions taken from that file): crossed hierarchical bootstrap over seeds
# (resampled jointly) and clusters (patients; DMID: source images), B = 10,000, numpy default_rng(0) (one generator per stage),
# Holm within each family of four contrasts per test set and reference; equivalence when the 95% CI lies within +-0.02 Dice.
# Primary learning-rate schedule only.
#
# Usage (from the repository root):
#   python analysis/additional_experiments.py results/five_seed/per_image results/additional_experiments/per_image \
#          results/additional_experiments/stats
# Further folders with per-image tables can be appended (they are searched in order). CSAW-S per-image scores are not in the
# public release; without them the CSAW-S rows are skipped and the bootstrap draws of the following set (CDD-CESM) differ from
# the released stage3_contrasts.csv and stage4_contrasts.csv, which were computed with all sets.
import os, sys, glob, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'five_seed_statistics.py')).read()
exec(src[src.index('B = 10_000'):src.index('def load(name)')])          # B, EQ, holm, matrix, boot, verdict, family

*DIRS, OUT = sys.argv[1:] if len(sys.argv) > 2 else ['results/five_seed/per_image', 'results/additional_experiments/per_image',
                                                      'results/additional_experiments/stats']
os.makedirs(OUT, exist_ok=True)
SEL = json.load(open(os.path.join(HERE, '..', 'results', 'additional_experiments', 'E3_selection.json')))
SETS = ['cbis', 'frozen', 'dmid', 'csaws', 'cddcesm']
STANDARD = ['g_q', 'g_cl', 'g_tr', 'g_id', 'p_q', 'p_cl', 'p_tr', 'p_id']


def read(prefix, key):
    for d in DIRS:
        f = f'{d}/{prefix}_{key}_per_image.csv.gz'; parts = sorted(glob.glob(f'{d}/{prefix}_{key}_per_image.part*.csv.gz'))
        if os.path.exists(f): return pd.read_csv(f)
        if parts: return pd.concat([pd.read_csv(x) for x in parts], ignore_index=True)
    return None


def with_cluster(d):
    d = d[d.schedule == 'train'].copy()
    d['cluster'] = d['patient'].astype(str) if 'patient' in d and d['patient'].notna().all() else d.roi_id.str.split('_').str[0]
    return d


def references(d, key):
    return [('dice', 'consensus')] + ([(c, c.replace('dice_', '')) for c in d.columns if c.startswith('dice_annotator')]
                                       if key == 'csaws' else [])


def levels(d, M, codes, models, key, stage, rng):
    rows = []
    for m in models:
        ss, X = M[m]; r = boot(X, np.zeros_like(X), codes, rng); r.pop('boot')
        lv = dict(stage=stage, set=key, model=m, dice=r['diff'], ci_lo=r['ci_lo'], ci_hi=r['ci_hi'], n_seeds=len(ss))
        sub = d[d.model == m]
        if 'empty' in sub: lv['empty_rate'] = float(sub.groupby(['seed', 'roi_id']).empty.mean().mean())
        if 'hd95' in sub: lv['hd95_median'] = float(sub.groupby(['seed', 'roi_id']).hd95.mean().median())
        rows.append(lv)
    return rows


def run_families(M, codes, fams, rng, key, ref, stage):
    rows = []
    for label, pairs, seeds in fams:
        rows += [dict(stage=stage, set=key, reference=ref, **r) for r in family(M, codes, pairs, rng, label, seeds=seeds)]
    return rows


# ---------------------------------------------------------------- stage 3: E1 (seeds 0-2) and E2 (seeds 0-4)
E1 = [(f'{p}_q_fast', f'{p}_{b}_fast') for p in 'gp' for b in ('cl', 'tr')]
E1D = [(f'{p}_{b}_fast', f'{p}_{b}') for b in ('q', 'cl', 'tr') for p in 'gp']
E2 = [(f'{p}_q', f'{p}_{b}_shared') for p in 'gp' for b in ('tr', 'id')]
E2D = [(f'{p}_{b}_shared', f'{p}_{b}') for b in ('tr', 'id') for p in 'gp']
FAM3 = [('E1 main', E1, None), ('E1 descriptive', E1D, None), ('E2 main', E2, None), ('E2 descriptive', E2D, None)]

rng = np.random.default_rng(0); C3, L3 = [], []
for key in SETS:
    st, s3 = read('rev5', key), read('stage3', key)
    if st is None or s3 is None: print('stage 3: skipped', key); continue
    st = st[st.model.isin(STANDARD)]
    d = with_cluster(pd.concat([st, s3], ignore_index=True))
    for col, ref in references(d, key):
        M, codes = matrix(d, col)
        C3 += run_families(M, codes, FAM3, rng, key, ref, 3)
        if ref == 'consensus':
            L3 += levels(d, M, codes, [m for m in M if 'fast' in m or 'shared' in m], key, 3, rng)
    print('stage 3 done:', key, flush=True)

# ---------------------------------------------------------------- stage 4: E3 (seeds 0-2), E1+ (seeds 0-4), E4 (seeds 0-2)
def name(block, factor): return f'{block}_fast' if factor == 100 else f'{block}_fast{factor}'    # factor 100 = the E1 models
E3 = [(name(f'{p}_q', SEL[f'{p}_q']['selected']), name(f'{p}_{b}', SEL[f'{p}_{b}']['selected'])) for p in 'gp' for b in ('cl', 'tr')]
E3D = [(name(f'{p}_{b}', f), f'{p}_{b}_fast') for p in 'gp' for b in ('q', 'cl', 'tr') for f in (10, 30)]
E1P = E1
E4 = [(f'{p}_q_wide', f'{p}_{b}') for p in 'gp' for b in ('cl_fast', 'tr_wide')]
E4D = [(f'{p}_{b}_wide', f'{p}_{b}_fast') for b in ('q', 'tr') for p in 'gp']
FAM4 = [('E3 main', E3, [0, 1, 2]), ('E3 seed-0 descriptive', E3D, [0]), ('E1+ main', E1P, None),
        ('E4 main', E4, [0, 1, 2]), ('E4 descriptive', E4D, [0, 1, 2])]

rng = np.random.default_rng(0); C4, L4 = [], []
for key in SETS:
    s3, s4 = read('stage3', key), read('stage4', key)
    if s3 is None or s4 is None: print('stage 4: skipped', key); continue
    d = pd.concat([s3, s4], ignore_index=True); d = with_cluster(d[d.model.str.contains('fast|wide')])
    for col, ref in references(d, key):
        M, codes = matrix(d, col)
        C4 += run_families(M, codes, FAM4, rng, key, ref, 4)
        if ref == 'consensus':
            L4 += levels(d, M, codes, list(M), key, 4, rng)
    print('stage 4 done:', key, flush=True)

for nm_, rows in (('stage3_contrasts', C3), ('stage3_levels', L3), ('stage4_contrasts', C4), ('stage4_levels', L4)):
    pd.DataFrame(rows).to_csv(f'{OUT}/{nm_}.csv', index=False)
pd.set_option('display.width', 250)
for nm_, rows in (('stage 3', C3), ('stage 4', C4)):
    t = pd.DataFrame(rows); t = t[(t.set == 'cbis') & t.family.str.contains('main')]
    print(f'\n## {nm_}, CBIS-DDSM test folds\n', t[['family', 'contrast', 'n_seeds', 'diff', 'ci_lo', 'ci_hi', 'p_holm', 'verdict']].round(4).to_string(index=False))
