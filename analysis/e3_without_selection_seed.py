# E3 contrasts on the CBIS-DDSM test folds with and without seed 0, the seed that also selected the block learning-rate
# factors (descriptive; not prespecified). Same estimator as analysis/additional_experiments.py: crossed hierarchical bootstrap
# over seeds and patients, B = 10,000, numpy default_rng(0), Holm within the four E3 contrasts.
# With seeds 0-2 the output reproduces the E3 rows of results/additional_experiments/stats/stage4_contrasts.csv.
# Usage (from the repository root): python analysis/e3_without_selection_seed.py
import os, json, numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
src = open(f'{R}/analysis/five_seed_statistics.py').read()
exec(src[src.index('B = 10_000'):src.index('def load(name)')])          # B, EQ, holm, matrix, boot, verdict, family
SEL = json.load(open(f'{R}/results/additional_experiments/E3_selection.json'))
def read(p, k): return pd.read_csv(f'{R}/results/additional_experiments/per_image/{p}_{k}_per_image.csv.gz')
def name(b, f): return f'{b}_fast' if f == 100 else f'{b}_fast{f}'
E3 = [(name(f'{p}_q', SEL[f'{p}_q']['selected']), name(f'{p}_{b}', SEL[f'{p}_{b}']['selected'])) for p in 'gp' for b in ('cl', 'tr')]
d = pd.concat([read('stage3', 'cbis'), read('stage4', 'cbis')], ignore_index=True)
d = d[(d.schedule == 'train') & d.model.str.contains('fast|wide')].copy()
d['cluster'] = d['patient'].astype(str) if 'patient' in d and d['patient'].notna().all() else d.roi_id.str.split('_').str[0]
M, codes = matrix(d, 'dice')
rows = []
for seeds in ([0, 1, 2], [1, 2]):
    rng = np.random.default_rng(0)
    rows += [dict(seeds='-'.join(map(str, seeds)), **{k: v for k, v in r.items() if k != 'boot'}) for r in family(M, codes, E3, rng, 'E3', seeds=seeds)]
t = pd.DataFrame(rows)
print(t[['seeds', 'contrast', 'diff', 'ci_lo', 'ci_hi', 'p_holm', 'verdict']].round(4).to_string(index=False))
t.to_csv(f'{R}/results/additional_experiments/stats/E3_without_selection_seed.csv', index=False)
