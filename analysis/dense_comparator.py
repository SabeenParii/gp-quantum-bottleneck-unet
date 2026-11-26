# Descriptive analysis (not specified in the protocols): effect of the faster block learning rate on the global dense
# block and on the global circuit, and the trained circuits against the dense block at the shared rate (main text,
# Section 4.4). Same crossed hierarchical bootstrap as analysis/five_seed_statistics.py, CBIS-DDSM folds, primary schedule.
# Usage: python analysis/dense_comparator.py   -> results/additional_experiments/stats/dense_comparator.csv
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, 'analysis', 'five_seed_statistics.py')).read()
exec(src[src.index('B = 10_000'):src.index('def load(name)')])          # B, EQ, holm, matrix, boot, verdict, family
R = os.path.join(ROOT, 'results')
m = pd.read_csv(f'{R}/five_seed/per_image/rev5_cbis_per_image.csv.gz'); m = m[m.schedule == 'train']
s3 = pd.read_csv(f'{R}/additional_experiments/per_image/stage3_cbis_per_image.csv.gz')
s4 = pd.read_csv(f'{R}/additional_experiments/per_image/stage4_cbis_per_image.csv.gz')
cols = ['roi_id', 'patient', 'dice', 'model', 'seed']
allp = pd.concat([m[cols], s3[cols], s4[cols]]).drop_duplicates(['roi_id', 'model', 'seed'])
rois = sorted(allp.roi_id.unique())
pat = allp.drop_duplicates('roi_id').set_index('roi_id').patient.reindex(rois).values


def M(model, seeds):
    return allp[allp.model == model].pivot_table(index='seed', columns='roi_id', values='dice').reindex(index=seeds, columns=rois).values


rng = np.random.default_rng(0); rows = []
for a, b, seeds in [('g_cl_fast', 'g_cl', [0, 1, 2, 3, 4]), ('g_q_fast', 'g_q', [0, 1, 2, 3, 4]), ('g_q_fast', 'g_cl', [0, 1, 2, 3, 4]),
                    ('g_q_wide', 'g_cl', [0, 1, 2]), ('g_cl_fast', 'g_cl', [0, 1, 2]), ('g_q_wide', 'g_q', [0, 1, 2])]:
    r = boot(M(a, seeds), M(b, seeds), pat, rng)
    rows.append(dict(contrast=f'{a} - {b}', seeds=''.join(map(str, seeds)), diff=r['diff'], ci_lo=r['ci_lo'], ci_hi=r['ci_hi'], p=r['p']))
out = pd.DataFrame(rows)
out.to_csv(f'{R}/additional_experiments/stats/dense_comparator.csv', index=False)
print(out.round(4).to_string(index=False))
