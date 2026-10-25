# Code-replacement (severing) tests with Holm correction.
import numpy as np, pandas as pd
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, 'analysis')
_f = os.path.join(HERE, 'hierarchical_stats.py')
src = open(_f).read().split('CONTRASTS = [')[0]
ns = {'__file__': _f}; exec(src, ns)
df, unit_table, boot_stat, holm = ns['df'], ns['unit_table'], ns['boot_stat'], ns['holm']
rows = []
for sched in ('train', 'val'):
    for ds in ('CBIS', 'DMID'):
        res = []
        rng = np.random.default_rng(0)
        for m in ('g_q', 'g_cl', 'g_id', 'p_q', 'p_cl', 'p_id'):
            tt = df[(df.set == sched) & (df.dataset == ds) & (df.model == m)]
            for col, nm in (('dice_const', 'const'), ('dice_swapavg', 'swap')):
                u = unit_table(tt.assign(v=tt.dice - tt[col]), 'v')
                e, lo, hi, p, _ = boot_stat(u, rng)
                res.append(dict(schedule=sched, dataset=ds, model=m, test=nm, drop=e, ci_low=lo, ci_high=hi, p=p))
        r = pd.DataFrame(res); r['p_holm'] = holm(r.p.values); rows.append(r)
out = pd.concat(rows); out.to_csv(os.path.join(ns['OUT'], 'severing_tests.csv'), index=False)
pd.set_option('display.width', 200); print(out.round(4).to_string())
