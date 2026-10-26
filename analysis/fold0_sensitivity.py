# Fold-0 exclusion under both schedules, with the interaction and the schedule/dataset changes of the
# encoding effect recomputed on the same retained subset.
import numpy as np, pandas as pd
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, 'analysis')
_f = os.path.join(HERE, 'additional_analyses.py')
src = open(_f).read().split('rows = []')[0]
ns = {'__file__': _f}; exec(src, ns)
df, boot_stat, holm, B, paired_units, BLOCK, PLACE = (ns[k] for k in ['df', 'boot_stat', 'holm', 'B', 'paired_units', 'BLOCK', 'PLACE'])
mats_src = open(_f).read()
exec(mats_src[mats_src.index('def mats'):mats_src.index('for sched in')], ns); mats = ns['mats']
d0 = df[df.fold != 0]
rows = []
for sched in ('train', 'val'):
    for ds in ('CBIS', 'DMID'):
        t = d0[(d0.set == sched) & (d0.dataset == ds)]
        rng = np.random.default_rng(0); res = []
        for a, b in BLOCK + PLACE:
            u = paired_units(t, a, b); e, lo, hi, p, _ = boot_stat(u, rng)
            res.append(dict(analysis='without fold 0', schedule=sched, dataset=ds, contrast=f'{a} - {b}', diff=e, ci_low=lo, ci_high=hi, p=p, n_units=len(u)))
        r = pd.DataFrame(res); r['p_holm'] = holm(r.p.values); rows.append(r)
        # interaction (p_q - g_q) - (p_cl - g_cl), exploratory
        x = paired_units(t, 'p_q', 'g_q'); y = paired_units(t, 'p_cl', 'g_cl')
        m = x.merge(y, on=['roi_id', 'cluster', 'seed'], suffixes=('_q', '_c')); m['v'] = m.v_q - m.v_c
        e, lo, hi, p, _ = boot_stat(m[['roi_id', 'cluster', 'seed', 'v']], rng)
        rows.append(pd.DataFrame([dict(analysis='without fold 0, interaction (exploratory)', schedule=sched, dataset=ds,
                                       contrast='(p_q - g_q) - (p_cl - g_cl)', diff=e, ci_low=lo, ci_high=hi, p=p, n_units=len(m))]))
# schedule change of the placement effect, fold 0 removed
for ds in ('CBIS', 'DMID'):
    rng = np.random.default_rng(0); res = []
    for a, b in PLACE:
        u1 = paired_units(d0[(d0.set == 'train') & (d0.dataset == ds)], a, b); u2 = paired_units(d0[(d0.set == 'val') & (d0.dataset == ds)], a, b)
        m = u1.merge(u2, on=['roi_id', 'cluster', 'seed'], suffixes=('_1', '_2')); m['v'] = m.v_1 - m.v_2
        e, lo, hi, p, _ = boot_stat(m[['roi_id', 'cluster', 'seed', 'v']], rng)
        res.append(dict(analysis='without fold 0, schedule change (exploratory)', schedule='both', dataset=ds, contrast=f'({a} - {b}) primary - second', diff=e, ci_low=lo, ci_high=hi, p=p, n_units=len(m)))
    rows.append(pd.DataFrame(res))
# dataset change of the placement effect, fold 0 removed
for sched in ('train', 'val'):
    rng = np.random.default_rng(0); res = []
    for a, b in PLACE:
        Sc, Nc = mats(paired_units(d0[(d0.set == sched) & (d0.dataset == 'CBIS')], a, b)); Sd, Nd = mats(paired_units(d0[(d0.set == sched) & (d0.dataset == 'DMID')], a, b))
        est = Sc.sum() / Nc.sum() - Sd.sum() / Nd.sum()
        wc = rng.multinomial(Sc.shape[0], np.ones(Sc.shape[0]) / Sc.shape[0], size=B); wd = rng.multinomial(Sd.shape[0], np.ones(Sd.shape[0]) / Sd.shape[0], size=B)
        ws = rng.multinomial(3, np.ones(3) / 3, size=B)
        bs = np.einsum('bc,cs,bs->b', wc, Sc, ws) / np.einsum('bc,cs,bs->b', wc, Nc, ws) - np.einsum('bc,cs,bs->b', wd, Sd, ws) / np.einsum('bc,cs,bs->b', wd, Nd, ws)
        p = max(1 / B, min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
        res.append(dict(analysis='without fold 0, dataset change (exploratory)', schedule=sched, dataset='both', contrast=f'({a} - {b}) CBIS - DMID', diff=est, ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5), p=p))
    rows.append(pd.DataFrame(res))
out = pd.concat(rows, ignore_index=True); out.to_csv(os.path.join(ns['OUT'], 'fold0_excluded_both_schedules.csv'), index=False)
pd.set_option('display.width', 250); print(out.drop(columns='analysis').round(3).to_string())
