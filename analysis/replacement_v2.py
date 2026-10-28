# Code-replacement tests, revised protocol (notebook 24): the bank holds the codes of all 240 unique training images of
# the fold; const = their exact mean (per cell for patch-wise models); swap draw r of case i uses training image
# default_rng(int(md5(f"{case_id}|{r}")[:8], 16)).integers(240) of the fold's sorted training identifiers, so donors do not
# depend on activations or batch composition and are the same for every model and seed of a fold.
# Same crossed hierarchical bootstrap as hierarchical_stats.py (10,000 replicates, rng(0)); Holm over the 12 tests of each
# schedule and dataset. Drops are paired with the normal prediction recomputed in the same evaluation.
# Also: exploratory paired comparisons of drops between blocks at the same placement (common donors), not Holm-adjusted.
import numpy as np, pandas as pd, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_f = os.path.join(ROOT, 'analysis', 'hierarchical_stats.py')
ns = {'__file__': _f}; exec(open(_f).read().split('CONTRASTS = [')[0], ns)
df, unit_table, boot_stat, holm, OUT = ns['df'], ns['unit_table'], ns['boot_stat'], ns['holm'], ns['OUT']
K = ['set', 'model', 'fold', 'seed', 'dataset', 'roi_id']
SW = [f'dice_swap{r}' for r in range(10)]
ra = pd.read_csv(os.path.join(ROOT, 'results', 'replacement_v2_per_image.csv.gz'))
m = ra.merge(df[K + ['dice', 'cluster']].rename(columns={'dice': 'dice_stored'}), on=K, how='inner')
assert len(m) == len(ra[ra.roi_id.isin(df.roi_id)]), 'unmatched rows'
m['drop_const'] = m.dice - m.dice_const
m['drop_swap'] = m.dice - m[SW].mean(axis=1)
chk = m.assign(d=m.dice - m.dice_stored).groupby('dataset').d.agg(max_abs=lambda x: x.abs().max(), share_changed=lambda x: (x.abs() > 1e-6).mean())
chk.to_csv(os.path.join(OUT, 'replacement_v2_normal_check.csv'))

rows, cmp_rows, dist = [], [], []
for sched in ('train', 'val'):
    for ds in ('CBIS', 'DMID'):
        res = []; rng = np.random.default_rng(0)
        for mdl in ('g_q', 'g_cl', 'g_id', 'p_q', 'p_cl', 'p_id'):
            tt = m[(m.set == sched) & (m.dataset == ds) & (m.model == mdl)]
            for col, nm in (('drop_const', 'const'), ('drop_swap', 'swap')):
                e, lo, hi, p, _ = boot_stat(unit_table(tt.assign(v=tt[col]), 'v'), rng)
                res.append(dict(schedule=sched, dataset=ds, model=mdl, test=nm, drop=e, ci_low=lo, ci_high=hi, p=p,
                                draw_sd=tt.groupby(['fold', 'seed'])[SW].mean().std(axis=1).mean() if nm == 'swap' else np.nan))
            dist.append(dict(schedule=sched, dataset=ds, model=mdl,
                             const_gt05=(tt.drop_const > 0.05).mean(), const_lt_m05=(tt.drop_const < -0.05).mean(),
                             swap_gt05=(tt.drop_swap > 0.05).mean(), swap_lt_m05=(tt.drop_swap < -0.05).mean(),
                             flip_const=tt.flip_const.mean(), flip_swap=tt.flip_swap.mean(),
                             absdp_const=tt.absdp_const.mean(), absdp_swap=tt.absdp_swap.mean()))
        r = pd.DataFrame(res); r['p_holm'] = holm(r.p.values); rows.append(r)
        rng = np.random.default_rng(0)
        for a, b in (('p_q', 'p_cl'), ('p_q', 'p_id'), ('g_q', 'g_cl'), ('g_q', 'g_id')):
            for col, nm in (('drop_const', 'const'), ('drop_swap', 'swap')):
                t = m[(m.set == sched) & (m.dataset == ds)]
                x = unit_table(t[t.model == a], col).merge(unit_table(t[t.model == b], col), on=['roi_id', 'cluster', 'seed'], suffixes=('_a', '_b'))
                e, lo, hi, p, _ = boot_stat(x.assign(v=x[f'{col}_a'] - x[f'{col}_b'])[['roi_id', 'cluster', 'seed', 'v']], rng)
                cmp_rows.append(dict(schedule=sched, dataset=ds, contrast=f'{a} - {b}', test=nm, diff=e, ci_low=lo, ci_high=hi, p=p))
out = pd.concat(rows); out.to_csv(os.path.join(OUT, 'replacement_v2_tests.csv'), index=False)
cmp_ = pd.DataFrame(cmp_rows); cmp_.to_csv(os.path.join(OUT, 'replacement_v2_drop_contrasts.csv'), index=False)
dist = pd.DataFrame(dist); dist.to_csv(os.path.join(OUT, 'replacement_v2_distribution.csv'), index=False)
# comparison with the original protocol (256-entry bank sampled with replacement, activation-seeded donors)
old = pd.read_csv(os.path.join(OUT, 'severing_tests.csv'))
both = out.merge(old, on=['schedule', 'dataset', 'model', 'test'], suffixes=('', '_v1'))
both[['schedule', 'dataset', 'model', 'test', 'drop', 'drop_v1']].assign(diff=both['drop'] - both['drop_v1']).to_csv(
    os.path.join(OUT, 'replacement_v2_vs_v1.csv'), index=False)
pd.set_option('display.width', 220)
print(chk.to_string()); print(out.round(4).to_string()); print(cmp_.round(4).to_string()); print(dist.round(3).to_string())
print('largest |v2 - v1| drop:', round(float((both['drop'] - both['drop_v1']).abs().max()), 4))
