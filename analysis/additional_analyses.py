# Additional analyses: interaction, between-schedule and between-dataset changes of the encoding effect,
# collapsed-run sensitivity analysis and seed-specific contrasts.
import numpy as np, pandas as pd
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, 'analysis')
_f = os.path.join(HERE, 'hierarchical_stats.py')
src = open(_f).read().split('CONTRASTS = [')[0]
ns = {'__file__': _f}; exec(src, ns)
df, unit_table, boot_stat, holm, B = ns['df'], ns['unit_table'], ns['boot_stat'], ns['holm'], ns['B']
OUT = ns['OUT']
BLOCK = [('g_q', 'g_cl'), ('g_q', 'g_id'), ('g_cl', 'g_id'), ('p_q', 'p_cl'), ('p_q', 'p_id'), ('p_cl', 'p_id')]
PLACE = [('p_q', 'g_q'), ('p_cl', 'g_cl'), ('p_id', 'g_id')]


def paired_units(t, a, b, drop_runs=()):
    """Per image x seed difference a - b; runs (fold, seed) in drop_runs are removed from BOTH models before averaging."""
    t = t[~t[['fold', 'seed']].apply(tuple, axis=1).isin(drop_runs)] if drop_runs else t
    x = unit_table(t[t.model == a]).rename(columns={'dice': 'a'}); y = unit_table(t[t.model == b]).rename(columns={'dice': 'b'})
    m = x.merge(y, on=['roi_id', 'cluster', 'seed']); m['v'] = m.a - m.b
    return m[['roi_id', 'cluster', 'seed', 'v']]


rows = []
# ---- M5: collapsed run removed from both members of every contrast (paired), primary schedule
for ds in ('CBIS', 'DMID'):
    t = df[(df.set == 'train') & (df.dataset == ds)]
    rng = np.random.default_rng(0); res = []
    for a, b in BLOCK + PLACE:
        u = paired_units(t, a, b, drop_runs=((0, 1),))
        e, lo, hi, p, _ = boot_stat(u, rng)
        res.append(dict(analysis='M5 paired exclusion of fold 0 / seed 1', schedule='train', dataset=ds, contrast=f'{a} - {b}',
                        diff=e, ci_low=lo, ci_high=hi, p=p, n_units=len(u)))
    r = pd.DataFrame(res); r['p_holm'] = holm(r.p.values); rows.append(r)

# ---- M4: fold 0 (used in the pilot) excluded: CBIS fold-0 images and DMID fold-0 models removed, primary schedule
for ds in ('CBIS', 'DMID'):
    t = df[(df.set == 'train') & (df.dataset == ds) & (df.fold != 0)]
    rng = np.random.default_rng(0); res = []
    for a, b in BLOCK + PLACE:
        u = paired_units(t, a, b); e, lo, hi, p, _ = boot_stat(u, rng)
        res.append(dict(analysis='M4 without fold 0', schedule='train', dataset=ds, contrast=f'{a} - {b}', diff=e, ci_low=lo, ci_high=hi, p=p, n_units=len(u)))
    r = pd.DataFrame(res); r['p_holm'] = holm(r.p.values); rows.append(r)

# ---- M3a: change of the placement effect between schedules, (P-G)primary - (P-G)second, same images, seed index aligned
for ds in ('CBIS', 'DMID'):
    rng = np.random.default_rng(0); res = []
    for a, b in PLACE:
        u1 = paired_units(df[(df.set == 'train') & (df.dataset == ds)], a, b)
        u2 = paired_units(df[(df.set == 'val') & (df.dataset == ds)], a, b)
        m = u1.merge(u2, on=['roi_id', 'cluster', 'seed'], suffixes=('_1', '_2')); m['v'] = m.v_1 - m.v_2
        e, lo, hi, p, _ = boot_stat(m[['roi_id', 'cluster', 'seed', 'v']], rng)
        res.append(dict(analysis='M3 schedule change of placement effect (primary - second)', schedule='both', dataset=ds,
                        contrast=f'({a} - {b}) primary - second', diff=e, ci_low=lo, ci_high=hi, p=p, n_units=len(m)))
    rows.append(pd.DataFrame(res))


# ---- M3b: change of the placement effect between datasets, (P-G)CBIS - (P-G)DMID; clusters resampled per dataset, seeds jointly
def mats(u):
    seeds = np.sort(u.seed.unique()); cl = np.sort(u.cluster.unique()); ci = {c: i for i, c in enumerate(cl)}; si = {s: i for i, s in enumerate(seeds)}
    S = np.zeros((len(cl), len(seeds))); N = np.zeros_like(S)
    for c, s, v in zip(u.cluster, u.seed, u.v): S[ci[c], si[s]] += v; N[ci[c], si[s]] += 1
    return S, N
for sched in ('train', 'val'):
    rng = np.random.default_rng(0); res = []
    for a, b in PLACE:
        Sc, Nc = mats(paired_units(df[(df.set == sched) & (df.dataset == 'CBIS')], a, b))
        Sd, Nd = mats(paired_units(df[(df.set == sched) & (df.dataset == 'DMID')], a, b))
        est = Sc.sum() / Nc.sum() - Sd.sum() / Nd.sum()
        wc = rng.multinomial(Sc.shape[0], np.ones(Sc.shape[0]) / Sc.shape[0], size=B)
        wd = rng.multinomial(Sd.shape[0], np.ones(Sd.shape[0]) / Sd.shape[0], size=B)
        ws = rng.multinomial(3, np.ones(3) / 3, size=B)
        bc = np.einsum('bc,cs,bs->b', wc, Sc, ws) / np.einsum('bc,cs,bs->b', wc, Nc, ws)
        bd = np.einsum('bc,cs,bs->b', wd, Sd, ws) / np.einsum('bc,cs,bs->b', wd, Nd, ws)
        bs = bc - bd; p = max(1 / B, min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
        res.append(dict(analysis='M3 dataset change of placement effect (CBIS - DMID)', schedule=sched, dataset='both',
                        contrast=f'({a} - {b}) CBIS - DMID', diff=est, ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5), p=p))
    rows.append(pd.DataFrame(res))

out = pd.concat(rows, ignore_index=True); out.to_csv(f'{OUT}/additional_analyses.csv', index=False)
pd.set_option('display.width', 250); print(out.round(4).to_string())

# ---- M2: seed-specific circuit - Dense(4) means (paired, all images of a seed), per setting
sd = []
for sched in ('train', 'val'):
    for ds in ('CBIS', 'DMID'):
        t = df[(df.set == sched) & (df.dataset == ds)]
        for a, b in (('g_q', 'g_cl'), ('p_q', 'p_cl')):
            u = paired_units(t, a, b)
            for s, g in u.groupby('seed'): sd.append(dict(schedule=sched, dataset=ds, contrast=f'{a} - {b}', seed=s, mean_diff=g.v.mean()))
sd = pd.DataFrame(sd); sd.to_csv(f'{OUT}/seed_contrasts.csv', index=False)
print(sd.pivot_table(index=['schedule', 'dataset', 'contrast'], columns='seed', values='mean_diff').round(4).to_string())
