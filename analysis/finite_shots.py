# Finite-shot evaluation of the circuit models (notebook 23): Dice change from exact expectation values to
# N-shot estimates (N = 100, 1,000, 10,000; three draws each), and the circuit contrasts recomputed with N-shot circuit Dice.
# Same crossed hierarchical bootstrap as hierarchical_stats.py (10,000 replicates, rng(0)).
# --joint: the joint-bitstring re-evaluation of notebook 24 (all four Z expectations from the same N bitstrings)
#          instead of the independent per-qubit sampling of notebook 23; outputs get the suffix _joint.
import numpy as np, pandas as pd, os, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_f = os.path.join(ROOT, 'analysis', 'hierarchical_stats.py')
ns = {'__file__': _f}; exec(open(_f).read().split('CONTRASTS = [')[0], ns)
df, unit_table, boot_stat, holm, OUT = ns['df'], ns['unit_table'], ns['boot_stat'], ns['holm'], ns['OUT']
_ap = argparse.ArgumentParser(); _ap.add_argument('--joint', action='store_true'); _a, _ = _ap.parse_known_args()
TAG = '_joint' if _a.joint else ''
LEVELS, DRAWS = (100, 1000, 10000), 3
K = ['set', 'model', 'fold', 'seed', 'dataset', 'roi_id']
sh = pd.read_csv(os.path.join(ROOT, 'results', 'finite_shots_joint_per_image.csv.gz' if _a.joint else 'finite_shots_per_image.csv.gz'))
for n in LEVELS:
    sh[f'dice_s{n}'] = sh[[f'dice_s{n}_d{r}' for r in range(DRAWS)]].mean(axis=1)
    sh[f'flip_s{n}'] = sh[[f'flip_s{n}_d{r}' for r in range(DRAWS)]].mean(axis=1)
# exact predictions of notebook 23 vs the stored results (numerical differences between evaluation sessions)
m = sh.merge(df[K + ['dice']], on=K, how='left')
assert m.dice.notna().all()
chk = m.assign(d=m.dice_exact - m.dice).groupby(['dataset']).d.agg(mean_diff='mean', max_abs=lambda x: x.abs().max(),
                                                                    share_changed=lambda x: (x.abs() > 1e-6).mean())
chk.to_csv(os.path.join(OUT, f'finite_shots{TAG}_exact_check.csv'))
sh = m.assign(cluster=df.set_index(K).loc[pd.MultiIndex.from_frame(m[K]), 'cluster'].values)

# 1) Dice drop exact - N shots, per schedule x dataset x circuit model (Holm over the 6 tests of each schedule x dataset)
rows = []
for sched in ('train', 'val'):
    for ds in ('CBIS', 'DMID'):
        rng = np.random.default_rng(0); res = []
        for mod in ('g_q', 'p_q'):
            t = sh[(sh.set == sched) & (sh.dataset == ds) & (sh.model == mod)]
            for n in LEVELS:
                e, lo, hi, p, _ = boot_stat(unit_table(t.assign(v=t.dice_exact - t[f'dice_s{n}']), 'v'), rng)
                dr = t[[f'dice_s{n}_d{r}' for r in range(DRAWS)]].groupby([t.fold, t.seed]).mean().std(axis=1).mean()
                res.append(dict(schedule=sched, dataset=ds, model=mod, shots=n, exact=t.dice_exact.mean(), drop=e, ci_low=lo, ci_high=hi, p=p,
                                flip=t[f'flip_s{n}'].mean(), run_draw_sd=dr))
        r = pd.DataFrame(res); r['p_holm'] = holm(r.p.values); rows.append(r)
drop = pd.concat(rows); drop.to_csv(os.path.join(OUT, f'finite_shots{TAG}_drop.csv'), index=False)

# 2) Circuit contrasts with N-shot circuit Dice: circuit Dice = stored exact Dice + (N-shot - exact) of notebook 23
CON = [('g_q', 'g_cl'), ('g_q', 'g_id'), ('p_q', 'p_cl'), ('p_q', 'p_id'), ('p_q', 'g_q')]
rows = []
for sched in ('train', 'val'):
    for ds in ('CBIS', 'DMID'):
        t = df[(df.set == sched) & (df.dataset == ds)]
        for lab in ['exact'] + [f's{n}' for n in LEVELS]:
            tt = t.copy()
            if lab != 'exact':
                s = sh[(sh.set == sched) & (sh.dataset == ds)].set_index(K)
                idx = pd.MultiIndex.from_frame(tt[K]); hit = idx.isin(s.index)
                tt.loc[hit, 'dice'] = tt.loc[hit, 'dice'].values + (s.loc[idx[hit], f'dice_{lab}'] - s.loc[idx[hit], 'dice_exact']).values
            rng = np.random.default_rng(0); res = []
            for a, b in CON:
                x = unit_table(tt[tt.model == a]).merge(unit_table(tt[tt.model == b]), on=['roi_id', 'cluster', 'seed'], suffixes=('_a', '_b'))
                e, lo, hi, p, _ = boot_stat(x.assign(v=x.dice_a - x.dice_b)[['roi_id', 'cluster', 'seed', 'v']], rng)
                res.append(dict(schedule=sched, dataset=ds, circuit=lab, contrast=f'{a} - {b}', diff=e, ci_low=lo, ci_high=hi, p=p))
            rows += res
con = pd.DataFrame(rows); con.to_csv(os.path.join(OUT, f'finite_shots{TAG}_contrasts.csv'), index=False)
pd.set_option('display.width', 220)
print(chk.to_string()); print(drop.round(4).to_string()); print(con.round(4).to_string())
