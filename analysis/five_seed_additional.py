# Additional analyses of notebook 33 (not prespecified): A circuit function class / entanglement (initial vs trained); B noisy re-evaluation (crossed bootstrap, B = 10,000, rng(0), seeds x clusters); C training dynamics and seed variability. Reads and writes results/five_seed/additional/.
import numpy as np, pandas as pd, os
import sys
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'five_seed', 'additional'); os.chdir(HERE)
NB = 10_000

def boot(mat, codes, rng):
    """mat: (seeds, items) per-image deltas; codes: cluster index per item. Crossed bootstrap over seeds and clusters."""
    S, n = mat.shape; ncl = codes.max() + 1
    sums = np.vstack([np.bincount(codes, r, ncl) for r in mat]); cnt = np.bincount(codes, minlength=ncl)
    est = mat.mean()
    out = np.empty(NB)
    for b in range(NB):
        s = rng.integers(0, S, S); w = np.bincount(rng.integers(0, ncl, ncl), minlength=ncl)
        out[b] = (sums[s].mean(0) @ w) / (cnt @ w)
    lo, hi = np.percentile(out, [2.5, 97.5]); p = min(1.0, 2 * min((out <= 0).mean(), (out >= 0).mean())); p = max(p, 1 / NB)
    return est, lo, hi, p

# ---------------- A
A = pd.read_csv('A_circuit_function.csv')
A['placement'] = A.model.str[0]
cols = ['code_sd', 'out_sd', 'r2_order1', 'r2_order2', 'r2_trig_fit', 'r2_dense_fit', 'meyer_wallach']
SA = A.groupby(['model', 'schedule', 'state'])[cols].agg(['median', lambda v: v.quantile(.25), lambda v: v.quantile(.75)])
SA.columns = [f'{c}_{s}' for c, s in zip(SA.columns.get_level_values(0), ['med', 'q25', 'q75'] * len(cols))]
SA.round(4).to_csv('A_summary.csv')
k = ['run', 'schedule']
t = A[A.state == 'trained'].set_index(k)[cols]; i = A[A.state == 'init'].set_index(k)[cols]
D = (t - i).abs(); D.describe().round(4).to_csv('A_trained_minus_init.csv')
print('A: median by model/schedule/state'); print(A.groupby(['model', 'schedule', 'state'])[cols].median().round(3).to_string())
print('A: max |trained - init| per metric'); print(D.max().round(4).to_string())

# ---------------- B
Bn = pd.read_csv('B_noise_per_image.csv.gz')
Bn['cluster'] = np.where(Bn.dataset == 'CBIS', Bn.roi_id.str.extract(r'(P_\d+)')[0], Bn.roi_id.str.split('_').str[0])
rows = []; rng = np.random.default_rng(0)
for (ds, sch, mod), g in Bn.groupby(['dataset', 'schedule', 'model']):
    g = g.groupby(['seed', 'roi_id', 'cluster'], as_index=False)[['dice', 'dice_low', 'dice_high']].mean()
    items = g[['roi_id', 'cluster']].drop_duplicates('roi_id').sort_values('roi_id').reset_index(drop=True)
    codes = pd.factorize(items.cluster)[0]
    for lvl in ('low', 'high'):
        piv = g.pivot(index='seed', columns='roi_id', values=f'dice_{lvl}').reindex(columns=items.roi_id) - \
              g.pivot(index='seed', columns='roi_id', values='dice').reindex(columns=items.roi_id)
        est, lo, hi, p = boot(piv.values, codes, rng)
        rows.append(dict(dataset=ds, schedule=sch, model=mod, noise=lvl, dice_exact=g.dice.mean(), delta=est, ci_lo=lo, ci_hi=hi, p=p,
                         n_items=len(items), n_clusters=codes.max() + 1))
RB = pd.DataFrame(rows); RB.to_csv('B_noise_summary.csv', index=False)
print('B: Dice change under noise (noisy - exact), 95% CI'); print(RB.round(4).to_string(index=False))
print('B: max |exact_np - exact|', float((Bn.dice_exact_np - Bn.dice).abs().max()))

# ---------------- C
C = pd.read_csv('C_training_dynamics.csv')
C['seed_block'] = np.where(C.seed <= 2, '0-2', '3-4')
sd = C.groupby(['schedule', 'model', 'fold']).dice_test_fold.std().reset_index(name='sd')
# bootstrap over folds of the mean seed SD, and paired fold-level differences q - cl and q - tr
rowsC = []; rng = np.random.default_rng(0)
for sch in ('train', 'val'):
    for pl in ('g', 'p'):
        P = sd[(sd.schedule == sch)].pivot(index='fold', columns='model', values='sd')
        for ctl in ('cl', 'tr', 'id'):
            if f'{pl}_{ctl}' not in P: continue
            d = (P[f'{pl}_q'] - P[f'{pl}_{ctl}']).values
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(NB)])
            rowsC.append(dict(schedule=sch, contrast=f'{pl}_q - {pl}_{ctl}', seed_sd_q=P[f'{pl}_q'].mean(), seed_sd_ctl=P[f'{pl}_{ctl}'].mean(),
                              diff=d.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
RC = pd.DataFrame(rowsC); RC.to_csv('C_seed_sd_contrasts.csv', index=False)
print('C: seed-to-seed SD of test-fold Dice (mean over folds), circuit minus control, 95% CI over folds'); print(RC.round(4).to_string(index=False))
dyn = C.groupby(['schedule', 'model']).agg(best_epoch_median=('best_epoch', 'median'), q25=('best_epoch', lambda v: v.quantile(.25)),
                                          q75=('best_epoch', lambda v: v.quantile(.75)), min_test_dice=('dice_test_fold', 'min'))
dyn.to_csv('C_dynamics.csv'); print(dyn.round(3).to_string())
