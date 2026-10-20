# 07 · Sensitivity analysis (CPU): main seed-0 runs vs. the 02c runs with the learning-rate schedule on the
# training loss. Same seed, folds and models; only the schedule differs. Reads only; writes results/p2_sensitivity_lr/.
import os, glob, re, json
import numpy as np, pandas as pd

PROJECT = globals().get('PROJECT', '/content/drive/MyDrive/quantum-placement')
MAIN_RUNS = f'{PROJECT}/results/p2_cbis_runs_e200'
SENS_RUNS = f'{PROJECT}/results/p2_cbis_runs_e200_lrtrain'
MAIN_DMID = f'{PROJECT}/results/p2_external_dmid_roi'
SENS_DMID = f'{PROJECT}/results/p2_external_dmid_roi_lrtrain'
DEDUP = f'{PROJECT}/results/p2_analysis_v2/dmid_dedup.csv'
OUT = f'{PROJECT}/results/p2_sensitivity_lr'
os.makedirs(OUT, exist_ok=True)
MODELS = ['g_q', 'g_cl', 'g_id', 'p_q', 'p_cl', 'p_id']
B = 10000
rng = np.random.default_rng(0)
keep = pd.read_csv(DEDUP)
keep = keep.loc[keep['keep'] == True, ['roi_id', 'img']]


def per_image(root, schedule):
    rows = []
    for m in MODELS:
        for f in range(5):
            p = f'{root}/{m}_fold{f}_seed0/per_image.csv'
            if os.path.exists(p):
                t = pd.read_csv(p); t.columns = [c.strip() for c in t.columns]
                if 'roi_id' not in t.columns:
                    t = t.rename(columns={t.columns[0]: 'roi_id'})
                rows.append(t.assign(model=m, fold=f, schedule=schedule))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def dynamics(root, schedule):
    rows = []
    for m in MODELS:
        for f in range(5):
            p = f'{root}/{m}_fold{f}_seed0/history.csv'
            if not os.path.exists(p):
                continue
            h = pd.read_csv(p); h.columns = [c.strip() for c in h.columns]
            lr = 'learning_rate' if 'learning_rate' in h else 'lr'
            b = int(h.val_loss.idxmin()); low = h.index[h[lr] <= 1e-6]
            rows.append(dict(schedule=schedule, model=m, fold=f, epochs=len(h), best_epoch=b + 1,
                             lr_le_1e6_epoch=(int(low[0]) + 1 if len(low) else np.nan),
                             lr_at_best=float(h[lr].iloc[b]), gap=float(h.soft_dice[b] - h.val_soft_dice[b]),
                             train_dice_best=float(h.soft_dice[b]), val_dice_best=float(h.val_soft_dice[b]),
                             val_loss_max_first10=float(h.val_loss[:10].max())))
    return pd.DataFrame(rows)


cb = pd.concat([per_image(MAIN_RUNS, 'main'), per_image(SENS_RUNS, 'lr_train')], ignore_index=True)
dm = pd.concat([per_image(MAIN_DMID, 'main'), per_image(SENS_DMID, 'lr_train')], ignore_index=True)
dm = dm.drop(columns=[c for c in ('img',) if c in dm.columns]).merge(keep, on='roi_id', how='inner')
dm = dm.groupby(['schedule', 'model', 'roi_id', 'img'], as_index=False)[['dice', 'dice_const', 'dice_swap']].mean()  # mean of 5 fold models
dyn = pd.concat([dynamics(MAIN_RUNS, 'main'), dynamics(SENS_RUNS, 'lr_train')], ignore_index=True)
n_sens = cb[cb.schedule == 'lr_train'].groupby('model').fold.nunique()
print('sensitivity runs found (folds per model):', n_sens.to_dict())

# ---- 1. per-model summary, main vs lr_train
summ = []
for (s, m), t in cb.groupby(['schedule', 'model']):
    d = dm[(dm.schedule == s) & (dm.model == m)]
    y = dyn[(dyn.schedule == s) & (dyn.model == m)]
    summ.append(dict(schedule=s, model=m, cbis_dice=t.dice.mean(), cbis_drop_const=(t.dice - t.dice_const).mean(),
                     cbis_drop_swap=(t.dice - t.dice_swap).mean(), dmid_dice=d.dice.mean() if len(d) else np.nan,
                     dmid_drop_const=(d.dice - d.dice_const).mean() if len(d) else np.nan,
                     lr_le_1e6_epoch_median=y.lr_le_1e6_epoch.median(), runs_lr_le_1e6_by_30=int((y.lr_le_1e6_epoch <= 30).sum()),
                     best_epoch_mean=y.best_epoch.mean(), gap_mean=y.gap.mean(), val_loss_max_first10=y.val_loss_max_first10.median()))
summ = pd.DataFrame(summ).set_index(['model', 'schedule']).sort_index().round(4)
summ.to_csv(f'{OUT}/summary_main_vs_lrtrain.csv')
pd.set_option('display.width', 250)
print('\n## Per model: main seed-0 runs vs lr_train runs\n', summ.to_string())


# ---- 2. contrasts within each schedule (cluster bootstrap; one seed, so no seed variance)
def boot(diff, cl):
    d = pd.DataFrame(dict(d=diff, cl=cl)); s = d.groupby('cl').d.sum(); n = d.groupby('cl').size(); c = s.index.to_numpy()
    w_all = [np.bincount(rng.integers(0, len(c), len(c)), minlength=len(c)) for _ in range(B)]
    bs = np.array([(w @ s.values) / (w @ n.values) for w in w_all])
    return diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


CONTRASTS = [('p_q', 'g_q'), ('p_cl', 'g_cl'), ('p_id', 'g_id'), ('g_q', 'g_cl'), ('g_q', 'g_id'), ('p_q', 'p_cl'), ('p_q', 'p_id')]
con = []
for ds, df, key, cl in (('CBIS', cb, 'roi_id', 'patient'), ('DMID', dm, 'roi_id', 'img')):
    for s in ('main', 'lr_train'):
        t = df[df.schedule == s]
        for a, b in CONTRASTS:
            x = t[t.model == a].set_index(key); y = t[t.model == b].set_index(key); k = x.index.intersection(y.index)
            if len(k) == 0:
                continue
            est, lo, hi = boot((x.loc[k, 'dice'] - y.loc[k, 'dice']).to_numpy(), x.loc[k, cl].to_numpy())
            con.append(dict(dataset=ds, schedule=s, contrast=f'{a} - {b}', diff=est, ci_low=lo, ci_high=hi))
con = pd.DataFrame(con)
wide = con.pivot_table(index=['dataset', 'contrast'], columns='schedule', values=['diff', 'ci_low', 'ci_high'])
wide.columns = [f'{v}_{s}' for v, s in wide.columns]
wide = wide[[c for c in ['diff_main', 'ci_low_main', 'ci_high_main', 'diff_lr_train', 'ci_low_lr_train', 'ci_high_lr_train'] if c in wide.columns]]
if {'diff_main', 'diff_lr_train'} <= set(wide.columns):
    wide['same_sign'] = np.sign(wide.diff_main) == np.sign(wide.diff_lr_train)
wide = wide.round(4)
wide.to_csv(f'{OUT}/contrasts_main_vs_lrtrain.csv')
print('\n## Contrasts (A - B), seed 0, cluster bootstrap 95% CI\n', wide.to_string())

# ---- 3. decision rule (fixed in notebook 02c before the runs)
print('\n## Decision rule')
lt = summ.xs('lr_train', level='schedule')
c = wide.loc['CBIS'] if 'CBIS' in wide.index.get_level_values(0) else pd.DataFrame()
d = wide.loc['DMID'] if 'DMID' in wide.index.get_level_values(0) else pd.DataFrame()
r1 = all(lt.loc[m, 'runs_lr_le_1e6_by_30'] == 0 for m in ('p_q', 'p_cl', 'p_id') if m in lt.index)
print('(1) fix worked (no patch-wise run at lr <= 1e-6 by epoch 30):', r1)
r2 = all(c.loc[k, 'diff_lr_train'] < 0 for k in ('p_q - g_q', 'p_cl - g_cl') if k in c.index)
print('(2) CBIS patch - global < 0 for circuit and Dense(4):', r2, '|', {k: c.loc[k, 'diff_lr_train'] for k in ('p_q - g_q', 'p_cl - g_cl') if k in c.index})
better = [f'{ds} {k}' for ds, t in (('CBIS', c), ('DMID', d)) for k in ('g_q - g_cl', 'p_q - p_cl') if k in t.index and t.loc[k, 'ci_low_lr_train'] > 0]
small = abs(c.loc['g_q - g_cl', 'diff_lr_train']) <= 0.01 if 'g_q - g_cl' in c.index else None
print('(3) circuit null holds:', (not better) and bool(small), '| circuit better than Dense(4) (CI > 0):', better or 'none',
      '| CBIS g_q - g_cl =', c.loc['g_q - g_cl', 'diff_lr_train'] if 'g_q - g_cl' in c.index else None)
r4 = all(lt.loc[f'g_{b}', 'cbis_drop_const'] > lt.loc[f'p_{b}', 'cbis_drop_const'] for b in ('q', 'cl', 'id'))
print('(4) global const-drop > patch-wise const-drop for every block:', r4, '|',
      {b: (lt.loc[f'g_{b}', 'cbis_drop_const'], lt.loc[f'p_{b}', 'cbis_drop_const']) for b in ('q', 'cl', 'id')})
print('\nfiles written to', OUT)
