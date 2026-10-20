"""
# Fig. 5 · Qualitative examples (CBIS-DDSM fold 0 and DMID), main runs, seed 0, fold-0 models
Reads only. Writes results/figures/fig5_examples.{pdf,png} and fig5_selection.csv.
Examples are NOT hand-picked: for each dataset the crops are ranked by their mean Dice over the four models and the
crops at the 20th, 50th and 80th percentile are shown (a hard, a typical and an easy case).
Runtime: CPU is enough (~5-10 min); T4 GPU is faster.
"""
# 1 · Drive, paths, imports
from google.colab import drive
drive.mount('/content/drive')
import os, sys, glob
import numpy as np, pandas as pd

PROJECT  = '/content/drive/MyDrive/quantum-placement'
CBIS_DIR = f'{PROJECT}/data/processed/cbis_mass_256'
DMID_DIR = f'{PROJECT}/data/external/dmid_mass_256/roi'
RUNS_DIR = f'{PROJECT}/results/p2_cbis_runs_e200'           # main runs, read only
DEDUP    = f'{PROJECT}/results/p2_analysis_v2/dmid_dedup.csv'
OUT_DIR  = f'{PROJECT}/results/figures'
CODE_DIR = f'{PROJECT}/code'
QSEG_DIR = '/content/drive/MyDrive/QSEG_FINAL/code'      # Paper 1 package, read only
if not os.path.exists(f'{QSEG_DIR}/qseg/quantum_sim.py'):
    hits = glob.glob('/content/drive/MyDrive/**/qseg/quantum_sim.py', recursive=True)
    assert hits, 'Paper 1 qseg package not found on Drive'
    QSEG_DIR = os.path.dirname(os.path.dirname(hits[0]))
for p in (QSEG_DIR, CODE_DIR):
    if p not in sys.path: sys.path.insert(0, p)
os.makedirs(OUT_DIR, exist_ok=True)

import importlib, tensorflow as tf, qseg2; importlib.reload(qseg2)
from qseg.config import Config
cfg = Config(work_dir=RUNS_DIR)
MODELS = ['g_q', 'g_cl', 'p_q', 'p_cl']
TITLES = {'g_q': 'global, circuit', 'g_cl': 'global, Dense(4)', 'p_q': 'patch-wise, circuit', 'p_cl': 'patch-wise, Dense(4)'}
FOLD, SEED = 0, 0


# 2 · Data
def _png(p, mask):
    a = tf.image.decode_png(tf.io.read_file(p), channels=1).numpy()
    return (a > 127).astype(np.float32) if mask else a.astype(np.float32) / 255.0

XC, YC, DFC = qseg2.load_cbis(CBIS_DIR)
tr, va, te = qseg2.fold_split(DFC, FOLD)
meta = pd.read_csv(f'{DMID_DIR}/metadata.csv')
keep = pd.read_csv(DEDUP); keep = set(keep.loc[keep['keep'] == True, 'roi_id'])
meta = meta[meta.roi_id.isin(keep)].reset_index(drop=True)
XD = np.stack([_png(f'{DMID_DIR}/images/{r}.png', False) for r in meta.roi_id])
YD = np.stack([_png(f'{DMID_DIR}/masks/{r}.png', True) for r in meta.roi_id])
print('CBIS test fold', FOLD, len(te), '| DMID crops', len(XD))


# 3 · Predictions of the four fold-0 / seed-0 models
pred_c, pred_d, dice_c, dice_d = {}, {}, {}, {}
for name in MODELS:
    run = f'{RUNS_DIR}/{name}_fold{FOLD}_seed{SEED}'
    tf.keras.backend.clear_session()
    m = qseg2.build_p2_model(name, (cfg.image_size, cfg.image_size, cfg.in_channels), cfg.n_qubits, cfg.n_q_layers)
    m.load_weights(f'{run}/best.weights.h5')
    pc = m.predict(XC[te], batch_size=cfg.batch_size, verbose=0)
    pdm = m.predict(XD, batch_size=cfg.batch_size, verbose=0)
    pred_c[name] = (pc[..., 0] if pc.ndim == 4 else pc) > cfg.threshold
    pred_d[name] = (pdm[..., 0] if pdm.ndim == 4 else pdm) > cfg.threshold
    dice_c[name] = qseg2.per_image_scores(YC[te], pc, cfg.threshold).dice.values
    dice_d[name] = qseg2.per_image_scores(YD, pdm, cfg.threshold).dice.values
    print(name, 'CBIS fold-0 Dice', round(dice_c[name].mean(), 4), '| DMID Dice', round(dice_d[name].mean(), 4))


# 4 · Unbiased selection: crops at the 20th, 50th and 80th percentile of the mean Dice over the four models
def pick(dmat):
    mean = np.mean(np.stack([dmat[m] for m in MODELS]), 0)
    order = np.argsort(mean)
    return [int(order[int(round(q * (len(order) - 1)))]) for q in (0.2, 0.5, 0.8)], mean

sel_c, mean_c = pick(dice_c)
sel_d, mean_d = pick(dice_d)
rows = []
for ds, sel, mean, ids in (('CBIS-DDSM', sel_c, mean_c, DFC.roi_id.values[te]), ('DMID', sel_d, mean_d, meta.roi_id.values)):
    for q, i in zip((20, 50, 80), sel):
        r = dict(dataset=ds, percentile=q, index=i, roi_id=ids[i], mean_dice=round(float(mean[i]), 4))
        for mname in MODELS:
            r[f'dice_{mname}'] = round(float((dice_c if ds == 'CBIS-DDSM' else dice_d)[mname][i]), 4)
        rows.append(r)
sel_df = pd.DataFrame(rows)
sel_df.to_csv(f'{OUT_DIR}/fig5_selection.csv', index=False)
print(sel_df.to_string())


# 5 · Figure: rows = 3 CBIS + 3 DMID crops; columns = image with ground truth, then each model's prediction
import matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 7, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
GT_C, PR_C = '#00c853', '#ff1744'
fig, axes = plt.subplots(6, 5, figsize=(7.0, 8.6), gridspec_kw={'wspace': 0.04, 'hspace': 0.06})
for r_i, r in sel_df.iterrows():
    cbis = r.dataset == 'CBIS-DDSM'
    img = np.squeeze(XC[te][r['index']] if cbis else XD[r['index']])
    gt = (YC[te][r['index']] if cbis else YD[r['index']]).squeeze()
    for c_i in range(5):
        ax = axes[r_i, c_i]
        ax.imshow(np.squeeze(img), cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax.contour(gt, levels=[0.5], colors=GT_C, linewidths=1.0)
        if c_i > 0:
            mname = MODELS[c_i - 1]
            pr = (pred_c if cbis else pred_d)[mname][r['index']].squeeze()
            if pr.any():
                ax.contour(pr.astype(float), levels=[0.5], colors=PR_C, linewidths=1.0)
            ax.text(0.03, 0.04, f"Dice {r[f'dice_{mname}']:.2f}", transform=ax.transAxes, color='white', fontsize=7,
                    bbox=dict(fc='black', alpha=0.6, pad=1.2, lw=0))
        ax.set_xticks([]); ax.set_yticks([])
        if r_i == 0:
            ax.set_title('image + ground truth' if c_i == 0 else TITLES[MODELS[c_i - 1]], fontsize=7.5)
    axes[r_i, 0].set_ylabel(f"{r.dataset}\n{r.percentile}th pct.", fontsize=7.5)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=GT_C, lw=1.5, label='ground truth'), Line2D([], [], color=PR_C, lw=1.5, label='prediction')],
           loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.04)
fig.savefig(f'{OUT_DIR}/fig5_examples.pdf')
fig.savefig(f'{OUT_DIR}/fig5_examples.png', dpi=400)
print('saved to', OUT_DIR)
