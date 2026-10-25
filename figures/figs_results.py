"""Figs. 3-6 (randomized-crop experiment). Run from the repository root: python figures/figs_results.py"""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

fam = 'Liberation Sans' if any('Liberation Sans' in f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
mpl.rcParams.update({'font.family': fam, 'font.size': 7.5, 'pdf.fonttype': 42, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST = os.path.join(ROOT, 'results', 'stats'); O = os.path.join(ROOT, 'figures', 'out'); os.makedirs(O, exist_ok=True)
COL = {'q': '#0072B2', 'cl': '#D55E00', 'id': '#009E73', 'ref': '#444444'}
MK = {'q': 'o', 'cl': 's', 'id': '^', 'ref': 'D'}
LAB = {'q': 'Circuit', 'cl': 'Dense(4)', 'id': 'Identity'}
GRID = '#e3e3e3'
sm = pd.read_csv(f'{ST}/summary_models.csv')
con = pd.read_csv(f'{ST}/contrasts.csv'); con = con[~con.excl_failed]
df = pd.read_csv(os.path.join(ROOT, 'results', 'per_image_results.csv.gz')); df = df[df.keep == True]
base = pd.read_csv(os.path.join(ROOT, 'results', 'no_network_baselines.csv'))
SCHED = {'train': 'LR schedule on training loss (primary)', 'val': 'LR schedule on validation loss'}

# ---------------- Fig. 3: example crops (panels cut from the QC grid, identifiers added)
# Needs the QC grid written by notebook 11 (image crops are not distributed with this repository).
QC = os.path.join(ROOT, 'figures', 'jit_qc.png')
if os.path.exists(QC):
    im = Image.open(QC).convert('RGB')
    ROWS_Y = [(31, 286), (313, 569)]; COLS_X = [(17, 273), (289, 545), (561, 817), (833, 1089), (1106, 1362), (1378, 1634)]
    IDS = [['P_00419_RIGHT_CC_1', 'P_01363_RIGHT_MLO_1', 'P_01544_RIGHT_MLO_1', 'P_00765_RIGHT_MLO_1', 'P_01039_RIGHT_MLO_3', 'P_01210_RIGHT_MLO_1'],
           ['IMG440_279', 'IMG495_340', 'IMG420_261', 'IMG367_215', 'IMG280_155', 'IMG485_330']]
    fig, axs = plt.subplots(2, 6, figsize=(7.16, 2.9))
    for r, (y0, y1) in enumerate(ROWS_Y):
        for c, (x0, x1) in enumerate(COLS_X):
            ax = axs[r, c]; ax.imshow(np.asarray(im.crop((x0, y0, x1, y1)))); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.set_xlabel(IDS[r][c], fontsize=5.6, labelpad=1.5)
        axs[r, 0].set_title(['(a) CBIS-DDSM (development and internal test)', '(b) DMID (external test)'][r], fontsize=7.5, fontweight='bold', loc='left', x=0.0)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.05, wspace=0.04, hspace=0.32)
    fig.savefig(f'{O}/fig3_examples.png', dpi=400); fig.savefig(f'{O}/fig3_examples.pdf'); plt.close(fig)


# ---------------- Fig. 4: Dice per model (dots = runs, bar = mean with 95% CI), baselines
run = df.groupby(['set', 'dataset', 'model', 'fold', 'seed']).dice.mean().reset_index()
run_dmid = df[df.dataset == 'DMID'].groupby(['set', 'model', 'seed', 'fold']).dice.mean().reset_index()
ORDER = [('g_q', 'q'), ('g_cl', 'cl'), ('g_id', 'id'), ('p_q', 'q'), ('p_cl', 'cl'), ('p_id', 'id'), ('full', 'ref'), ('unet', 'ref')]
XT = ['G-Q', 'G-Cl', 'G-Id', 'P-Q', 'P-Cl', 'P-Id', 'full', 'unet']
fig, axs = plt.subplots(2, 2, figsize=(7.16, 4.6), sharey=True)
for i, sched in enumerate(('train', 'val')):
    for j, ds in enumerate(('CBIS', 'DMID')):
        ax = axs[i, j]; ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
        for k, (m, b) in enumerate(ORDER):
            s = sm[(sm.schedule == sched) & (sm.dataset == ds) & (sm.model == m)]
            if s.empty: continue
            x = k + (0.45 if k >= 3 else 0) + (0.45 if k >= 6 else 0)
            v = run[(run.set == sched) & (run.dataset == ds) & (run.model == m)].dice.values
            jit = np.random.default_rng(k).uniform(-0.17, 0.17, len(v))
            ax.scatter(x + jit, v, s=5, color=COL[b], alpha=0.35, lw=0, zorder=2)
            e = s.iloc[0]
            ax.errorbar(x, e.dice, yerr=[[e.dice - e.ci_low], [e.ci_high - e.dice]], fmt=MK[b], ms=4.5, color=COL[b],
                        mfc='white' if b == 'ref' else COL[b], mew=0.9, elinewidth=1.0, capsize=2, zorder=3)
        bl = base[base.dataset == ds].groupby('baseline').dice.mean()
        ax.axhline(bl['centred_disc'], color='#888888', ls='--', lw=0.8, zorder=1)
        ax.text(-0.5, bl['centred_disc'] - 0.008, 'centered disc (no network)', fontsize=6.2, color='#666666', ha='left', va='top')
        xs = [k + (0.45 if k >= 3 else 0) + (0.45 if k >= 6 else 0) for k in range(8)]
        ax.set_xticks(xs); ax.set_xticklabels(XT, fontsize=6.8); ax.set_xlim(-0.6, 9.5); ax.set_ylim(0.0, 0.85)
        if sched == 'val':
            for t in ax.get_xticklabels()[6:]: t.set_color('#bbbbbb')
            ax.text(8.45, 0.05, 'not trained\nunder this schedule', fontsize=5.8, color='#999999', ha='center')
        ax.set_title(f"{'CBIS-DDSM (internal test)' if ds == 'CBIS' else 'DMID (external, no fine-tuning)'}", fontsize=7.5, loc='left')
        if j == 0: ax.set_ylabel(f"Dice\n{SCHED[sched]}", fontsize=7)
h = [plt.Line2D([], [], marker=MK[b], color=COL[b], ls='', ms=4.5, label=LAB.get(b, 'Reference (seed 0)'), mfc='white' if b == 'ref' else COL[b]) for b in ('q', 'cl', 'id', 'ref')]
fig.legend(handles=h, loc='upper center', ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f'{O}/fig4_dice.png', dpi=400); fig.savefig(f'{O}/fig4_dice.pdf'); plt.close(fig)

# ---------------- Fig. 5: forest plot of paired contrasts
ROWS = [('p_q - g_q', 'Encoding: P-Q − G-Q', 'q'), ('p_cl - g_cl', 'Encoding: P-Cl − G-Cl', 'cl'), ('p_id - g_id', 'Encoding: P-Id − G-Id', 'id'),
        ('g_q - g_cl', 'Global: Q − Cl', 'q'), ('g_q - g_id', 'Global: Q − Id', 'q'), ('g_cl - g_id', 'Global: Cl − Id', 'cl'),
        ('p_q - p_cl', 'Patch-wise: Q − Cl', 'q'), ('p_q - p_id', 'Patch-wise: Q − Id', 'q'), ('p_cl - p_id', 'Patch-wise: Cl − Id', 'cl')]
fig, axs = plt.subplots(1, 4, figsize=(7.16, 3.0), sharey=True)
for a, (sched, ds) in zip(axs, (('train', 'CBIS'), ('train', 'DMID'), ('val', 'CBIS'), ('val', 'DMID'))):
    t = con[(con.schedule == sched) & (con.dataset == ds)].set_index('contrast')
    for y, (c, lab, b) in enumerate(ROWS):
        yy = len(ROWS) - 1 - y - (0.4 if y >= 3 else 0) - (0.4 if y >= 6 else 0)
        r = t.loc[c]; sig = r.p_holm < 0.05
        a.plot([r.ci_low, r.ci_high], [yy, yy], color=COL[b], lw=1.1)
        a.plot(r['diff'], yy, MK[b], color=COL[b], ms=4.2, mfc=COL[b] if sig else 'white', mew=0.9)
    a.axvline(0, color='#555555', lw=0.7); a.grid(axis='x', color=GRID, lw=0.5); a.set_axisbelow(True)
    a.set_xlim(-0.2, 0.2); a.set_xticks([-0.2, -0.1, 0, 0.1, 0.2]); a.tick_params(axis='x', labelsize=6.5)
    a.set_title(f"{'CBIS-DDSM' if ds == 'CBIS' else 'DMID'}\n{'training-loss LR (primary)' if sched == 'train' else 'validation-loss LR'}", fontsize=7)
    a.set_xlabel('Δ Dice (A − B)', fontsize=7)
ys = [len(ROWS) - 1 - y - (0.4 if y >= 3 else 0) - (0.4 if y >= 6 else 0) for y in range(len(ROWS))]
axs[0].set_yticks(ys); axs[0].set_yticklabels([r[1] for r in ROWS], fontsize=6.8)
fig.tight_layout(w_pad=0.6); fig.savefig(f'{O}/fig5_contrasts.png', dpi=400); fig.savefig(f'{O}/fig5_contrasts.pdf'); plt.close(fig)

# ---------------- Fig. 6: severing drops
MODS = [('g_q', 'q'), ('g_cl', 'cl'), ('g_id', 'id'), ('p_q', 'q'), ('p_cl', 'cl'), ('p_id', 'id')]
fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.5), sharey=True)
for a, (sched, ds) in zip(axs, (('train', 'CBIS'), ('train', 'DMID'), ('val', 'CBIS'), ('val', 'DMID'))):
    for k, (m, b) in enumerate(MODS):
        r = sm[(sm.schedule == sched) & (sm.dataset == ds) & (sm.model == m)].iloc[0]
        x = k + (0.5 if k >= 3 else 0)
        for dx, col, mfc in ((-0.16, 'drop_const', 'white'), (0.16, 'drop_swapavg', COL[b])):
            a.errorbar(x + dx, r[col], yerr=[[r[col] - r[col + '_lo']], [r[col + '_hi'] - r[col]]], fmt=MK[b], ms=4, color=COL[b],
                       mfc=mfc, mew=0.9, elinewidth=1.0, capsize=1.5)
    a.axhline(0, color='#555555', lw=0.7); a.grid(axis='y', color=GRID, lw=0.5); a.set_axisbelow(True)
    a.set_xticks([k + (0.5 if k >= 3 else 0) for k in range(6)]); a.set_xticklabels(['G-Q', 'G-Cl', 'G-Id', 'P-Q', 'P-Cl', 'P-Id'], fontsize=6.3, rotation=0)
    a.set_title(f"{'CBIS-DDSM' if ds == 'CBIS' else 'DMID'}\n{'training-loss LR (primary)' if sched == 'train' else 'validation-loss LR'}", fontsize=7)
axs[0].set_ylabel('Dice drop when the code is replaced', fontsize=7); axs[0].set_ylim(-0.06, 0.45)
h = [plt.Line2D([], [], marker='o', color='#444444', ls='', ms=4, mfc='white', label='const (mean training code)'),
     plt.Line2D([], [], marker='o', color='#444444', ls='', ms=4, label='swap (another image’s code, mean of 10 draws)')]
fig.legend(handles=h, loc='upper center', ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.6); fig.savefig(f'{O}/fig6_severing.png', dpi=400); fig.savefig(f'{O}/fig6_severing.pdf'); plt.close(fig)
print('ok')
