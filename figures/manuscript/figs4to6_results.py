"""Figs. 4-6 at Springer double-column width (174 mm), lettering 8-10 pt."""
import os, sys, numpy as np, pandas as pd, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
ST = os.path.join(HERE, '..', '..', 'results', 'five_seed', 'stats'); O = os.path.join(HERE, 'out'); os.makedirs(O, exist_ok=True)
W = 174 / 25.4
mpl.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
                     'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5, 'pdf.fonttype': 42,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})
lv = pd.read_csv(f'{ST}/rev5_levels.csv')
co = pd.read_csv(f'{ST}/rev5_contrasts.csv'); co = co[(co.analysis == 'main') & (co.reference == 'consensus')]
dr = pd.read_csv(f'{ST}/rev5_cbis_drops.csv')
SETS = [('cbis', 'CBIS-DDSM', '#1f4e79'), ('frozen', 'Frozen', '#5b9bd5'), ('dmid', 'DMID', '#c55a11'),
        ('csaws', 'CSAW-S', '#7f6000'), ('cddcesm', 'CDD-CESM', '#548235')]
NAME = {'g_q': 'G circuit', 'g_cl': 'G Dense(4)', 'g_tr': 'G trigonometric', 'g_id': 'G identity', 'p_q': 'P circuit',
        'p_cl': 'P Dense(4)', 'p_tr': 'P trigonometric', 'p_id': 'P identity', 'full': 'full', 'unet': 'unet'}
SCH = {'train': 'Primary schedule', 'val': 'Second schedule'}
def save(fig, n):
    fig.savefig(f'{O}/{n}.pdf', metadata={'Creator': None, 'Producer': None, 'CreationDate': None}); fig.savefig(f'{O}/{n}.png', dpi=600); plt.close(fig)

# Fig. 4
M = ['g_q', 'g_cl', 'g_tr', 'g_id', 'p_q', 'p_cl', 'p_tr', 'p_id', 'full', 'unet']
fig, axs = plt.subplots(1, 2, figsize=(W, 3.7), sharey=True, gridspec_kw={'width_ratios': [10, 8]})
for ax, s in zip(axs, ('train', 'val')):
    mods = [m for m in M if not lv[(lv.schedule == s) & (lv.model == m)].empty]
    for k, m in enumerate(mods):
        for j, (st, lab, c) in enumerate(SETS):
            r = lv[(lv.schedule == s) & (lv.model == m) & (lv.set == st)].iloc[0]
            x = k + (j - 2) * 0.15
            ax.errorbar(x, r.dice, yerr=[[r.dice - r.ci_lo], [r.ci_hi - r.dice]], fmt='o', ms=3.6, color=c, elinewidth=1.1, capsize=0)
    ax.set_xticks(range(len(mods))); ax.set_xticklabels([NAME[m] for m in mods], rotation=60, ha='right', rotation_mode='anchor')
    ax.set_title(SCH[s]); ax.grid(axis='y', color='#e6e6e6', lw=0.5); ax.set_axisbelow(True); ax.set_xlim(-0.6, len(mods) - 0.4)
axs[0].set_ylabel('Mean per-image Dice'); axs[0].set_ylim(0.38, 0.78)
h = [plt.Line2D([], [], marker='o', color=c, ls='-', lw=1.1, ms=3.6, label=l) for _, l, c in SETS]
fig.legend(handles=h, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.0); save(fig, 'fig4')

# Fig. 5
ROWS = [('g_q - g_cl', 'Global: Q − Cl'), ('g_q - g_tr', 'Global: Q − Tr'), ('g_q - g_id', 'Global: Q − Id'),
        ('p_q - p_cl', 'Patch-wise: Q − Cl'), ('p_q - p_tr', 'Patch-wise: Q − Tr'), ('p_q - p_id', 'Patch-wise: Q − Id'),
        ('p_q - g_q', 'circuit: P − G'), ('p_cl - g_cl', 'Dense(4): P − G'), ('p_id - g_id', 'identity: P − G')]
fig, axs = plt.subplots(1, 2, figsize=(W, 5.4), sharey=True)
for ax, s in zip(axs, ('train', 'val')):
    ax.axvspan(-0.02, 0.02, color='#e3e3e3', lw=0, zorder=0); ax.axvline(0, color='#333333', lw=0.7, zorder=1)
    for i, (cn, lab) in enumerate(ROWS):
        y0 = len(ROWS) - 1 - i
        for j, (st, l, c) in enumerate(SETS):
            r = co[(co.schedule == s) & (co.contrast == cn) & (co.set == st)]
            if r.empty: continue
            r = r.iloc[0]; y = y0 + (2 - j) * 0.15
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=c, lw=1.1, zorder=2); ax.plot(r['diff'], y, 'o', color=c, ms=3.6, zorder=3)
    ax.set_title(SCH[s]); ax.set_xlabel('Dice difference (A − B)'); ax.set_xlim(-0.205, 0.145)
    ax.set_xticks([-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1]); ax.set_xticklabels(['−0.20', '−0.15', '−0.10', '−0.05', '0', '0.05', '0.10'])
axs[0].set_yticks(range(len(ROWS))); axs[0].set_yticklabels([l for _, l in ROWS][::-1]); axs[0].set_ylim(-0.6, len(ROWS) - 0.4)
h = [plt.Line2D([], [], marker='o', color=c, ls='-', lw=1.1, ms=3.6, label=l) for _, l, c in SETS]
fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.0); save(fig, 'fig5')

# Fig. 6
M8 = ['g_q', 'g_cl', 'g_tr', 'g_id', 'p_q', 'p_cl', 'p_tr', 'p_id']
SER = [('train', 'const', 'o', 'white', '#1f4e79', 'Primary, const'), ('train', 'swap', 's', None, '#1f4e79', 'Primary, swap'),
       ('val', 'const', 'o', 'white', '#c55a11', 'Second, const'), ('val', 'swap', 's', None, '#c55a11', 'Second, swap')]
fig, ax = plt.subplots(figsize=(W, 3.0))
for k, m in enumerate(M8):
    for j, (s, rep, mk, mfc, c, l) in enumerate(SER):
        r = dr[(dr.schedule == s) & (dr.model == m) & (dr.replacement == rep)].iloc[0]
        ax.errorbar(k + (j - 1.5) * 0.17, r['diff'], yerr=[[r['diff'] - r.ci_lo], [r.ci_hi - r['diff']]], fmt=mk, ms=4.2,
                    color=c, mfc=mfc or c, mew=1.0, elinewidth=1.1, capsize=0)
ax.axhline(0, color='#333333', lw=0.7); ax.grid(axis='y', color='#e6e6e6', lw=0.5); ax.set_axisbelow(True)
ax.set_xticks(range(8)); ax.set_xticklabels([NAME[m].replace(' ', '\n', 1) for m in M8]); ax.set_ylabel('Dice drop (CBIS-DDSM)'); ax.set_ylim(-0.01, 0.24)
h = [plt.Line2D([], [], marker=mk, color=c, mfc=mfc or c, ls='-', lw=1.1, ms=4.2, label=l) for _, _, mk, mfc, c, l in SER]
fig.legend(handles=h, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, 'fig6')
print('ok')
