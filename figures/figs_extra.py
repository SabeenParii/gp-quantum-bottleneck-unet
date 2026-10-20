"""Fig. 4 (severing), Fig. 6 (seed-to-seed spread), Fig. 7 (training dynamics under the two LR schedules)."""
import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager

fam = 'Liberation Sans' if any('Liberation Sans' in f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
mpl.rcParams.update({'font.family': fam, 'font.size': 8, 'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
M = ['g_q', 'g_cl', 'g_id', 'p_q', 'p_cl', 'p_id']
LAB = {'g_q': 'global, circuit', 'g_cl': 'global, Dense(4)', 'g_id': 'global, identity',
       'p_q': 'patch-wise, circuit', 'p_cl': 'patch-wise, Dense(4)', 'p_id': 'patch-wise, identity'}
MINUS = lambda t: f'{t:+.2f}'.replace('-', '−') if t else '0'


def save(fig, name):
    for ext in ('pdf', 'svg'):
        fig.savefig(f'out/{name}.{ext}')
    fig.savefig(f'out/{name}.png', dpi=600)


def clean(ax, keep_left=True):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if not keep_left:
        ax.spines['left'].set_visible(False)

# ---------------- Fig. 4: severing ----------------
s = pd.read_csv('data/severing_hierarchical.csv')
C = {'const': '#1f5aa6', 'swap': '#c2571a'}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True, gridspec_kw={'wspace': 0.08})
ys = np.array([0, 1, 2, 3.7, 4.7, 5.7])
for ax, ds, title in zip(axes, ['CBIS', 'DMID'], ['In-domain: CBIS-DDSM', 'External: DMID']):
    ax.axvspan(-0.03, 0, color='#f2f2f2', zorder=0)
    for y, m in zip(ys, M):
        for k, (mode, off) in enumerate((('const', -0.17), ('swap', 0.17))):
            r = s[(s.dataset == ds) & (s.model == m) & (s['mode'] == mode)].iloc[0]
            ax.plot([r.ci95_low, r.ci95_high], [y + off] * 2, color=C[mode], lw=1.3, zorder=2)
            ax.plot(r['drop'], y + off, 'o', ms=4.6, mec=C[mode], mew=1.1, mfc=C[mode] if r.p_holm < 0.05 else 'white', zorder=3)
    ax.axvline(0, color='#555555', lw=0.7)
    ax.set_xlim(-0.03, 0.16); ax.set_xticks([0, 0.05, 0.10, 0.15])
    ax.set_xticklabels([MINUS(t) if t else '0' for t in ax.get_xticks()])
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel('Dice drop when the code is severed (95% CI)')
    ax.grid(axis='x', color='#e3e3e3', lw=0.5); ax.set_axisbelow(True)
    clean(ax, keep_left=False); ax.tick_params(axis='y', length=0)
axes[0].set_yticks(ys); axes[0].set_yticklabels([LAB[m] for m in M]); axes[0].invert_yaxis()
fig.legend(handles=[Line2D([], [], marker='o', ls='-', color=C['const'], ms=4.5, label='const: mean training code'),
                    Line2D([], [], marker='o', ls='-', color=C['swap'], ms=4.5, label='swap: code of a random training image'),
                    Line2D([], [], marker='o', ls='', mfc='white', mec='#444444', ms=4.5, label='open marker: Holm p ≥ 0.05')],
           loc='lower center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.58, -0.01))
fig.subplots_adjust(left=0.17, right=0.99, top=0.91, bottom=0.27)
save(fig, 'fig4_severing')

# ---------------- Fig. 6: seed-to-seed spread ----------------
runs = pd.read_csv('data/runs.csv')
ps = pd.read_csv('data/per_seed.csv')
SC = ['#1b9e77', '#d95f02', '#7570b3']
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={'wspace': 0.28})
xs = np.arange(len(M))
rng = np.random.default_rng(1)
ax = axes[0]
for i, m in enumerate(M):
    for s_ in (0, 1, 2):
        v = runs[(runs.model == m) & (runs.seed == s_)].dice_per_image.values
        ax.plot(i + (s_ - 1) * 0.2 + rng.uniform(-0.04, 0.04, len(v)), v, 'o', ms=2.6, color=SC[s_], alpha=0.45, mew=0)
        mu = ps[(ps.model == m) & (ps.seed == s_)].cbis_dice.values[0]
        ax.plot([i + (s_ - 1) * 0.2 - 0.08, i + (s_ - 1) * 0.2 + 0.08], [mu, mu], color=SC[s_], lw=2)
ax.set_title('CBIS-DDSM: per fold (dots) and per seed (bars)', fontsize=8)
ax2 = axes[1]
for i, m in enumerate(M):
    for s_ in (0, 1, 2):
        mu = ps[(ps.model == m) & (ps.seed == s_)].dmid_dice.values[0]
        ax2.plot(i + (s_ - 1) * 0.2, mu, 'o', ms=4.5, color=SC[s_])
ax2.set_title('DMID: per seed (mean of 5 fold models)', fontsize=8)
for a in axes:
    a.set_xticks(xs); a.set_xticklabels([LAB[m].replace('patch-wise', 'patch').replace(', ', '\n') for m in M], fontsize=6.8)
    a.set_ylabel('Mean per-image Dice'); clean(a)
    a.axvline(2.5, color='#cccccc', lw=0.6)
axes[0].set_ylim(0.68, 0.90)
fig.legend(handles=[Line2D([], [], marker='o', ls='', color=c, ms=4.5, label=f'seed {k}') for k, c in enumerate(SC)],
           loc='lower center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.26)
save(fig, 'fig6_seeds')

# ---------------- Fig. 7: training dynamics, fold 0 seed 0 ----------------
H = {('g_q', 'main'): 'hist_gq_main.csv', ('p_q', 'main'): 'hist_pq_main.csv',
     ('g_q', 'lr_train'): 'hist_gq_lr.csv', ('p_q', 'lr_train'): 'hist_pq_lr.csv'}
CC = {'g_q': '#1f5aa6', 'p_q': '#c2571a'}
LS = {'main': '-', 'lr_train': (0, (4, 2))}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={'wspace': 0.3})
for (m, sch), f in H.items():
    h = pd.read_csv('data/' + f); h.columns = [c.strip() for c in h.columns]
    lr = 'learning_rate' if 'learning_rate' in h else 'lr'
    e = h.epoch + 1
    axes[0].plot(e, h.val_loss, color=CC[m], ls=LS[sch], lw=1.1)
    b = int(h.val_loss.idxmin())
    axes[0].plot(e[b], h.val_loss[b], marker='v', color=CC[m], ms=4.5, mfc='white' if sch == 'lr_train' else CC[m])
    axes[1].plot(e, h[lr], color=CC[m], ls=LS[sch], lw=2.4 if m == 'p_q' else 1.0, zorder=3 if m == 'g_q' else 2)
axes[0].set_yscale('log'); axes[0].set_ylabel('Validation loss'); axes[0].set_title('Validation loss (▼ = best epoch)', fontsize=8)
axes[1].set_yscale('log'); axes[1].set_ylabel('Learning rate'); axes[1].set_title('Learning rate', fontsize=8)
axes[1].axhline(1e-6, color='#999999', lw=0.6, ls=':'); axes[1].text(165, 1.25e-6, r'$10^{-6}$', fontsize=6.5, color='#777777', ha='right')
for a in axes:
    a.set_xlabel('Epoch'); clean(a); a.set_xlim(0, 168)
fig.legend(handles=[Line2D([], [], color=CC['g_q'], lw=1.5, label='global circuit (g_q)'),
                    Line2D([], [], color=CC['p_q'], lw=1.5, label='patch-wise circuit (p_q)'),
                    Line2D([], [], color='#444444', lw=1.1, label='main runs: schedule on validation loss'),
                    Line2D([], [], color='#444444', lw=1.1, ls=(0, (4, 2)), label='sensitivity: schedule on training loss')],
           loc='lower center', ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.03))
fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.33)
save(fig, 'fig7_training')
print('ok')
