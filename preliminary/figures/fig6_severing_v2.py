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
    fig.savefig(f'out/{name}.png', dpi=600); fig.savefig(f'out/{name}_300.png', dpi=300)


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
    ax.axvspan(-0.04, 0, color='#f2f2f2', zorder=0)
    for y, m in zip(ys, M):
        for k, (mode, off) in enumerate((('const', -0.17), ('swap', 0.17))):
            r = s[(s.dataset == ds) & (s.model == m) & (s['mode'] == mode)].iloc[0]
            ax.plot([r.ci95_low, r.ci95_high], [y + off] * 2, color=C[mode], lw=1.3, solid_capstyle='butt', zorder=2)
            for xe in (r.ci95_low, r.ci95_high):
                ax.plot([xe, xe], [y + off - 0.09, y + off + 0.09], color=C[mode], lw=1.0, zorder=2)
            ax.plot(r['drop'], y + off, 'o', ms=4.6, mec=C[mode], mew=1.1, mfc=C[mode] if r.p_holm < 0.05 else 'white', zorder=3)
    ax.axvline(0, color='#555555', lw=0.7)
    ax.set_xlim(-0.04, 0.16); ax.set_xticks([0, 0.05, 0.10, 0.15])
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
save(fig, 'fig6_severing_v2')

