"""Fig. 3: paired Dice differences (hierarchical bootstrap, 3 seeds) as a forest plot, CBIS-DDSM vs DMID."""
import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager

fam = 'Liberation Sans' if any('Liberation Sans' in f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
mpl.rcParams.update({'font.family': fam, 'font.size': 8, 'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'xtick.major.width': 0.6, 'ytick.major.width': 0, 'axes.titlesize': 9})

d = pd.read_csv('data/paired_hierarchical.csv')
NAME = {'q': 'circuit', 'cl': 'Dense(4)', 'id': 'identity'}
rows = [  # (group label, A, B, row label)
    ('Circuit vs controls, global', 'g_q', 'g_cl', 'circuit − Dense(4)'),
    ('Circuit vs controls, global', 'g_q', 'g_id', 'circuit − identity'),
    ('Circuit vs controls, global', 'g_cl', 'g_id', 'Dense(4) − identity'),
    ('Circuit vs controls, patch-wise', 'p_q', 'p_cl', 'circuit − Dense(4)'),
    ('Circuit vs controls, patch-wise', 'p_q', 'p_id', 'circuit − identity'),
    ('Circuit vs controls, patch-wise', 'p_cl', 'p_id', 'Dense(4) − identity'),
    ('Placement: patch-wise − global', 'p_q', 'g_q', 'circuit'),
    ('Placement: patch-wise − global', 'p_cl', 'g_cl', 'Dense(4)'),
    ('Placement: patch-wise − global', 'p_id', 'g_id', 'identity'),
]
# y positions with a gap between groups
ys, labels, groups, y = [], [], [], 0.0
prev = None
for g, a, b, lab in rows:
    if prev is not None and g != prev:
        y += 0.9
    ys.append(y); labels.append(lab); groups.append(g); prev = g; y += 1
ys = np.array(ys)

C = {'CBIS': '#1f5aa6', 'DMID': '#c2571a'}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.9), sharey=True, gridspec_kw={'wspace': 0.08})
for ax, ds, title in zip(axes, ['CBIS', 'DMID'], ['In-domain: CBIS-DDSM (400 ROIs, 208 patients)',
                                               'External: DMID (291 crops, 237 images)']):
    for yy, (g, a, b, lab) in zip(ys, rows):
        r = d[(d.dataset == ds) & (d.A == a) & (d.B == b)].iloc[0]
        sig = r.p_holm < 0.05
        ax.plot([r.ci95_low, r.ci95_high], [yy, yy], color=C[ds], lw=1.4, solid_capstyle='butt', zorder=2)
        ax.plot([r.ci95_low] * 2, [yy - 0.16, yy + 0.16], color=C[ds], lw=1.0, zorder=2)
        ax.plot([r.ci95_high] * 2, [yy - 0.16, yy + 0.16], color=C[ds], lw=1.0, zorder=2)
        ax.plot(r.mean_diff, yy, marker='o', ms=5.2, mec=C[ds], mew=1.2, mfc=C[ds] if sig else 'white', zorder=3)
        txt = '0.000' if abs(r.mean_diff) < 0.0005 else f'{r.mean_diff:+.3f}'.replace('-', '−')
        ax.text(0.088, yy, txt + ('*' if sig else ''), va='center', ha='right', fontsize=7.2, color='#333333')
    ax.axvline(0, color='#555555', lw=0.7, zorder=1)
    ax.set_xlim(-0.085, 0.09)
    ax.set_xticks([-0.08, -0.04, 0, 0.04])
    ax.set_xticklabels([f'{t:+.2f}'.replace('-', '−') if t else '0' for t in ax.get_xticks()])
    ax.set_title(title, fontsize=8.5, pad=14)
    ax.set_xlabel('Difference in per-image Dice (A − B), 95% CI')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='x', color='#e3e3e3', lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)

axes[0].set_yticks(ys); axes[0].set_yticklabels(labels)
axes[0].invert_yaxis()
# group headers
gpos = {}
for yy, g in zip(ys, groups):
    gpos.setdefault(g, yy)
for g, yy in gpos.items():
    axes[0].text(-0.42, yy - 0.75, g, transform=axes[0].get_yaxis_transform(), fontsize=8, fontweight='bold', ha='left', va='center')
leg = [Line2D([], [], marker='o', ls='', mfc='#444444', mec='#444444', ms=5, label='Holm p < 0.05'),
       Line2D([], [], marker='o', ls='', mfc='white', mec='#444444', ms=5, label='not significant')]
fig.legend(handles=leg, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.6, 0.0), fontsize=7.5)
fig.subplots_adjust(left=0.2, right=0.99, top=0.86, bottom=0.18)
for ext in ('pdf', 'svg'):
    fig.savefig(f'out/fig3_forest.{ext}')
fig.savefig('out/fig3_forest.png', dpi=600)
print('ok', fam)
