"""Fig. 1: study overview."""
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager

fam = 'Liberation Sans' if any('Liberation Sans' in f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
mpl.rcParams.update({'font.family': fam, 'font.size': 7.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(7.16, 2.75))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 143.2); ax.set_ylim(0, 55); ax.axis('off')
E = '#333333'
DATA, MOD, TEST, STAT, SENS = '#dbe6f4', '#fbe3cf', '#d9ead3', '#eeeeee', '#fff2b3'


def box(x, y, w, h, head, body, fc, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.0', fc=fc, ec=E, lw=0.7, ls=ls))
    ax.text(x + w / 2, y + h - 2.2, head, ha='center', va='top', fontsize=7.6, fontweight='bold')
    ax.text(x + w / 2, y + h - 6.6, body, ha='center', va='top', fontsize=6.6, linespacing=1.25)


def arr(x1, y1, x2, y2, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color=E, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=8))

y1, h1 = 29, 23
box(2, y1, 25, h1, 'Development data', 'CBIS-DDSM\n400 mass ROIs\n208 patients\n256×256 crops', DATA)
box(31, y1, 27, h1, 'Training', 'patient-level 5-fold CV\nseeds 0, 1, 2\nsame settings for all\n(100 runs)', MOD)
box(62, y1, 33, h1, 'Eight models', 'global or patch-wise bottleneck\n× circuit / Dense(4) / identity\n+ full (no compression)\n+ unet (classical, 31M)', MOD)
box(99, y1, 20, h1, 'In-domain test', 'CBIS-DDSM\ntest folds\n(every ROI\nonce per seed)', TEST)
box(122, y1, 19.5, h1, 'External test', 'DMID\n291 crops\n237 images\nno fine-tuning', TEST)
for a, b in ((27, 31), (58, 62), (95, 99), (119, 122)):
    arr(a + 0.3, y1 + h1 / 2, b - 0.3, y1 + h1 / 2)

y2, h2 = 2, 21
box(2, y2, 35, h2, 'Sensitivity analysis', 'six G/P models, seed 0 (30 runs)\nLR schedule on training loss\n4 decision rules fixed in advance', SENS, ls=(0, (3, 2)))
box(41, y2, 31, h2, 'Is the code used?', 'severing tests at test time:\nconst (mean code) and swap\n(random training code)', STAT)
box(76, y2, 26, h2, 'Trainability', 'gradient audit of the\n20 effective circuit\nweights (init, best)', STAT)
box(106, y2, 35.5, h2, 'Statistics', 'paired per image; hierarchical\nbootstrap over seeds and patients\n(CBIS) or images (DMID); Holm', STAT)
arr(78.5, y1 - 0.3, 56.5, y2 + h2 + 0.3)
arr(78.5, y1 - 0.3, 89, y2 + h2 + 0.3)
arr(130.5, y1 - 0.3, 123.75, y2 + h2 + 0.3)
arr(109, y1 - 0.3, 117, y2 + h2 + 0.3)
arr(44.5, y1 - 0.3, 19.5, y2 + h2 + 0.3, ls=(0, (3, 2)))
for ext in ('pdf', 'svg'):
    fig.savefig(f'out/fig1_overview.{ext}')
fig.savefig('out/fig1_overview.png', dpi=600)
print('ok')
