"""Fig. 1: study overview at 174 mm, lettering 8.5-9.5 pt."""
import sys, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, 'fig1')
mpl.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42})
WM, HM = 174, 82
fig = plt.figure(figsize=(WM / 25.4, HM / 25.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WM); ax.set_ylim(0, HM); ax.axis('off')
E = '#333333'; DATA, MOD, TEST, STAT, SENS = '#dbe6f4', '#fbe3cf', '#d9ead3', '#eeeeee', '#fff2b3'
BOX = {}; TXT = []
def box(key, x, y, w, h, head, body, fc, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.5', fc=fc, ec=E, lw=0.8, ls=ls))
    TXT.append((ax.text(x + w / 2, y + h - 2.6, head, ha='center', va='top', fontsize=9.5, fontweight='bold'), x, w))
    TXT.append((ax.text(x + w / 2, y + h - 9.0, body, ha='center', va='top', fontsize=8.5, linespacing=1.2), x, w))
    BOX[key] = (x, y, w, h)
def arr(x1, y1, x2, y2, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', lw=0.9, color=E, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=9))
y1, h1 = 45, 36.5
r1 = [('dev', 31, 'Development data', 'CBIS-DDSM\n400 mass ROIs,\n208 patients\nrandomized crops\n(lesion position\nand scale vary)', DATA),
      ('tr', 33, 'Training', 'patient-level\n5-fold CV\nLR schedule on\ntraining loss\nseeds 0–4, all models\n(250 runs)', MOD),
      ('mod', 41, 'Ten models', 'global or patch-wise\nbottleneck × circuit /\nDense(4) / trig. / identity\n+ full (no compression)\n+ unet (classical, 31M)', MOD),
      ('ind', 29, 'In-domain test', 'CBIS-DDSM\ntest folds\n(every ROI once\nper seed)', TEST),
      ('ext', 29, 'External test', 'frozen CBIS\ncohort, DMID,\nCSAW-S,\nCDD-CESM\nno fine-tuning', TEST)]
x = 0.5
for k, w, hd, bd, fc in r1:
    box(k, x, y1, w, h1, hd, bd, fc); x += w + (174 - 1 - sum(r[1] for r in r1)) / 4
for a, b in zip(r1[:-1], r1[1:]):
    xa = BOX[a[0]][0] + BOX[a[0]][2]; xb = BOX[b[0]][0]; arr(xa, y1 + h1 / 2, xb, y1 + h1 / 2)
y2, h2 = 0.5, 33
r2 = [('sec', 36, 'Second LR schedule', 'eight G/P models,\nseeds 0–4 (200 runs)\nLR schedule on\nvalidation loss', SENS, (0, (3, 2))),
      ('code', 32, 'Code sensitivity', 'test-time replacement:\nconst (mean code),\nswap (training code,\n10 draws)', STAT, '-'),
      ('trn', 28, 'Trainability', 'gradient and init\naudits; circuit\nfunction class;\nnoisy re-evaluation', STAT, '-'),
      ('add', 41, 'Additional experiments', '410 runs, primary LR:\nfaster or per-block LR,\nwider encoding, shared\ncircuit initialization', SENS, (0, (3, 2))),
      ('st', 29, 'Statistics', 'written protocols;\npaired per image;\nbootstrap over\nseeds and patients;\nHolm; margin\n±0.02 Dice', STAT, '-')]
x = 0.5
for k, w, hd, bd, fc, ls in r2:
    box(k, x, y2, w, h2, hd, bd, fc, ls); x += w + (174 - 1 - sum(r[1] for r in r2)) / 4
def bot(k, f=0.5): x, y, w, h = BOX[k]; return x + w * f, y
def top(k, f=0.5): x, y, w, h = BOX[k]; return x + w * f, y + h
arr(*bot('tr'), *top('sec'), ls=(0, (3, 2)))
arr(*bot('mod'), *top('code')); arr(*bot('mod'), *top('trn')); arr(*bot('mod'), *top('add'), ls=(0, (3, 2)))
arr(*bot('ext'), *top('st'))
fig.canvas.draw(); r = fig.canvas.get_renderer(); bad = 0
for t, x0, w in TXT:
    bb = t.get_window_extent(r).transformed(ax.transData.inverted())
    if bb.x0 < x0 + 0.8 or bb.x1 > x0 + w - 0.8: print('overflow:', t.get_text()[:30], round(bb.x0, 1), round(bb.x1, 1), x0, x0 + w); bad += 1
fig.savefig(out + '.pdf', metadata={'Creator': None, 'Producer': None, 'CreationDate': None}); fig.savefig(out + '.png', dpi=600); print('overflows', bad)
