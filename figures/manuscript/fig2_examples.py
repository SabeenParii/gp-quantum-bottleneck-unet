"""Fig. 2 re-laid out at 174 mm from the existing crop tiles (pixels unchanged)."""
import sys, numpy as np, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
src = os.path.join(HERE, 'fig2_tiles.png'); out = os.path.join(OUT, 'fig2')
im = np.array(Image.open(src).convert('RGB'))
ROWS = [(81, 521), (662, 1102)]
COLS = [[(22, 464), (498, 939), (973, 1415), (1449, 1891), (1925, 2366), (2400, 2842)],
        [(23, 463), (499, 939), (974, 1414), (1450, 1890), (1925, 2365), (2401, 2841)]]
LAB = [['P_00419_RIGHT_CC_1', 'P_01363_RIGHT_MLO_1', 'P_01544_RIGHT_MLO_1', 'P_00765_RIGHT_MLO_1', 'P_01039_RIGHT_MLO_3', 'P_01210_RIGHT_MLO_1'],
       ['IMG440_279', 'IMG495_340', 'IMG420_261', 'IMG367_215', 'IMG280_155', 'IMG485_330']]
TIT = ['(a) CBIS-DDSM (development and internal test)', '(b) DMID (external test)']
mpl.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'pdf.fonttype': 42})
W = 174 / 25.4
H = 3.4; fig = plt.figure(figsize=(W, H))
gap = 0.012; tw = (1 - 5 * gap) / 6; twi = tw * W
for r in range(2):
    top = H - r * 1.78                      # inches from bottom
    fig.text(0, (top - 0.02) / H, TIT[r], fontsize=9.5, fontweight='bold', va='top')
    ybot = top - 0.24 - twi
    for c in range(6):
        y0, y1 = ROWS[r]; x0, x1 = COLS[r][c]
        ax = fig.add_axes([c * (tw + gap), ybot / H, tw, twi / H]); ax.imshow(im[y0:y1, x0:x1], interpolation='none'); ax.axis('off')
        p = LAB[r][c].split('_'); lab = '_'.join(p[:2]) + '\n' + '_'.join(p[2:]) if r == 0 else LAB[r][c]
        fig.text(c * (tw + gap) + tw / 2, (ybot - 0.04) / H, lab, ha='center', va='top', fontsize=8, linespacing=1.1)
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf', dpi=600, metadata={'Creator': None, 'Producer': None, 'CreationDate': None})
