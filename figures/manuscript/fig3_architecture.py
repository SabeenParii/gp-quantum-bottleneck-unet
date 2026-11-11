"""Fig. 3: architecture at 174 mm, lettering 8-10 pt (coordinates in mm)."""
import sys, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, 'fig3')
mpl.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42, 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic', 'mathtext.cal': 'Liberation Sans', 'mathtext.fallback': 'stixsans'})
WM, HM = 174, 165
fig = plt.figure(figsize=(WM / 25.4, HM / 25.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WM); ax.set_ylim(0, HM); ax.axis('off')
ENC, SLOT, Q, CL, TR, ID, SEV, E = '#dbe6f4', '#fbe3cf', '#f6d3d3', '#d9ead3', '#dde3f3', '#eeeeee', '#fff2b3', '#333333'
TXT = []
def box(x, y, w, h, text, fc, fs=8, lw=0.8, ec=E, r=1.0, weight='normal', ls='-', sq=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='square,pad=0' if sq else f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls))
    if text:
        t = ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight=weight, linespacing=1.15); TXT.append((t, x, y, w, h))
def arrow(x1, y1, x2, y2, ls='-', color=E, lw=0.9):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=8))
def label(x, y, t, title):
    ax.text(x, y, t, fontsize=10, fontweight='bold', ha='left', va='top'); ax.text(x + 8, y, title, fontsize=9.5, ha='left', va='top')
G = '#555555'
# ---------------- (a) U-Net ----------------
label(1, HM - 1, '(a)', 'Hybrid U-Net (7.9M parameters)')
bw, bh, dy, sh = 17, 8.6, 10, 3.2
ytop = HM - 19; xe0, xd0 = 3, 76
lev = [('32', '256²'), ('64', '128²'), ('128', '64²'), ('256', '32²')]
for i, (f, r) in enumerate(lev):
    y = ytop - i * dy; xe = xe0 + i * sh; xd = xd0 - i * sh
    box(xe, y, bw, bh, f'Conv {f}\n{r}', ENC); box(xd, y, bw, bh, f'Conv {f}\n{r}', ENC)
    arrow(xe + bw, y + bh / 2, xd, y + bh / 2, ls=(0, (3, 2)), color='#6b6b6b', lw=0.8)
    if i < 3:
        arrow(xe + bw / 2, y, xe + sh + bw / 2, y - dy + bh); arrow(xd - sh + bw / 2, y - dy + bh, xd + bw / 2, y)
ax.text((xe0 + bw + xd0) / 2, ytop + bh / 2 + 1.0, 'skip connections', fontsize=8, color=G, ha='center', va='bottom')
ax.text(xe0, ytop + bh + 1.2, 'input 256×256 crop (randomized)', fontsize=8, color=G, va='bottom')
ax.text(xd0 + bw, ytop + bh + 1.2, 'mask (sigmoid)', fontsize=8, color=G, va='bottom', ha='right')
yb = ytop - 4 * dy
box(16, yb, 19, bh, 'Conv 512\n16²×512', ENC)
arrow(xe0 + 3 * sh + bw / 2, ytop - 3 * dy, 25.5, yb + bh)
box(40.5, yb - 0.5, 16, bh + 1, 'bottleneck\nslot', SLOT, weight='bold', lw=1.3, ec='#c0601a')
arrow(35, yb + bh / 2, 40.5, yb + bh / 2); arrow(56.5, yb + bh / 2, 62, yb + bh / 2)
box(62, yb, 16, bh, '16²×512', ENC)
arrow(70, yb + bh, xd0 - 3 * sh + bw / 2, ytop - 3 * dy)
ax.text(1, yb - 3, 'unet: classical U-Net (31M); full: no compression in the slot', fontsize=8, color=G, va='top')
# code-replacement explanation (right of (a))
rx, rw, ry, rh = 103, 70, HM - 52, 44
box(rx, ry, rw, rh, '', SEV, ls=(0, (2, 1.5)))
ax.text(rx + rw / 2, ry + rh - 3, 'replace code (test time only)', ha='center', va='top', fontsize=9, fontweight='bold')
ax.text(rx + 3, ry + rh - 10, 'Inactive during training. At test time it\ncan replace the BLOCK output with:\n\n• const: mean training code\n• swap: code of a random training image\n\nDice drop = sensitivity of the output\nto the code.',
        fontsize=8.5, va='top', linespacing=1.25)
# ---------------- (b), (c) slots ----------------
def row(y, steps, lab, title):
    label(1, y + 15.5, lab, title)
    w = (WM - 2 - 6 * 3.6) / 7; x = 1
    for k, (t, fc, kw) in enumerate(steps):
        box(x, y, w, 11, t, fc, **kw)
        if k < 6: arrow(x + w, y + 5.5, x + w + 3.6, y + 5.5)
        x += w + 3.6
blk = dict(lw=1.4, ec='#9b1c1c'); rep = dict(ls=(0, (2, 1.5)))
row(77, [('16²×512', ENC, {}), ('global avg pool\n→ 512', SLOT, {}), ('Dense(4)\ntanh', SLOT, {}), ('BLOCK\n1 code', '#ffffff', blk),
               ('replace\ncode', SEV, rep), ('Dense(512)\nReLU', SLOT, {}), ('tile\n→ 16²×512', SLOT, {})], '(b)', 'Global slot: one 4-d code per image')
row(53, [('16²×512', ENC, {}), ('avg pool 4×4\n→ 4²×512', SLOT, {}), ('1×1 conv (4)\ntanh', SLOT, {}), ('BLOCK\n16 codes', '#ffffff', blk),
                ('replace\ncode', SEV, rep), ('1×1 conv (512)\nReLU', SLOT, {}), ('upsample 4×\n→ 16²×512', SLOT, {})], '(c)', 'Patch-wise slot: one 4-d code per pooled cell (nominal 64×64 px)')
ax.text(WM / 2, 50.5, 'Both placements: 7,856,581–7,856,605 trainable parameters', fontsize=8, color=G, ha='center', va='top')
# ---------------- (d) blocks ----------------
label(1, 46.5, '(d)', 'BLOCK options (4 → 4)')
cx, cy, cw, ch = 1, 1, 113, 40
box(cx, cy, cw, ch, '', Q, r=1.5)
ax.text(cx + 2, cy + ch - 2.2, 'Q: 4-qubit circuit', fontsize=9, fontweight='bold', va='top')
ax.text(cx + 33, cy + ch - 2.4, '24 weights, 20 effective (final ω inert)', fontsize=8, va='top')
qy = [cy + 28.5, cy + 22, cy + 15.5, cy + 9]
for i, y in enumerate(qy):
    ax.plot([cx + 6, cx + cw - 7], [y, y], color=E, lw=0.6, zorder=1)
    ax.text(cx + 1.8, y, f'$c_{i}$', va='center', fontsize=9)
    box(cx + 7, y - 1.9, 15, 3.8, 'RY(πc/2)', '#ffffff', sq=True)
    ax.text(cx + cw - 6.5, y, r'$\langle Z\rangle$', va='center', fontsize=8.5)
def layer(x0, name):
    for y in qy: box(x0, y - 1.9, 17, 3.8, 'Rot(φ,θ,ω)', '#ffffff', sq=True)
    for xx, (c, t) in zip([x0 + 20, x0 + 24, x0 + 28, x0 + 32], [(0, 1), (1, 2), (2, 3), (3, 0)]):
        ax.plot([xx, xx], [qy[c], qy[t]], color=E, lw=0.6, zorder=2); ax.add_patch(Circle((xx, qy[c]), 0.6, color=E, zorder=3))
        ax.add_patch(Circle((xx, qy[t]), 1.2, fc='white', ec=E, lw=0.6, zorder=3))
        ax.plot([xx - 1.2, xx + 1.2], [qy[t]] * 2, color=E, lw=0.6, zorder=4); ax.plot([xx, xx], [qy[t] - 1.2, qy[t] + 1.2], color=E, lw=0.6, zorder=4)
    ax.text(x0 + 17, qy[-1] - 3.2, name, ha='center', va='top', fontsize=8)
layer(cx + 25, 'layer 1: rotations + CNOT ring'); layer(cx + 62, 'layer 2')
bx, bw2 = 117, 56
box(bx, 28, bw2, 13, 'Cl: Dense(4), tanh\n20 weights (same count)', CL)
box(bx, 14.5, bw2, 13, 'Tr: tanh(A cos(πc/2 + φ))\n20 weights, circuit encoding', TR)
box(bx, 1, bw2, 12.5, 'Id: identity\nno weights', ID)
fig.canvas.draw(); rr = fig.canvas.get_renderer(); bad = 0
for t, x, y, w, h in TXT:
    bb = t.get_window_extent(rr).transformed(ax.transData.inverted())
    if bb.x0 < x + 0.5 or bb.x1 > x + w - 0.5 or bb.y0 < y + 0.3 or bb.y1 > y + h - 0.3: print('overflow', repr(t.get_text()), [round(v, 1) for v in (bb.x0, bb.x1, bb.y0, bb.y1)], (x, x + w, y, y + h)); bad += 1
fig.savefig(out + '.pdf', metadata={'Creator': None, 'Producer': None, 'CreationDate': None}); fig.savefig(out + '.png', dpi=600); print('overflows', bad)
