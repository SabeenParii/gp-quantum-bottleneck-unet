import os
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'); os.makedirs(OUTDIR, exist_ok=True)
"""Fig. 2: hybrid U-Net with a bottleneck slot; global vs patch-wise placement; block types; severing layer."""
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib import font_manager

fam = 'Liberation Sans' if any('Liberation Sans' in f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
mpl.rcParams.update({'font.family': fam, 'font.size': 7.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})

W, H = 7.16, 4.6
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 143.2); ax.set_ylim(0, 92); ax.axis('off')

ENC, DEC, SLOT, Q, CL, ID, SEV = '#dbe6f4', '#dbe6f4', '#fbe3cf', '#f6d3d3', '#d9ead3', '#eeeeee', '#fff2b3'
EDGE = '#333333'


def box(x, y, w, h, text, fc, fs=7.2, lw=0.7, ec=EDGE, style='round,pad=0.25,rounding_size=0.8', weight='normal', ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight=weight, linespacing=1.15)


def arrow(x1, y1, x2, y2, ls='-', color=EDGE, lw=0.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=7))


def label(x, y, t):
    ax.text(x, y, t, fontsize=9, fontweight='bold', ha='left', va='top')

# ---------------- (a) U-Net overview ----------------
label(1, 91, '(a)')
ax.text(8, 91, 'Hybrid U-Net (7.9M parameters)', fontsize=8, va='top')
levels = [('32', '256²'), ('64', '128²'), ('128', '64²'), ('256', '32²')]
bw, bh = 11, 6.2
xs_enc, xs_dec = 4, 46
ytop = 77
for i, (f, r) in enumerate(levels):
    y = ytop - i * 9.2
    xe = xs_enc + i * 2.2; xd = xs_dec - i * 2.2
    box(xe, y, bw, bh, f'Conv {f}\n{r}', ENC, fs=6.6)
    box(xd, y, bw, bh, f'Conv {f}\n{r}', DEC, fs=6.6)
    arrow(xe + bw, y + bh / 2, xd, y + bh / 2, ls=(0, (3, 2)), color='#6b6b6b', lw=0.7)
    if i < 3:
        arrow(xe + bw / 2, y, xe + 2.2 + bw / 2, y - 9.2 + bh, color=EDGE)
        arrow(xd - 2.2 + bw / 2, y - 9.2 + bh, xd + bw / 2, y, color=EDGE)
ax.text(30.5, ytop + bh / 2 + 1.2, 'skip connections', fontsize=6.5, color='#555555', ha='center')
# bottleneck conv + slot
yb = ytop - 4 * 9.2 + 1.0
box(14, yb, 11, bh, 'Conv 512\n16²×512', ENC, fs=6.6)
arrow(xs_enc + 3 * 2.2 + bw / 2, ytop - 3 * 9.2, 19.5, yb + bh)
box(29.5, yb - 0.4, 10.5, bh + 0.8, 'bottleneck\nslot', SLOT, fs=6.8, weight='bold', lw=1.1, ec='#c0601a')
arrow(25, yb + bh / 2, 29.5, yb + bh / 2)
arrow(40, yb + bh / 2, 44.2, yb + bh / 2)
box(44.2, yb, 9.5, bh, '16²×512', ENC, fs=6.6)
arrow(48.95, yb + bh, xs_dec - 3 * 2.2 + bw / 2, ytop - 3 * 9.2)
ax.text(4, ytop + bh + 1.0, 'input 256×256 crop (randomized)', fontsize=6.5, color='#555555')
ax.text(46, ytop + bh + 1.0, 'mask (sigmoid)', fontsize=6.5, color='#555555')
ax.text(29, 38.5, 'unet: classical U-Net (31M)   full: no compression in the slot', fontsize=6.3, color='#555555', ha='center')

# ---------------- (b) global, (c) patch-wise ----------------
def row(y, steps, title, lab):
    label(62, y + 13, lab)
    ax.text(69, y + 12.3, title, fontsize=8, va='top')
    x = 63
    for k, (t, fc, w, kw) in enumerate(steps):
        box(x, y, w, 8.2, t, fc, fs=6.3, **kw)
        if k < len(steps) - 1:
            arrow(x + w, y + 4.1, x + w + 2.2, y + 4.1)
        x += w + 2.2

slotkw = dict(lw=1.1, ec='#c0601a')
row(70, [('16²×512', ENC, 8.5, {}),
         ('global avg\npool → 512', SLOT, 10.5, {}),
         ('Dense(4)\ntanh', SLOT, 8.0, {}),
         ('BLOCK\n1 code', '#ffffff', 8.6, dict(lw=1.2, ec='#9b1c1c')),
         ('replace\ncode', SEV, 7.5, dict(ls=(0, (2, 1.5)))),
         ('Dense(512)\nReLU', SLOT, 10.5, {}),
         ('tile\n→ 16²×512', SLOT, 10.5, {})],
    'Global slot: one 4-d code per image', '(b)')
row(47, [('16²×512', ENC, 8.5, {}),
         ('avg pool 4×4\n→ 4²×512', SLOT, 10.5, {}),
         ('1×1 conv\n(4), tanh', SLOT, 9.0, {}),
         ('BLOCK\n16 codes', '#ffffff', 8.6, dict(lw=1.2, ec='#9b1c1c')),
         ('replace\ncode', SEV, 7.5, dict(ls=(0, (2, 1.5)))),
         ('1×1 conv\n(512), ReLU', SLOT, 10.5, {}),
         ('upsample 4×\n→ 16²×512', SLOT, 10.5, {})],
    'Patch-wise slot: one 4-d code per pooled cell (nominal 64×64 px)', '(c)')
ax.text(103.5, 44.2, 'Both placements: 7,856,581–7,856,605 trainable parameters', fontsize=6.3, color='#555555', ha='center')

# ---------------- (d) block types ----------------
label(1, 33, '(d)')
ax.text(8, 32.3, 'BLOCK options (4 → 4)', fontsize=8, va='top')
# circuit diagram
cx0, cy0 = 4, 4
box(cx0 - 1.5, cy0 - 2.5, 66, 28.5, '', Q, lw=0.7, style='round,pad=0.3,rounding_size=1.2')
ax.text(cx0, cy0 + 24.2, 'Q: 4-qubit circuit', fontsize=7.5, fontweight='bold', va='center')
ax.text(cx0 + 20, cy0 + 24.2, '24 weights, 20 effective (final ω inert)', fontsize=6.5, va='center', color='#333333')
qy = [cy0 + 18, cy0 + 13, cy0 + 8, cy0 + 3]
for i, y in enumerate(qy):
    ax.plot([cx0 + 4, cx0 + 62], [y, y], color=EDGE, lw=0.6, zorder=1)
    ax.text(cx0 + 0.5, y, f'x{i + 1}', va='center', fontsize=6.6)
    box(cx0 + 5.5, y - 1.7, 8.8, 3.4, 'RY(πx/2)', '#ffffff', fs=6, style='square,pad=0.05')
    ax.text(cx0 + 63, y, r'$\langle Z\rangle$', va='center', fontsize=6.8)


def layer(x0, name):
    for y in qy:
        box(x0, y - 1.7, 9.6, 3.4, 'Rot(φ,θ,ω)', '#ffffff', fs=5.9, style='square,pad=0.05')
    # CNOT ring 1->2->3->4->1
    xs = [x0 + 12.5, x0 + 15, x0 + 17.5, x0 + 20]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for xx, (c, t) in zip(xs, pairs):
        ax.plot([xx, xx], [qy[c], qy[t]], color=EDGE, lw=0.6, zorder=2)
        ax.add_patch(Circle((xx, qy[c]), 0.55, color=EDGE, zorder=3))
        ax.add_patch(Circle((xx, qy[t]), 1.05, fc='white', ec=EDGE, lw=0.6, zorder=3))
        ax.plot([xx - 1.05, xx + 1.05], [qy[t]] * 2, color=EDGE, lw=0.6, zorder=4)
        ax.plot([xx, xx], [qy[t] - 1.05, qy[t] + 1.05], color=EDGE, lw=0.6, zorder=4)
    ax.text(x0 + 11, qy[-1] - 3.6, name, ha='center', fontsize=6.2, color='#333333')

layer(cx0 + 16.5, 'layer 1: rotations + CNOT ring')
layer(cx0 + 40, 'layer 2')
# classical and identity
box(74, 17, 30, 13, 'Cl: Dense(4), tanh\n20 weights\n(same count as the circuit)', CL, fs=6.8)
box(74, 1.5, 30, 12.5, 'Id: identity\nno weights\n(is a trainable map needed?)', ID, fs=6.8)
# code-replacement explanation
box(108, 1.5, 33.5, 28.5, '', SEV, ls=(0, (2, 1.5)))
ax.text(124.75, 27.5, 'replace code (test time only)', ha='center', va='top', fontsize=7.2, fontweight='bold')
ax.text(110, 22.5, 'Inactive during training.\nAt test time it can replace\nthe BLOCK output with:\n\n• const: mean training code\n• swap: code of a random\n   training image\n\nDice drop = sensitivity of\nthe output to the code.',
        fontsize=6.4, va='top', linespacing=1.2)

for ext in ('pdf', 'svg'):
    fig.savefig(f'{OUTDIR}/fig2_architecture.{ext}')
fig.savefig(OUTDIR + '/fig2_architecture.png', dpi=600)
print('ok')
