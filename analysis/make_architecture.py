"""Fig. 2, architectures of the hybrid model, TokenMLP and the EfficientNetB0 baseline.
Run from the repository root:  python analysis/make_architecture.py   (writes to figures/)"""
# Unpack the stored results (results/*.zip) on first use and run from the repository root
import os, zipfile, glob
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
for _z in glob.glob(os.path.join("results", "*.zip")):
    if not os.path.isdir(os.path.splitext(_z)[0]):
        zipfile.ZipFile(_z).extractall("results")
os.makedirs("figures", exist_ok=True)
"""Architecture figure: shared EfficientNetB0 trunk (top), then (a) hybrid transformer block, (b) parameter-matched TokenMLP block,
(c) EfficientNetB0 baseline without a block; shared pooling and classification head. No red."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
plt.rcParams.update({"font.family": "DejaVu Serif", "pdf.fonttype": 42})
INK, MUTED, LINE = "#1f2937", "#4b5563", "#4b5563"
GREY = ("#f3f4f6", "#6b7280"); SLATE = ("#ffffff", "#334155"); KEY = ("#e6f4f1", "#0f766e"); HEAD = ("#f1effa", "#4a3aa7")
W, H = 6.85, 4.25
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
FS = 7.3

def box(x, y, w, h, txt, pal, sub=None, bold=False, ls="-"):
    fc, ec = pal
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=1.0, ls=ls))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, txt, ha="center", va="center", fontsize=FS, color=INK, weight="bold" if bold else "normal")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=6.3, color=MUTED)
    else:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=FS, color=INK, weight="bold" if bold else "normal")
def arr(x1, y1, x2, y2, col=LINE, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, ls=ls, mutation_scale=8, shrinkA=0, shrinkB=0))
def plus(x, y):
    ax.add_patch(Circle((x, y), 0.08, fc="white", ec=LINE, lw=0.9, zorder=3)); ax.text(x, y, "+", ha="center", va="center", fontsize=8, color=INK, zorder=4)
def residual(x0, x1, y):                       # dashed skip connection drawn above a sub-layer
    ax.plot([x0, x0, x1], [y, y + 0.36, y + 0.36], color=MUTED, lw=0.8, ls="--"); arr(x1, y + 0.36, x1, y + 0.08, col=MUTED, ls="--")

# ---- shared trunk (top row)
yt = 3.78
box(0.55, yt - 0.25, 0.9, 0.5, "Input tile", GREY, "160 × 160 × 3"); arr(1.45, yt, 1.62, yt)
box(1.62, yt - 0.3, 1.45, 0.6, "EfficientNetB0", GREY, "ImageNet-pretrained · 4.05M", bold=True); arr(3.07, yt, 3.24, yt)
gx, cs = 3.27, 0.105; gy = yt - 2.5 * cs
for i in range(5):
    for j in range(5):
        ax.add_patch(Rectangle((gx + j * cs, gy + i * cs), cs, cs, fc="#dbeafe" if (i + j) % 2 else "#bfdbfe", ec="white", lw=0.5))
ax.text(gx + 5 * cs + 0.08, yt + 0.09, "5 × 5 × 1280 feature map", ha="left", va="center", fontsize=6.5, color=MUTED)
ax.text(gx + 5 * cs + 0.08, yt - 0.11, "= 25 tokens of dimension 1280", ha="left", va="center", fontsize=6.5, color=MUTED)
xsplit = 0.32
ax.plot([gx + 2.5 * cs, gx + 2.5 * cs, xsplit, xsplit], [gy, 3.28, 3.28, 0.62], color=LINE, lw=0.9)

# ---- three branches
rows = {"a": 2.72, "b": 1.72, "c": 0.62}
lab = {"a": "Hybrid: transformer block, 499,424 added parameters",
       "b": "TokenMLP: parameter-matched control, 495,552 added parameters",
       "c": "EfficientNetB0 baseline: no added block"}
x0 = 0.5; bh = 0.42
for k, y in rows.items():
    arr(xsplit, y, x0, y)
    ax.text(0.08, y + (0.5 if k != "c" else 0.22), f"({k})", ha="left", va="center", fontsize=7.8, color=INK, weight="bold")
    ax.text(0.36, y + (0.5 if k != "c" else 0.22), lab[k], ha="left", va="center", fontsize=6.9, color=MUTED, style="italic")
LN, AT, MLPW = 0.7, 1.36, 1.0
# (a)
y = rows["a"]; x = x0
box(x, y - bh / 2, LN, bh, "LayerNorm", SLATE); arr(x + LN, y, x + LN + 0.1, y)
xa = x + LN + 0.1
box(xa, y - bh / 2 - 0.05, AT, bh + 0.1, "Self-attention", KEY, "2 heads, d$_k$ = 16 · 165,216 ", bold=True)
xp1 = xa + AT + 0.2; arr(xa + AT, y, xp1 - 0.08, y); plus(xp1, y); residual(x - 0.05, xp1, y)
xl2 = xp1 + 0.18; arr(xp1 + 0.08, y, xl2, y)
box(xl2, y - bh / 2, LN, bh, "LayerNorm", SLATE); xm = xl2 + LN + 0.1; arr(xl2 + LN, y, xm, y)
box(xm, y - bh / 2 - 0.05, MLPW, bh + 0.1, "Token MLP", SLATE, "hidden 128 · 329,088")
xp2 = xm + MLPW + 0.2; arr(xm + MLPW, y, xp2 - 0.08, y); plus(xp2, y); residual(xp1 + 0.1, xp2, y)
xend = xp2 + 0.08
# (b)
y = rows["b"]
box(x, y - bh / 2, LN, bh, "LayerNorm", SLATE); arr(x + LN, y, xa, y)
box(xa, y - bh / 2 - 0.05, AT, bh + 0.1, "no token mixing", KEY, "attention removed", ls="--")
arr(xa + AT, y, xm, y)
box(xm, y - bh / 2 - 0.05, MLPW, bh + 0.1, "Token MLP", SLATE, "hidden 192 · 492,992")
arr(xm + MLPW, y, xp2 - 0.08, y); plus(xp2, y); residual(x - 0.05, xp2, y)
# (c)
y = rows["c"]
ax.plot([x0, xend], [y, y], color=LINE, lw=0.9)
# ---- head
xh = xend + 0.28; ym = (rows["a"] + rows["c"]) / 2
for y in rows.values():
    ax.plot([xend, xh - 0.12, xh - 0.12], [y, y, ym], color=LINE, lw=0.9)
arr(xh - 0.12, ym, xh, ym)
hw = W - 0.08 - xh
box(xh, ym - 0.95, hw, 1.9, "", HEAD)
ax.text(xh + hw / 2, ym + 0.66, "Head", ha="center", va="center", fontsize=FS, weight="bold", color=INK)
for dy, t in [(0.3, "global average\npooling"), (-0.08, "dropout 0.3"), (-0.45, "dense + softmax\n3 classes · 3,843")]:
    ax.text(xh + hw / 2, ym + dy, t, ha="center", va="center", fontsize=6.4, color=INK)
ax.text(W - 0.08, 0.08, "teal: token mixing (present in a, absent in b; MLP width adjusted to match parameters)  ·  dashed: residual connections  ·  numbers: trainable parameters",
        ha="right", va="bottom", fontsize=6.1, color=MUTED)
fig.savefig("figures/fig_arch.pdf"); fig.savefig("figures/fig_arch.png", dpi=600); plt.close(fig); print("ok", round(xend, 2), round(xh, 2))
