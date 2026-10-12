"""Fig. 1, study design.
Run from the repository root:  python analysis/make_study_design.py   (writes to figures/)"""
# Unpack the stored results (results/*.zip) on first use and run from the repository root
import os, zipfile, glob
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
for _z in glob.glob(os.path.join("results", "*.zip")):
    if not os.path.isdir(os.path.splitext(_z)[0]):
        zipfile.ZipFile(_z).extractall("results")
os.makedirs("figures", exist_ok=True)
"""Fig. 1 (study design), colour-coded version: every box has a coloured title band; protocols, models, evaluation and the
external cohort each have their own hue (Okabe-Ito based, no red). Content identical to make_design2.py --tokA."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
plt.rcParams.update({"font.family": "DejaVu Serif", "pdf.fonttype": 42})
INK, MUTED, LINE = "#111827", "#475569", "#334155"
C = {"data": "#334155", "A": "#1d5fa8", "B": "#0b7a6e", "frozen": "#5b3cc4", "eval": "#1e3a8a", "ext": "#b45309", "rq": "#1f2937", "sec": "#64748b"}
TINT = {"data": "#eef2f6", "A": "#e6f0fb", "B": "#e3f4f1", "frozen": "#efebfb", "eval": "#e8edf8", "ext": "#fdf1e3", "rq": "#f3f4f6", "sec": "#f8fafc"}
GRP = ["#0072B2", "#E69F00", "#009E73", "#56B4E9"]
W = 7.0; H = 10.4
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
TF, BF, HB = 8.6, 7.7, 0.30

def card(x, y, w, h, title, lines, key, ls="-", tcol="white", band=True):
    col, tint = C[key], TINT[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=tint, ec=col, lw=1.3, ls=ls, zorder=2))
    if band:
        ax.add_patch(FancyBboxPatch((x, y + h - HB), w, HB, boxstyle="round,pad=0,rounding_size=0.06", fc=col, ec=col, lw=1.3, zorder=3))
        ax.add_patch(Rectangle((x, y + h - HB), w, HB / 2, fc=col, ec="none", zorder=3))
        ax.text(x + w / 2, y + h - HB / 2, title, ha="center", va="center", fontsize=TF, weight="bold", color=tcol, zorder=4)
        top = y + h - HB - 0.2
    else:
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="center", fontsize=TF, weight="bold", color=col, zorder=4); top = y + h - 0.45
    for i, t in enumerate(lines):
        ax.text(x + w / 2, top - i * 0.2, t, ha="center", va="center", fontsize=BF, color=INK, zorder=4)
def arrow(x1, y1, x2, y2, style="-|>", ls="-", col=LINE, lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle=style, color=col, lw=lw, ls=ls, mutation_scale=11, shrinkA=0, shrinkB=0), zorder=1)
def badge(n, yc, key):
    ax.add_patch(Circle((0.24, yc), 0.16, fc=C[key], ec="white", lw=1.2, zorder=5))
    ax.text(0.24, yc, str(n), ha="center", va="center", fontsize=8.5, color="white", weight="bold", zorder=6)
hb = lambda n: HB + 0.18 + 0.2 * n + 0.08

m = 0.52; gap = 0.8; cw = (W - m - 0.1 - gap) / 2; xa = m; xb = m + cw + gap; cx = (xa + xb + cw) / 2
y = H - 0.1
# 1 data
h = hb(1); y -= h; card(cx - 2.2, y, 4.4, h, "LC25000 lung subset", ["15,000 images · adenocarcinoma, benign, squamous"], "data"); badge(1, y + h / 2, "data"); prev = y
# 2 dedup
h = hb(2); y -= 0.28 + h; arrow(cx, prev, cx, y + h)
card(cx - 2.2, y, 4.4, h, "Deduplication and source-tile grouping", ["MD5 deduplication: 14,195 images", "LC25000-clean grouping: 740 source-tile groups"], "data")
badge(2, y + h / 2, "data"); prev = y
# 3 partitioning
h = 1.62; y -= 0.38 + h
arrow(cx - 0.7, prev, xa + cw / 2, y + h); arrow(cx + 0.7, prev, xb + cw / 2, y + h)
SPLIT = {"A": {"Train": [0, 1, 2, 3, 0, 1, 2, 3], "Val": [1, 3], "Test": [0, 2, 3]},
         "B": {"Train": [0, 0, 1, 1, 0, 1, 0, 1], "Val": [2, 2], "Test": [3, 3, 3]}}
for key, x0, title, sub, note in [("A", xa, "Protocol A: image-level split", "class-stratified 70/15/15 · seeds 42–46", "groups shared across\npartitions (leakage)"),
                                  ("B", xb, "Protocol B: source-group split", "whole groups per partition · seeds 42–46", "each group in one\npartition only")]:
    card(x0, y, cw, h, title, [sub], key)
    sq = 0.15; yy = y + h - HB - 0.62
    for part, cols in SPLIT[key].items():
        ax.text(x0 + 0.18, yy + sq / 2, part, ha="left", va="center", fontsize=BF, color=INK, zorder=4)
        for k, c in enumerate(cols):
            ax.add_patch(Rectangle((x0 + 0.72 + k * (sq + 0.045), yy), sq, sq, fc=GRP[c], ec="white", lw=0.5, zorder=4))
        yy -= 0.23
    ax.text(x0 + cw - 0.12, y + 0.3, note, ha="right", va="center", fontsize=7.0, color=C[key], style="italic", weight="bold", zorder=4)
badge(3, y + h / 2, "A"); prev = y
ax.text(xa + cw / 2 + 0.1, y - 0.07, "squares: images · colour: source group", ha="left", va="top", fontsize=6.5, color=MUTED, style="italic")
# 4 models + frozen band
h = hb(2); y -= 0.42 + h
arrow(xa + cw / 2, prev, xa + cw / 2, y + h); arrow(xb + cw / 2, prev, xb + cw / 2, y + h)
for key, x0 in [("A", xa), ("B", xb)]:
    card(x0, y, cw, h, "Fine-tuned models (20 networks)", ["MobileNetV2 · EfficientNetB0", "Hybrid (attention) · TokenMLP (matched)"], key)
ytop4 = y + h
fh = hb(1); y -= 0.12 + fh
card(xa, y, xb + cw - xa, fh, "Frozen-feature baselines · both protocols · 30 classifiers",
     ["Phikon (pathology) · ViT-B/16 and EfficientNetB0 (ImageNet) + logistic regression"], "frozen")
badge(4, (ytop4 + y) / 2, "frozen"); prev = y
# 5 internal evaluation
h = hb(2); y -= 0.38 + h
arrow(xa + cw / 2, prev, xa + cw / 2, y + h); arrow(xb + cw / 2, prev, xb + cw / 2, y + h)
card(xa, y, cw, h, "Internal evaluation", ["mean ± SD over 5 partitions; leakage audit", "clustered permutation tests (Holm)"], "A")
card(xb, y, cw, h, "Internal evaluation", ["mean ± SD over 5 partitions", "clustered permutation tests (Holm)"], "B")
arrow(xa + cw + 0.02, y + h / 2 - 0.05, xb - 0.02, y + h / 2 - 0.05, style="<|-|>", col=C["eval"], lw=1.4)
ax.text(cx, y + h / 2 + 0.06, "paired\nby seed", ha="center", va="bottom", fontsize=6.7, color=C["eval"], weight="bold", linespacing=1.0)
ax.text(cx, y + h / 2 - 0.15, "Δ = B − A", ha="center", va="top", fontsize=6.6, color=C["eval"], weight="bold")
badge(5, y + h / 2, "eval"); prev = y
# secondary analyses
sh = hb(2); y -= 0.24 + sh; sw = cw - 0.75
card(xa, y, sw, sh, "Secondary analyses", ["initial seed-42 run: curves,", "efficiency, saliency, sensitivity"], "sec", ls="--")
arrow(xa + sw / 2, prev, xa + sw / 2, y + sh, ls="--", col=C["sec"])
# barrier
yb = y - 0.26
ax.plot([m, W - 0.1], [yb, yb], ls=(0, (5, 3)), color=C["ext"], lw=1.1)
ax.text(W - 0.1, yb + 0.07, "external cohort: no retraining, tuning or model selection", ha="right", va="bottom", fontsize=7.0, color=C["ext"], style="italic", weight="bold", bbox=dict(fc="white", ec="none", pad=1.5), zorder=6)
# 6 external
h = hb(3); y = yb - 0.24 - h
arrow(xa + cw - 0.35, prev, xa + cw - 0.35, y + h, col=C["ext"]); arrow(xb + 0.35, prev, xb + 0.35, y + h, col=C["ext"])
card(m + 0.35, y, W - m - 0.1 - 0.7, h, "External evaluation on LungHist700",
     ["691 images · 45 patients · 20× and 40× · 7 classes mapped to 3",
      "12 tiles of 400 × 400 px per image, probabilities averaged",
      "patient-cluster bootstrap CIs · patient–class aggregation"], "ext")
badge(6, y + h / 2, "ext")
# research questions
rh = 0.9; y -= 0.24 + rh
card(m, y, W - m - 0.1, rh, "Research questions and the stages that answer them", [], "rq")
ax.text(cx, y + 0.38, "RQ1 leakage (3, 5)  ·  RQ2 ranking stability (5)  ·  RQ3 attention vs capacity (4, 5)", ha="center", va="center", fontsize=7.4, color=INK)
ax.text(cx, y + 0.16, "RQ4 external transfer and saliency (6)  ·  RQ5 pretrained representation (4, 6)", ha="center", va="center", fontsize=7.4, color=INK)
ax.set_ylim(y - 0.08, H); fig.set_size_inches(W, H - (y - 0.08)); ax.set_position([0, 0, 1, 1])
fig.savefig("figures/fig_design.pdf"); fig.savefig("figures/fig_design.png", dpi=600); plt.close(fig); print("ok", round(H - (y - 0.08), 2))
