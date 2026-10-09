"""Fig. 1, study design (run with --tokA, as in the manuscript).
Run from the repository root:  python analysis/make_study_design.py   (writes to figures/)"""
# Unpack the stored results (results/*.zip) on first use and run from the repository root
import os, zipfile, glob
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
for _z in glob.glob(os.path.join("results", "*.zip")):
    if not os.path.isdir(os.path.splitext(_z)[0]):
        zipfile.ZipFile(_z).extractall("results")
os.makedirs("figures", exist_ok=True)
"""Fig. 1 (study design), redesigned: numbered stages, drawn leakage mechanism, shared frozen-baseline band,
secondary analyses in a dashed box, visible 'no retraining' barrier. No red anywhere."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
plt.rcParams.update({"font.family": "DejaVu Serif", "pdf.fonttype": 42})
TOK_A = "--tokA" in sys.argv          # TokenMLP also trained under Protocol A
INK, MUTED, LINE = "#1f2937", "#4b5563", "#4b5563"
SLATE, FILL_B, GREY = "#334155", "#e2e8f0", ("#f3f4f6", "#6b7280")
FM = ("#f1effa", "#4a3aa7"); EXT = ("#fdf3e1", "#9a6412"); RQ = ("#f8fafc", "#64748b")
GRP = ["#0072B2", "#E69F00", "#009E73", "#56B4E9"]          # Okabe-Ito, no red
W = 6.85; H = 9.6
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
TF, BF = 8.6, 7.6

def rbox(x, y, w, h, fc, ec, ls="-", lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07", fc=fc, ec=ec, lw=lw, ls=ls))
def text_box(x, y, w, h, title, lines, fc, ec, tcol=None, ls="-"):
    rbox(x, y, w, h, fc, ec, ls)
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="center", fontsize=TF, weight="bold", color=tcol or ec)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.45 - i * 0.19, t, ha="center", va="center", fontsize=BF, color=INK)
def arrow(x1, y1, x2, y2, style="-|>", ls="-", col=LINE):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=col, lw=1.0, ls=ls, mutation_scale=9, shrinkA=0, shrinkB=0))
def badge(n, yc):
    ax.add_patch(Circle((0.2, yc), 0.13, fc=SLATE, ec="none")); ax.text(0.2, yc, str(n), ha="center", va="center", fontsize=7.5, color="white", weight="bold")
hb = lambda n: 0.34 + 0.19 * n + 0.1

m = 0.48; gap = 0.62; cw = (W - m - 0.12 - gap) / 2; xa = m; xb = m + cw + gap; cx = (xa + xb + cw) / 2
y = H - 0.1
# 1 data
h = hb(1); y -= h; text_box(cx - 2.1, y, 4.2, h, "LC25000 lung subset", ["15,000 images · adenocarcinoma, benign, squamous"], *GREY, tcol=INK); badge(1, y + h / 2); prev = y
# 2 dedup
h = hb(2); y -= 0.26 + h; arrow(cx, prev, cx, y + h)
text_box(cx - 2.1, y, 4.2, h, "Deduplication and source-tile grouping", ["MD5 deduplication: 14,195 images", "LC25000-clean grouping: 740 source-tile groups"], *GREY, tcol=INK)
badge(2, y + h / 2); prev = y
# 3 partitioning with drawn leakage schematic
h = 1.42; y -= 0.34 + h
arrow(cx - 0.6, prev, xa + cw / 2, y + h); arrow(cx + 0.6, prev, xb + cw / 2, y + h)
SPLIT = {"A": {"Train": [0, 1, 2, 3, 0, 1, 2, 3], "Val": [1, 3], "Test": [0, 2, 3]},
         "B": {"Train": [0, 0, 1, 1, 0, 1, 0, 1], "Val": [2, 2], "Test": [3, 3, 3]}}
for key, x0, title, sub in [("A", xa, "Protocol A: image-level split", "class-stratified 70 / 15 / 15 · seeds 42–46"),
                             ("B", xb, "Protocol B: source-group split", "whole groups per partition · seeds 42–46")]:
    rbox(x0, y, cw, h, "white" if key == "A" else FILL_B, SLATE)
    ax.text(x0 + cw / 2, y + h - 0.2, title, ha="center", va="center", fontsize=TF, weight="bold", color=SLATE)
    ax.text(x0 + cw / 2, y + h - 0.42, sub, ha="center", va="center", fontsize=BF, color=INK)
    sq = 0.13; yy = y + h - 0.68
    for part, cols in SPLIT[key].items():
        ax.text(x0 + 0.18, yy + sq / 2, part, ha="left", va="center", fontsize=BF, color=INK)
        for k, c in enumerate(cols):
            ax.add_patch(Rectangle((x0 + 0.68 + k * (sq + 0.04), yy), sq, sq, fc=GRP[c], ec="white", lw=0.4))
        yy -= 0.2
    ax.text(x0 + cw - 0.12, y + 0.3, "groups shared across\npartitions (leakage)" if key == "A" else "each group in one\npartition only",
            ha="right", va="center", fontsize=7.0, color=MUTED, style="italic")
badge(3, y + h / 2); prev = y
ax.text(xa, y - 0.07, "squares: images; colour: source group", ha="left", va="top", fontsize=6.6, color=MUTED)
# 4 models + shared frozen-baseline band
h = hb(2); y -= 0.38 + h
arrow(xa + cw / 2, prev, xa + cw / 2, y + h); arrow(xb + cw / 2, prev, xb + cw / 2, y + h)
la = (["MobileNetV2, EfficientNetB0, Hybrid,", "TokenMLP: 20 fine-tuned networks"] if TOK_A else ["MobileNetV2 · EfficientNetB0 · Hybrid", "15 fine-tuned networks"])
text_box(xa, y, cw, h, "Fine-tuned models", la, "white", SLATE)
text_box(xb, y, cw, h, "Fine-tuned models", ["MobileNetV2, EfficientNetB0, Hybrid,", "TokenMLP: 20 fine-tuned networks"], FILL_B, SLATE)
ytop4 = y + h
fh = hb(1) - 0.02; y -= 0.1 + fh
text_box(xa, y, xb + cw - xa, fh, "Frozen-feature baselines (both protocols, 30 classifiers)",
         ["Phikon (pathology) · ViT-B/16 and EfficientNetB0 (ImageNet) + logistic regression"], *FM)
badge(4, (ytop4 + y) / 2); prev = y
# 5 internal evaluation with paired arrow
h = hb(2); y -= 0.34 + h
arrow(xa + cw / 2, prev, xa + cw / 2, y + h); arrow(xb + cw / 2, prev, xb + cw / 2, y + h)
text_box(xa, y, cw, h, "Internal evaluation", ["mean ± SD over 5 partitions; leakage audit", "clustered permutation tests (Holm)"], "white", SLATE)
text_box(xb, y, cw, h, "Internal evaluation", ["mean ± SD over 5 partitions", "clustered permutation tests (Holm)"], FILL_B, SLATE)
arrow(xa + cw, y + h / 2, xb, y + h / 2, style="<|-|>")
ax.text(cx, y + h / 2 + 0.16, "paired\nby seed", ha="center", va="bottom", fontsize=6.6, color=MUTED, linespacing=1.0)
ax.text(cx, y + h / 2 - 0.1, "Δ = B − A", ha="center", va="top", fontsize=6.6, color=MUTED)
badge(5, y + h / 2); prev = y
# secondary analyses (dashed)
sh = hb(2); y -= 0.22 + sh
sw = cw - 0.75
text_box(xa, y, sw, sh, "Secondary analyses", ["initial seed-42 run: curves,", "efficiency, saliency, sensitivity"], "white", MUTED, tcol=MUTED, ls="--")
arrow(xa + sw / 2, prev, xa + sw / 2, y + sh, ls="--", col=MUTED)
prev2 = y
# no-retraining barrier
yb = y - 0.24
ax.plot([m, W - 0.12], [yb, yb], ls=(0, (4, 3)), color=MUTED, lw=0.9)
ax.text(W - 0.12, yb + 0.07, "external cohort: no retraining, tuning or model selection", ha="right", va="bottom", fontsize=6.8, color=MUTED, style="italic")
# 6 external
h = hb(3); y = yb - 0.22 - h
arrow(xa + cw - 0.35, prev, xa + cw - 0.35, y + h); arrow(xb + 0.35, prev, xb + 0.35, y + h)
text_box(m + 0.4, y, W - m - 0.12 - 0.8, h, "External evaluation on LungHist700",
         ["691 images · 45 patients · 20× and 40× · 7 classes mapped to 3",
          "12 tiles of 400 × 400 px per image, probabilities averaged",
          "patient-cluster bootstrap CIs · patient–class aggregation"], *EXT)
badge(6, y + h / 2); prev = y
# research questions linked to stages
rh = 0.8; y -= 0.22 + rh
rbox(m, y, W - m - 0.12, rh, *RQ)
ax.text(cx, y + rh - 0.18, "Research questions (answering stages)", ha="center", va="center", fontsize=TF, weight="bold", color=SLATE)
ax.text(cx, y + 0.36, "RQ1 leakage (stages 3, 5)  ·  RQ2 ranking stability (5)  ·  RQ3 attention vs capacity (4, 5)", ha="center", va="center", fontsize=7.2, color=INK)
ax.text(cx, y + 0.15, "RQ4 external transfer and saliency (6)  ·  RQ5 pretrained representation (4, 6)", ha="center", va="center", fontsize=7.2, color=INK)
ax.set_ylim(y - 0.08, H)
fig.set_size_inches(W, H - (y - 0.08)); ax.set_position([0, 0, 1, 1])
fig.savefig("figures/fig_design.pdf"); fig.savefig("figures/fig_design.png", dpi=600); plt.close(fig); print("ok, height", round(H - (y - 0.08), 2))
