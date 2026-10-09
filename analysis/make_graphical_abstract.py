"""Graphical abstract.
Run from the repository root:  python analysis/make_graphical_abstract.py   (writes to figures/)"""
# Unpack the stored results (results/*.zip) on first use and run from the repository root
import os, zipfile, glob
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
for _z in glob.glob(os.path.join("results", "*.zip")):
    if not os.path.isdir(os.path.splitext(_z)[0]):
        zipfile.ZipFile(_z).extractall("results")
os.makedirs("figures", exist_ok=True)
"""Graphical abstract (separate submission file): four panels summarising the main findings. Values from the result files."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85"})
INK, MUTED = "#1f2937", "#4b5563"
R = "results"; S = [42, 43, 44, 45, 46]
M = ["MobileNetV2", "EfficientNetB0", "Hybrid_EffNetB0_Transformer", "EffNetB0_TokenMLP"]; LAB = ["MobileNetV2", "EfficientNetB0", "Hybrid", "TokenMLP"]
COL = ["#2a78d6", "#E69F00", "#1baf7a", "#4d4d4d"]
J = lambda p: json.load(open(p))
acc = lambda P, m, k: np.mean([J(f"{R}/protocol{P}/seed{s}/{m}/{k}.json")["accuracy"] for s in S]) * 100
fig = plt.figure(figsize=(10.0, 4.0))
fig.text(0.5, 0.955, "Beyond the LC25000 benchmark: what changes when evaluation is made stricter", ha="center", fontsize=12, weight="bold", color=INK)
# panel 1: leakage schematic
ax = fig.add_axes([0.015, 0.1, 0.22, 0.74]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.5, 0.97, "1  Source-tile leakage", ha="center", fontsize=10, weight="bold", color=INK)
GRP = ["#0072B2", "#E69F00", "#009E73", "#56B4E9"]
for yy, (name, A, B) in zip([0.68, 0.48], [("Train", [0, 1, 2, 3, 0, 1], [0, 0, 1, 1, 0, 1]), ("Test", [0, 2, 3], [3, 3, 2])]):
    pass
def bars(x0, y0, title, split):
    ax.text(x0 + 0.2, y0 + 0.2, title, ha="center", fontsize=8.5, color=INK)
    for r, (part, cols) in enumerate(split):
        ax.text(x0 - 0.02, y0 + 0.07 - r * 0.12, part, ha="right", va="center", fontsize=7.5, color=MUTED)
        for k, c in enumerate(cols):
            ax.add_patch(Rectangle((x0 + k * 0.07, y0 + 0.03 - r * 0.12), 0.06, 0.08, fc=GRP[c], ec="white", lw=0.5))
bars(0.2, 0.62, "Image-level split", [("Train", [0, 1, 2, 3, 0, 1]), ("Test", [0, 2, 3])])
bars(0.2, 0.25, "Source-group split", [("Train", [0, 0, 1, 1, 0, 1]), ("Test", [2, 3, 3])])
ax.text(0.55, 0.02, "≥ 99.95% of image-level\ntest images share a source\ntile with training images", ha="center", fontsize=8.0, color=INK)
# panel 2: internal accuracy A vs B
ax = fig.add_axes([0.29, 0.2, 0.2, 0.6])
x = np.arange(4)
a = [acc("A", m, "internal") for m in M]; b = [acc("B", m, "internal") for m in M]
ax.bar(x - 0.19, a, 0.36, color="white", edgecolor=COL, linewidth=1.4, label="image-level")
ax.bar(x + 0.19, b, 0.36, color=COL, label="source-group")
ax.set_ylim(93, 101.8); ax.set_xticks(x); ax.set_xticklabels(LAB, rotation=30, ha="right", fontsize=7.5); ax.set_ylabel("Internal accuracy (%)", fontsize=8)
ax.tick_params(labelsize=7.5)
ax.set_title("2  Accuracy falls in every\npartition (−2.6 to −3.8 points)", fontsize=10, weight="bold", color=INK, pad=6)
ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, handlelength=1.2, columnspacing=0.8)
# panel 3: attention vs capacity
ax = fig.add_axes([0.545, 0.2, 0.17, 0.6])
h = [acc("A", M[2], "internal"), acc("B", M[2], "internal")]; t = [acc("A", M[3], "internal"), acc("B", M[3], "internal")]
x = np.arange(2)
ax.bar(x - 0.19, h, 0.36, color=COL[2], label="Hybrid (attention)"); ax.bar(x + 0.19, t, 0.36, color=COL[3], label="TokenMLP (no attention)")
for xi, (hv, tv) in enumerate(zip(h, t)):
    ax.text(xi - 0.19, hv + 0.12, f"{hv:.2f}", ha="center", fontsize=7, color=INK); ax.text(xi + 0.19, tv + 0.12, f"{tv:.2f}", ha="center", fontsize=7, color=INK)
ax.set_ylim(93, 101.8); ax.set_xticks(x); ax.set_xticklabels(["image-level", "source-group"], fontsize=7.5); ax.tick_params(labelsize=7.5)
ax.set_title("3  Attention adds nothing\nover matched capacity", fontsize=10, weight="bold", color=INK, pad=6)
ax.legend(frameon=False, fontsize=6.8, loc="upper right", handlelength=1.2)
# panel 4: external
ax = fig.add_axes([0.775, 0.2, 0.2, 0.6])
cnn = [acc(P, m, "external") for P in "AB" for m in M]
im = [np.mean([J(f"{R}/imagenet_frozen/{bb}/protocol{P}_seed{s}.json")["external"]["accuracy"] for s in S]) * 100 for bb in ["ViT-B16_ImageNet", "EfficientNetB0_ImageNet"] for P in "AB"]
ph = [np.mean([J(f"{R}/phikon/protocol{P}_seed{s}.json")["external"]["accuracy"] for s in S]) * 100 for P in "AB"]
vals = [np.mean(cnn), np.mean(im), np.mean(ph)]; lo = [min(cnn), min(im), min(ph)]; hi = [max(cnn), max(im), max(ph)]
cols = ["#94a3b8", "#64748b", "#4a3aa7"]
ax.bar(range(3), vals, 0.6, color=cols)
ax.errorbar(range(3), vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none", ecolor=INK, capsize=3, lw=1)
for i, v in enumerate(vals): ax.text(i, hi[i] + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=INK, weight="bold")
ax.axhline(40.5, ls="--", color=MUTED, lw=0.9); ax.text(0.5, 34.5, "majority\nclass", ha="center", va="center", fontsize=6.2, color=MUTED, linespacing=0.9)
ax.set_xticks(range(3)); ax.set_xticklabels(["fine-tuned\nCNNs", "frozen\nImageNet", "frozen\nPhikon"], fontsize=7.3)
ax.set_ylim(30, 95); ax.set_ylabel("LungHist700 accuracy (%)", fontsize=8); ax.tick_params(labelsize=7.5)
ax.set_title("4  External cohort: pathology\npretraining matters most", fontsize=10, weight="bold", color=INK, pad=6)
for ext in ["pdf", "png"]:
    fig.savefig(f"figures/graphical_abstract.{ext}", dpi=300)
from PIL import Image
Image.open("figures/graphical_abstract.png").convert("RGB").save("figures/graphical_abstract.tif", compression="tiff_lzw", dpi=(300, 300))
print("ok", vals, lo, hi)
