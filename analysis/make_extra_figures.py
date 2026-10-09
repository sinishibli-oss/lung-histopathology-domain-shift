"""Training curves and external sensitivity analyses of the initial seed-42 run.
Run from the repository root:  python analysis/make_extra_figures.py   (writes to figures/)"""
# Unpack the stored results (results/*.zip) on first use and run from the repository root
import os, zipfile, glob
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
for _z in glob.glob(os.path.join("results", "*.zip")):
    if not os.path.isdir(os.path.splitext(_z)[0]):
        zipfile.ZipFile(_z).extractall("results")
os.makedirs("figures", exist_ok=True)
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#2b2b29", "xtick.color": "#55554f", "ytick.color": "#55554f"})
INK, MUTED, GRID = "#2b2b29", "#8a8a85", "#e6e6e1"
N = ["MobileNetV2", "EfficientNetB0", "Hybrid_EffNetB0_Transformer"]; LAB = ["MobileNetV2", "EfficientNetB0", "Hybrid"]
COL = ["#2a78d6", "#E69F00", "#1baf7a"]
H = {n: json.load(open(f"results/initial_run_seed42/history_{n}.json")) for n in N}

# ---------- training curves ----------
fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.6), sharex=True)
for j, n in enumerate(N):
    h = H[n]; ep = np.arange(1, len(h["accuracy"]) + 1); s1 = h["stage1_epochs"]
    for r, (k, lab) in enumerate([("accuracy", "Accuracy"), ("loss", "Loss")]):
        a = axs[r, j]
        a.axvspan(s1 + 0.5, ep[-1] + 0.5, color="#f1f1ee", zorder=0, lw=0)
        a.plot(ep, h[k], color=COL[j], lw=1.6, zorder=3)
        a.plot(ep, h["val_" + k], color=COL[j], lw=1.3, ls=(0, (3, 1.6)), marker="o", ms=2.6, mfc="white", mew=0.9, zorder=3)
        a.grid(axis="y", color=GRID, lw=0.7); a.set_axisbelow(True); a.set_xlim(0.5, ep[-1] + 0.5)
        if r == 0:
            a.set_ylim(0.70, 1.005); a.set_title(LAB[j], fontsize=9, color=INK)
            a.text(s1 + 0.9, 0.715, "Stage 2", fontsize=7, color=MUTED, ha="left", va="bottom")
            a.text(s1 + 0.1, 0.715, "Stage 1", fontsize=7, color=MUTED, ha="right", va="bottom")
        else:
            a.set_ylim(0, 0.65); a.set_xlabel("Epoch")
        a.set_xticks([2, 4, 6, 8, 10, 12, 14, 16])
        if j == 0: a.set_ylabel(lab)
        else: a.tick_params(labelleft=False)
axs[0, 0].text(-0.3, 1.08, "(a)", transform=axs[0, 0].transAxes, fontsize=9.5, weight="bold", color=INK)
axs[1, 0].text(-0.3, 1.08, "(b)", transform=axs[1, 0].transAxes, fontsize=9.5, weight="bold", color=INK)
hd = [Line2D([], [], color=INK, lw=1.6, label="Training"),
      Line2D([], [], color=INK, lw=1.3, ls=(0, (3, 1.6)), marker="o", ms=2.6, mfc="white", label="Validation"),
      Patch(facecolor="#f1f1ee", label="Stage 2: fine-tuning")]
fig.legend(handles=hd, loc="lower center", ncol=3, frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.6)
fig.savefig("figures/fig_curves.pdf"); fig.savefig("figures/fig_curves.png", dpi=600); plt.close(fig)

# ---------- external sensitivity analyses ----------
E = json.load(open("results/initial_run_seed42/external_summary.json"))["accuracy_table"]
S = json.load(open("results/initial_run_seed42/summary.json"))["models"]
conds = [("tiled|none|all", "12 tiles,\nno normalisation"), ("tiled|reinhard|all", "12 tiles,\nReinhard"),
         ("whole|none|all", "Whole image,\nno normalisation"), ("whole|reinhard|all", "Whole image,\nReinhard")]
key = {"MobileNetV2": "MobileNetV2", "EfficientNetB0": "EfficientNetB0", "Hybrid_EffNetB0_Transformer": "Hybrid"}
fig, ax = plt.subplots(figsize=(7.4, 3.3)); w = 0.25
for j, n in enumerate(N):
    v = np.array([E[c][key[n]][0] for c, _ in conds]); lo = np.array([E[c][key[n]][1][0] for c, _ in conds]); hi = np.array([E[c][key[n]][1][1] for c, _ in conds])
    x = np.arange(len(conds)) + (j - 1) * w
    ax.bar(x, v, w * 0.92, color=COL[j], label=LAB[j], zorder=2)
    ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=3)
    for xi, vi, hii in zip(x, v, hi): ax.text(xi, hii + 1.2, f"{vi:.1f}", ha="center", va="bottom", fontsize=6.6, color=INK)
ia = 100 * S["Hybrid_EffNetB0_Transformer"]["accuracy"]
ax.axhline(40.5, ls="--", color=MUTED, lw=1, zorder=1, label="Majority-class rate (40.5%)")
ax.axhline(ia, ls=":", color=INK, lw=1.2, zorder=1, label=f"Internal accuracy, hybrid ({ia:.1f}%)")
ax.set_xticks(range(len(conds))); ax.set_xticklabels([l for _, l in conds], fontsize=7.8); ax.set_xlim(-0.5, 3.65)
ax.set_ylim(0, 104); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("LungHist700 accuracy (%)")
ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
hh, ll = ax.get_legend_handles_labels(); o = [2, 3, 4, 0, 1]
ax.legend([hh[i] for i in o], [ll[i] for i in o], loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False, fontsize=7.6, columnspacing=1.4)
fig.tight_layout(); fig.savefig("figures/fig_ext_conditions.pdf"); fig.savefig("figures/fig_ext_conditions.png", dpi=600); plt.close(fig)
print("ok", round(ia, 2))
