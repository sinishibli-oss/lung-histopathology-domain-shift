"""Recompute the statistics reported in the manuscript from the files in results/ and splits/.

Run from the repository root:  python analysis/reproduce_statistics.py
Requires numpy, pandas, scipy, statsmodels.
"""
# Unpack the stored results (results/*.zip) on first use and run from the repository root
import os, zipfile, glob
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
for _z in glob.glob(os.path.join("results", "*.zip")):
    if not os.path.isdir(os.path.splitext(_z)[0]):
        zipfile.ZipFile(_z).extractall("results")

import itertools, json
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

S = [42, 43, 44, 45, 46]
CNN = ["MobileNetV2", "EfficientNetB0", "Hybrid_EffNetB0_Transformer"]
M4 = CNN + ["EffNetB0_TokenMLP"]
J = lambda p: json.load(open(p))
IA = {(s, m): J(f"results/protocolA/seed{s}/{m}/internal.json") for s in S for m in M4}
EA = {(s, m): J(f"results/protocolA/seed{s}/{m}/external.json") for s in S for m in M4}
IB = {(s, m): J(f"results/protocolB/seed{s}/{m}/internal.json") for s in S for m in M4}
EB = {(s, m): J(f"results/protocolB/seed{s}/{m}/external.json") for s in S for m in M4}
for s in S:
    for m in M4:
        pp = pd.read_csv(f"results/protocolB/seed{s}/{m}/external_patient_predictions.csv")
        EB[s, m].setdefault("patient_class_accuracy", float((pp.y_pred == pp.y_true).mean()))
PH = {(P, s): J(f"results/phikon/protocol{P}_seed{s}.json") for P in "AB" for s in S}


def ms(v, sc=100, d=2):
    v = np.asarray(v, float) * sc
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f} [{v.min():.{d}f}–{v.max():.{d}f}]"


def signflip(d):
    d = np.asarray(d, float); obs = abs(d.mean())
    return np.mean([abs((d * np.array(g)).mean()) >= obs - 1e-12 for g in itertools.product([1, -1], repeat=len(d))])


print("== Leakage in Protocol A (share of test images whose source group occurs in training)")
for s in S:
    tr = pd.read_csv(f"splits/protocolA_seed{s}/train.csv"); te = pd.read_csv(f"splits/protocolA_seed{s}/test.csv")
    print(f"  seed {s}: {100 * te.group_id.isin(set(tr.group_id)).mean():.2f}%  (test groups: {te.group_id.nunique()})")
for s in S:
    g = {k: set(pd.read_csv(f"splits/protocolB_seed{s}/{k}.csv").group_id) for k in ["train", "val", "test"]}
    assert not (g["train"] & g["test"]) and not (g["train"] & g["val"]) and not (g["val"] & g["test"])
print("  Protocol B: no source group shared between partitions (all seeds)")

for name, I, E, models in [("Protocol A", IA, EA, M4), ("Protocol B", IB, EB, M4)]:
    print(f"\n== {name}: internal (Tables 2-3) and external (Tables 5-6)")
    for m in models:
        print(f"  {m}")
        for k, sc, d in [("accuracy", 100, 2), ("macro_f1", 1, 4), ("balanced_accuracy", 100, 2), ("ece", 1, 4), ("brier", 1, 4)]:
            print(f"    internal {k:18s} {ms([I[s, m][k] for s in S], sc, d)}")
        print(f"    internal errors      {[I[s, m]['errors'] for s in S]}")
        for k, sc, d in [("accuracy", 100, 1), ("macro_f1", 1, 3), ("balanced_accuracy", 100, 1), ("ece", 1, 3), ("auc_ovr_macro", 1, 3), ("patient_class_accuracy", 100, 1)]:
            print(f"    external {k:18s} {ms([E[s, m][k] for s in S], sc, d)}")

print("\n== Paired change, Protocol B minus Protocol A (same seed)")
for m in M4:
    for lab, I1, I0 in [("internal", IB, IA), ("external", EB, EA)]:
        b = np.array([I1[s, m]["accuracy"] for s in S]); a = np.array([I0[s, m]["accuracy"] for s in S]); d = 100 * (b - a)
        print(f"  {m:28s} {lab}: {d.mean():+.2f} ± {d.std(ddof=1):.2f} pp; paired t p = {stats.ttest_rel(b, a).pvalue:.4f}; exact sign-flip p = {signflip(d):.4f}; per seed {np.round(d, 2)}")

def cluster_signflip(ca, cb, groups, rng, M=100_000):
    """Paired sign-flip permutation test that keeps each source group intact (primary test)."""
    d = pd.Series(ca.astype(float) - cb.astype(float)).groupby(groups).sum().values
    null = np.abs(rng.choice([-1.0, 1.0], size=(M, len(d))) @ d)
    return (1 + (null >= abs(d.sum()) - 1e-9).sum()) / (M + 1)


print("\n== Within-partition comparisons with Holm correction: source-group-clustered sign-flip test (primary) and image-level McNemar (secondary)")
for name, base, pairs in [("Protocol A", "results/protocolA", [(M4[2], M4[0]), (M4[2], M4[1]), (M4[1], M4[0]), (M4[2], M4[3]), (M4[3], M4[1])]),
                          ("Protocol B", "results/protocolB", [(M4[2], M4[0]), (M4[2], M4[1]), (M4[1], M4[0]), (M4[2], M4[3]), (M4[3], M4[1])])]:
    rng = np.random.default_rng(0)
    P = np.zeros((len(S), len(pairs))); C = np.zeros_like(P); D = np.zeros_like(P)
    for j, (a, b) in enumerate(pairs):
        for i, s in enumerate(S):
            ta = pd.read_csv(f"{base}/seed{s}/{a}/test_predictions.csv"); tb = pd.read_csv(f"{base}/seed{s}/{b}/test_predictions.csv")
            ca = (ta.y_pred == ta.y_true).values; cb = (tb.y_pred == tb.y_true).values
            P[i, j] = mcnemar([[(ca & cb).sum(), (ca & ~cb).sum()], [(~ca & cb).sum(), (~ca & ~cb).sum()]], exact=True).pvalue
            C[i, j] = cluster_signflip(ca, cb, ta.group_id.values, rng)
            D[i, j] = 100 * (ca.mean() - cb.mean())
    H = np.array([multipletests(P[i], method="holm")[1] for i in range(len(S))])
    HC = np.array([multipletests(C[i], method="holm")[1] for i in range(len(S))])
    print(f"  {name}")
    for j, (a, b) in enumerate(pairs):
        print(f"    {a} - {b}: {D[:, j].mean():+.2f} ± {D[:, j].std(ddof=1):.2f} pp; clustered Holm-significant in {(HC[:, j] < 0.05).sum()}/5 (seeds {[S[i] for i in range(len(S)) if HC[i, j] < 0.05]}); McNemar Holm-significant in {(H[:, j] < 0.05).sum()}/5")


print("\n== Rankings (first place per partition) and external paired differences (patient bootstrap, uncorrected)")
for name, I, E, base in [("Protocol A", IA, EA, "results/protocolA"), ("Protocol B", IB, EB, "results/protocolB")]:
    firsts = [max(M4, key=lambda m: (I[s, m]["accuracy"])) for s in S]
    ties = [[m for m in M4 if abs(I[s, m]["accuracy"] - max(I[s, x]["accuracy"] for x in M4)) < 1e-12] for s in S]
    print(f"  {name} top model(s) per seed: {ties}")
    for a, b in [(M4[2], M4[0]), (M4[2], M4[1]), (M4[1], M4[0]), (M4[2], M4[3]), (M4[3], M4[1])]:
        out = []
        for s in S:
            fa = pd.read_csv(f"{base}/seed{s}/{a}/external_predictions.csv"); fb = pd.read_csv(f"{base}/seed{s}/{b}/external_predictions.csv")
            ca = (fa.y_pred == fa.y_true).values.astype(float); cb = (fb.y_pred == fb.y_true).values.astype(float); pat = fa.patient_id.values
            rng = np.random.default_rng(0); u = np.unique(pat); idx = {k: np.where(pat == k)[0] for k in u}; bs = []
            for _ in range(2000):
                ss = np.concatenate([idx[k] for k in rng.choice(u, len(u))]); bs.append(ca[ss].mean() - cb[ss].mean())
            out.append((round(100 * (ca.mean() - cb.mean()), 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)))
        d = [o[0] for o in out]
        print(f"    {a} - {b} external: {np.mean(d):+.1f} ± {np.std(d, ddof=1):.1f} pp; CIs excluding zero: {sum(1 for o in out if o[1] > 0 or o[2] < 0)}/5 {out}")

print("\n== Phikon frozen-feature baseline (Table 7)")
for P in "AB":
    print(f"  Protocol {P}: internal {ms([PH[P, s]['internal']['accuracy'] for s in S])}; external {ms([PH[P, s]['external']['accuracy'] for s in S], 100, 1)}; "
          f"balanced {ms([PH[P, s]['external']['balanced_accuracy'] for s in S], 100, 1)}; patient-class {ms([PH[P, s]['external']['patient_class_accuracy'] for s in S], 100, 1)}")

print("\n== Spearman correlation, internal versus external accuracy")
for name, I, E, models in [("Protocol A (20 models)", IA, EA, M4), ("Protocol B (20 models)", IB, EB, M4)]:
    r = stats.spearmanr([I[s, m]["accuracy"] for s in S for m in models], [E[s, m]["accuracy"] for s in S for m in models])
    print(f"  {name}: rho = {r.statistic:.2f}, p = {r.pvalue:.2f}")

print("\n== Phikon versus CNN-based models on LungHist700: Bonferroni-adjusted patient-bootstrap lower bounds (40 comparisons)")
ext = pd.read_csv("results/phikon/protocolA_seed42_external_predictions.csv")
pid = ext.patient_id.values; up = np.unique(pid); pos = [np.where(pid == p)[0] for p in up]
rng = np.random.default_rng(0); NB = 20000
W = np.stack([np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up)) for _ in range(NB)])
cnt = np.array([len(i) for i in pos]); alpha = 0.05 / 40
for P, base, models in [("A", "results/protocolA", M4), ("B", "results/protocolB", M4)]:
    lows = []
    for s in S:
        f = pd.read_csv(f"results/phikon/protocol{P}_seed{s}_external_predictions.csv")
        cf = (f.y_pred == f.y_true).values.astype(float)
        for m in models:
            c = pd.read_csv(f"{base}/seed{s}/{m}/external_predictions.csv"); cc = (c.y_pred == c.y_true).values.astype(float)
            d = cf - cc; sums = np.array([d[i].sum() for i in pos])
            lows.append(np.percentile(100 * (W @ sums) / (W @ cnt), 100 * alpha / 2))
    print(f"  Protocol {P}: adjusted lower bounds {min(lows):.1f}–{max(lows):.1f} percentage points")

print("\n== Frozen ImageNet controls through the Phikon pipeline (Table 7)")
for b in ["ViT-B16_ImageNet", "EfficientNetB0_ImageNet"]:
    IM = {(P, s): J(f"results/imagenet_frozen/{b}/protocol{P}_seed{s}.json") for P in "AB" for s in S}
    for P in "AB":
        print(f"  {b} Protocol {P}: internal {ms([IM[P, s]['internal']['accuracy'] for s in S])}; external {ms([IM[P, s]['external']['accuracy'] for s in S], 100, 1)}; "
              f"balanced {ms([IM[P, s]['external']['balanced_accuracy'] for s in S], 100, 1)}; normal recall {np.mean([IM[P, s]['external']['recall']['lung_n'] for s in S]):.2f}")
    d = [100 * (IM['B', s]['internal']['accuracy'] - IM['A', s]['internal']['accuracy']) for s in S]
    print(f"    paired internal change B - A: {np.mean(d):+.2f} pp, per seed {np.round(d, 2)}")
    for P in "AB":
        diffs, lows = [], []
        for s in S:
            f = pd.read_csv(f"results/phikon/protocol{P}_seed{s}_external_predictions.csv")
            y, pat = f.y_true.values, f.patient_id.values
            ca = (f.y_pred.values == y).astype(float)
            cb = (np.load(f"results/imagenet_frozen/{b}/protocol{P}_seed{s}_external_probs.npy").argmax(1) == y).astype(float)
            rng = np.random.default_rng(0); u = np.unique(pat); idx = {k: np.where(pat == k)[0] for k in u}; bs = []
            for _ in range(2000):
                ss = np.concatenate([idx[k] for k in rng.choice(u, len(u))]); bs.append(ca[ss].mean() - cb[ss].mean())
            diffs.append(100 * (ca.mean() - cb.mean())); lows.append(100 * np.percentile(bs, 2.5))
        print(f"    Phikon minus {b}, Protocol {P}: {min(diffs):.1f}–{max(diffs):.1f} pp (mean {np.mean(diffs):.1f}); 95% CI lower bounds {min(lows):.1f}–{max(lows):.1f}")
