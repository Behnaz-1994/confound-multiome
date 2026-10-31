"""Fixed-effect within-line Fisher-z combination vs linear mixed model (prostate, GSE251977).
300 pairs drawn at random from the pairs with |r_pool| > 0.15 in the original link table; for each pair
expression ~ accessibility + (1 | cell line) is fitted and the fixed-effect slope is put on the correlation
scale (slope * sd(accessibility) / sd(expression)). Reported in the manuscript (Section 3.1, Supplementary
Fig. S2): Pearson 0.663, Spearman 0.698; pairs with |estimate| > 0.1: 2/300 (Fisher-z) vs 15/300 (LMM).
Usage:  python 05_lmm_validation.py --data "<folder that contains checkpoints/>"
"""
import os, argparse, warnings
import numpy as np, pandas as pd, scanpy as sc, matplotlib.pyplot as plt
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"C:\Users\Digikey\Desktop\prostate_rebuild")
args, _ = ap.parse_known_args()
CK = os.path.join(args.data, "checkpoints")
OUT = os.path.join(args.data, "revision_out"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(args.data, "revision_figures"); os.makedirs(FIG, exist_ok=True)

rna = sc.read_h5ad(os.path.join(CK, "rna_processed.h5ad"))
atac = sc.read_h5ad(os.path.join(CK, "atac_processed.h5ad"))
sc.pp.normalize_total(atac, target_sum=1e4); sc.pp.log1p(atac)
links = pd.read_csv(os.path.join(CK, "grn_links_fisher.csv"))     # original link table (corr_naive, corr_partial)
Rx = rna.X.tocsc() if issparse(rna.X) else rna.X
Ax = atac.X.tocsc() if issparse(atac.X) else atac.X
gi = {g: i for i, g in enumerate(rna.var_names)}; pj = {p: i for i, p in enumerate(atac.var_names)}
cell_line = rna.obs["cell_line"].values

np.random.seed(42)
strong = links[links["corr_naive"].abs() > 0.15]
print(f"pairs with |r_pool| > 0.15: {len(strong):,}")
sub = strong.sample(n=min(300, len(strong)), random_state=42).reset_index(drop=True)

def col(M, j):
    v = M[:, j]; return np.asarray(v.todense()).ravel() if issparse(M) else np.asarray(v).ravel()

rows = []
for k, row in sub.iterrows():
    if row["gene"] not in gi or row["peak"] not in pj: continue
    ge = col(Rx, gi[row["gene"]]); pa = col(Ax, pj[row["peak"]])
    if ge.std() < 1e-6 or pa.std() < 1e-6: continue
    df = pd.DataFrame({"expression": ge, "accessibility": pa, "cell_line": cell_line})
    try:
        m = smf.mixedlm("expression ~ accessibility", df, groups=df["cell_line"]).fit(reml=False, method="lbfgs")
        rows.append({"gene": row["gene"], "peak": row["peak"], "naive": row["corr_naive"], "fisher_z": row["corr_partial"],
                     "lmm": m.fe_params["accessibility"] * (pa.std() / ge.std())})
    except Exception:
        continue
    if (k + 1) % 50 == 0: print(f"  {k + 1}/{len(sub)}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "lmm_vs_fisherz_strong.csv"), index=False)
r_p, _ = pearsonr(res["fisher_z"], res["lmm"]); r_s, _ = spearmanr(res["fisher_z"], res["lmm"])
hc_f = int((res["fisher_z"].abs() > 0.1).sum()); hc_l = int((res["lmm"].abs() > 0.1).sum()); n = len(res)
print(f"\n{n} pairs fitted | Pearson {r_p:.3f} | Spearman {r_s:.3f}")
print(f"|estimate| > 0.1: Fisher-z {hc_f}/{n} ({100 * (1 - hc_f / n):.0f}% lost), LMM {hc_l}/{n} ({100 * (1 - hc_l / n):.0f}% lost)")

# ---------- Supplementary Fig. S2 ----------
plt.rcParams.update({"font.size": 11, "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 0.85]})
a1.scatter(res["fisher_z"], res["lmm"], s=28, alpha=0.55, color="#2980b9", edgecolors="none")
lim = [min(res["fisher_z"].min(), res["lmm"].min()) - 0.02, max(res["fisher_z"].max(), res["lmm"].max()) + 0.02]
a1.plot(lim, lim, "k--", lw=1, label="y = x")
a1.axhline(0.1, color="#c0392b", ls=":", lw=1, alpha=0.7); a1.axvline(0.1, color="#c0392b", ls=":", lw=1, alpha=0.7)
a1.set_xlabel("Within-group Fisher-z estimate"); a1.set_ylabel("Linear mixed-effects estimate")
a1.set_title(f"(a) Per-pair agreement\nPearson r = {r_p:.2f}, Spearman ρ = {r_s:.2f}", fontsize=11)
a1.legend(frameon=False, fontsize=9); a1.set_xlim(lim); a1.set_ylim(lim)
bars = a2.bar(["Naive\n(pooled)", "Fisher-z\n(control)", "LMM\n(control)"], [n, hc_f, hc_l],
              color=["#c0392b", "#2980b9", "#27ae60"], alpha=0.85, width=0.6)
for b, c in zip(bars, [n, hc_f, hc_l]):
    a2.text(b.get_x() + b.get_width() / 2, c + 4, f"{c}", ha="center", fontweight="bold")
a2.set_ylabel("Pairs with |estimate| > 0.1"); a2.set_title("(b) Same conclusion under both estimators", fontsize=11)
a2.set_ylim(0, n * 1.15)
plt.tight_layout()
for ext in ("png", "pdf"): plt.savefig(os.path.join(FIG, f"FigS2_LMM_validation.{ext}"), bbox_inches="tight")
plt.close()
print("saved:", os.path.join(FIG, "FigS2_LMM_validation.png"))
