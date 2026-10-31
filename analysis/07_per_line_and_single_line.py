"""Per-line correlations of the pooled calls and single-line associations (prostate, GSE251977).
Part 1: distribution of within-line r_k for the pooled calls (Supplementary Fig. S3), count of pooled calls with
|r_k| > 0.1 in exactly one line, and the withdrawn resistance programme of the original submission
(Supplementary Data 1). Part 2: the same single-line count on the matched trans pairs (Section 3.1).
Reads only cached results of 01_prostate_revision.py; nothing is recomputed.
Usage:  python 07_per_line_and_single_line.py --data "<folder that contains checkpoints/ and revision_out/>"
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"C:\Users\Digikey\Desktop\prostate_rebuild")
args, _ = ap.parse_known_args()

# ======================= Part 1 =======================
import os, pickle, json, numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp, matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROS = args.data
CK = os.path.join(PROS, "checkpoints"); OUT = os.path.join(PROS, "revision_out")
FIG = os.path.join(PROS, "revision_figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight"})
THR, FDR, CLIP = 0.1, 0.05, 0.999
S = {}

# ---------- (A) per-line correlations of the pooled calls (cell-line grouping, revised estimator, min_nz = 1) ----------
res = pickle.load(open(os.path.join(OUT, "P1_scan_cellline.pkl"), "rb"))
rn = res["r_naive"]; N = res["N"]; lines = [str(g) for g in res["group_names"]]
pn = 2 * stats.norm.sf(np.abs(np.arctanh(np.clip(rn, -CLIP, CLIP))) * np.sqrt(N - 3))
hn = (np.abs(rn) > THR) & (multipletests(pn, method="fdr_bh")[1] < FDR)          # pooled calls
r = res["r_k"].astype(float)
use = res["gok_k"] & ~np.isnan(r) & (res["nz_k"] >= 1) & (res["gnz_k"] >= 1)    # informative groups
R = np.where(use, r, np.nan)[hn]; rp = rn[hn]
S["pooled_calls"] = int(hn.sum()); print("pooled calls:", S["pooled_calls"], "(expected 5,631)")

strong = np.abs(R) > THR                                  # |r_k| > 0.1 in that line
same = np.sign(R) == np.sign(rp)[:, None]
n_strong = strong.sum(1)
S["n_lines_with_abs_rk_gt_0.1"] = {int(k): int((n_strong == k).sum()) for k in range(len(lines) + 1)}
single = (n_strong == 1)
which = np.where(single, np.argmax(strong, 1), -1)
S["single_line_calls"] = int(single.sum())
S["single_line_same_sign_as_pooled"] = int((single & (strong & same).any(1)).sum())
S["single_line_by_line"] = {lines[k]: int((which == k).sum()) for k in range(len(lines))}
S["per_line"] = {}
for k, L in enumerate(lines):
    v = R[:, k]; ok = ~np.isnan(v)
    S["per_line"][L] = dict(n_cells=int(res["n_k"][k]), informative=int(ok.sum()),
                            median_rk=float(np.nanmedian(v)), median_abs_rk=float(np.nanmedian(np.abs(v))),
                            pct_abs_gt_0_1=float(100 * (np.abs(v[ok]) > THR).mean()) if ok.any() else None)
print(json.dumps(S, indent=1))
pd.DataFrame(np.c_[rp, R], columns=["r_pool"] + [f"r_{L}" for L in lines]).assign(
    gene=np.asarray(res["gene"])[hn], peak=np.asarray(res["peak"])[hn]).to_csv(
    os.path.join(OUT, "S3_pooled_calls_r_by_line.csv.gz"), index=False)

# ---------- Supplementary Fig. S3 ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
data = [rp] + [R[~np.isnan(R[:, k]), k] for k in range(len(lines))]
labs = ["Pooled\n(all cells)"] + [f"{L}\n(n = {int(res['n_k'][k]):,})" for k, L in enumerate(lines)]
vp = a1.violinplot(data, showextrema=False, widths=0.8)
for i, b in enumerate(vp["bodies"]):
    b.set_facecolor("#c0392b" if i == 0 else "#2980b9"); b.set_alpha(0.55)
a1.boxplot(data, widths=0.15, showfliers=False, medianprops=dict(color="k"))
for yv in (THR, -THR): a1.axhline(yv, color="#555555", ls=":", lw=0.8)
a1.axhline(0, color="k", lw=0.5)
a1.set_xticks(range(1, len(labs) + 1)); a1.set_xticklabels(labs, fontsize=8)
a1.set_ylabel("Peak–gene correlation"); a1.set_title(f"(a) Pooled and within-line r for the {hn.sum():,} pooled calls", fontsize=10)
ks = list(range(len(lines) + 1)); cnt = [S["n_lines_with_abs_rk_gt_0.1"][k] for k in ks]
bars = a2.bar(ks, cnt, color="#2980b9")
for b, c in zip(bars, cnt): a2.text(b.get_x() + b.get_width() / 2, c, f"{c:,}", ha="center", va="bottom", fontsize=8)
a2.set_xlabel("Number of lines with |r_k| > 0.1"); a2.set_ylabel("Pooled calls")
a2.set_title("(b) Lines in which each pooled call is strong", fontsize=10)
plt.tight_layout()
for ext in ("png", "pdf"): plt.savefig(os.path.join(FIG, f"FigS3_within_line_r_of_pooled_calls.{ext}"))
plt.close()

# ---------- (B) withdrawn resistance programme: complete lists for transparency ----------
prog = pd.read_csv(os.path.join(CK, "resistance_program_64genes.csv"))
AR_EXCL = {"AR", "KLK2", "KLK3", "FOLH1", "STEAP2", "NKX3-1", "TMPRSS2", "PMEPA1", "FKBP5", "NDRG1"}   # as in the original analysis
prog["in_33_gene_list"] = prog["direction"].astype(str).str.contains("sensitive") & ~prog["gene"].isin(AR_EXCL)
rna = sc.read_h5ad(os.path.join(CK, "rna_processed.h5ad"))
X = rna.X.tocsr() if sp.issparse(rna.X) else sp.csr_matrix(rna.X); ln = rna.obs["cell_line"].astype(str).values
gs = [g for g in prog["gene"] if g in rna.var_names]; gi = [list(rna.var_names).index(g) for g in gs]
for L in ["LNCaP", "VCaP", "22Rv1", "DU145", "MDA-PCa-2B", "H660"]:
    m = np.asarray(X[ln == L][:, gi].mean(0)).ravel()
    prog[f"mean_logexpr_{L}"] = prog["gene"].map(dict(zip(gs, m)))
prog = prog.rename(columns={"diff": "signed_effect_resistant_minus_sensitive"})
prog.insert(0, "status", "withdrawn analysis, provided for transparency")
prog.to_csv(os.path.join(OUT, "Supplementary_withdrawn_resistance_program.csv"), index=False)
print(f"\n64-gene list: {len(prog)} genes | 33-gene list: {int(prog['in_33_gene_list'].sum())} genes")
print(prog.loc[prog["gene"].isin(["KDM6A", "GRHL2", "SCHLAP1"]),
               ["gene", "direction", "signed_effect_resistant_minus_sensitive"]].to_string(index=False))
json.dump(S, open(os.path.join(OUT, "STEP2_SUMMARY.json"), "w"), indent=1)
print("\nsaved: STEP2_SUMMARY.json, S3_pooled_calls_r_by_line.csv.gz, Supplementary_withdrawn_resistance_program.csv,",
      "FigS3_within_line_r_of_pooled_calls.png/pdf")


# ======================= Part 2: single-line associations, cis vs matched trans =======================
import os, pickle, json, numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROS = args.data
OUT = os.path.join(PROS, "revision_out")
THR, FDR, CLIP = 0.1, 0.05, 0.999

cis = pickle.load(open(os.path.join(OUT, "P1_scan_cellline.pkl"), "rb"))
tr = pickle.load(open(os.path.join(OUT, "P3b_scan_trans_matched.pkl"), "rb"))
lines = [str(g) for g in cis["group_names"]]
assert [str(g) for g in tr["group_names"]] == lines

def pval(r, n): return 2 * stats.norm.sf(np.abs(np.arctanh(np.clip(r, -CLIP, CLIP))) * np.sqrt(n - 3))
def rk_informative(res):
    r = res["r_k"].astype(float)
    use = res["gok_k"] & ~np.isnan(r) & (res["nz_k"] >= 1) & (res["gnz_k"] >= 1)
    return np.where(use, r, np.nan)

# pooled calling rule of the main analysis (|r| > 0.1, raw p <= largest raw p among the cis pooled calls)
pc_all = pval(cis["r_naive"], cis["N"])
hit = (np.abs(cis["r_naive"]) > THR) & (multipletests(pc_all, method="fdr_bh")[1] < FDR)
cut = pc_all[hit].max()
same = np.isin(cis["gene"], np.unique(tr["gene"]))           # cis pairs of the same genes as the trans set
Rc, Rt = rk_informative(cis)[same], rk_informative(tr)
rc, rt = cis["r_naive"][same], tr["r_naive"]
callc = (np.abs(rc) > THR) & (pval(rc, cis["N"]) <= cut)
callt = (np.abs(rt) > THR) & (pval(rt, tr["N"]) <= cut)
S = {"cis_pairs_same_genes": int(same.sum()), "trans_pairs": int(len(rt)),
     "cis_pooled_calls": int(callc.sum()), "trans_pooled_calls": int(callt.sum())}

# (A) all pairs, each line: fraction with |r_k| > 0.1 (cis vs matched trans) -> per-line enrichment and FDR estimate
S["A_all_pairs_per_line"] = {}
for k, L in enumerate(lines):
    c, t = Rc[:, k], Rt[:, k]; c, t = c[~np.isnan(c)], t[~np.isnan(t)]
    fc, ft = (np.abs(c) > THR).mean(), (np.abs(t) > THR).mean()
    pc_, pt_ = (c > THR).mean(), (t > THR).mean()
    S["A_all_pairs_per_line"][L] = dict(n_cis=int(len(c)), n_trans=int(len(t)),
        pct_abs_gt_cis=round(100 * fc, 3), pct_abs_gt_trans=round(100 * ft, 3),
        enrichment_abs=round(fc / ft, 2) if ft > 0 else None, est_fdr_abs=round(min(1, ft / fc), 2) if fc > 0 else None,
        enrichment_pos=round(pc_ / pt_, 2) if pt_ > 0 else None)

# (B) among pooled calls (same selection for cis and trans): single-line associations
def single_line(R, rp):
    strong = (np.abs(R) > THR) & (np.sign(R) == np.sign(rp)[:, None])   # same sign as the pooled r
    n = strong.sum(1); s = n == 1
    return s, np.where(s, np.argmax(strong, 1), -1), n
sc_, wc, nc = single_line(Rc[callc], rc[callc]); st_, wt, nt = single_line(Rt[callt], rt[callt])
tab = [[int(sc_.sum()), int((~sc_).sum())], [int(st_.sum()), int((~st_).sum())]]
odds, p_fisher = stats.fisher_exact(tab)
S["B_pooled_calls_single_line"] = dict(
    cis=dict(n=int(callc.sum()), single=int(sc_.sum()), pct=round(100 * sc_.mean(), 1) if callc.any() else None),
    trans=dict(n=int(callt.sum()), single=int(st_.sum()), pct=round(100 * st_.mean(), 1) if callt.any() else None),
    fisher_odds_ratio=round(float(odds), 2), fisher_p=float(p_fisher),
    by_line_cis={L: int((wc == k).sum()) for k, L in enumerate(lines)},
    by_line_trans={L: int((wt == k).sum()) for k, L in enumerate(lines)})
# rate-based estimate, as for the matched trans null: single-line calls per pair, trans vs cis (same genes)
rate_c, rate_t = sc_.sum() / len(rc), st_.sum() / len(rt)
B = S["B_pooled_calls_single_line"]
B["single_line_rate_per_pair_cis"] = float(rate_c); B["single_line_rate_per_pair_trans"] = float(rate_t)
B["enrichment_cis_over_trans"] = round(float(rate_c / rate_t), 2) if rate_t > 0 else None
B["est_fdr_single_line"] = round(float(min(1, rate_t / rate_c)), 2) if rate_c > 0 else None
B["expected_single_line_from_trans_background"] = round(float(rate_t * len(rc)), 1)

print(json.dumps(S, indent=1))
json.dump(S, open(os.path.join(OUT, "STEP3_SUMMARY.json"), "w"), indent=1)
print("\nsaved:", os.path.join(OUT, "STEP3_SUMMARY.json"))
