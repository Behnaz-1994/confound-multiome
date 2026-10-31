"""Revision analyses, prostate (GSE251977): reproduction, revised estimator, call decomposition,
sensitivity, trans and matched-trans nulls, permutation on the same pairs, treatment/library designs,
cis-vs-trans continuous comparison.
Usage:  python 01_prostate_revision.py --data "<folder that contains checkpoints/>"
"""
import os, sys, json, pickle, argparse, gc
import numpy as np, pandas as pd, scanpy as sc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from confound.core import *

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"C:\Users\Digikey\Desktop\prostate_rebuild")
args, _ = ap.parse_known_args()
PROJECT = args.data

# ---------------- stage 1: design, reproduction, main result ----------------
CKPT = os.path.join(PROJECT, "checkpoints")
OUT = os.path.join(PROJECT, "revision_out"); os.makedirs(OUT, exist_ok=True)
SUMMARY_PATH = os.path.join(OUT, "REVISION_SUMMARY_prostate.json")
SUMMARY = json.load(open(SUMMARY_PATH)) if os.path.exists(SUMMARY_PATH) else {}

cache = cache_factory(OUT)

log("loading ...")
rna = sc.read_h5ad(os.path.join(CKPT, "rna_processed.h5ad"))
atac = sc.read_h5ad(os.path.join(CKPT, "atac_processed.h5ad"))
sc.pp.normalize_total(atac, target_sum=1e4); sc.pp.log1p(atac)
tss = pickle.load(open(os.path.join(CKPT, "tss.pkl"), "rb"))
D = Data(rna, atac, tss)
obs = D.obs
line = obs["cell_line"].astype(str).values
drug = obs["drug"].astype(str).values
log(f"cells={D.N}, genes={len(D.genes)}, peaks={len(D.peaks)}")

design = pd.crosstab([obs["cell_line"], obs["drug"]], obs["sample"], margins=True)
design.to_csv(os.path.join(OUT, "P0_design_line_drug_library.csv"))
print(design.to_string())
markers = [g for g in ["AR","KLK3","KLK2","TMPRSS2","NKX3-1","FOLH1","ERG","HOXB13",
                       "ASCL1","CHGA","SYP","INSM1","VIM","KRT8","KRT18","KRT5"] if g in D.gi]
rows = []
for (cl, dr), _ in obs.groupby(["cell_line", "drug"], observed=True):
    ix = np.where((line == cl) & (drug == dr))[0]
    row = {"cell_line": cl, "drug": dr, "n_cells": len(ix)}
    for gname in markers:
        v = D.gene_vec(D.gi[gname], ix)
        row[f"{gname}_mean"] = float(v.mean()); row[f"{gname}_pct"] = float((v > 0).mean() * 100)
    rows.append(row)
mk = pd.DataFrame(rows)
mk.to_csv(os.path.join(OUT, "P0_marker_check_line_x_drug.csv"), index=False)
print(mk[["cell_line","drug","n_cells"] + [f"{g}_mean" for g in markers[:8]]].round(2).to_string())

allc = np.arange(D.N); ar = D.gene_vec(D.gi["AR"], allc); arc = {}
for tg in ["KLK2", "KLK3", "TMPRSS2", "NKX3-1", "FKBP5"]:
    if tg not in D.gi: continue
    te = D.gene_vec(D.gi[tg], allc)
    row = {"all_cells": float(np.corrcoef(ar, te)[0, 1])}
    m = np.isin(line, ["LNCaP", "VCaP"])
    row["LNCaP+VCaP_pooled"] = float(np.corrcoef(ar[m], te[m])[0, 1])
    for cl in sorted(set(line)):
        mm = line == cl
        if ar[mm].std() > 0 and te[mm].std() > 0:
            row[f"within_{cl}"] = float(np.corrcoef(ar[mm], te[mm])[0, 1])
    arc[tg] = row
SUMMARY["P0_AR_target_correlations"] = arc
SUMMARY["P0_design"] = {f"{a}|{b}": int(v) for (a, b), v in pd.crosstab(obs["cell_line"], obs["drug"]).stack().items()}
print(pd.DataFrame(arc).round(3).to_string())
save_json(SUMMARY, SUMMARY_PATH); log("=== P0 saved ===")

res = cache("P1_scan_cellline.pkl", lambda: scan(D, line, tag="cell_line"))
e_orig = combine(res, "orig"); e_rev = combine(res, "revised", min_nz=1)
pt = pd.DataFrame({"gene": D.genes[res["gene"]], "peak": D.peaks[res["peak"]], "dist": res["dist"],
                   "r_naive": res["r_naive"], "r_wg_orig": e_orig["r_wg"], "r_wg_revised": e_rev["r_wg"],
                   "K_informative": e_rev["K_inf"], "I2": e_rev["I2"],
                   "n_zero_peak_groups": e_orig["n_zero_peak_groups"], "nnz_peak": res["nnz_peak"]})
pt.to_csv(os.path.join(OUT, "P1_pairs_cellline.csv.gz"), index=False)

old = pd.read_csv(os.path.join(CKPT, "grn_links_fisher.csv"), usecols=["gene","peak","corr_naive","corr_partial"])
rep = reproduction_check(old, pt)
rep["published_naive_calls"] = int(((bh(p_ttest_orig(old["corr_naive"].values, D.N)) < .05) & (old["corr_naive"].abs() > .1)).sum())
rep["published_partial_calls"] = int(((bh(p_ttest_orig(old["corr_partial"].values, D.N)) < .05) & (old["corr_partial"].abs() > .1)).sum())
SUMMARY["P1_reproduction"] = rep
print("\nREPRODUCTION:", rep)

SUMMARY["P1_main"] = standard_bundle(res)
print()
for k, v in SUMMARY["P1_main"].items():
    print(f"{k:16s}", {x: v[x] for x in ["pooled_calls","wg_calls","reduction_count_based",
                                         "shared","pooled_only","pooled_only_not_estimable","wg_only"]})
save_json(SUMMARY, SUMMARY_PATH)
log("=== STAGE 1 FINISHED -> " + SUMMARY_PATH)

# ---------------- stage 2: sensitivity, trans null, permutation ----------------
# ---------- P2: sensitivity (window, peak filter, minimum group size), no rescan ----------
sens = {}
for win in (100_000, 250_000):
    for minpk in (20, 50, 100):
        r2 = subset_res(res, (res["dist"] <= win) & (res["nnz_peak"] >= minpk))
        for nm, est in [("orig", combine(r2, "orig")), ("revised", combine(r2, "revised", min_nz=1))]:
            s, _ = summarize(r2, est)
            sens[f"win{win//1000}kb_minpeak{minpk}_{nm}"] = {x: s[x] for x in
                ["n_pairs", "pooled_calls", "wg_calls", "shared", "reduction_count_based",
                 "frac_pooled_calls_not_supported_within"]}
for mg in (30, 600):   # 600 drops MDA-PCa-2B (573 cells)
    s, _ = summarize(res, combine(res, "revised", min_nz=1, min_group_cells=mg))
    sens[f"min_group_cells{mg}_revised"] = {x: s[x] for x in ["pooled_calls", "wg_calls", "shared",
                                                              "reduction_count_based"]}
SUMMARY["P2_sensitivity"] = sens
print(pd.DataFrame(sens).T.to_string())
save_json(SUMMARY, SUMMARY_PATH); log("=== P2 saved ===")

# ---------- P3: trans negative control (peak and gene on different chromosomes) ----------
tr = cache("P3_scan_trans.pkl", lambda: scan_trans(D, line))
SUMMARY["P3_trans_null"] = {
    "orig": trans_null(res, e_orig, tr, combine(tr, "orig")),
    "revised_minnz1": trans_null(res, e_rev, tr, combine(tr, "revised", min_nz=1))}
for k, v in SUMMARY["P3_trans_null"].items(): print(k, v)
save_json(SUMMARY, SUMMARY_PATH); log("=== P3 saved ===")

# ---------- P4: permutation with observed and null on the SAME pairs and SAME rule ----------
rng = np.random.default_rng(42)
pick = np.sort(rng.choice(len(res["r_naive"]), size=20000, replace=False))
perm = cache("P4_permutation_same_set.pkl",
             lambda: permutation_same_set(D, line, res["gene"][pick], res["peak"][pick], B=1000))
SUMMARY["P4_permutation"] = {k: v for k, v in perm.items() if not k.endswith("_null")}
print("PERMUTATION:", SUMMARY["P4_permutation"])
save_json(SUMMARY, SUMMARY_PATH)
log("=== STAGE 2 FINISHED -> " + SUMMARY_PATH)

# ---------------- stage 3: matched trans null, treatment / library designs ----------------
# ---------- P3b: matched trans null for the main (cell-line) analysis ----------
trm = cache("P3b_scan_trans_matched.pkl", lambda: scan_trans_matched(D, line))
SUMMARY["P3b_trans_matched"] = matched_null(res, trm)
for k, v in SUMMARY["P3b_trans_matched"].items(): print(k, v)
save_json(SUMMARY, SUMMARY_PATH); log("=== P3b saved ===")

# ---------- stage 3: other groupings, each with its own matched trans null ----------
def run_grouping(tag, labels, cell_idx=None):
    r = cache(f"{tag}.pkl", lambda: scan(D, labels, cell_idx=cell_idx, tag=tag))
    b = standard_bundle(r, min_nz_list=(1,))
    b["groups"] = {str(g_): int(n_) for g_, n_ in zip(r["group_names"], r["n_k"])}
    t = cache(f"{tag}_trans_matched.pkl", lambda: scan_trans_matched(D, labels, cell_idx=cell_idx, n_genes=1000))
    b["trans_matched"] = matched_null(r, t)
    print(f"\n[{tag}] groups={len(b['groups'])}")
    for k in ("orig", "revised_minnz1"):
        v = b[k]; print(f"  {k:15s} pooled={v['pooled_calls']} wg={v['wg_calls']} shared={v['shared']} "
                        f"reduction={v['reduction_count_based']:.3f} not_supported={v['frac_pooled_calls_not_supported_within']:.3f}")
    print("  matched-trans FDR  pooled:", b["trans_matched"]["revised"]["pooled"]["estimated_fdr"],
          "| within:", b["trans_matched"]["revised"]["within_group"]["estimated_fdr"])
    return b

line_drug = (obs["cell_line"].astype(str) + "|" + obs["drug"].astype(str)).values
line_lib = (obs["cell_line"].astype(str) + "|" + obs["sample"].astype(str)).values

SUMMARY["P5_line_x_drug"] = run_grouping("P5_scan_line_x_drug", line_drug)
save_json(SUMMARY, SUMMARY_PATH); log("=== P5 saved ===")
SUMMARY["P6_line_x_library"] = run_grouping("P6_scan_line_x_library", line_lib)
save_json(SUMMARY, SUMMARY_PATH); log("=== P6 saved ===")
SUMMARY["P7_by_treatment"] = {}
for dr in sorted(set(drug)):
    ix = np.where(drug == dr)[0]
    if int((pd.Series(line[ix]).value_counts() >= 30).sum()) < 2: continue
    SUMMARY["P7_by_treatment"][dr] = run_grouping(f"P7_scan_only_{dr}".replace(" ", "_"), line, cell_idx=ix)
    save_json(SUMMARY, SUMMARY_PATH); log(f"=== P7 {dr} saved ===")
log("=== STAGE 3 FINISHED ===")

# ---------------- stage 4: continuous cis vs matched-trans ----------------
SUMMARY["P8_cis_vs_trans"] = {}
SUMMARY["P8_cis_vs_trans"]["cell_line"] = cis_vs_trans(res, trm, "cell line (all cells)")
for tag, nm in [("P5_scan_line_x_drug", "line x drug"), ("P7_scan_only_DMSO", "DMSO only, by line"),
                ("P7_scan_only_Enzalutamide", "Enzalutamide only, by line")]:
    r_ = cache(f"{tag}.pkl", lambda: None); t_ = cache(f"{tag}_trans_matched.pkl", lambda: None)
    SUMMARY["P8_cis_vs_trans"][tag] = cis_vs_trans(r_, t_, nm)
save_json(SUMMARY, SUMMARY_PATH)
log("=== P8 saved ===")
