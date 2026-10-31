"""Revision analyses, brain (GSE214979): reproduction, revised estimator, matched-trans null,
permutation on the same pairs, cell type / donor x cell type / within-cell-type / within-diagnosis designs.
Usage:  python 02_brain_revision.py --data "<folder that contains checkpoints/>"
"""
import os, sys, json, pickle, argparse, gc
import numpy as np, pandas as pd, scanpy as sc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from confound.core import *

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"C:\Users\Digikey\Desktop\brain_confound")
args, _ = ap.parse_known_args()
BRAIN = args.data

BCK = os.path.join(BRAIN, "checkpoints")
OUT = os.path.join(BRAIN, "revision_out"); os.makedirs(OUT, exist_ok=True)
cache = cache_factory(OUT)
SUMMARY_PATH = os.path.join(OUT, "REVISION_SUMMARY_brain.json")
SUMMARY = json.load(open(SUMMARY_PATH)) if os.path.exists(SUMMARY_PATH) else {}

log("loading brain ...")
rna = sc.read_h5ad(os.path.join(BCK, "rna_brain_processed.h5ad"))
atac = sc.read_h5ad(os.path.join(BCK, "atac_brain_raw.h5ad"))
sc.pp.normalize_total(atac, target_sum=1e4); sc.pp.log1p(atac)          # as in the notebook
tss = pickle.load(open(os.path.join(BCK, "tss_brain.pkl"), "rb"))
D = Data(rna, atac, tss); del atac; gc.collect()
obs = D.obs
donor = obs["donor"].astype(str).values
diag = obs["diagnosis"].astype(str).values
ctype = obs["celltype"].astype(str).values
log(f"cells={D.N}, genes={len(D.genes)}, peaks={len(D.peaks)}")

# ---------- B0: design table ----------
meta = pd.read_csv(os.path.join(BRAIN, "data", "GSE214979_cell_metadata.csv.gz"), index_col=0).loc[obs.index]
dt = (meta.assign(donor=donor).groupby("donor")
          .agg(diagnosis=("Diagnosis", "first"), sex=("Sex", "first"), age=("Age", "first"),
               sub_batch=("Sub.batch", lambda s: ";".join(sorted(map(str, s.dropna().unique())))),
               repository=("Repository", lambda s: ";".join(sorted(map(str, s.dropna().unique())))),
               structure=("structure", lambda s: ";".join(sorted(map(str, s.unique())))),
               n_cells=("Diagnosis", "size")))
design = dt.join(pd.crosstab(donor, ctype))
design.to_csv(os.path.join(OUT, "B0_design_donor_table.csv"))
print(design.to_string())
SUMMARY["B0_celltype_counts"] = pd.Series(ctype).value_counts().to_dict()

# ---------- B1: main donor analysis + reproduction ----------
res = cache("B1_scan_donor.pkl", lambda: scan(D, donor, tag="donor"))
e_orig = combine(res, "orig"); e_rev = combine(res, "revised", min_nz=1)
pt = pd.DataFrame({"gene": D.genes[res["gene"]], "peak": D.peaks[res["peak"]],
                   "r_naive": res["r_naive"], "r_wg_orig": e_orig["r_wg"]})
old = pd.read_csv(os.path.join(BCK, "grn_brain_links.csv"), usecols=["gene","peak","corr_naive","corr_partial"])
rep = reproduction_check(old, pt)
rep["published_naive_calls"] = int(((bh(p_ttest_orig(old["corr_naive"].values, D.N)) < .05) & (old["corr_naive"].abs() > .1)).sum())
rep["published_partial_calls"] = int(((bh(p_ttest_orig(old["corr_partial"].values, D.N)) < .05) & (old["corr_partial"].abs() > .1)).sum())
SUMMARY["B1_reproduction"] = rep; print("\nREPRODUCTION:", rep)
del pt, old; gc.collect()
SUMMARY["B1_main"] = standard_bundle(res)
for k, v in SUMMARY["B1_main"].items():
    print(f"{k:16s}", {x: v[x] for x in ["pooled_calls","wg_calls","reduction_count_based","shared",
                                         "pooled_only","pooled_only_not_estimable","wg_only",
                                         "frac_pooled_calls_not_supported_within"]})
save_json(SUMMARY, SUMMARY_PATH); log("=== B0/B1 saved ===")

# ---------- B2: matched trans null + continuous cis-vs-trans ----------
trm = cache("B2_scan_trans_matched.pkl", lambda: scan_trans_matched(D, donor))
SUMMARY["B2_trans_matched"] = matched_null(res, trm)
for k, v in SUMMARY["B2_trans_matched"].items(): print(k, v)
SUMMARY["B2_cis_vs_trans"] = cis_vs_trans(res, trm, "brain, donor")
save_json(SUMMARY, SUMMARY_PATH); log("=== B2 saved ===")

# ---------- B3: permutation, same pairs + same rule (10,000 pairs; 105k cells) ----------
rng = np.random.default_rng(42)
pick = np.sort(rng.choice(len(res["r_naive"]), size=10000, replace=False))
perm = cache("B3_permutation_same_set.pkl",
             lambda: permutation_same_set(D, donor, res["gene"][pick], res["peak"][pick], B=1000, chunk=250))
SUMMARY["B3_permutation"] = {k: v for k, v in perm.items() if not k.endswith("_null")}
print("PERMUTATION:", SUMMARY["B3_permutation"])
save_json(SUMMARY, SUMMARY_PATH)
log("=== BRAIN CELL A FINISHED ===")

def run_grouping_b(tag, labels, cell_idx=None):
    r = cache(f"{tag}.pkl", lambda: scan(D, labels, cell_idx=cell_idx, tag=tag))
    b = standard_bundle(r, min_nz_list=(1,))
    b["groups"] = {str(g_): int(n_) for g_, n_ in zip(r["group_names"], r["n_k"])}
    t = cache(f"{tag}_trans_matched.pkl", lambda: scan_trans_matched(D, labels, cell_idx=cell_idx, n_genes=1000))
    b["trans_matched"] = matched_null(r, t)
    b["cis_vs_trans"] = cis_vs_trans(r, t, tag)
    v = b["revised_minnz1"]
    print(f"[{tag}] groups={len(b['groups'])} pooled={v['pooled_calls']} wg={v['wg_calls']} shared={v['shared']} "
          f"reduction={v['reduction_count_based']:.3f} not_supported={v['frac_pooled_calls_not_supported_within']:.3f} | "
          f"trans-FDR pooled={b['trans_matched']['revised']['pooled']['estimated_fdr']} "
          f"within={b['trans_matched']['revised']['within_group']['estimated_fdr']}")
    del r, t; gc.collect()
    return b

# B4: cell type as the grouping variable
SUMMARY["B4_celltype"] = run_grouping_b("B4_scan_celltype", ctype)
save_json(SUMMARY, SUMMARY_PATH); log("=== B4 saved ===")
# B5: joint donor x cell type
SUMMARY["B5_donor_x_celltype"] = run_grouping_b("B5_scan_donor_x_celltype",
                                                (obs["donor"].astype(str) + "|" + obs["celltype"].astype(str)).values)
save_json(SUMMARY, SUMMARY_PATH); log("=== B5 saved ===")
# B6: donor confound WITHIN each well-sampled cell type
SUMMARY["B6_donor_within_celltype"] = {}
for t_, n_t in pd.Series(ctype).value_counts().items():
    if n_t < 3000: continue
    ix = np.where(ctype == t_)[0]
    SUMMARY["B6_donor_within_celltype"][t_] = run_grouping_b(f"B6_scan_donor_in_{t_}", donor, ix)
    save_json(SUMMARY, SUMMARY_PATH); log(f"=== B6 {t_} saved ===")
# B7: donor confound WITHIN each diagnosis group
SUMMARY["B7_donor_within_diagnosis"] = {}
for dg in sorted(set(diag)):
    ix = np.where(diag == dg)[0]
    SUMMARY["B7_donor_within_diagnosis"][dg] = run_grouping_b(
        f"B7_scan_donor_in_{dg}".replace("'", "").replace(" ", "_"), donor, ix)
    save_json(SUMMARY, SUMMARY_PATH); log(f"=== B7 {dg} saved ===")
log("=== BRAIN CELL B FINISHED ===")
