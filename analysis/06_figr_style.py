"""Simplified FigR-style gene–peak linking (prostate, GSE251977).
Re-implements only the core correlation step of FigR (Kartha et al. 2022): Spearman correlation between gene
expression and peak accessibility for peaks whose start lies within ±100 kb of the TSS (peaks accessible in
>= 20 cells), positive correlations only, rho > 0.1. FigR's nearest-neighbour smoothing and background-peak
significance test are NOT included. The within-line version combines within-line Spearman correlations by
sample-size-weighted Fisher-z. Reported in the manuscript (Section 3.1): 579,340 pairs; 4,569 pooled vs
68 within-line links (98.5% reduction).
Usage:  python 06_figr_style.py --data "<folder that contains checkpoints/>"
"""
import os, pickle, argparse
import numpy as np, pandas as pd, scanpy as sc
from scipy.sparse import issparse
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"C:\Users\Digikey\Desktop\prostate_rebuild")
args, _ = ap.parse_known_args()
CK = os.path.join(args.data, "checkpoints")
OUT = os.path.join(args.data, "revision_out"); os.makedirs(OUT, exist_ok=True)
WINDOW, MIN_PEAK_CELLS, MIN_GROUP, POS = 100_000, 20, 30, 0.1

rna = sc.read_h5ad(os.path.join(CK, "rna_processed.h5ad"))
atac = sc.read_h5ad(os.path.join(CK, "atac_processed.h5ad"))
sc.pp.normalize_total(atac, target_sum=1e4); sc.pp.log1p(atac)
tss = pickle.load(open(os.path.join(CK, "tss.pkl"), "rb"))
Ax = atac.X.tocsc() if issparse(atac.X) else atac.X
Rx = rna.X.tocsc() if issparse(rna.X) else rna.X
gi = {g: i for i, g in enumerate(rna.var_names)}
cell_line = rna.obs["cell_line"].values
groups = [g for g in np.unique(cell_line) if (cell_line == g).sum() >= MIN_GROUP]
masks = {g: np.where(cell_line == g)[0] for g in groups}
pk_chrom = np.array([p.split(":")[0] for p in atac.var_names])
pk_start = np.array([int(p.split(":")[1].split("-")[0]) for p in atac.var_names])

def dense(M, cols):
    X = M[:, cols]; return np.asarray(X.todense()) if issparse(M) else np.asarray(X)

def spearman_vec(x, Y):
    """Spearman correlation of x with every column of Y (Pearson on ranks)."""
    xr = rankdata(x); Yr = rankdata(Y, axis=0)
    xc = xr - xr.mean(); Yc = Yr - Yr.mean(0)
    return (xc @ Yc) / (np.sqrt((xc ** 2).sum() * (Yc ** 2).sum(0)) + 1e-12)

rows = []
ts = tss[tss["gene"].isin(gi)].reset_index(drop=True)
for cnt, gr in ts.iterrows():
    near = np.where((pk_chrom == gr["chrom"]) & (np.abs(pk_start - gr["tss"]) <= WINDOW))[0]
    if len(near) == 0: continue
    ge = dense(Rx, [gi[gr["gene"]]]).ravel()
    if ge.std() < 1e-6: continue
    P = dense(Ax, near); good = (P > 0).sum(0) >= MIN_PEAK_CELLS
    if not good.any(): continue
    near, P = near[good], P[:, good]
    rn = spearman_vec(ge, P)                                       # pooled (FigR-style)
    z = np.zeros(P.shape[1]); w = 0.0
    for g in groups:                                               # within-line, Fisher-z combined
        idx = masks[g]; geg = ge[idx]
        if geg.std() < 1e-8: continue
        r = np.clip(spearman_vec(geg, P[idx]), -0.999, 0.999)
        z += (len(idx) - 3) * np.arctanh(r); w += len(idx) - 3
    rp = np.tanh(z / w)
    for k in range(len(near)):
        rows.append((gr["gene"], atac.var_names[near[k]], rn[k], rp[k]))
    if cnt % 3000 == 0: print(f"  {cnt}/{len(ts)}", flush=True)

figr = pd.DataFrame(rows, columns=["gene", "peak", "spearman_pooled", "spearman_within_line"])
figr.to_csv(os.path.join(OUT, "figr_style_links.csv.gz"), index=False)
n_pool = int((figr["spearman_pooled"] > POS).sum()); n_wg = int((figr["spearman_within_line"] > POS).sum())
print(f"pairs (±{WINDOW // 1000} kb): {len(figr):,}")
print(f"FigR-style links (rho > {POS}): pooled {n_pool:,} | within-line {n_wg:,} | reduction {100 * (1 - n_wg / max(n_pool, 1)):.1f}%")
