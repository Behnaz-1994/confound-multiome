"""
confound/core.py -- group-structure control for peak-gene association inference
in single-cell multiome data (revision of "Real signal, false links").

Estimators
  pooled      : Pearson r across all cells
  ORIG        : sample-size-weighted Fisher-z combination of within-group r
                (as in the submitted manuscript; a group in which the peak never
                varies contributes r = 0)
  REVISED     : as ORIG, but a group contributes only if gene AND peak vary in it
                (min_nz non-zero cells each); pairs without an informative group
                are 'not estimable'
Calibration
  permutation_same_set : label permutation, observed and null on the same pairs/rule
  scan_trans_matched   : matched trans (cannot-be-cis) null; same genes, peaks on
                         other chromosomes matched on accessibility
"""
import os, time, json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

CLIP = 0.999


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def peak_coords(peak_names):
    chrom = np.array([p.split(":")[0] for p in peak_names])
    start = np.array([int(p.split(":")[1].split("-")[0]) for p in peak_names])
    return chrom, start


def bh(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def p_fisher(r, n):
    r = np.clip(np.asarray(r, float), -CLIP, CLIP)
    return 2 * stats.norm.sf(np.abs(np.arctanh(r)) * np.sqrt(max(n - 3, 1)))


def p_ttest_orig(r, n):
    r = np.clip(np.asarray(r, float), -CLIP, CLIP)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return 2 * stats.t.sf(np.abs(t), df=n - 2)


def encode_groups(labels, cell_idx, min_cells=30):
    lab = pd.Series(np.asarray(labels)[cell_idx]).astype(str).values
    uniq, inv, counts = np.unique(lab, return_inverse=True, return_counts=True)
    keep = counts >= min_cells
    gid = np.full(len(uniq), -1)
    gid[keep] = np.arange(keep.sum())
    return gid[inv], uniq[keep], counts[keep]


def onehot(g, K):
    m = g >= 0
    n = len(g)
    return sp.csr_matrix((np.ones(m.sum()), (g[m], np.where(m)[0])), shape=(K, n))


def group_pearson(H, n_k, x, Y):
    x = x.astype(np.float64)
    Y = Y.astype(np.float64)
    nk = n_k.astype(np.float64)[:, None]
    Sx = H @ x
    Sxx = H @ (x * x)
    SY = H @ Y
    SYY = H @ (Y * Y)
    SXY = H @ (x[:, None] * Y)
    nzY = np.asarray(H @ (Y > 0).astype(np.float64))
    nzx = np.asarray(H @ (x > 0).astype(np.float64))
    vx = Sxx - Sx ** 2 / nk[:, 0]
    vY = SYY - SY ** 2 / nk
    cov = SXY - Sx[:, None] * SY / nk
    x_ok = np.sqrt(np.maximum(vx, 0) / nk[:, 0]) >= 1e-8
    Y_ok = (np.sqrt(np.maximum(vY, 0) / nk) >= 1e-8) & (nzY > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(np.maximum(vx, 0)[:, None] * np.maximum(vY, 0))
    r[~(x_ok[:, None] & Y_ok)] = np.nan
    return r, nzY, x_ok, nzx


def pooled_r(x, Y):
    x = x.astype(np.float64); Y = Y.astype(np.float64)
    xc = x - x.mean()
    Yc = Y - Y.mean(0)
    return (xc @ Yc) / (np.sqrt((xc ** 2).sum() * (Yc ** 2).sum(0)) + 1e-12)


class Data:
    def __init__(self, rna, atac, tss):
        if not np.array_equal(np.asarray(rna.obs_names), np.asarray(atac.obs_names)):
            raise ValueError("RNA and ATAC cells are not in the same order")
        self.obs = rna.obs.copy()
        self.N = rna.n_obs
        self.Rx = rna.X.tocsc() if sp.issparse(rna.X) else sp.csc_matrix(rna.X)
        self.Ax = atac.X.tocsc() if sp.issparse(atac.X) else sp.csc_matrix(atac.X)
        self.genes = np.asarray(rna.var_names)
        self.peaks = np.asarray(atac.var_names)
        self.gi = {g: i for i, g in enumerate(self.genes)}
        self.pk_chrom, self.pk_start = peak_coords(self.peaks)
        self.tss = tss[tss["gene"].isin(self.gi)].reset_index(drop=True)

    def gene_vec(self, j, idx):
        return np.asarray(self.Rx[:, j].todense()).ravel()[idx]

    def peak_mat(self, cols, idx):
        return np.asarray(self.Ax[:, cols].todense())[idx]


def scan(D, labels, cell_idx=None, window=250_000, min_peak_cells=20,
         min_group_cells=30, every=3000, tag=""):
    idx = np.arange(D.N) if cell_idx is None else np.asarray(cell_idx)
    g, gnames, n_k = encode_groups(labels, idx, min_group_cells)
    K = len(gnames)
    H = onehot(g, K)
    Asub = D.Ax[idx] if cell_idx is not None else D.Ax
    peak_nnz = np.asarray((Asub > 0).sum(0)).ravel()
    del Asub
    rows = D.tss
    out = {k: [] for k in ["gene", "peak", "dist", "r_naive", "nnz_peak", "r_k", "nz_k",
                           "gok_k", "gnz_k"]}
    t0 = time.time()
    for cnt, gr in enumerate(rows.itertuples(index=False)):
        near = np.where((D.pk_chrom == gr.chrom) & (np.abs(D.pk_start - gr.tss) <= window))[0]
        if len(near) == 0:
            continue
        ge = D.gene_vec(D.gi[gr.gene], idx)
        if ge.std() < 1e-6:
            continue
        near = near[peak_nnz[near] >= min_peak_cells]
        if len(near) == 0:
            continue
        P = D.peak_mat(near, idx)
        rn = pooled_r(ge, P)
        r_k, nz_k, gok, gnz = group_pearson(H, n_k, ge, P)
        m = len(near)
        out["gene"].append(np.full(m, D.gi[gr.gene], dtype=np.int32))
        out["peak"].append(near.astype(np.int32))
        out["dist"].append(np.abs(D.pk_start[near] - gr.tss).astype(np.int64))
        out["r_naive"].append(rn.astype(np.float64))
        out["nnz_peak"].append(peak_nnz[near].astype(np.int32))
        out["r_k"].append(r_k.T.astype(np.float32))
        out["nz_k"].append(nz_k.T.astype(np.int32))
        out["gok_k"].append(np.repeat(gok[None, :], m, 0))
        out["gnz_k"].append(np.repeat(gnz[None, :], m, 0).astype(np.int32))
        if cnt % every == 0:
            log(f"  [{tag}] gene {cnt}/{len(rows)}  ({time.time() - t0:.0f}s)")
    res = {k: np.concatenate(v) for k, v in out.items()}
    res.update(dict(group_names=gnames, n_k=n_k, N=len(idx), window=window, tag=tag))
    log(f"  [{tag}] done: {len(res['r_naive']):,} pairs, {K} groups, {time.time() - t0:.0f}s")
    return res


def combine(res, estimator="revised", min_nz=1, min_group_cells=None):
    r = res["r_k"].astype(np.float64)
    nz = res["nz_k"]; gok = res["gok_k"]; gnz = res["gnz_k"]
    n_k = res["n_k"].astype(np.float64)
    w = (n_k - 3)[None, :]
    if estimator == "orig":
        use = gok.copy()
        z = np.where(np.isnan(r), 0.0, np.arctanh(np.clip(np.nan_to_num(r), -CLIP, CLIP)))
    else:
        use = gok & ~np.isnan(r) & (nz >= min_nz) & (gnz >= min_nz)
        z = np.arctanh(np.clip(np.nan_to_num(r), -CLIP, CLIP))
    if min_group_cells is not None:
        use = use & (res["n_k"] >= min_group_cells)[None, :]
    wu = w * use
    W = wu.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        zbar = (wu * z).sum(1) / W
    zbar[W <= 0] = np.nan
    Z = zbar * np.sqrt(np.where(W > 0, W, np.nan))
    p = 2 * stats.norm.sf(np.abs(Z))
    K_inf = use.sum(1)
    dev = np.where(use, z - np.nan_to_num(zbar)[:, None], 0.0)
    Q = (w * dev ** 2).sum(1)
    df = K_inf - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        I2 = np.where(Q > 0, np.clip((Q - df) / Q, 0, 1), 0.0)
        pQ = np.where(df >= 1, stats.chi2.sf(Q, np.maximum(df, 1)), np.nan)
    sgn = np.sign(np.nan_to_num(zbar))[:, None]
    n_opp = (use & (np.sign(z) == -sgn) & (np.abs(np.nan_to_num(r)) >= 0.05)).sum(1)
    n_zero_peak_groups = (gok & np.isnan(r)).sum(1)
    return dict(r_wg=np.tanh(zbar), p=p, W=W, K_inf=K_inf, Q=Q, I2=I2, pQ=pQ,
                n_opposite=n_opp, n_zero_peak_groups=n_zero_peak_groups,
                estimator=estimator, min_nz=min_nz)


GRID = np.round(np.arange(0.02, 0.401, 0.01), 2)


def calls(r, p_adj, thr, fdr):
    return (~np.isnan(r)) & (np.abs(r) > thr) & (p_adj < fdr)


def summarize(res, est, thr=0.1, fdr=0.05):
    rn = res["r_naive"]; pn = bh(p_fisher(rn, res["N"]))
    rw = est["r_wg"]; pw = bh(est["p"])
    estimable = ~np.isnan(rw)
    hn = calls(rn, pn, thr, fdr); hw = calls(rw, pw, thr, fdr)
    n_hn = int(hn.sum())
    s = dict(n_pairs=int(len(rn)), n_cells=int(res["N"]), n_groups=int(len(res["n_k"])),
             estimator=est["estimator"], min_nz=est["min_nz"], thr=thr, fdr=fdr,
             n_not_estimable=int((~estimable).sum()),
             pooled_calls=n_hn, wg_calls=int(hw.sum()),
             reduction_count_based=float(1 - hw.sum() / max(n_hn, 1)),
             shared=int((hn & hw).sum()),
             pooled_only=int((hn & ~hw & estimable).sum()),
             pooled_only_not_estimable=int((hn & ~estimable).sum()),
             wg_only=int((hw & ~hn).sum()),
             neither=int((~hn & ~hw).sum()),
             frac_pooled_calls_not_supported_within=float((hn & ~hw).sum() / max(n_hn, 1)))
    s["pooled_calls_r_only"] = int(((~np.isnan(rn)) & (np.abs(rn) > thr)).sum())
    s["wg_calls_r_only"] = int((estimable & (np.abs(rw) > thr)).sum())
    if n_hn:
        sel = hn & estimable
        s["pooled_calls_median_K_informative"] = float(np.median(est["K_inf"][hn]))
        s["pooled_calls_frac_with_zero_peak_group"] = float((est["n_zero_peak_groups"][hn] > 0).mean())
        if sel.any():
            s["pooled_calls_median_I2"] = float(np.median(est["I2"][sel]))
            s["pooled_calls_frac_heterogeneous_Q_fdr05"] = float((bh(est["pQ"][sel]) < 0.05).mean())
            s["pooled_calls_frac_with_opposite_sign_group"] = float((est["n_opposite"][sel] > 0).mean())
            with np.errstate(invalid="ignore", divide="ignore"):
                s["pooled_calls_median_attenuation_ratio"] = float(np.nanmedian(np.abs(rw[sel]) / np.abs(rn[sel])))
    ok = estimable & ~np.isnan(rn)
    s["spearman_rnaive_rwg_all_pairs"] = float(stats.spearmanr(rn[ok], rw[ok])[0]) if ok.sum() > 2 else None
    s["curve"] = [dict(thr=float(t), pooled=int(calls(rn, pn, t, fdr).sum()),
                       wg=int(calls(rw, pw, t, fdr).sum()),
                       shared=int((calls(rn, pn, t, fdr) & calls(rw, pw, t, fdr)).sum())) for t in GRID]
    return s, dict(hit_naive=hn, hit_wg=hw)


def standard_bundle(res, thr=0.1, fdr=0.05, min_nz_list=(1, 5, 10)):
    bundle = {}
    bundle["orig"], _ = summarize(res, combine(res, "orig"), thr, fdr)
    for mnz in min_nz_list:
        bundle[f"revised_minnz{mnz}"], _ = summarize(res, combine(res, "revised", min_nz=mnz), thr, fdr)
    return bundle


def save_json(obj, path):
    def conv(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def reproduction_check(old, pt):
    same = (len(old) == len(pt) and
            np.array_equal(old["gene"].astype(str).values, pt["gene"].astype(str).values) and
            np.array_equal(old["peak"].astype(str).values, pt["peak"].astype(str).values))
    if same:
        a = b = None; a, b, how = old, pt, "row-by-row"
    else:
        o = old.drop_duplicates(["gene", "peak"], keep=False)
        p = pt.drop_duplicates(["gene", "peak"], keep=False)
        m = o.merge(p, on=["gene", "peak"]); a = b = m
        how = f"merged on unique (gene, peak): {len(m)} pairs"
    return dict(compared=how, pairs_old=len(old), pairs_new=len(pt),
                max_abs_diff_naive=float(np.nanmax(np.abs(a["corr_naive"].values - b["r_naive"].values))),
                max_abs_diff_partial=float(np.nanmax(np.abs(a["corr_partial"].values - b["r_wg_orig"].values))))


import torch
from scipy.stats import mannwhitneyu

def subset_res(res, mask):
    n = len(res["r_naive"])
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v)
            for k, v in res.items()}


def scan_trans(D, labels, n_genes=2000, peaks_per_gene=100, min_peak_cells=20,
               min_group_cells=30, seed=7):
    idx = np.arange(D.N)
    g, gnames, n_k = encode_groups(labels, idx, min_group_cells)
    H = onehot(g, len(gnames))
    peak_nnz = np.asarray((D.Ax > 0).sum(0)).ravel()
    eligible = np.where(peak_nnz >= min_peak_cells)[0]
    rng = np.random.default_rng(seed)
    rows = D.tss.iloc[rng.permutation(len(D.tss))].reset_index(drop=True)
    out = {k: [] for k in ["r_naive", "r_k", "nz_k", "gok_k", "gnz_k"]}
    done_g = 0
    for gr in rows.itertuples(index=False):
        if done_g >= n_genes: break
        ge = D.gene_vec(D.gi[gr.gene], idx)
        if ge.std() < 1e-6: continue
        other = eligible[D.pk_chrom[eligible] != gr.chrom]
        near = np.sort(rng.choice(other, size=min(peaks_per_gene, len(other)), replace=False))
        P = D.peak_mat(near, idx)
        r_k, nz_k, gok, gnz = group_pearson(H, n_k, ge, P)
        m = len(near)
        out["r_naive"].append(pooled_r(ge, P)); out["r_k"].append(r_k.T.astype(np.float32))
        out["nz_k"].append(nz_k.T.astype(np.int32)); out["gok_k"].append(np.repeat(gok[None, :], m, 0))
        out["gnz_k"].append(np.repeat(gnz[None, :], m, 0).astype(np.int32))
        done_g += 1
        if done_g % 500 == 0: log(f"  trans genes {done_g}/{n_genes}")
    t = {k: np.concatenate(v) for k, v in out.items()}
    t.update(dict(group_names=gnames, n_k=n_k, N=len(idx)))
    return t


def trans_null(cis_res, cis_est, tr_res, tr_est, thr=0.1, fdr=0.05):
    """Apply the SAME calling rule as for cis pairs (|r| > thr and raw p below the cis
    BH cut-off) to trans pairs, which cannot be cis-regulatory."""
    out = {}
    rn_c = cis_res["r_naive"]; p_c = p_fisher(rn_c, cis_res["N"]); hit_c = calls(rn_c, bh(p_c), thr, fdr)
    cut_n = p_c[hit_c].max() if hit_c.any() else 0
    pw_c = cis_est["p"]; hw_c = calls(cis_est["r_wg"], bh(pw_c), thr, fdr)
    cut_w = pw_c[hw_c].max() if hw_c.any() else 0
    for name, r_t, p_t, cut, ncalls, ntested in [
        ("pooled", tr_res["r_naive"], p_fisher(tr_res["r_naive"], tr_res["N"]), cut_n,
         int(hit_c.sum()), len(rn_c)),
        ("within_group", tr_est["r_wg"], tr_est["p"], cut_w,
         int(hw_c.sum()), int((~np.isnan(cis_est["r_wg"])).sum()))]:
        ok = ~np.isnan(r_t)
        hit = ok & (np.abs(r_t) > thr) & (p_t <= cut)
        rate = hit.sum() / max(ok.sum(), 1)
        out[name] = dict(trans_pairs=int(ok.sum()), trans_calls=int(hit.sum()), trans_call_rate=float(rate),
                         cis_calls=ncalls, expected_from_background=float(rate * ntested),
                         estimated_fdr=float(min(1.0, rate * ntested / max(ncalls, 1))))
    return out


def permutation_same_set(D, labels, pg, pp, B=1000, min_group_cells=30, thr=0.1, fdr=0.05,
                         min_nz=1, seed=42, chunk=500):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    idx = np.arange(D.N)
    g, gnames, n_k = encode_groups(labels, idx, min_group_cells)
    K = len(gnames); n = len(idx); m = len(pg)
    rng = np.random.default_rng(seed)
    labs = np.empty((B + 1, n), dtype=np.int16); labs[0] = g
    for b in range(1, B + 1): labs[b] = rng.permutation(g)
    labs[labs < 0] = K
    nk = torch.tensor(n_k, dtype=torch.float64, device=dev)[:, None]
    w = nk - 3
    zb = {e: np.full((B + 1, m), np.nan, np.float32) for e in ("orig", "revised")}
    Wb = {e: np.zeros((B + 1, m), np.float32) for e in ("orig", "revised")}
    r_naive = np.empty(m); t0 = time.time()
    order = np.argsort(pg, kind="stable")
    for c0 in range(0, m, chunk):
        sel = order[c0:c0 + chunk]
        gc = {j: D.gene_vec(j, idx) for j in np.unique(pg[sel])}
        G = np.stack([gc[j] for j in pg[sel]], 1).astype(np.float64)
        P = D.peak_mat(pp[sel], idx).astype(np.float64)
        Gc = G - G.mean(0); Pc = P - P.mean(0)
        r_naive[sel] = (Gc * Pc).sum(0) / (np.sqrt((Gc ** 2).sum(0) * (Pc ** 2).sum(0)) + 1e-12)
        Gt = torch.tensor(G, device=dev); Pt = torch.tensor(P, device=dev)
        feats = [Gt, Gt * Gt, Pt, Pt * Pt, Gt * Pt, (Pt > 0).double(), (Gt > 0).double()]
        mc = Gt.shape[1]
        for b in range(B + 1):
            lab = torch.tensor(labs[b].astype(np.int64), device=dev)
            S = [torch.zeros(K + 1, mc, dtype=torch.float64, device=dev).index_add_(0, lab, f)[:K] for f in feats]
            Sx, Sxx, Sy, Syy, Sxy, nzy, nzx = S
            vx = Sxx - Sx ** 2 / nk; vy = Syy - Sy ** 2 / nk; cov = Sxy - Sx * Sy / nk
            x_ok = torch.sqrt(torch.clamp(vx, min=0) / nk) >= 1e-8
            y_ok = (torch.sqrt(torch.clamp(vy, min=0) / nk) >= 1e-8) & (nzy > 0)
            r = torch.nan_to_num(cov / torch.sqrt(torch.clamp(vx, min=0) * torch.clamp(vy, min=0)),
                                 nan=0.0, posinf=0.0, neginf=0.0)
            z = torch.atanh(torch.clamp(r, -CLIP, CLIP))
            for e in ("orig", "revised"):
                if e == "orig":
                    use = x_ok; ze = torch.where(y_ok, z, torch.zeros_like(z))
                else:
                    use = x_ok & y_ok & (nzy >= min_nz) & (nzx >= min_nz); ze = z
                wu = w * use; W = wu.sum(0); zbar = (wu * ze).sum(0) / W
                zbar = torch.where(W > 0, zbar, torch.full_like(zbar, float("nan")))
                zb[e][b, sel] = zbar.float().cpu().numpy(); Wb[e][b, sel] = W.float().cpu().numpy()
        log(f"  perm chunk {c0 // chunk + 1}/{(m + chunk - 1) // chunk} ({time.time() - t0:.0f}s)")
    hn = calls(r_naive, bh(p_fisher(r_naive, n)), thr, fdr)
    out = dict(n_pairs=int(m), B=int(B), pooled_calls=int(hn.sum()))
    for e in ("orig", "revised"):
        red = np.empty(B + 1)
        for b in range(B + 1):
            zz = zb[e][b].astype(np.float64); WW = Wb[e][b].astype(np.float64)
            p = 2 * stats.norm.sf(np.abs(zz * np.sqrt(WW))); p[np.isnan(zz)] = np.nan
            red[b] = 1 - calls(np.tanh(zz), bh(p), thr, fdr).sum() / max(hn.sum(), 1)
        null = red[1:]
        out[e] = dict(observed=float(red[0]), null_mean=float(null.mean()), null_sd=float(null.std()),
                      null_max=float(null.max()), p_emp=float((1 + (null >= red[0]).sum()) / (B + 1)))
        out[e + "_null"] = null.tolist()
    return out


def scan_trans_matched(D, labels, cell_idx=None, n_genes=2000, window=250_000, min_peak_cells=20,
                       min_group_cells=30, nbins=20, seed=11):
    """For each sampled gene, every cis peak is replaced by a random peak on ANOTHER chromosome
    with similar accessibility (same quantile bin of non-zero cells). Same genes, same number of
    peaks, same sparsity profile -> a matched background that cannot be cis-regulatory."""
    idx = np.arange(D.N) if cell_idx is None else np.asarray(cell_idx)
    g, gnames, n_k = encode_groups(labels, idx, min_group_cells)
    H = onehot(g, len(gnames))
    Asub = D.Ax[idx] if cell_idx is not None else D.Ax
    peak_nnz = np.asarray((Asub > 0).sum(0)).ravel(); del Asub
    elig = np.where(peak_nnz >= min_peak_cells)[0]
    edges = np.quantile(peak_nnz[elig], np.linspace(0, 1, nbins + 1))
    pbin = np.clip(np.searchsorted(edges, peak_nnz, side="right") - 1, 0, nbins - 1)
    pools = {b: elig[pbin[elig] == b] for b in range(nbins)}
    rng = np.random.default_rng(seed)
    rows = D.tss.iloc[rng.permutation(len(D.tss))].reset_index(drop=True)
    out = {k: [] for k in ["gene", "r_naive", "r_k", "nz_k", "gok_k", "gnz_k"]}
    done_g = 0
    for gr in rows.itertuples(index=False):
        if done_g >= n_genes: break
        near = np.where((D.pk_chrom == gr.chrom) & (np.abs(D.pk_start - gr.tss) <= window))[0]
        near = near[peak_nnz[near] >= min_peak_cells]
        if len(near) == 0: continue
        ge = D.gene_vec(D.gi[gr.gene], idx)
        if ge.std() < 1e-6: continue
        tp = []
        for pk in near:
            pool = pools[pbin[pk]]
            for _ in range(50):
                c = pool[rng.integers(len(pool))]
                if D.pk_chrom[c] != gr.chrom: tp.append(c); break
        tp = np.array(tp)
        if len(tp) == 0: continue
        P = D.peak_mat(tp, idx)
        r_k, nz_k, gok, gnz = group_pearson(H, n_k, ge, P)
        m = len(tp)
        out["gene"].append(np.full(m, D.gi[gr.gene], dtype=np.int32))
        out["r_naive"].append(pooled_r(ge, P)); out["r_k"].append(r_k.T.astype(np.float32))
        out["nz_k"].append(nz_k.T.astype(np.int32)); out["gok_k"].append(np.repeat(gok[None, :], m, 0))
        out["gnz_k"].append(np.repeat(gnz[None, :], m, 0).astype(np.int32))
        done_g += 1
    t = {k: np.concatenate(v) for k, v in out.items()}
    t.update(dict(group_names=gnames, n_k=n_k, N=len(idx)))
    return t


def matched_null(cis_res, tr, thr=0.1, fdr=0.05):
    """Call rate of matched trans pairs vs call rate of the cis pairs of the SAME genes,
    using the cis calling rule (|r| > thr and raw p below the cis BH cut-off)."""
    out = {}
    same = np.isin(cis_res["gene"], np.unique(tr["gene"]))
    for est_name, ce, te in [("orig", combine(cis_res, "orig"), combine(tr, "orig")),
                             ("revised", combine(cis_res, "revised", min_nz=1), combine(tr, "revised", min_nz=1))]:
        o = {}
        for kind, rc, pc, rt, pt_ in [
            ("pooled", cis_res["r_naive"], p_fisher(cis_res["r_naive"], cis_res["N"]),
             tr["r_naive"], p_fisher(tr["r_naive"], tr["N"])),
            ("within_group", ce["r_wg"], ce["p"], te["r_wg"], te["p"])]:
            hit_all = calls(rc, bh(pc), thr, fdr)
            cut = pc[hit_all].max() if hit_all.any() else 0
            okc = same & ~np.isnan(rc); okt = ~np.isnan(rt)
            cis_rate = (okc & (np.abs(rc) > thr) & (pc <= cut)).sum() / max(okc.sum(), 1)
            tr_rate = (okt & (np.abs(rt) > thr) & (pt_ <= cut)).sum() / max(okt.sum(), 1)
            o[kind] = dict(cis_pairs_same_genes=int(okc.sum()), cis_call_rate=float(cis_rate),
                           trans_pairs=int(okt.sum()), trans_call_rate=float(tr_rate),
                           enrichment_cis_over_trans=float(cis_rate / tr_rate) if tr_rate > 0 else None,
                           estimated_fdr=float(min(1.0, tr_rate / cis_rate)) if cis_rate > 0 else None)
        out[est_name] = o
    return out


def cis_vs_trans(cis_res, tr, label):
    same = np.isin(cis_res["gene"], np.unique(tr["gene"]))
    ce = combine(cis_res, "revised", min_nz=1); te = combine(tr, "revised", min_nz=1)
    out = {}
    for kind, rc, rt in [("pooled", cis_res["r_naive"][same], tr["r_naive"]),
                         ("within_group", ce["r_wg"][same], te["r_wg"])]:
        rc = rc[~np.isnan(rc)]; rt = rt[~np.isnan(rt)]
        u = mannwhitneyu(rc, rt, alternative="two-sided")
        o = dict(n_cis=int(len(rc)), n_trans=int(len(rt)),
                 mean_r_cis=float(rc.mean()), mean_r_trans=float(rt.mean()),
                 frac_positive_cis=float((rc > 0).mean()), frac_positive_trans=float((rt > 0).mean()),
                 auc_cis_vs_trans=float(u.statistic / (len(rc) * len(rt))), mwu_p=float(u.pvalue))
        curve = []
        for t in (0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2):
            pc, pt_ = (rc > t).mean(), (rt > t).mean()
            nc, nt = (rc < -t).mean(), (rt < -t).mean()
            curve.append(dict(thr=t, cis_pos=float(pc), trans_pos=float(pt_),
                              enrich_pos=float(pc / pt_) if pt_ > 0 else None,
                              cis_neg=float(nc), trans_neg=float(nt),
                              enrich_neg=float(nc / nt) if nt > 0 else None))
        o["curve"] = curve
        out[kind] = o
    print(f"\n=== {label} ===")
    for kind, o in out.items():
        print(f"  {kind:13s} mean r cis={o['mean_r_cis']:+.4f} trans={o['mean_r_trans']:+.4f} | "
              f"%pos cis={100*o['frac_positive_cis']:.1f} trans={100*o['frac_positive_trans']:.1f} | "
              f"AUC={o['auc_cis_vs_trans']:.3f}")
        print("      |r|>t  enrichment(+)  enrichment(-):",
              "  ".join(f"{c['thr']}: {c['enrich_pos'] if c['enrich_pos'] is None else round(c['enrich_pos'],2)}"
                        f"/{c['enrich_neg'] if c['enrich_neg'] is None else round(c['enrich_neg'],2)}" for c in o["curve"]))
    return out



def cache_factory(out_dir):
    import pickle
    def cache(name, fn):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            log(f"checkpoint: {name}"); return pickle.load(open(p, "rb"))
        log(f"running: {name}"); r = fn(); pickle.dump(r, open(p, "wb"), protocol=4); return r
    return cache
