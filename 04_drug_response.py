"""Within-line drug response, prostate (GSE251977): AR-response score, pseudobulk log2 fold changes,
VCaP technical-replicate check, between-line contrast vs within-line response, chromatin near AR targets.
Writes revision_out/DRUG_RESPONSE_SUMMARY.json, revision_out/D1_within_line_log2FC.csv.gz and
revision_figures/DR1_*.png/pdf, DR2_*.png/pdf (manuscript Figs. 9 and 10).
Usage:  python 04_drug_response.py --data "<folder that contains checkpoints/>"
"""
# Self-contained: reads rna_processed.h5ad / atac_processed.h5ad / tss.pkl. Safe to run in a NEW notebook.
import os, json, pickle, numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp, matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"C:\Users\Digikey\Desktop\prostate_rebuild")
args, _ = ap.parse_known_args()
PROS = args.data
CK = os.path.join(PROS, "checkpoints")
OUT = os.path.join(PROS, "revision_out"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(PROS, "revision_figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight"})

LINES = ["VCaP", "MDA-PCa-2B", "H660"]                       # the three lines that have a DMSO (vehicle) arm
AR_STATUS = {"VCaP": "AR+", "MDA-PCa-2B": "AR+", "H660": "AR−"}
VEH = "DMSO"
DRUGS = ["Enzalutamide", "ARV-110", "SMARCA2.4_1"]
MIN_CELLS = 30
# canonical androgen-responsive genes, fixed BEFORE looking at the drug contrasts
AR_SET = ["KLK3", "KLK2", "TMPRSS2", "NKX3-1", "FKBP5", "PMEPA1", "NDRG1", "ZBTB16", "STEAP4", "ACSL3"]
S = {}

rna = sc.read_h5ad(os.path.join(CK, "rna_processed.h5ad"))
obs = rna.obs.copy()
line, drug, lib = (obs[c].astype(str).values for c in ("cell_line", "drug", "sample"))
AR_SET = [g for g in AR_SET if g in rna.var_names]
S["ar_set_used"] = AR_SET
print("AR target genes used:", AR_SET)

# ---------- design actually available ----------
keep = np.isin(line, LINES)
design = pd.crosstab([obs["cell_line"][keep], obs["drug"][keep]], obs["sample"][keep])
print(design.to_string()); design.to_csv(os.path.join(OUT, "D0_design_vehicle_lines.csv"))

def ok(cl, dr): return ((line == cl) & (drug == dr)).sum() >= MIN_CELLS

# ---------- (1) per-cell AR-response score, within line, drug vs vehicle ----------
Xl = rna[:, AR_SET].X; Xl = Xl.toarray() if sp.issparse(Xl) else np.asarray(Xl)
Z = (Xl - Xl.mean(0)) / (Xl.std(0) + 1e-8)          # each gene weighted equally
score = Z.mean(1)
S["ar_score"] = {}
for cl in LINES:
    v = score[(line == cl) & (drug == VEH)]
    for dr in DRUGS:
        if not (ok(cl, VEH) and ok(cl, dr)): continue
        d = score[(line == cl) & (drug == dr)]
        sd = np.sqrt((v.var() + d.var()) / 2)
        rec = dict(n_vehicle=int(len(v)), n_drug=int(len(d)), mean_vehicle=float(v.mean()), mean_drug=float(d.mean()),
                   cohens_d=float((d.mean() - v.mean()) / sd), mwu_p_cell_level=float(mannwhitneyu(d, v).pvalue))
        # replicate check: per library (only lines sequenced in >1 library)
        libs = sorted(set(lib[(line == cl)]))
        if len(libs) > 1:
            rec["per_library_delta"] = {L: float(score[(line == cl) & (drug == dr) & (lib == L)].mean() -
                                                 score[(line == cl) & (drug == VEH) & (lib == L)].mean()) for L in libs}
        S["ar_score"][f"{cl}|{dr}"] = rec
        print(f"{cl:11s} {dr:13s} d = {rec['cohens_d']:+.2f}  (n={len(v)} vs {len(d)})",
              rec.get("per_library_delta", ""))

# ---------- (2) pseudobulk within-line log2FC (drug vs vehicle), genome-wide ----------
C = rna.layers["counts"]; C = C.tocsr() if sp.issparse(C) else sp.csr_matrix(C)
def pb(mask):
    s = np.asarray(C[mask].sum(0)).ravel().astype(float)
    return s / s.sum() * 1e6
genes = np.asarray(rna.var_names)
lfc = {}
for cl in LINES:
    v = pb((line == cl) & (drug == VEH)) if ok(cl, VEH) else None
    for dr in DRUGS:
        if v is None or not ok(cl, dr): continue
        d = pb((line == cl) & (drug == dr))
        expressed = (v >= 10) | (d >= 10)
        lfc[f"{cl}|{dr}"] = pd.Series(np.where(expressed, np.log2(d + 1) - np.log2(v + 1), np.nan), index=genes)
LFC = pd.DataFrame(lfc); LFC.to_csv(os.path.join(OUT, "D1_within_line_log2FC.csv.gz"))
S["ar_targets_log2FC"] = LFC.loc[AR_SET].round(3).to_dict()
print("\nlog2FC of AR targets (drug vs DMSO, within line):\n", LFC.loc[AR_SET].round(2).to_string())

# VCaP library replicates: does the drug response reproduce between the two libraries?
S["vcap_replicates"] = {}
vl = sorted(set(lib[line == "VCaP"]))
if len(vl) == 2:
    for dr in DRUGS:
        r_ = []
        for L in vl:
            v = pb((line == "VCaP") & (drug == VEH) & (lib == L)); d = pb((line == "VCaP") & (drug == dr) & (lib == L))
            r_.append(pd.Series(np.where((v >= 10) | (d >= 10), np.log2(d + 1) - np.log2(v + 1), np.nan), index=genes))
        both = r_[0].notna() & r_[1].notna()
        top = r_[0][both].abs().nlargest(200).index
        S["vcap_replicates"][dr] = dict(n_genes=int(both.sum()),
                                        spearman_all=float(spearmanr(r_[0][both], r_[1][both])[0]),
                                        sign_agreement_top200=float((np.sign(r_[0][top]) == np.sign(r_[1][top])).mean()))
        print(f"VCaP {dr}: replicate Spearman = {S['vcap_replicates'][dr]['spearman_all']:.2f}, "
              f"sign agreement (top 200) = {S['vcap_replicates'][dr]['sign_agreement_top200']:.2f}")

# between-line contrast (the design of the submitted manuscript) vs within-line drug response
grpA = np.isin(line, ["LNCaP", "VCaP"]); grpB = np.isin(line, ["DU145", "H660", "MDA-PCa-2B"])
a_, b_ = pb(grpA), pb(grpB)
cross = pd.Series(np.where((a_ >= 10) | (b_ >= 10), np.log2(b_ + 1) - np.log2(a_ + 1), np.nan), index=genes)
S["cross_vs_within"] = {}
for k in LFC.columns:
    both = cross.notna() & LFC[k].notna()
    hits_cross = set(cross[both][cross[both].abs() > 1].index); hits_within = set(LFC[k][both][LFC[k][both].abs() > 1].index)
    S["cross_vs_within"][k] = dict(spearman=float(spearmanr(cross[both], LFC[k][both])[0]),
                                  n_cross_hits=len(hits_cross), n_within_hits=len(hits_within),
                                  overlap=len(hits_cross & hits_within))
print("\nbetween-line contrast vs within-line drug response:")
for k, v in S["cross_vs_within"].items(): print(f"  {k:28s}", v)

# ---------- (3) chromatin: accessibility near AR targets, drug vs vehicle, within line ----------
atac = sc.read_h5ad(os.path.join(CK, "atac_processed.h5ad"))
assert np.array_equal(np.asarray(atac.obs_names), np.asarray(rna.obs_names))
tss = pickle.load(open(os.path.join(CK, "tss.pkl"), "rb"))
A = atac.X.tocsr() if sp.issparse(atac.X) else sp.csr_matrix(atac.X)
pk = np.asarray(atac.var_names)
pc = np.array([p.split(":")[0] for p in pk]); ps = np.array([int(p.split(":")[1].split("-")[0]) for p in pk])
near = np.zeros(len(pk), bool)
for g in AR_SET:
    t = tss[tss["gene"] == g]
    if len(t): near |= (pc == t.iloc[0]["chrom"]) & (np.abs(ps - t.iloc[0]["tss"]) <= 50_000)
def pba(mask):
    s = np.asarray(A[mask].sum(0)).ravel().astype(float); return s / s.sum() * 1e6
S["atac_ar_proximal"] = {"n_peaks": int(near.sum())}
rng = np.random.default_rng(1)
atac_rows = []
for cl in LINES:
    if not ok(cl, VEH): continue
    v = pba((line == cl) & (drug == VEH))
    for dr in DRUGS:
        if not ok(cl, dr): continue
        d = pba((line == cl) & (drug == dr))
        l2 = np.log2(d + 1) - np.log2(v + 1)
        acc = (v + d) / 2; open_ = acc > 0
        # background: peaks matched on accessibility (same decile), not near AR targets
        dec = pd.qcut(np.log1p(acc[open_]), 10, labels=False, duplicates="drop")
        dmap = np.full(len(pk), -1); dmap[open_] = dec
        sel = near & open_
        if sel.sum() == 0: continue
        bg = np.concatenate([rng.choice(np.where((dmap == q) & ~near)[0], size=20 * (sel & (dmap == q)).sum(), replace=True)
                             for q in np.unique(dmap[sel]) if ((dmap == q) & ~near).any()])
        rec = dict(median_log2FC_AR_proximal=float(np.median(l2[sel])), median_log2FC_background=float(np.median(l2[bg])),
                   mwu_p=float(mannwhitneyu(l2[sel], l2[bg]).pvalue), n=int(sel.sum()))
        S["atac_ar_proximal"][f"{cl}|{dr}"] = rec
        atac_rows += [(cl, dr, "AR-target proximal", x) for x in l2[sel]] + [(cl, dr, "matched background", x) for x in l2[bg]]
        print(f"ATAC {cl:11s} {dr:13s} median log2FC near AR targets = {rec['median_log2FC_AR_proximal']:+.3f} "
              f"vs background {rec['median_log2FC_background']:+.3f} (p = {rec['mwu_p']:.2g}, n = {rec['n']})")
AT = pd.DataFrame(atac_rows, columns=["line", "drug", "set", "log2FC"])

json.dump(S, open(os.path.join(OUT, "DRUG_RESPONSE_SUMMARY.json"), "w"), indent=1, default=float)

# ---------- figures ----------
conds = [VEH] + DRUGS
cols = {"DMSO": "#bdc3c7", "Enzalutamide": "#c0392b", "ARV-110": "#e67e22", "SMARCA2.4_1": "#2980b9"}
fig, axes = plt.subplots(1, len(LINES), figsize=(12, 3.8), sharey=True)
for ax, cl in zip(axes, LINES):
    data, labs, cc = [], [], []
    for dr in conds:
        if ok(cl, dr): data.append(score[(line == cl) & (drug == dr)]); labs.append(dr.replace("SMARCA2.4_1", "SMARCA2/4d")); cc.append(cols[dr])
    if not data:
        ax.set_title(f"{cl}: no condition with >= {MIN_CELLS} cells", fontsize=9); ax.axis("off"); continue
    bp = ax.boxplot(data, tick_labels=labs, showfliers=False, patch_artist=True)
    for p_, c_ in zip(bp["boxes"], cc): p_.set_facecolor(c_)
    ax.set_title(f"{cl} ({AR_STATUS[cl]})", fontsize=10); ax.tick_params(axis="x", rotation=30)
axes[0].set_ylabel("AR-response score (per cell)")
fig.suptitle("Within-line drug response: AR-target expression vs vehicle", y=1.03)
plt.tight_layout()
for ext in ("png", "pdf"): plt.savefig(os.path.join(FIG, f"DR1_AR_score_within_line.{ext}"))
plt.close()


# ---------- label placement (non-overlapping gene labels in Fig. 10a) ----------
def place_labels(ax, xs, ys, names, fontsize=8, dx=8, pad=6, max_shift=140, iters=500, flip_px=10):
    """Labels beside each point (right by default, left if crowded); overlaps resolved in pixel space."""
    fig_ = ax.figure
    anns = [ax.annotate(n, (x, y), xytext=(dx, 0), textcoords="offset points", fontsize=fontsize, va="center",
                        zorder=5, arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555", shrinkA=0, shrinkB=3))
            for x, y, n in zip(xs, ys, names)]
    fig_.canvas.draw(); r = fig_.canvas.get_renderer()
    P = ax.transData.transform(np.c_[xs, ys])
    W = np.array([a.get_window_extent(r).width for a in anns]) + 4
    H = np.array([a.get_window_extent(r).height for a in anns]) + pad
    px = dx * fig_.dpi / 72
    def solve(side):
        x0 = np.where(side > 0, P[:, 0] + px, P[:, 0] - px - W); y = P[:, 1].astype(float).copy()
        for _ in range(iters):
            moved = False
            for i in range(len(anns)):
                for j in range(i + 1, len(anns)):
                    if x0[i] < x0[j] + W[j] and x0[j] < x0[i] + W[i]:
                        need = (H[i] + H[j]) / 2 - abs(y[i] - y[j])
                        if need > 0:
                            s_ = 1 if (y[i] > y[j] or (y[i] == y[j] and i < j)) else -1
                            y[i] += s_ * (need / 2 + 0.5); y[j] -= s_ * (need / 2 + 0.5); moved = True
                for k in range(len(anns)):
                    if k != i and x0[i] - 6 < P[k, 0] < x0[i] + W[i] + 6:
                        need = H[i] / 2 + 6 - abs(P[k, 1] - y[i])
                        if need > 0:
                            y[i] += (1 if y[i] >= P[k, 1] else -1) * (need + 0.5); moved = True
            y = np.clip(y, P[:, 1] - max_shift, P[:, 1] + max_shift)
            if not moved: break
        return y
    side = np.ones(len(anns))
    y = solve(side)
    for _ in range(len(anns)):                       # move the most displaced label to the left, one at a time
        cand = [k for k in range(len(anns)) if side[k] > 0 and abs(y[k] - P[k, 1]) > flip_px]
        if not cand: break
        k = max(cand, key=lambda k: abs(y[k] - P[k, 1])); trial = side.copy(); trial[k] = -1
        y2 = solve(trial)
        if np.abs(y2 - P[:, 1]).sum() < np.abs(y - P[:, 1]).sum(): side, y = trial, y2
        else: break
    for a, yy, p, sd in zip(anns, y, P, side):
        a.xyann = (dx * sd, (yy - p[1]) * 72 / fig_.dpi); a.set_ha("left" if sd > 0 else "right")
    return anns

k0 = "VCaP|Enzalutamide"
both = cross.notna() & LFC[k0].notna()

# ---------- figure ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
ax.scatter(cross[both], LFC[k0][both], s=3, alpha=0.3, color="#7f8c8d", edgecolors="none", rasterized=True)
lab = [g for g in AR_SET if g in both.index and both[g]]
ax.scatter(cross[lab], LFC[k0][lab], s=26, color="#c0392b", zorder=4)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("Between-line contrast, log2FC\n(DU145+H660+MDA vs LNCaP+VCaP)")
ax.set_ylabel("Within-line response, log2FC\n(VCaP: Enzalutamide vs DMSO)")
ax.set_title(f"(a) Spearman ρ = {S['cross_vs_within'][k0]['spearman']:.2f}", fontsize=10)
place_labels(ax, cross[lab].values, LFC[k0][lab].values, lab)

ax = axes[1]
order = [(cl, dr) for cl in LINES for dr in DRUGS if f"{cl}|{dr}" in S["atac_ar_proximal"]]
pos = np.arange(len(order))
for off, st, c_ in [(-0.18, "AR-target proximal", "#c0392b"), (0.18, "matched background", "#95a5a6")]:
    dd = [AT[(AT.line == cl) & (AT.drug == dr) & (AT.set == st)]["log2FC"].values for cl, dr in order]
    b2 = ax.boxplot(dd, positions=pos + off, widths=0.32, showfliers=False, patch_artist=True)
    for p_ in b2["boxes"]: p_.set_facecolor(c_)
ax.set_xticks(pos)
ax.set_xticklabels([f"{cl}\n{dr.replace('SMARCA2.4_1', 'SMARCA2/4d')}" for cl, dr in order], fontsize=7.5, rotation=45, ha="right")
ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("Peak accessibility log2FC (drug vs DMSO)")
ax.set_title("(b) Chromatin near AR targets vs matched background", fontsize=10)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#c0392b", label="AR-target proximal"), Patch(color="#95a5a6", label="Matched background")],
          frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
plt.tight_layout()
for ext in ("png", "pdf"): plt.savefig(os.path.join(FIG, f"DR2_cross_vs_within_and_chromatin.{ext}"))
plt.close()
print("\nsaved: DRUG_RESPONSE_SUMMARY.json, D1_within_line_log2FC.csv.gz, figures DR1 and DR2 in", FIG)
