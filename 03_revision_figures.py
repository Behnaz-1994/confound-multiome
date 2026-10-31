"""Revision figures. Reads only saved results.
Usage: python 03_revision_figures.py --prostate <dir> --brain <dir>
"""
import os, json, pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--prostate", default=r"C:\Users\Digikey\Desktop\prostate_rebuild")
ap.add_argument("--brain", default=r"C:\Users\Digikey\Desktop\brain_confound")
args, _ = ap.parse_known_args()
PROS, BRAIN = args.prostate, args.brain
PO, BO = os.path.join(PROS, "revision_out"), os.path.join(BRAIN, "revision_out")
FIG = os.path.join(PROS, "revision_figures"); os.makedirs(FIG, exist_ok=True)
SP = json.load(open(os.path.join(PO, "REVISION_SUMMARY_prostate.json")))
SB = json.load(open(os.path.join(BO, "REVISION_SUMMARY_brain.json")))
plt.rcParams.update({"font.size": 10, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
C_POOL, C_WG, C_SH, C_TR = "#c0392b", "#2980b9", "#7f8c8d", "#95a5a6"

def save(name):
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(); print("saved", name)

def bundle(d):   # revised estimator (min_nz = 1) is the one reported in the revision
    return d["revised_minnz1"]

# ---------------------------------------------------------------------------------------------
# R1: decomposition of calls (R1.6) -- prostate cell line, brain donor, brain cell type
# ---------------------------------------------------------------------------------------------
sets = [("Prostate\ncell line", bundle(SP["P1_main"])),
        ("Brain\ndonor", bundle(SB["B1_main"])),
        ("Brain\ncell type", bundle(SB["B4_celltype"])),
        ("Brain\ndonor × cell type", bundle(SB["B5_donor_x_celltype"]))]
fig, ax = plt.subplots(figsize=(7.2, 4))
x = np.arange(len(sets)); w = 0.6
sh = np.array([s["shared"] for _, s in sets]); po = np.array([s["pooled_only"] + s["pooled_only_not_estimable"] for _, s in sets])
wo = np.array([s["wg_only"] for _, s in sets])
ax.bar(x, sh, w, color=C_SH, label="Called by both")
ax.bar(x, po, w, bottom=sh, color=C_POOL, label="Pooled only (not supported within groups)")
ax.bar(x, wo, w, bottom=sh + po, color=C_WG, label="Within-group only")
for i, (_, s) in enumerate(sets):
    ax.text(i, sh[i] + po[i] + wo[i] + 80, f"{100*s['frac_pooled_calls_not_supported_within']:.1f}%\nnot supported",
            ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([n for n, _ in sets])
ax.set_ylabel("Peak–gene calls (|r| > 0.1, FDR < 0.05)")
ax.set_ylim(0, max(sh + po + wo) * 1.25)
ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
ax.set_title("Pooled calls, within-group calls and their overlap")
save("R1_call_decomposition")

# ---------------------------------------------------------------------------------------------
# R2: cis vs matched-trans enrichment of positive associations (R3.3, R1.1)
# ---------------------------------------------------------------------------------------------
panels = [("Prostate: cell line", SP["P8_cis_vs_trans"]["cell_line"]),
          ("Brain: donor", SB["B2_cis_vs_trans"]),
          ("Brain: cell type", SB["B4_celltype"]["cis_vs_trans"])]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, (ttl, d) in zip(axes, panels):
    for kind, col, lab in [("pooled", C_POOL, "Pooled"), ("within_group", C_WG, "Within group")]:
        cv = [c for c in d[kind]["curve"] if c["enrich_pos"] is not None]
        ax.plot([c["thr"] for c in cv], [c["enrich_pos"] for c in cv], "o-", color=col,
                label=f"{lab} (AUC = {d[kind]['auc_cis_vs_trans']:.3f})")
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set_yscale("log"); ax.set_xlabel("Correlation threshold r > t"); ax.set_title(ttl, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("Enrichment of positive r\n(cis / matched trans)")
fig.suptitle("Cis pairs are enriched for positive peak–gene correlation relative to matched trans pairs", y=1.03)
plt.tight_layout(); save("R2_cis_vs_matched_trans")

# ---------------------------------------------------------------------------------------------
# R3: estimated FDR of calls from the matched trans null, across designs (R3.3, R1.7, R2.1)
# ---------------------------------------------------------------------------------------------
rows = [("Prostate: cell line", SP["P3b_trans_matched"]["revised"]),
        ("Prostate: line × drug", SP["P5_line_x_drug"]["trans_matched"]["revised"]),
        ("Prostate: line × library", SP["P6_line_x_library"]["trans_matched"]["revised"])]
rows += [(f"Prostate: {dr} only", v["trans_matched"]["revised"]) for dr, v in SP["P7_by_treatment"].items()]
rows += [("Brain: donor", SB["B2_trans_matched"]["revised"]),
         ("Brain: cell type", SB["B4_celltype"]["trans_matched"]["revised"]),
         ("Brain: donor × cell type", SB["B5_donor_x_celltype"]["trans_matched"]["revised"])]
rows += [(f"Brain: donor within {ct}", v["trans_matched"]["revised"]) for ct, v in SB["B6_donor_within_celltype"].items()]
rows += [(f"Brain: donor within {dg}", v["trans_matched"]["revised"]) for dg, v in SB["B7_donor_within_diagnosis"].items()]
lab = [r[0] for r in rows]
fp = [r[1]["pooled"]["estimated_fdr"] for r in rows]; fw = [r[1]["within_group"]["estimated_fdr"] for r in rows]
fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1.2))
y = np.arange(len(rows))[::-1]
ax.barh(y + 0.2, [np.nan if v is None else v for v in fp], 0.4, color=C_POOL, label="Pooled calls")
ax.barh(y - 0.2, [np.nan if v is None else v for v in fw], 0.4, color=C_WG, label="Within-group calls")
for off, vals, c_ in ((0.2, fp, C_POOL), (-0.2, fw, C_WG)):        # print every value, so zeros stay visible
    for yy, v in zip(y + off, vals):
        txt = "n/a" if v is None else f"{v:.2f}"
        xx = 0.012 if v is None else (max(v, 1.0) if v > 0.85 else v) + 0.012
        ax.text(xx, yy, txt, va="center", fontsize=6.8, color="#333333")
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8.5)
ax.set_xlim(0, 1.18); ax.axvline(1, color="k", lw=0.5, ls=":"); ax.set_xlabel("Estimated FDR (matched-trans call rate / cis call rate)")
ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.4, -0.08 - 0.6 / len(rows)), ncol=2)
ax.set_title("Empirical false-discovery estimate from a matched trans (cannot-be-cis) null")
save("R3_matched_trans_FDR")

# ---------------------------------------------------------------------------------------------
# R4: how much of the pooled signal each grouping accounts for (R1.7, R2.1, R2.4)
# ---------------------------------------------------------------------------------------------
des = [("Prostate: cell line", bundle(SP["P1_main"])), ("Prostate: line × drug", bundle(SP["P5_line_x_drug"])),
       ("Prostate: line × library", bundle(SP["P6_line_x_library"]))]
des += [(f"Prostate: {dr} only", bundle(v)) for dr, v in SP["P7_by_treatment"].items()]
des += [("Brain: donor", bundle(SB["B1_main"])), ("Brain: cell type", bundle(SB["B4_celltype"])),
        ("Brain: donor × cell type", bundle(SB["B5_donor_x_celltype"]))]
des += [(f"Brain: donor within {ct}", bundle(v)) for ct, v in SB["B6_donor_within_celltype"].items()]
des += [(f"Brain: donor within {dg}", bundle(v)) for dg, v in SB["B7_donor_within_diagnosis"].items()]
fig, ax = plt.subplots(figsize=(7, 0.32 * len(des) + 1.2))
y = np.arange(len(des))[::-1]
vals = [100 * s["frac_pooled_calls_not_supported_within"] for _, s in des]
ax.barh(y, vals, 0.6, color=[C_POOL if n.startswith("Prostate") else "#8e44ad" for n, _ in des])
for yy, v, (_, s) in zip(y, vals, des):
    ax.text(v + 1, yy, f"{v:.0f}%  (n = {s['pooled_calls']:,})", va="center", fontsize=7.5)
ax.set_yticks(y); ax.set_yticklabels([n for n, _ in des], fontsize=8.5)
ax.set_xlim(0, 125); ax.set_xlabel("Pooled calls not supported within groups (%)")
ax.set_title("Dependence on the grouping variable and design")
save("R4_designs_not_supported")

# ---------------------------------------------------------------------------------------------
# R5: permutation null and observed value on the SAME pairs (R3 minor 1)
# ---------------------------------------------------------------------------------------------
pp = pickle.load(open(os.path.join(PO, "P4_permutation_same_set.pkl"), "rb"))
pb = pickle.load(open(os.path.join(BO, "B3_permutation_same_set.pkl"), "rb"))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, (ttl, p, col) in zip(axes, [("Prostate (cell line)", pp, C_POOL), ("Brain (donor)", pb, "#8e44ad")]):
    null = 100 * np.array(p["revised_null"]); obs = 100 * p["revised"]["observed"]
    ax.hist(null, bins=30, color=C_TR, label=f"Permuted labels (n = {len(null)})")
    ax.axvline(obs, color=col, lw=2.5, label=f"Observed = {obs:.1f}%")
    ax.set_xlabel("Reduction in calls on the same pairs (%)"); ax.set_ylabel("Count")
    ax.set_title(f"{ttl}: {p['n_pairs']:,} pairs, p = {p['revised']['p_emp']:.4f}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
plt.tight_layout(); save("R5_permutation_same_set")

# ---------------------------------------------------------------------------------------------
# R6: continuous summary -- calls across a fine grid of |r| thresholds (R3.1); replaces old Fig 3
# ---------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
for ax, (ttl, s) in zip(axes, [("Prostate: cell line", bundle(SP["P1_main"])),
                                ("Brain: donor", bundle(SB["B1_main"])),
                                ("Brain: cell type", bundle(SB["B4_celltype"]))]):
    cv = s["curve"]; t = [c["thr"] for c in cv]
    ax.plot(t, [max(c["pooled"], 0.8) for c in cv], color=C_POOL, label="Pooled")
    ax.plot(t, [max(c["wg"], 0.8) for c in cv], color=C_WG, label="Within group")
    ax.plot(t, [max(c["shared"], 0.8) for c in cv], color=C_SH, ls="--", label="Both")
    for tt in (0.05, 0.1, 0.15, 0.2, 0.3): ax.axvline(tt, color="k", lw=0.3, alpha=0.3)
    ax.set_yscale("log"); ax.set_xlabel("Correlation threshold |r|"); ax.set_title(ttl, fontsize=10)
axes[0].set_ylabel("Calls (FDR < 0.05)"); axes[0].legend(frameon=False, fontsize=8)
plt.tight_layout(); save("R6_calls_across_thresholds")

# ---------------------------------------------------------------------------------------------
# FIX of old Fig 3: five thresholds (0.05-0.3), with the 'called by both' counts
# ---------------------------------------------------------------------------------------------
cv = {round(c["thr"], 2): c for c in bundle(SP["P1_main"])["curve"]}
th = [0.05, 0.1, 0.15, 0.2, 0.3]
fig, ax = plt.subplots(figsize=(6.5, 3.8)); x = np.arange(len(th)); w = 0.27
ax.bar(x - w, [cv[t]["pooled"] for t in th], w, color=C_POOL, label="Pooled")
ax.bar(x, [max(cv[t]["wg"], 0.8) for t in th], w, color=C_WG, label="Within cell line")
ax.bar(x + w, [max(cv[t]["shared"], 0.8) for t in th], w, color=C_SH, label="Called by both")
for i, t in enumerate(th):
    ax.text(i - w, cv[t]["pooled"] * 1.15, f"{cv[t]['pooled']:,}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f">{t}" for t in th]); ax.set_yscale("log")
ax.set_xlabel("Correlation threshold |r|"); ax.set_ylabel("Calls (FDR < 0.05)")
ax.legend(frameon=False, fontsize=8, loc="upper right"); ax.set_title("Prostate: calls at five thresholds")
save("FIX_Fig3_five_thresholds")

# ---------------------------------------------------------------------------------------------
# FIX of old Fig 6: neutral title, legend outside the bars
# ---------------------------------------------------------------------------------------------
pt = pd.read_csv(os.path.join(PO, "P1_pairs_cellline.csv.gz"), usecols=["gene", "r_naive", "r_wg_revised"])
std = pt[pt["r_naive"].abs() > 0.2]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
a1.scatter(std["r_naive"].abs(), std["r_wg_revised"].abs(), s=14, alpha=0.5, color=C_SH, edgecolors="none")
a1.axhline(0.1, color=C_POOL, ls="--", lw=1.2); a1.set_ylim(0, max(0.15, float(std["r_wg_revised"].abs().max()) * 1.1))
a1.set_xlabel("|r| pooled"); a1.set_ylabel("|r| within cell line")
a1.set_title(f"(a) Strongest pooled associations (|r| > 0.2; n = {len(std)})", fontsize=10)
ex = ["AR", "VIM", "EPHB1", "FOSL1", "S100A6", "CALML3", "MARK2", "RTN3"]; ed = []
for g in ex:
    r = std[std["gene"] == g]
    if len(r):
        b = r.loc[r["r_naive"].abs().idxmax()]; ed.append((g, abs(b["r_naive"]), abs(b["r_wg_revised"])))
ed = pd.DataFrame(ed, columns=["gene", "pooled", "within"]).sort_values("pooled"); yy = np.arange(len(ed))
a2.barh(yy + 0.2, ed["pooled"], 0.4, color=C_POOL, label="Pooled")
a2.barh(yy - 0.2, ed["within"], 0.4, color=C_WG, label="Within cell line")
a2.set_yticks(yy); a2.set_yticklabels(ed["gene"]); a2.axvline(0.1, color="k", ls="--", lw=0.8)
a2.set_xlabel("|Peak–gene correlation|"); a2.set_title("(b) Examples: strongest peak per gene", fontsize=10)
a2.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
fig.suptitle("Strong pooled peak–gene associations are not supported within cell lines", y=1.02)
plt.tight_layout(); save("FIX_Fig6_strong_pooled_links")

# ---------------------------------------------------------------------------------------------
# FIX of old Fig 8: dominant grouping in each dataset
# ---------------------------------------------------------------------------------------------
bars = [("Prostate\ncell line", bundle(SP["P1_main"]), C_POOL), ("Brain\ndonor", bundle(SB["B1_main"]), "#b39ddb"),
        ("Brain\ncell type", bundle(SB["B4_celltype"]), "#8e44ad")]
fig, ax = plt.subplots(figsize=(5.8, 4))
v = [100 * s["frac_pooled_calls_not_supported_within"] for _, s, _ in bars]
ax.bar([b[0] for b in bars], v, color=[b[2] for b in bars], width=0.6)
for i, vv in enumerate(v): ax.text(i, vv + 2, f"{vv:.1f}%", ha="center", fontweight="bold")
ax.set_ylim(0, 112); ax.set_ylabel("Pooled calls not supported within groups (%)")
ax.set_title("The dominant biological grouping accounts for most pooled calls", fontsize=10.5)
save("FIX_Fig8_cross_dataset")
print("\nAll figures in:", FIG)
