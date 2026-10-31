# Group structure dominates pooled peak–gene associations in single-cell multi-omics

Code accompanying the manuscript by Shirgir, Asir, Sekeroglu and Mirzaei (Near East University),
*Functional & Integrative Genomics* (revised version).

The analyses compare **pooled** peak–gene correlations (computed across all cells) with a
**within-group** estimate (fixed-effect, sample-size-weighted Fisher-z combination of within-group
correlations, *r*<sub>WG</sub>) in paired scRNA/scATAC (10x Multiome) data. Calls are decomposed into
pooled-only, within-only, shared and neither sets, and both estimators are calibrated with

* **label permutation**, with the observed value and the null computed on the same pairs and with the same calling rule, and
* a **matched trans null**: for each sampled gene, every cis peak is replaced by a peak on another
  chromosome with similar accessibility. These pairs cannot be cis-regulatory, so their call rate
  gives an empirical false-discovery estimate for cis calls.

All peak–gene pairs are treated as candidate associations, not demonstrated regulatory links.

## Data

Both datasets are public (GEO):

| Dataset | Accession | Grouping variables |
|---|---|---|
| Six prostate cancer cell lines, drug perturbations (10x Multiome, HTO-multiplexed) | GSE251977 | cell line, treatment, library |
| Human dorsolateral prefrontal cortex, 15 donors (7 AD, 8 control) | GSE214979 | donor, cell type, diagnosis |

## Repository layout

```
confound/core.py                    estimators, calibration (permutation, trans and matched-trans nulls), summaries
analysis/01_prostate_revision.py    prostate: reproduction of the original estimator, revised estimator, call decomposition,
                                    heterogeneity, sensitivity, trans / matched-trans nulls, permutation,
                                    treatment and library designs, cis-vs-trans comparison
analysis/02_brain_revision.py       brain: the same, plus cell type, donor x cell type,
                                    donor within each cell type, donor within each diagnosis
analysis/03_revision_figures.py     figures from the saved results only (Figs. 1-8, Supplementary Fig. S1)
analysis/04_drug_response.py        within-line drug response: AR-response score, pseudobulk log2FC,
                                    VCaP technical-replicate check, chromatin near AR targets (Figs. 9 and 10)
analysis/05_lmm_validation.py       Fisher-z vs linear mixed model on 300 pairs (Supplementary Fig. S2)
analysis/06_figr_style.py           simplified FigR-style linking (Spearman, ±100 kb, positive only, no smoothing)
preprocessing/                      description of the preprocessing that produces checkpoints/ (see preprocessing/README.md)
results/                            place for the summary files produced by the analysis scripts (see below)
```

## Running

1. Preprocess each dataset as described in `preprocessing/README.md` and in Sections 2.2–2.3 of the
   manuscript. This gives, for each dataset, a `checkpoints/` folder with the processed RNA and ATAC
   matrices (`*.h5ad`), `tss*.pkl` and the original link table.
2. Run the analyses:

```bash
python analysis/01_prostate_revision.py --data /path/to/prostate_rebuild
python analysis/02_brain_revision.py    --data /path/to/brain_confound
python analysis/03_revision_figures.py  --prostate /path/to/prostate_rebuild --brain /path/to/brain_confound
python analysis/04_drug_response.py     --data /path/to/prostate_rebuild
python analysis/05_lmm_validation.py    --data /path/to/prostate_rebuild
python analysis/06_figr_style.py        --data /path/to/prostate_rebuild
```

Every expensive step is cached in `<data>/revision_out/`, so re-running resumes where it stopped.
The first step of scripts 01 and 02 recomputes the estimator of the original submission and checks it
against the saved link table (prostate: 5,631 pooled / 180 within-line calls; brain: 6,309 / 5,031).

Runtime on one workstation (RTX 5090, 64 GB RAM): about 1.5 h (prostate) and 4 h (brain).
The permutation test uses a CUDA GPU when available and falls back to the CPU.

## Results files

The analysis scripts write the following files to `<data>/revision_out/`; the numbers in the manuscript
are taken from them. Copy them to `results/` to archive a run:

| File | Produced by | Content |
|---|---|---|
| `REVISION_SUMMARY_prostate.json` | 01 | all prostate analyses |
| `REVISION_SUMMARY_brain.json` | 02 | all brain analyses |
| `DRUG_RESPONSE_SUMMARY.json` | 04 | within-line drug response |
| `P1_pairs_cellline.csv.gz` | 01 | per-pair r_pool, r_WG, informative groups, I², closed-peak groups (prostate, cell line) |
| `B0_design_donor_table.csv` | 02 | brain design by donor (Supplementary Table S2) |

## Requirements

Python ≥ 3.11; see `requirements.txt`.

## Licence

MIT (see `LICENSE`).
