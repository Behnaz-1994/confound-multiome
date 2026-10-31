# Preprocessing

The analysis scripts start from a `checkpoints/` folder for each dataset, produced by the preprocessing
described in Sections 2.2–2.3 of the manuscript:

| Dataset | Steps | Files in `checkpoints/` |
|---|---|---|
| GSE251977 (prostate) | HTO demultiplexing (best HTO ≥ 2 × second best), RNA/ATAC alignment (18,291 cells), genes in ≥ 10 cells, normalisation to 10,000 counts + log1p, unified peak matrix (237,856 peaks), original link table | `rna_processed.h5ad`, `atac_processed.h5ad`, `tss.pkl`, `grn_links_fisher.csv` |
| GSE214979 (brain) | metadata from GEO (donor, diagnosis, cell type), QC, normalisation to 10,000 counts + log1p, original link table | `rna_brain_processed.h5ad`, `atac_brain_raw.h5ad`, `tss_brain.pkl`, `grn_brain_links.csv` |

Multiome-GraphVAE was used for embedding, quality control and visualisation only; peak–gene inference in
`analysis/` uses the normalised matrices directly and does not use the learned embedding.
