# PRISM reproducibility metadata

This directory contains position-level metadata for selected PRISM benchmark examples presented in the main figures: human tonsil S2, human lymph node S1, embryonic mouse brain E15.5, human PD striatum C1 and adjacent P22 mouse brain sections. The CSV files record spatial coordinates, modality availability, registration correspondences and supervision/evaluation assignments. Location identifiers and the original row order of the corresponding input or final registration AnnData objects are retained.

### Inputs and public sources

| Tutorials | Local input path(s) | Original source |
| --- | --- | --- |
| 1, 2.1-2.3 | `Datasets/human lymphoid organs/{tonsil,lymph}/...` | Processed human tonsil and lymph node datasets from the SpaMosaic data deposit ([Zenodo record 18946723](https://zenodo.org/records/18946723)). |
| 3.1-3.2 | `Datasets/embryonic mouse brain/E15.5/adata_{atac,rna}.h5ad` | The original MISAR-seq data are available under accession OEP003285. Jiang, F. et al. Simultaneous profiling of spatial gene expression and chromatin accessibility during mouse brain development. *Nature Methods* **20**, 1048-1057 (2023). Tutorial objects are reused from the SpaMosaic experimental-data deposit, [Zenodo record 18946723](https://zenodo.org/records/18946723). |
| 5.1-5.2 | `Datasets/PD human brain/C1/C1_MSI_reg_annot.h5ad`, `C1_RNA.h5ad` | Visium RNA [DOI 10.17044/scilifelab.22778920](https://doi.org/10.17044/scilifelab.22778920) and MSI [DOI 10.17044/scilifelab.22770161](https://doi.org/10.17044/scilifelab.22770161); source-specific citations and terms apply. |
| 6.1-6.2 | `Datasets/P22 mouse brain_adjacent sections/S3_adata_RNA_reg.h5ad` and source/query AnnData inputs | GEO [GSE205055](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055). The exported registration object is an existing tutorial dependency. |

Input datasets and preparation steps are described in [`Datasets/README.md`](../Datasets/README.md). The source AnnData object corresponding to each exported CSV is identified in [`export_summary.csv`](export_summary.csv).

## Mask and registration configuration

- **Tutorial1 (Tonsil_S2)**: source RNA, complete modality; no simulated mask; target ADT, direction=H, width=0.5, step_ratio=0.1, step_id=0.
- **Tutorial2_1 (Lymph_node_S1)**: source RNA, complete modality; no simulated mask; target ADT, direction=H, width=0.5, step_ratio=0.1, step_id=0.
- **Tutorial2_2 (Lymph_node_S1)**: source RNA, complete modality; no simulated mask; target ADT, direction=R, width=0.5, step_ratio=0.1, step_id=0. Random seed is the helper's step_id fallback (0); random_seed was not set in the tutorial.
- **Tutorial2_3 (Lymph_node_S1)**: source RNA, direction=H, width=0.3, step_ratio=0.1, step_id=0; target ADT, direction=H, width=0.4, step_ratio=0.1, step_id=8.
- **Tutorial3_1 (Embryonic_mouse_brain_E15.5)**: source ATAC, complete modality; no simulated mask; target RNA, direction=H, width=0.5, step_ratio=0.1, step_id=0.
- **Tutorial3_2 (Embryonic_mouse_brain_E15.5)**: source ATAC, complete modality; no simulated mask; target RNA, direction=R, width=0.5, step_ratio=0.1, step_id=0. Random seed is the helper's step_id fallback (0); random_seed was not set in the tutorial.
- **Tutorial5.1-5.2 (PD C1)**: one registration CSV exported from the final `C1_MSI_reg_annot.h5ad`; the saved final `missing` field is authoritative, including rows invalidated by missing `new_label`. No other PD sections are included.
- **Tutorial6.1-6.2 (P22)**: `Tutorial6_P22_registration.csv` records the saved `S3_adata_RNA_reg.h5ad` registration in S2 H3K27ac query order, including scSLAT matches retained at similarity threshold 0.85 and the saved availability labels.

The random-missing tutorials do not pass `random_seed` to `simulate_missing_sliding`; the helper therefore uses the tutorial's `step_id` as the random seed. This is recorded as seed 0 where applicable. Training seeds are not used to generate these masks.

## Fields

| Field | Meaning |
| --- | --- |
| `tutorial_id` | Tutorial that consumes or generates this table. |
| `section_id` | Slice, sample, or simulated subset identifier. |
| `modality` | RNA, ADT, ATAC, or MSI represented by the availability flag. |
| `row_index` | Zero-based row index in the corresponding source/final AnnData object. |
| `location_id` | Exact `obs_names` value at that row. |
| `coord_0`, `coord_1` | First two columns of the object's original `obsm['spatial']`, in their original order. |
| `missing` | Tutorial coding normalized to integer: `1` available, `0` unavailable. |
| `supervision_eligible` | `1` when the modality measurement is available to training; otherwise `0`. |
| `evaluation_heldout` | `1` only for simulated positions with known values withheld for quantitative evaluation. Real-unregistered positions are always `0`. |
| `matched_RNA_barcode` | RNA reference location matched to the query row, when applicable. |
| `matched_RNA_index` | Zero-based row index in the current RNA reference object, resolved by exact `matched_RNA_barcode` lookup when available. |
| `stored_matched_RNA_index` | Original saved index, included only where retained as an audit field (P22). |
| `stored_index_matches_barcode` | Whether that saved index points to the saved barcode in the current reference (P22 audit field only). |
| `SLAT_score` | Existing scSLAT similarity score, when available. |
| `original_omics2_barcode` | Original MSI location identifier in the PD registration. |
| `match_distance` | Existing PD spatial matching distance. |

For PD C1, modality availability is determined by the final `missing` field in `C1_MSI_reg_annot.h5ad`. Candidate-match identifiers and distances may remain populated for unavailable rows, including those excluded because of invalid `new_label` annotations.

For P22, `stored_matched_RNA_index = -1` denotes an unmatched row. The resolved RNA barcode and index fields are empty for these rows.

## Interpretation of availability fields

- Simulated missing positions have `missing=0`, `supervision_eligible=0`, and `evaluation_heldout=1`.
- Real-unregistered positions have `missing=0`, `supervision_eligible=0`, and `evaluation_heldout=0`.
- Observed positions have `missing=1`, `supervision_eligible=1`, and `evaluation_heldout=0`.
- A source modality is marked fully available only where the tutorial explicitly treats it as complete.
- Tutorial 2.3 exports independent source and target masks. They are not combined into a single modality mask.
- No separate validation set or per-epoch random center masks are included.

## File inventory and use

[`export_summary.csv`](export_summary.csv) lists the exported CSV files, their source AnnData objects, modalities, row counts and supervision/evaluation counts. These files define the fixed masking and registration configurations for the listed examples. The corresponding tutorials describe how the masks and registration correspondences are constructed and applied. Training-time random center masking is documented separately in the tutorials.