# Tutorial data

The notebooks use repository-relative paths below `Datasets/`. Third-party molecular data, processed AnnData objects and derived registration objects are intentionally excluded from this repository. Obtain each input from its original public repository and comply with the source-specific access and reuse terms.

| Tutorials | Dataset and public source | Expected local input | Preparation scope |
| --- | --- | --- | --- |
| 1; 2.1-2.3 | Human tonsil and lymph node RNA-ADT, [SpaMosaic Zenodo](https://zenodo.org/records/18946723) | `human lymphoid organs/{tonsil,lymph}/{S1,S2,S3}/adata_{RNA,ADT}.h5ad` | Use the paired source inputs for the controlled simulations. |
| 3.1-3.2 | Embryonic mouse brain MISAR-seq, [SpaMosaic Zenodo](https://zenodo.org/records/18946723) | `embryonic mouse brain/{E13.5,E15.5,E18.5}/adata_{atac,rna}.h5ad` | Use the paired h5ad objects from `processed/Misar-Stereo/Misar-E13`, `Misar-E15` and `Misar-E18` to prepare the selected developmental stage. |
| 4 | Mouse thymus Stereo-CITE-seq, [SpatialGlue Zenodo](https://doi.org/10.5281/zenodo.10362607) | `mouse thymus/adata_{RNA,ADT}.h5ad` | Retain the source-provided `reference_domain` annotation. |
| 5.1-5.3 | Human PD striatum SMA, [Visium RNA](https://doi.org/10.17044/scilifelab.22778920) and [MALDI-MSI](https://doi.org/10.17044/scilifelab.22770161) | `PD human brain/{A1,B1,C1}/...` | Download the SMA RNA and MSI measurements, then prepare the source-compatible `{TAG}_MSI.h5ad` input from the MSI intensity and coordinate files. `landmark/landmarks_noHE.csv` is included for each section; then run Tutorial 5.1. |
| 6.1-6.2 | P22 mouse brain adjacent RNA and CUT&Tag, [GEO GSE205055](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055) | `P22 mouse brain_adjacent sections/...` | Prepare the source-derived AnnData objects and compatible GLUE embeddings externally. Tutorial 6.1 then creates `S3_adata_RNA_reg.h5ad` for Tutorial 6.2. |
| 7.1-7.3 | COAD Xenium and CODEX, [SPATCH](http://spatch.pku-genomics.org/) | `COAD/adata.h5ad`, `COAD/adata_codex.h5ad` | Prepare source-derived AnnData objects and compatible GLUE embeddings externally. Tutorial 7.1 creates `adata_matched.h5ad` and `adata_reg.h5ad` for Tutorials 7.2 and 7.3. |

Tutorial 5.1 uses the repository-supplied landmark coordinate pairs selected through MAGPIE's [interactive landmark-selection tool](https://core-bioinformatics.github.io/magpie/shiny-app/shiny-app.html). These PRISM-authored supplemental coordinates contain no RNA or MSI measurements; the underlying SMA data are not redistributed. Tutorials 6.1 and 7.1 begin after GLUE preparation. No standalone GLUE preparation workflow or precomputed embeddings are supplied here, so these registration notebooks should not be represented as end-to-end reproductions from raw data.

Workflow dependencies are:

- Tutorial 5.1 prepares the registered SMA MSI object used by Tutorial 5.2.
- Tutorial 5.2 prepares the prediction files used by Tutorial 5.3 for PD brain sections A1, B1 and C1.
- Tutorial 6.1 prepares the registered P22 RNA object used by Tutorial 6.2.
- Tutorial 7.1 prepares `adata_matched.h5ad` and `adata_reg.h5ad`, used by Tutorials 7.2 and 7.3.
