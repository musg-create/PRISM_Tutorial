# PRISM

PRISM is a PyTorch framework for spatial multi-omics with incomplete registration. It combines modality-aware spatial graphs, similarity priors and Transformer-based context integration to support two linked analyses: spatial-domain identification and missing-omics imputation.

This repository provides the implementation and dataset-specific tutorials used to reproduce the PRISM workflows. The tutorials cover simulated and real incomplete-registration settings across spatial transcriptomic, chromatin-accessibility, protein and metabolite measurements.

## Installation

The supplied Conda environment is CPU-first and does not pin CUDA, glibc or other machine-specific runtime packages.

```bash
mamba env create -f environment.yml
mamba activate PRISM_Tutorial
pip install -e . --no-deps
python -m ipykernel install --user --name PRISM_Tutorial --display-name "PRISM_Tutorial"
```

For GPU execution, install the PyTorch build appropriate for the local CUDA driver, then install the matching PyG extension wheels or Conda packages. Keep `torch`, `torch-geometric`, `torch-scatter` and `torch-sparse` on compatible versions. The CPU environment remains the portable reference configuration.

Confirm the installation from the repository root:

```bash
python -c "import PRISM; print(PRISM.__version__)"
```

Launch Jupyter from the repository root so that all tutorial paths resolve relative to `Datasets/` and `Results/`:

```bash
jupyter lab
```

## Data and outputs

Large inputs and generated outputs are excluded from Git. See [Datasets/README.md](Datasets/README.md) for the expected data layout and [Results/README.md](Results/README.md) for generated artifacts. A versioned Zenodo data archive, including file checksums and expected output artifacts, will be linked here before public release.

## Tutorials

| Tutorials | Setting |
| --- | --- |
| 1 | FOV-induced RNA-ADT incomplete registration in human tonsil. |
| 2.1-2.3 | FOV, random and asymmetric incomplete registration in human lymph node. |
| 3.1-3.2 | FOV and randomly distributed ATAC-RNA incomplete registration in embryonic mouse brain. |
| 4 | Omics-specific domain unregistration in mouse thymus. |
| 5.1-5.3 | SMA/MAGPIE preparation, real PD-brain RNA-MSI completion and dopamine enrichment. |
| 6.1-6.2 | scSLAT registration and real incomplete registration in adjacent P22 mouse-brain RNA and ATAC sections. |
| 7.1-7.3 | scSLAT registration, real incomplete registration and cell-type-specific simulation in COAD RNA and CODEX data. |

Tutorials 5.1, 6.1 and 7.1 generate registered intermediate objects consumed by the following tutorial. The remaining tutorials can be run independently once their documented inputs are available.

## Optional dependencies

The core environment deliberately excludes external registration frameworks.

- **scSLAT and GLUE**: Tutorials 6.1 and 7.1 require a separately installed scSLAT/GLUE environment. Follow the [SLAT installation guide](https://slat.readthedocs.io/) and select that kernel only for these registration notebooks.
- **MAGPIE**: Tutorial 5.1 uses MAGPIE-derived landmark inputs. Install MAGPIE separately only when regenerating landmarks; the supplied prepared inputs do not require the MAGPIE package itself.
- **KEGG enrichment**: Tutorial 5.3 requires `Rscript`, `curl`, and the R packages `AnnotationDbi`, `org.Hs.eg.db` and `ggplot2`.

## Reproducibility scope

Every notebook imports the installed `PRISM` package directly and uses repository-relative data and output paths. The release will pair this repository with a versioned data archive containing the large inputs, derived registration objects and selected expected outputs needed for end-to-end verification.

## License

PRISM is released under the [SCUT License](LICENSE).

## Citation

Citation details and the repository release DOI will be added with the first public release.
