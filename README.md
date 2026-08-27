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

The repository does not redistribute third-party molecular data, processed AnnData objects or generated result files. Each tutorial identifies its original public dataset and the required local input layout; obtain the data from the cited source repository under its own access and reuse terms, then place the prepared inputs below `Datasets/`. [Datasets/README.md](Datasets/README.md) provides the source links, expected paths and workflow dependencies.

The MAGPIE landmark coordinate pairs required by Tutorial 5.1 are included at `Datasets/PD human brain/{A1,B1,C1}/landmark/landmarks_noHE.csv`. They are PRISM-authored coordinate pairs without molecular measurements.

## Tutorials

| Tutorials | Setting |
| --- | --- |
| 1 | FOV-induced RNA-ADT incomplete registration in human tonsil. |
| 2.1-2.3 | FOV, random and asymmetric incomplete registration in human lymph node. |
| 3.1-3.2 | FOV and randomly distributed ATAC-RNA incomplete registration in embryonic mouse brain. |
| 4 | Omics-specific domain unregistration in mouse thymus. |
| 5.1-5.3 | SMA/MAGPIE preparation, real PD-brain RNA-MSI completion and dopamine enrichment. |
| 6.1-6.2 | scSLAT registration and real incomplete registration in adjacent P22 mouse-brain RNA and CUT&Tag sections. |
| 7.1-7.3 | scSLAT registration, real incomplete registration and cell-type-specific simulation in COAD RNA and CODEX data. |

Tutorials 5.1, 6.1 and 7.1 generate registered intermediate objects consumed by the following tutorial. The remaining tutorials can be run independently once their documented inputs are available.

## Optional dependencies

The core environment deliberately excludes external registration frameworks.

- **scSLAT and GLUE**: Tutorials 6.1 and 7.1 require a separately installed scSLAT/GLUE environment. They start from source-derived AnnData inputs with precomputed `X_glue` embeddings; GLUE preparation is an external prerequisite and is not distributed or implemented in this repository. Follow the [SLAT installation guide](https://slat.readthedocs.io/) and select that kernel only for these registration notebooks.
- **MAGPIE**: Tutorial 5.1 uses repository-supplied landmark coordinate pairs obtained with the [MAGPIE interactive landmark-selection tool](https://core-bioinformatics.github.io/magpie/shiny-app/shiny-app.html). The underlying SMA [Visium RNA](https://doi.org/10.17044/scilifelab.22778920) and [MALDI-MSI](https://doi.org/10.17044/scilifelab.22770161) measurements must be downloaded separately. Install MAGPIE only when selecting new landmarks.
- **KEGG enrichment**: Tutorial 5.3 requires `Rscript`, `curl`, and the R packages `AnnotationDbi`, `org.Hs.eg.db` and `ggplot2`.

## Reproducibility scope

Every notebook imports the installed `PRISM` package directly and uses repository-relative data and output paths. Controlled simulations are reproducible after the paired source inputs have been obtained. Tutorial 5.1 reproduces the PRISM-specific SMA coordinate transformation and matching stage after users assemble the documented source-compatible RNA and MSI inputs; the required landmark files are included in this repository. Tutorials 5.2-5.3 use its outputs. Tutorials 6.1 and 7.1 reproduce the scSLAT registration stage after users prepare compatible GLUE embeddings in their own external environment.

## License

PRISM is released under the [SCUT License](LICENSE).

## Citation

If you use PRISM, please cite:

```bibtex
@article{mu2026prism,
  title={PRISM: Niche-informed Deciphering of Incomplete Spatial Multi-Omics Data},
  author={Mu, Shiguan and Wang, Zhikang and Liao, Yi and Liang, Jiaming and Zhang, Daoliang and Wang, Chuyao and Xie, Jiahui and Sheng, Xiaoqi and Zhang, Tinghe and Huang, Weitian and others},
  journal={bioRxiv},
  pages={2026--02},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Contact

For questions, please contact [sg_mu543@foxmail.com](mailto:sg_mu543@foxmail.com).
