# PRISM: Niche-informed Deciphering of Incomplete Spatial Multi-Omics Data

This repository contains the official PyTorch implementation and reproducibility tutorials for [PRISM](https://www.biorxiv.org/content/10.64898/2026.02.03.703456v1), a framework for spatial multi-omics with incomplete registration.

PRISM combines modality-aware spatial graphs, a niche-informed similarity prior and Transformer-based context integration to support two linked analyses:

- spatial-domain identification from a shared representation;
- missing-omics imputation at target-unregistered locations.

## Overview

![Overview of the PRISM framework](assets/prism-overview.png)

## Installation

Clone the repository and create the supplied CPU-first Conda environment:

```bash
git clone https://github.com/musg-create/PRISM_Tutorial.git
cd PRISM_Tutorial
mamba env create -f environment.yml
mamba activate PRISM_Tutorial
pip install -e . --no-deps
python -m ipykernel install --user --name PRISM_Tutorial --display-name "PRISM_Tutorial"
```

For GPU execution, install the PyTorch build appropriate for the local CUDA driver, followed by compatible PyG extension wheels or Conda packages. Confirm the installation from the repository root:

```bash
python -c "import PRISM; print(PRISM.__version__)"
```

Launch Jupyter from the repository root so tutorial paths resolve relative to `Datasets/` and `Results/`:

```bash
jupyter lab
```

## Data preparation

Third-party molecular data, processed AnnData objects and generated result files are not redistributed in this repository. Each tutorial links to its original public data source and specifies the expected local input layout. The [dataset guide](Datasets/README.md) provides source links, preparation requirements and workflow dependencies.

Tutorial 5.1 includes PRISM-authored MAGPIE landmark coordinate pairs at `Datasets/PD human brain/{A1,B1,C1}/landmark/landmarks_noHE.csv`. These files contain coordinates only, not RNA or MSI measurements.

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

Tutorials 5.1, 6.1 and 7.1 prepare registered intermediate objects used by their subsequent analysis tutorials. The remaining tutorials can be run independently once their documented inputs are available.

## Optional external tools

- **scSLAT and GLUE**: Tutorials 6.1 and 7.1 require a separately installed scSLAT/GLUE environment with source-derived inputs and precomputed `X_glue` embeddings. GLUE preparation is an external prerequisite and is not distributed in this repository. Follow the [SLAT installation guide](https://slat.readthedocs.io/).
- **MAGPIE**: The provided Tutorial 5.1 landmarks were selected with the [MAGPIE interactive landmark-selection tool](https://core-bioinformatics.github.io/magpie/shiny-app/shiny-app.html). Install MAGPIE only to generate landmarks for new samples. The underlying SMA [Visium RNA](https://doi.org/10.17044/scilifelab.22778920) and [MALDI-MSI](https://doi.org/10.17044/scilifelab.22770161) measurements must be downloaded separately.
- **KEGG enrichment**: Tutorial 5.3 requires `Rscript`, `curl`, and the R packages `AnnotationDbi`, `org.Hs.eg.db` and `ggplot2`.

## Reproducibility scope

All notebooks import the installed `PRISM` package directly and use repository-relative paths. Controlled simulations are reproducible after obtaining the paired source inputs. Tutorial 5.1 reproduces the PRISM-specific SMA coordinate transformation and matching stage after users assemble the documented RNA and MSI inputs. Tutorials 6.1 and 7.1 reproduce scSLAT registration after compatible GLUE embeddings have been prepared externally; they are not end-to-end workflows from raw data.

## License

PRISM is released under the [SCUT License](LICENSE).

## Citation

If you find this repository useful, please consider citing this paper:

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

If you have questions, please contact [sg_mu543@foxmail.com](mailto:sg_mu543@foxmail.com).
