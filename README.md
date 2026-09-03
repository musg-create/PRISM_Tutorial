# PRISM: Niche-informed Deciphering of Incomplete Spatial Multi-Omics Data

This repository contains the official PyTorch implementation and reproducibility tutorials for [PRISM](https://www.biorxiv.org/content/10.64898/2026.02.03.703456v1), a framework for spatial multi-omics with incomplete registration.

## Overview

![Overview of the PRISM framework](overview.png)

Spatial multi-omics, which integrates diverse molecular layers, has emerged as an essential tool for *in situ* characterization of tissue architecture and underlying biological processes. However, current spatial multi-omics sequencing protocols are often hindered by technical incompatibilities, resulting in incomplete spatial pairing due to inconsistent field-of-view or varying spatial resolutions. To address this, we present PRISM, a computational framework designed for misaligned spatial multi-omics data. By leveraging a niche-informed similarity prior, PRISM propagates information from registered to unregistered regions, enabling simultaneous spatial-domain identification and omics imputation. Extensive benchmarking across seven diverse simulated and real-world datasets demonstrates that PRISM consistently outperforms existing methods in spatial multi-omics analysis tasks. When applied to the human Parkinson's disease striatum, PRISM effectively delineated dopamine-associated spatial domains and inferred metabolite distributions previously obscured by incomplete data gaps. Overall, PRISM provides a robust solution for bridging the integration gaps inherent in spatial multi-omics protocols, thereby facilitating more precise downstream biological discovery and interpretation.

## Installation

The supplied Conda environment uses Python 3.10 and contains the analysis and notebook dependencies. PyTorch and PyTorch Geometric (PyG) are installed separately so that the runtime can match the local hardware.

```bash
git clone https://github.com/musg-create/PRISM_Tutorial.git
cd PRISM_Tutorial
conda env create -f environment.yml
conda activate PRISM_Tutorial
```

### GPU configuration

GPU execution is recommended for model training. The tutorials were executed with an NVIDIA A100 80 GB GPU, PyTorch 2.4.0 with CUDA 12.1, and PyG 2.7.0. Install this reference configuration with:

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.7.0
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
  --find-links https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

For another NVIDIA driver or CUDA configuration, select a compatible PyTorch build through the [official PyTorch selector](https://pytorch.org/get-started/locally/) and use the corresponding wheel index in the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). The PyTorch build and PyG wheel index must use the same PyTorch and CUDA tags.

### CPU configuration

For systems without an NVIDIA GPU, install the following CPU configuration:

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.7.0
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
  --find-links https://data.pyg.org/whl/torch-2.4.0+cpu.html
```

After installing either runtime configuration, install PRISM from the repository root:

```bash
pip install -e . --no-deps
```

## Data availability and preparation

Third-party molecular data, processed AnnData objects and generated result files are not redistributed in this repository. Links to the original public data sources are provided in the corresponding tutorials and dataset documentation.

## Quick-start

For a quick evaluation of PRISM, we recommend starting with **Tutorial 2.1: Simulation of FOV-Induced Incomplete Registration in Human Lymph Node**. The tutorial uses **section S1 of the human lymph node RNA-ADT dataset** from a [**publicly available repository**](https://doi.org/10.5281/zenodo.12654113), which provides a processed AnnData version for direct use with **Tutorial 2.1**, including `adata_RNA.h5ad` and `adata_ADT.h5ad`. The same S1 dataset can also be used with **Tutorial 2.2** and **Tutorial 2.3** to evaluate PRISM under random and asymmetric incomplete-registration settings, respectively.

**Expected outputs:** integrated semantic representations, spatial-domain assignments, reconstructed or imputed molecular profiles, and quantitative evaluation results at held-out locations.

**Typical runtime:** approximately 5 minutes using the reference NVIDIA A100 80 GB GPU environment.

## Tutorials

Start with Tutorial 1 for the complete simulated source-to-target workflow, then select a tutorial matching the acquisition setting and modality pair of interest.

| Notebook(s) | Setting |
| --- | --- |
| [Tutorial 1](<Tutorial1: Simulation of FOV-Induced Incomplete Registration in Human Tonsil.ipynb>) | FOV-induced RNA-ADT incomplete registration in human tonsil. |
| [2.1](<Tutorial2_1: Simulation of FOV-Induced Incomplete Registration in Human Lymph Node.ipynb>), [2.2](<Tutorial2_2: Simulation of Random Incomplete Registration in Human Lymph Node.ipynb>) and [2.3](<Tutorial2_3: Simulation of Asymmetric Incomplete Registration in Human Lymph Node.ipynb>) | FOV, random and asymmetric incomplete registration in human lymph node. |
| [3.1](<Tutorial3_1: Simulation of FOV-Induced Incomplete Registration in Embryonic Mouse Brain.ipynb>) and [3.2](<Tutorial3_2: Simulation of Random Incomplete Registration in Embryonic Mouse Brain.ipynb>) | FOV and randomly distributed ATAC-RNA incomplete registration in embryonic mouse brain. |
| [Tutorial 4](<Tutorial4: Simulation of Omics-Specific Domain Unregistration in Mouse Thymus.ipynb>) | Omics-specific domain unregistration in mouse thymus. |
| [5.1](<Tutorial5_1: MAGPIE Registration and Preparation of SMA.ipynb>), [5.2](<Tutorial5_2: Application of PRISM to Real Resolution-Induced Incomplete in Human PD Brain.ipynb>) and [5.3](<Tutorial5_3: Enrichment Analysis of PRISM-Imputed Dopamine-DD in Human PD Brain.ipynb>) | SMA/MAGPIE preparation, real PD-brain RNA-MSI completion and dopamine enrichment. |
| [6.1](<Tutorial6_1: scSLAT Registration of P22 Mouse Brain Adjacent RNA and ATAC Sections.ipynb>) and [6.2](<Tutorial6_2: Application of PRISM to P22 Mouse Brain Adjacent RNA and CUT&Tag Sections.ipynb>) | scSLAT registration and real incomplete registration in adjacent P22 mouse-brain H3K27ac (spatial CUT&Tag) and RNA sections. |
| [7.1](<Tutorial7_1: scSLAT Registration of COAD RNA and CODEX Data.ipynb>), [7.2](<Tutorial7_2: Application of PRISM to COAD Adjacent RNA and CODEX Sections.ipynb>) and [7.3](<Tutorial7_3: Simulation of Cell-Type-Specific Incomplete Registration in COAD.ipynb>) | scSLAT registration, real incomplete registration and cell-type-specific simulation in COAD RNA and CODEX data. |

Tutorials 5.1, 6.1 and 7.1 prepare registered intermediate objects used by their subsequent analysis tutorials. The remaining tutorials can be run independently once their documented inputs are available.

## External prerequisites and scope

- **GLUE and scSLAT**: Tutorials 6.1 and 7.1 begin with compatible, externally prepared GLUE inputs and reproduce the subsequent scSLAT registration stage. This repository does not provide a standalone GLUE workflow or precomputed `X_glue` embeddings; consult the [dataset guide](Datasets/README.md) and [GLUE documentation](https://scglue.readthedocs.io/en/latest/) for the required inputs and embedding preparation.
- **MAGPIE**: MAGPIE is required only to select landmarks for new samples. The underlying SMA [Visium RNA](https://doi.org/10.17044/scilifelab.22778920) and [MALDI-MSI](https://doi.org/10.17044/scilifelab.22770161) measurements must be downloaded separately.
- **KEGG enrichment**: Tutorial 5.3 requires `Rscript`, `curl`, and the R packages `AnnotationDbi`, `org.Hs.eg.db` and `ggplot2`.

## Citation

If you find this repository useful, please consider citing this paper:

```bibtex
@article{mu2026prism,
  title={PRISM: Niche-informed Deciphering of Incomplete Spatial Multi-Omics Data},
  author={Mu, Shiguan and Wang, Zhikang and Liao, Yi and Liang, Jiaming and Zhang, Daoliang and Wang, Chuyao and Xie, Jiahui and Sheng, Xiaoqi and Zhang, Tinghe and Huang, Weitian and others},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.02.03.703456},
  url={https://doi.org/10.64898/2026.02.03.703456}
}
```

## Contact

If you have questions, please contact [sg_mu543@foxmail.com](mailto:sg_mu543@foxmail.com).
