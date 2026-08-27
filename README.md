# PRISM: Niche-informed Deciphering of Incomplete Spatial Multi-Omics Data

This repository contains the official PyTorch implementation and reproducibility tutorials for [PRISM](https://www.biorxiv.org/content/10.64898/2026.02.03.703456v1), a framework for spatial multi-omics with incomplete registration.

## Overview

![Overview of the PRISM framework](overview.png)

Spatial multi-omics, which integrates diverse molecular layers, has emerged as an essential tool for *in situ* characterization of tissue architecture and underlying biological processes. However, current spatial multi-omics sequencing protocols are often hindered by technical incompatibilities, resulting in incomplete spatial pairing due to inconsistent field-of-view or varying spatial resolutions. To address this, we present PRISM, a computational framework designed for misaligned spatial multi-omics data. By leveraging a niche-informed similarity prior, PRISM propagates information from registered to unregistered regions, enabling simultaneous spatial-domain identification and omics imputation. Extensive benchmarking across seven diverse simulated and real-world datasets demonstrates that PRISM consistently outperforms existing methods in spatial multi-omics analysis tasks. When applied to the human Parkinson's disease striatum, PRISM effectively delineated dopamine-associated spatial domains and inferred metabolite distributions previously obscured by incomplete data gaps. Overall, PRISM provides a robust solution for bridging the integration gaps inherent in spatial multi-omics protocols, thereby facilitating more precise downstream biological discovery and interpretation.

## Installation

The recommended installation uses Conda or Mamba with Python 3.10. The shared environment specifies only platform-independent analysis and notebook dependencies. Install PyTorch and PyTorch Geometric (PyG) separately for the local CPU or CUDA configuration; this avoids coupling the repository to a particular driver, CUDA runtime or system library.

### 1. Create the base environment

```bash
git clone https://github.com/musg-create/PRISM_Tutorial.git
cd PRISM_Tutorial
mamba env create -f environment.yml
conda activate PRISM_Tutorial
```

Replace `mamba` with `conda` if Mamba is unavailable. `environment.yml` includes `scikit-misc`, which Scanpy requires for the Seurat v3 highly variable feature selection used in the tutorials, and R/mclust for the mclust-based spatial-domain evaluation in Tutorial 1.

### 2. Install the PRISM runtime

For the tested Linux x86_64 CPU configuration, install the matching PyTorch 2.4.0 and PyG packages:

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.7.0
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
  --find-links https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install -e . --no-deps
```

This CPU configuration was tested by running Tutorial 1 with its documented human tonsil inputs through preprocessing, prior construction, PRISM training, spatial-domain analysis and ADT imputation. For GPU execution, first install a PyTorch build selected for the local CUDA driver from the [official PyTorch selector](https://pytorch.org/get-started/locally/). Then install `torch-geometric`, `torch-scatter` and `torch-sparse` using the matching wheel set from the [PyG installation guide](https://data.pyg.org/whl/). Do not combine the CPU commands above with a CUDA-enabled PyTorch installation.

### 3. Verify and register a notebook kernel

```bash
python - <<'PY'
import PRISM
import scanpy
import skmisc
import torch
import torch_geometric
import torch_sparse

print(f"PRISM {PRISM.__version__}")
print(f"PyTorch {torch.__version__}")
print(f"PyG {torch_geometric.__version__}")
PY
python -m ipykernel install --user --name PRISM_Tutorial --display-name "PRISM_Tutorial"
```

Launch Jupyter from the repository root so tutorial paths resolve relative to `Datasets/` and `Results/`:

```bash
jupyter lab
```

For an existing Python environment, `requirements.txt` records the same general Python dependencies. It deliberately excludes PyTorch and PyG because those packages must match the local platform. A pip-only installation also requires a separately installed R runtime and the R package `mclust`; the Conda/Mamba route above provisions both.

## Quick start

For the complete simulated source-to-target workflow, follow the [dataset guide](Datasets/README.md) to prepare the paired human tonsil inputs, then run [Tutorial 1](<Tutorial1: Simulation of FOV-Induced Incomplete Registration in Human Tonsil.ipynb>) from top to bottom. Select subsequent tutorials after preparing their documented inputs.

## Data availability and preparation

Third-party molecular data, processed AnnData objects and generated result files are not redistributed in this repository. Each tutorial links to its original public data source and specifies the expected local input layout. The [dataset guide](Datasets/README.md) provides source links, preparation requirements and workflow dependencies.

Tutorial 5.1 uses repository-supplied landmark coordinate pairs selected through [MAGPIE's interactive landmark-selection tool](https://core-bioinformatics.github.io/magpie/shiny-app/shiny-app.html). These PRISM-authored supplementary coordinates at `Datasets/PD human brain/{A1,B1,C1}/landmark/landmarks_noHE.csv` contain no RNA or MSI measurements.

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
