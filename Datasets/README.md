# Tutorial data

The notebooks expect all inputs below `Datasets/` and use paths relative to the repository root. Large source and derived AnnData files are intentionally excluded from the Git repository.

A versioned Zenodo archive and file-level manifest will be added before public release. It will provide the required directory layout, checksums and links to original study repositories. Until then, retain the dataset layout supplied with this tutorial bundle.

Some workflows consume derived registration objects produced by earlier tutorials:

- Tutorial 5.1 prepares the registered SMA MSI object used by Tutorial 5.2.
- Tutorial 6.1 prepares the registered P22 RNA object used by Tutorial 6.2.
- Tutorial 7.1 prepares `adata_matched.h5ad` and `adata_reg.h5ad`, used by Tutorials 7.2 and 7.3.

The original studies and public source datasets are linked in the opening Markdown cell of each tutorial.
