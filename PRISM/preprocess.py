"""
omics_preprocessing.py
Unified preprocessing utilities for RNA, ADT / Protein, ATAC / Epigenome,
and Metabolomics modalities.
Main entry point:
    preprocess_omics(adata, modality="RNA" / "ADT" / "ATAC" / "METABOLOMICS", ...)
Design
------
1. RNA:
   filter features -> select HVGs -> normalize_total -> log1p -> subset HVGs
   -> save log layer -> PCA -> mask / indices.
2. Metabolomics:
   Same logic as RNA, but saved with metabolomics-specific keys.
3. ADT / Protein:
   normalize_total -> log1p -> save log layer -> optional PCA
   -> protein mask / indices.
   No HVG selection by default.
4. ATAC / Epigenome:
   save counts -> select variable peaks -> TF-IDF -> L1 normalize
   -> log1p scale -> LSI / SVD -> mask / indices.
   No batch correction is included.
Missing convention
------------------
    observed_label = 1 means observed / non-missing.
    Other values are treated as missing.
    If missing_key does not exist, all cells are treated as observed.
"""
from __future__ import annotations
from typing import Optional, Tuple, Sequence
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from .core import pca
def save_raw_layer_and_counts(adata, layer_key="raw_data", count_key="raw_total_counts"):
    """
    Save raw matrix and raw total counts before preprocessing.
    Your prism_eval_and_save(raw) needs these two fields.
    """
    adata.layers[layer_key] = adata.X.copy()
    raw_X = adata.layers[layer_key]
    adata.obs[count_key] = np.asarray(raw_X.sum(axis=1)).ravel() if sp.issparse(raw_X) else np.asarray(raw_X).sum(axis=1)
    return adata
def save_raw_eval_fields(
    adata,
    raw_X,
    raw_total_counts,
    layer_key="raw_data",
    count_key="raw_total_counts",
):
    """
    Save raw data and total counts for raw-space imputation evaluation.
    raw_X:
        Raw matrix with the same feature columns as final adata.X.
    raw_total_counts:
        Per-cell total counts used by normalize_total.
        It does not have to be the row sum of raw_X if normalization was done
        before feature subsetting, e.g. RNA / metabolomics.
    """
    adata.layers[layer_key] = raw_X.copy()
    adata.obs[count_key] = np.asarray(raw_total_counts, dtype=np.float32).ravel()
    return adata
# ============================================================
# 1. General utilities
# ============================================================
def get_missing_indices(
    adata,
    missing_key: str = "missing",
    observed_label=1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return missing and non-missing cell indices based on adata.obs[missing_key].
    Convention:
        observed_label = 1 means observed / non-missing.
        Other values are treated as missing.
    If missing_key does not exist:
        all cells are treated as observed.
    """
    n_obs = adata.n_obs
    if missing_key not in adata.obs.columns:
        missing_indices = np.array([], dtype=np.int64)
        non_missing_indices = np.arange(n_obs, dtype=np.int64)
        return missing_indices, non_missing_indices
    values = adata.obs[missing_key].to_numpy()
    try:
        values_numeric = np.asarray(values, dtype=float)
        observed = values_numeric == float(observed_label)
    except (ValueError, TypeError):
        observed = values.astype(str) == str(observed_label)
    non_missing_indices = np.flatnonzero(observed).astype(np.int64)
    missing_indices = np.flatnonzero(~observed).astype(np.int64)
    return missing_indices, non_missing_indices
def build_and_save_mask(
    adata,
    missing_indices,
    non_missing_indices,
    modality: str,
    mask_key: Optional[str] = None,
    data_role: Optional[str] = None,
    save_modality_mask: bool = False,
):
    """
    Save observed/missing mask.
    Preferred new logic:
        data_role="source" -> save adata.obsm["source_mask"]
        data_role="target" -> save adata.obsm["target_mask"]
    Optional compatibility:
        save_modality_mask=True also saves modality-specific masks such as
        rna_mask / adt_mask / protein_mask / atac_mask.
    Mask:
        1 = observed / non-missing
        0 = missing
    Since PRISM experiments use modality-level (whole-row) missingness, the
    saved mask is one boolean value per cell rather than a repeated
    ``n_obs x n_vars`` feature matrix.
    """
    modality = str(modality).lower()
    if data_role is not None:
        data_role = str(data_role).lower()
        if data_role not in ["source", "target"]:
            raise ValueError("data_role must be one of ['source', 'target', None].")
    mask = np.ones(adata.n_obs, dtype=bool)
    if len(missing_indices) > 0:
        mask[missing_indices] = False
    saved_mask_keys = []
    # New role-specific mask.
    role_mask_key = None
    if data_role is not None:
        role_mask_key = f"{data_role}_mask"
        adata.obsm[role_mask_key] = mask
        adata.uns[f"{data_role}_mask_key"] = role_mask_key
        adata.uns[f"{data_role}_missing_indices"] = missing_indices
        adata.uns[f"{data_role}_non_missing_indices"] = non_missing_indices
        saved_mask_keys.append(role_mask_key)
    # Optional modality-specific mask for backward compatibility.
    if save_modality_mask:
        if mask_key is None:
            mask_key = f"{modality}_mask"
        adata.obsm[mask_key] = mask
        adata.uns[f"{modality}_mask_key"] = mask_key
        adata.uns[f"{modality}_missing_indices"] = missing_indices
        adata.uns[f"{modality}_non_missing_indices"] = non_missing_indices
        saved_mask_keys.append(mask_key)
        # Compatibility alias for old ADT code.
        if modality in ["adt", "protein"] and mask_key != "protein_mask":
            adata.obsm["protein_mask"] = mask
            adata.uns["adt_legacy_protein_mask_key"] = "protein_mask"
            saved_mask_keys.append("protein_mask")
    # If neither role nor modality mask is requested, keep a safe generic fallback.
    if data_role is None and not save_modality_mask:
        if mask_key is None:
            mask_key = f"{modality}_mask"
        adata.obsm[mask_key] = mask
        adata.uns[f"{modality}_mask_key"] = mask_key
        saved_mask_keys.append(mask_key)
    # Generic indices. Useful when each AnnData stores only one modality.
    adata.uns["missing_indices"] = missing_indices
    adata.uns["non_missing_indices"] = non_missing_indices
    adata.uns["data_role"] = data_role
    adata.uns["modality"] = modality
    adata.uns["saved_mask_keys"] = saved_mask_keys
    info = {
        "modality": modality,
        "data_role": data_role,
        "mask_key": role_mask_key if role_mask_key is not None else mask_key,
        "saved_mask_keys": saved_mask_keys,
        "missing_indices": missing_indices,
        "non_missing_indices": non_missing_indices,
        "has_missing": len(missing_indices) > 0,
    }
    return info
def _row_sums(X):
    return np.asarray(X.sum(axis=1)).ravel()
def _unique_preserve_order(values):
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for v in values:
        v = str(v)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return np.asarray(out, dtype=object)
def _resolve_keep_features(
    var_names,
    keep_features: Optional[Sequence[str]] = None,
    case_sensitive: bool = True,
):
    """
    Resolve user-requested features against adata.var_names.
    Returns
    -------
    found_features:
        Feature names that exist in adata.var_names, ordered by original var_names.
    missing_features:
        Requested feature names that are not found.
    """
    if keep_features is None:
        return np.asarray([], dtype=object), np.asarray([], dtype=object)
    if isinstance(keep_features, str):
        keep_features = [keep_features]
    requested = [str(x) for x in keep_features]
    var_names_arr = np.asarray(var_names).astype(str)
    if len(requested) == 0:
        return np.asarray([], dtype=object), np.asarray([], dtype=object)
    if case_sensitive:
        var_set = set(var_names_arr)
        matched_set = {x for x in requested if x in var_set}
        missing = [x for x in requested if x not in var_set]
    else:
        lower_to_real = {}
        for name in var_names_arr:
            lower_to_real.setdefault(name.lower(), name)
        matched_set = set()
        missing = []
        for x in requested:
            key = x.lower()
            if key in lower_to_real:
                matched_set.add(lower_to_real[key])
            else:
                missing.append(x)
    found = [x for x in var_names_arr if x in matched_set]
    return np.asarray(found, dtype=object), np.asarray(missing, dtype=object)
def normalize_total_log1p_safe(
    adata,
    target_sum: float = 1e4,
    dtype=np.float32,
):
    """
    Fast normalize_total + log1p.
    For each non-zero cell:
        X_i = X_i / sum(X_i) * target_sum
        X_i = log1p(X_i)
    All-zero rows are kept as zero.
    """
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr(copy=False)
        X = X.astype(dtype, copy=False)
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        valid = row_sums > 0
        if valid.sum() == 0:
            adata.X = X
            return adata
        scale = np.ones(X.shape[0], dtype=dtype)
        scale[valid] = target_sum / row_sums[valid].astype(dtype)
        nnz_per_row = np.diff(X.indptr)
        row_scale_for_data = np.repeat(scale, nnz_per_row)
        X.data *= row_scale_for_data
        np.log1p(X.data, out=X.data)
        adata.X = X
        return adata
    else:
        X = np.asarray(X, dtype=dtype)
        row_sums = X.sum(axis=1)
        valid = row_sums > 0
        if valid.sum() > 0:
            X_valid = X[valid, :].copy()
            X_valid *= (target_sum / row_sums[valid])[:, None]
            X_valid = np.log1p(X_valid)
            X[valid, :] = X_valid.astype(dtype)
        adata.X = X
        return adata
def normalize_by_missing_status(
    adata,
    target_sum: float = 1e4,
):
    """
    Fast replacement.
    Cell-wise normalization is equivalent whether missing and observed cells
    are processed together or in separate groups when ``target_sum`` is fixed.
    """
    return normalize_total_log1p_safe(
        adata,
        target_sum=target_sum,
        dtype=np.float32,
    )
def _compute_pca_missing_aware(
    adata,
    missing_indices,
    non_missing_indices,
    pca_key: str = "X_pca",
    max_pcs: int = 50,
):
    """
    Compute PCA and save it to adata.obsm[pca_key].
    If missing exists:
        fit / compute PCA only on non-missing cells,
        then write PCA coordinates back to original cell space.
        Missing cells are filled with zeros.
    If no missing:
        compute PCA on all cells.
    """
    has_missing = len(missing_indices) > 0
    if has_missing:
        n_ref = len(non_missing_indices)
        n_pcs = min(max_pcs, n_ref - 1, adata.n_vars - 1)
        if n_pcs > 0:
            adata_ref = adata[non_missing_indices, :].copy()
            pca_ref = pca(adata_ref, n_comps=n_pcs).astype("float32")
            pca_full = np.zeros((adata.n_obs, n_pcs), dtype=np.float32)
            pca_full[non_missing_indices, :] = pca_ref
            adata.obsm[pca_key] = pca_full
        else:
            adata.obsm[pca_key] = np.empty((adata.n_obs, 0), dtype=np.float32)
        return "non_missing_cells_only"
    n_pcs = min(max_pcs, adata.n_obs - 1, adata.n_vars - 1)
    if n_pcs > 0:
        adata.obsm[pca_key] = pca(adata, n_comps=n_pcs).astype("float32")
    else:
        adata.obsm[pca_key] = np.empty((adata.n_obs, 0), dtype=np.float32)
    return "all_cells"
# ============================================================
# 2. RNA-like preprocessing: RNA and Metabolomics
# ============================================================
def preprocess_rna_like(
    adata,
    modality: str,
    hv_features: int = 3000,
    min_cells: int = 10,
    target_sum: float = 1e4,
    missing_key: str = "missing",
    observed_label=1,
    log_layer_key: str = "log",
    pca_key: str = "X_pca",
    max_pcs: int = 50,
    compute_pca: bool = True,
    save_log_layer: bool = True,
    save_raw_eval: bool = True,
    mask_key: Optional[str] = None,
    data_role: Optional[str] = None,
    save_modality_mask: bool = False,
    keep_features: Optional[Sequence[str]] = None,
    keep_features_case_sensitive: bool = True,
    keep_features_strict: bool = False,
    copy: bool = True,
):
    """
    RNA-like preprocessing used by RNA and Metabolomics.
    Order:
        filter_features
        -> highly_variable_features
        -> normalize_total
        -> log1p
        -> subset highly_variable_features
        -> save log layer
        -> PCA
        -> mask / indices
    Missing-aware behavior:
        If missing exists, feature filtering and HVF selection use non-missing cells.
        Normalization is done separately for missing and non-missing parts.
    """
    modality = modality.lower()
    if copy:
        adata = adata.copy()
    # --------
    # 1. Missing / non-missing indices
    # --------
    missing_indices, non_missing_indices = get_missing_indices(
        adata,
        missing_key=missing_key,
        observed_label=observed_label,
    )
    has_missing = len(missing_indices) > 0
    # --------
    # 2. Reference for feature filtering and HVF selection
    # --------
    if has_missing and len(non_missing_indices) > 0:
        adata_ref = adata[non_missing_indices, :].copy()
    else:
        adata_ref = adata.copy()
    # --------
    # 3. Optional feature filtering
    # --------
    # RNA can use min_cells filtering.
    # Metabolomics can set min_cells=None to skip this step.
    use_feature_filter = min_cells is not None and int(min_cells) > 0
    if use_feature_filter:
        sc.pp.filter_genes(adata_ref, min_cells=int(min_cells))
    if adata_ref.n_vars == 0:
        if use_feature_filter:
            raise ValueError(
                f"No features remain after filter_genes for modality={modality}. "
                "Please reduce min_cells or check the input matrix."
            )
        raise ValueError(f"No features found for modality={modality}.")
    filtered_feature_names = adata_ref.var_names.to_numpy()
    # --------
    # 4. Highly variable feature selection
    # --------
    n_top_features = min(int(hv_features), adata_ref.n_vars)
    sc.pp.highly_variable_genes(
        adata_ref,
        flavor="seurat_v3",
        n_top_genes=n_top_features,
    )
    hv_feature_names = adata_ref.var_names[adata_ref.var["highly_variable"]].to_numpy()
    hvg_rank_by_feature = None
    if "highly_variable_rank" in adata_ref.var.columns:
        hvg_rank_by_feature = pd.to_numeric(
            adata_ref.var["highly_variable_rank"],
            errors="coerce",
        ).copy()
    if len(hv_feature_names) == 0:
        raise ValueError(f"No highly variable features were selected for {modality}.")
    # --------
    # 4.1 Force-keep user-specified features/metabolites
    # --------
    keep_feature_names, missing_keep_feature_names = _resolve_keep_features(
        adata.var_names,
        keep_features=keep_features,
        case_sensitive=keep_features_case_sensitive,
    )
    if len(missing_keep_feature_names) > 0:
        msg = (
            f"The following requested keep_features were not found in adata.var_names "
            f"for modality={modality}: {list(missing_keep_feature_names)}"
        )
        if keep_features_strict:
            raise ValueError(msg)
        warnings.warn(msg)
    # If feature filtering was used, force-kept features should be added back
    # to the processing feature space before normalization.
    filtered_feature_names = _unique_preserve_order(
        list(filtered_feature_names) + list(keep_feature_names)
    )
    # Final selected features = HV features + force-kept features.
    selected_feature_set = set(hv_feature_names).union(set(keep_feature_names))
    selected_feature_names = np.asarray(
        [x for x in filtered_feature_names if x in selected_feature_set],
        dtype=object,
    )
    if len(selected_feature_names) == 0:
        raise ValueError(f"No selected features remain for {modality}.")
    # --------
    # 5. Keep filtered features first, not selected features yet
    # --------
    adata = adata[:, filtered_feature_names].copy()
    adata.var["highly_variable"] = adata.var_names.isin(hv_feature_names)
    if hvg_rank_by_feature is not None:
        adata.var["highly_variable_rank"] = hvg_rank_by_feature.reindex(
            adata.var_names
        ).to_numpy(dtype=np.float32)
    adata.var["force_kept"] = adata.var_names.isin(keep_feature_names)
    adata.var["selected_feature"] = adata.var_names.isin(selected_feature_names)
    # --------
    # 5.1 Save raw fields for raw-space evaluation
    # --------
    # Important:
    # normalize_total is applied in filtered-feature space, before HV feature subsetting.
    # Therefore raw_total_counts must be computed from filtered features.
    # raw_data must be restricted to final HV features so that it matches final adata.X.
    if save_raw_eval:
        raw_total_counts_eval = _row_sums(adata.X).astype(np.float32)
        raw_X_eval = adata[:, selected_feature_names].X.copy()
    # --------
    # 6. Normalize + log1p before HV feature subsetting
    # --------
    normalize_by_missing_status(
        adata,
        target_sum=target_sum,
    )
    # # --------
    # # 7. Subset HV features
    # # --------
    # adata = adata[:, adata.var["highly_variable"]].copy()
    # adata.var["highly_variable"] = True
    # --------
    # 7. Subset selected features: HV features + force-kept features
    # --------
    adata = adata[:, selected_feature_names].copy()
    # Preserve the unforced highly-variable-feature selection.
    adata.var["highly_variable_raw"] = adata.var_names.isin(hv_feature_names)
    # Record features retained explicitly by the caller.
    adata.var["force_kept"] = adata.var_names.isin(keep_feature_names)
    # All retained features enter the model.
    adata.var["selected_feature"] = True
    adata.var["highly_variable"] = True
    # --------
    # 8. Save log layer
    # --------
    if save_log_layer:
        adata.layers[log_layer_key] = adata.X.copy()
    # --------
    # 8.1 Save raw data for raw-space evaluation
    # --------
    if save_raw_eval:
        adata = save_raw_eval_fields(
            adata,
            raw_X=raw_X_eval,
            raw_total_counts=raw_total_counts_eval,
            layer_key="raw_data",
            count_key="raw_total_counts",
        )
    # --------
    # 9. PCA
    # --------
    if compute_pca:
        pca_computed_on = _compute_pca_missing_aware(
            adata,
            missing_indices=missing_indices,
            non_missing_indices=non_missing_indices,
            pca_key=pca_key,
            max_pcs=max_pcs,
        )
    else:
        pca_computed_on = None
    # --------
    # 10. Mask and indices
    # --------
    info = build_and_save_mask(
        adata,
        missing_indices=missing_indices,
            non_missing_indices=non_missing_indices,
        modality=modality,
        mask_key=mask_key,
        data_role=data_role,
        save_modality_mask=save_modality_mask,
    )
    # --------
    # 11. Metadata
    # --------
    adata.uns[f"{modality}_filtered_features"] = filtered_feature_names
    adata.uns[f"{modality}_highly_variable_features"] = hv_feature_names
    adata.uns[f"{modality}_force_kept_features"] = keep_feature_names
    adata.uns[f"{modality}_missing_keep_features"] = missing_keep_feature_names
    adata.uns[f"{modality}_selected_features"] = selected_feature_names
    adata.uns[f"{modality}_log_layer_key"] = log_layer_key if save_log_layer else None
    adata.uns[f"{modality}_pca_key"] = pca_key if compute_pca else None
    adata.uns[f"{modality}_pca_computed_on"] = pca_computed_on
    filter_step = f"filter_features(min_cells={int(min_cells)})" if use_feature_filter else "no_filter"
    preprocess_steps = [
        filter_step,
        "highly_variable_features",
        "normalize_total",
        "log1p",
        "subset_hv_features",
    ]
    if save_log_layer:
        preprocess_steps.append("save_log_layer")
    if save_raw_eval:
        preprocess_steps.append("save_raw_eval")
    if compute_pca:
        preprocess_steps.append("pca")
    adata.uns[f"{modality}_preprocess_order"] = " -> ".join(preprocess_steps)
    info.update(
        {
            "filtered_features": filtered_feature_names,
            "highly_variable_features": hv_feature_names,
            "force_kept_features": keep_feature_names,
            "missing_keep_features": missing_keep_feature_names,
            "selected_features": selected_feature_names,
            "log_layer_key": log_layer_key if save_log_layer else None,
            "pca_key": pca_key if compute_pca else None,
            "pca_computed_on": pca_computed_on,
            "raw_layer_key": "raw_data" if save_raw_eval else None,
            "raw_total_key": "raw_total_counts" if save_raw_eval else None,
            "feature_filter_used": bool(use_feature_filter),
            "min_cells": int(min_cells) if use_feature_filter else None,
            "raw_total_counts_space": (
                "filtered_features" if use_feature_filter else "all_features_before_hv"
            ) if save_raw_eval else None,
            "raw_data_space": (
                "highly_variable_features_plus_force_kept_features"
                if len(keep_feature_names) > 0
                else "highly_variable_features"
            ) if save_raw_eval else None,
        }
    )
    return adata, info
def preprocess_rna(
    adata,
    hvgs: int = 3000,
    min_cells: int = 10,
    target_sum: float = 1e4,
    missing_key: str = "missing",
    observed_label=1,
    log_layer_key: str = "log",
    pca_key: str = "X_pca",
    max_pcs: int = 50,
    compute_pca: bool = True,
    save_log_layer: bool = True,
    save_raw_eval: bool = True,
    data_role: Optional[str] = None,
    mask_key: Optional[str] = None,
    save_modality_mask: bool = False,
    copy: bool = True,
):
    """RNA preprocessing wrapper."""
    return preprocess_rna_like(
        adata=adata,
        modality="rna",
        hv_features=hvgs,
        min_cells=min_cells,
        target_sum=target_sum,
        missing_key=missing_key,
        observed_label=observed_label,
        log_layer_key=log_layer_key,
        pca_key=pca_key,
        max_pcs=max_pcs,
        compute_pca=compute_pca,
        save_log_layer=save_log_layer,
        save_raw_eval=save_raw_eval,
        mask_key=mask_key,
        data_role=data_role,
        save_modality_mask=save_modality_mask,
        copy=copy,
    )
def preprocess_metabolomics(
    adata,
    hvms: int = 3000,
    min_cells: Optional[int] = None,
    target_sum: float = 1e4,
    missing_key: str = "missing",
    observed_label=1,
    log_layer_key: str = "log",
    pca_key: str = "X_pca",
    max_pcs: int = 50,
    compute_pca: bool = True,
    save_log_layer: bool = True,
    save_raw_eval: bool = True,
    data_role: Optional[str] = None,
    mask_key: Optional[str] = None,
    save_modality_mask: bool = False,
    keep_metabolites: Optional[Sequence[str]] = None,
    use_keep_metabolites: bool = False,
    keep_metabolites_case_sensitive: bool = True,
    keep_metabolites_strict: bool = False,
    copy: bool = True,
):
    """
    Metabolomics preprocessing wrapper.
    Logic:
        optional feature filtering
        -> highly variable metabolites
        -> normalize_total
        -> log1p
        -> subset highly variable metabolites
        -> save log layer
        -> PCA
        -> mask / indices
    By default, min_cells=None, so metabolomics does NOT run filter_genes.
    """
    return preprocess_rna_like(
        adata=adata,
        modality="metabolomics",
        hv_features=hvms,
        min_cells=min_cells,
        target_sum=target_sum,
        missing_key=missing_key,
        observed_label=observed_label,
        log_layer_key=log_layer_key,
        pca_key=pca_key,
        max_pcs=max_pcs,
        compute_pca=compute_pca,
        save_log_layer=save_log_layer,
        save_raw_eval=save_raw_eval,
        mask_key=mask_key,
        data_role=data_role,
        save_modality_mask=save_modality_mask,
        # Honor explicit metabolite retention only when enabled.
        keep_features=keep_metabolites if use_keep_metabolites else None,
        keep_features_case_sensitive=keep_metabolites_case_sensitive,
        keep_features_strict=keep_metabolites_strict,
        copy=copy,
    )
# ============================================================
# 3. ADT / Protein preprocessing
# ============================================================
def preprocess_adt(
    adata,
    target_sum: float = 1e4,
    missing_key: str = "missing",
    observed_label=1,
    log_layer_key: str = "log",
    pca_key: str = "X_pca",
    max_pcs: int = 50,
    compute_pca: bool = True,
    save_log_layer: bool = True,
    save_raw_eval: bool = True,
    data_role: Optional[str] = None,
    mask_key: Optional[str] = None,
    save_modality_mask: bool = False,
    copy: bool = True,
):
    """
    ADT / Protein preprocessing.
    Order:
        normalize_total
        -> log1p
        -> save log layer
        -> optional PCA
        -> protein_mask and indices
    No HVG selection is performed by default because protein feature number
    is usually small.
    """
    if copy:
        adata = adata.copy()
    # --------
    # 1. Missing / non-missing indices
    # --------
    missing_indices, non_missing_indices = get_missing_indices(
        adata,
        missing_key=missing_key,
        observed_label=observed_label,
    )
    # --------
    # 1.1 Save raw fields for raw-space evaluation
    # --------
    # ADT/protein keeps all features, so raw_data and raw_total_counts are both
    # computed from the full protein matrix before normalize_total/log1p.
    if save_raw_eval:
        raw_X_eval = adata.X.copy()
        raw_total_counts_eval = _row_sums(raw_X_eval).astype(np.float32)
    # --------
    # 2. Normalize + log1p
    # --------
    normalize_by_missing_status(
        adata,
        target_sum=target_sum,
    )
    # --------
    # 3. Save log layer
    # --------
    if save_log_layer:
        adata.layers[log_layer_key] = adata.X.copy()
    # --------
    # 3.1 Save raw data for raw-space evaluation
    # --------
    if save_raw_eval:
        adata = save_raw_eval_fields(
            adata,
            raw_X=raw_X_eval,
            raw_total_counts=raw_total_counts_eval,
            layer_key="raw_data",
            count_key="raw_total_counts",
        )
    # --------
    # 4. Optional PCA
    # --------
    if compute_pca:
        pca_computed_on = _compute_pca_missing_aware(
            adata,
            missing_indices=missing_indices,
        non_missing_indices=non_missing_indices,
            pca_key=pca_key,
            max_pcs=max_pcs,
        )
    else:
        pca_computed_on = None
    # --------
    # 5. Mask and indices
    # --------
    info = build_and_save_mask(
        adata,
        missing_indices=missing_indices,
        non_missing_indices=non_missing_indices,
        modality="adt",
        mask_key=mask_key,
        data_role=data_role,
        save_modality_mask=save_modality_mask,
    )
    # --------
    # 6. Metadata
    # --------
    adata.uns["adt_log_layer_key"] = log_layer_key if save_log_layer else None
    adata.uns["adt_pca_key"] = pca_key if compute_pca else None
    adata.uns["adt_features"] = adata.var_names.to_numpy()
    adata.uns["adt_pca_computed_on"] = pca_computed_on
    preprocess_steps = ["normalize_total", "log1p"]
    if save_log_layer:
        preprocess_steps.append("save_log_layer")
    if save_raw_eval:
        preprocess_steps.append("save_raw_eval")
    if compute_pca:
        preprocess_steps.append("pca_all_proteins")
    adata.uns["adt_preprocess_order"] = " -> ".join(preprocess_steps)
    info.update(
        {
            "log_layer_key": log_layer_key if save_log_layer else None,
            "pca_key": pca_key if compute_pca else None,
            "pca_computed_on": pca_computed_on,
            "raw_layer_key": "raw_data" if save_raw_eval else None,
            "raw_total_key": "raw_total_counts" if save_raw_eval else None,
            "raw_total_counts_space": "all_proteins" if save_raw_eval else None,
            "raw_data_space": "all_proteins" if save_raw_eval else None,
        }
    )
    return adata, info
# ============================================================
# 4. ATAC / Epigenome preprocessing
# ============================================================
def sparse_log1p_scale(X, scale: float = 1e4):
    """
    Run log1p(X * scale) for sparse or dense matrices.
    """
    if sp.issparse(X):
        X = X.copy()
        X.data = np.log1p(X.data * scale)
        return X
    return np.log1p(X * scale)
class TFIDFTransformer:
    """
    TF-IDF transformer for ATAC count matrix.
    TF:
        peak count / cell total count
    IDF:
        n_cells / peak total count
    """
    def __init__(self):
        self.idf = None
        self.fitted = False
    def fit(self, X):
        self.idf = X.shape[0] / (1e-8 + np.asarray(X.sum(axis=0)).ravel())
        self.fitted = True
        return self
    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("TFIDFTransformer has not been fitted.")
        if sp.issparse(X):
            row_sum = np.asarray(X.sum(axis=1)).ravel()
            tf = X.multiply(1.0 / (1e-8 + row_sum[:, None]))
            return tf.multiply(self.idf)
        row_sum = X.sum(axis=1, keepdims=True)
        tf = X / (1e-8 + row_sum)
        return tf * self.idf
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
class ATACLSITransformer:
    """
    ATAC / epigenome LSI preprocessing.
    Order:
        raw counts
        -> TF-IDF
        -> L1 normalize
        -> log1p(scale=1e4)
        -> TruncatedSVD
        -> drop first component
        -> row-wise z-score
    """
    def __init__(
        self,
        n_components: int = 50,
        drop_first: bool = True,
        tfidf: bool = True,
        norm: bool = True,
        log: bool = True,
        z_score: bool = True,
        random_state: int = 777,
        svd_algorithm: str = "arpack",
    ):
        self.n_components_output = int(n_components)
        self.drop_first = bool(drop_first)
        self.n_components_svd = self.n_components_output + int(self.drop_first)
        self.tfidf = tfidf
        self.norm = norm
        self.log = log
        self.z_score = z_score
        self.tfidf_transformer = TFIDFTransformer()
        self.normalizer = Normalizer(norm="l1")
        self.svd = TruncatedSVD(
            n_components=self.n_components_svd,
            random_state=random_state,
            algorithm=svd_algorithm,
        )
        self.fitted = False
    def _preprocess_before_svd(self, X, fit: bool):
        if self.tfidf:
            if fit:
                X = self.tfidf_transformer.fit_transform(X)
            else:
                X = self.tfidf_transformer.transform(X)
        if self.norm:
            if fit:
                X = self.normalizer.fit_transform(X)
            else:
                X = self.normalizer.transform(X)
        if self.log:
            X = sparse_log1p_scale(X, scale=1e4)
        return X
    def fit(self, X):
        X_pp = self._preprocess_before_svd(X, fit=True)
        self.svd.fit(X_pp)
        self.fitted = True
        return self
    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("ATACLSITransformer has not been fitted.")
        X_pp = self._preprocess_before_svd(X, fit=False)
        X_lsi = self.svd.transform(X_pp).astype("float32")
        # Match SpaMosaic: z-score before dropping the first component.
        if self.z_score and X_lsi.shape[1] > 0:
            X_lsi = X_lsi - X_lsi.mean(axis=1, keepdims=True)
            X_lsi = X_lsi / (1e-8 + X_lsi.std(axis=1, ddof=1, keepdims=True))
        if self.drop_first:
            X_lsi = X_lsi[:, 1:]
        return X_lsi.astype("float32")
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
def preprocess_atac(
    adata,
    n_peak: int = 100000,
    n_comps: int = 50,
    missing_key: str = "missing",
    observed_label=1,
    counts_layer_key: str = "counts",
    lsi_key: str = "X_lsi",
    dimred_key: str = "dimred",
    mask_key: Optional[str] = None,
    data_role: Optional[str] = None,
    save_modality_mask: bool = False,
    use_lsi_as_X: bool = False,
    lsi_var_prefix: str = "LSI",
    copy: bool = True,
):
    """
    ATAC / Epigenome preprocessing for a single AnnData.
    Default behavior:
        adata.X keeps selected peak counts;
        adata.obsm[lsi_key] stores LSI.
    If use_lsi_as_X=True:
        adata.X is replaced by LSI representation;
        adata.var is replaced by LSI dimensions;
        source_mask / target_mask is generated in LSI feature space.
    """
    if copy:
        adata = adata.copy()
    # --------
    # 1. Save raw counts
    # --------
    if counts_layer_key not in adata.layers:
        adata.layers[counts_layer_key] = adata.X.copy()
    # --------
    # 2. Missing / non-missing indices
    # --------
    missing_indices, non_missing_indices = get_missing_indices(
        adata,
        missing_key=missing_key,
        observed_label=observed_label,
    )
    has_missing = len(missing_indices) > 0
    if len(non_missing_indices) < 2:
        raise ValueError(
            f"Only {len(non_missing_indices)} non-missing ATAC cells are available. "
            "At least 2 cells are required for ATAC LSI."
        )
    # --------
    # 3. Reference for peak selection
    # --------
    adata_ref = adata[non_missing_indices, :].copy() if has_missing else adata.copy()
    if adata_ref.n_vars == 0:
        raise ValueError("No peaks found in the ATAC count matrix.")
    n_top_peaks = int(n_peak)
    sc.pp.highly_variable_genes(
        adata_ref,
        flavor="seurat_v3",
        n_top_genes=n_top_peaks,
    )
    peak_names = adata_ref.var_names[adata_ref.var["highly_variable"]].to_numpy()
    if len(peak_names) == 0:
        raise ValueError("No highly variable peaks were selected.")
    # --------
    # 4. Subset selected peaks
    # --------
    adata = adata[:, peak_names].copy()
    adata.var["highly_variable"] = True
    X_counts = adata.layers[counts_layer_key]
    # --------
    # 5. LSI
    # --------
    # Match SpaMosaic: do not automatically truncate n_comps.
    # ATACLSITransformer will internally compute n_comps + 1 components when drop_first=True.
    n_lsi = int(n_comps)
    transformer = ATACLSITransformer(
        n_components=n_lsi,
        drop_first=True,
        tfidf=True,
        norm=True,
        log=True,
        z_score=True,
        random_state=777,
        svd_algorithm="arpack",
    )
    if has_missing:
        X_ref = X_counts[non_missing_indices, :]
        X_lsi_ref = transformer.fit_transform(X_ref)
        X_lsi_full = np.zeros((adata.n_obs, X_lsi_ref.shape[1]), dtype=np.float32)
        X_lsi_full[non_missing_indices, :] = X_lsi_ref
        adata.obsm[lsi_key] = X_lsi_full
    else:
        adata.obsm[lsi_key] = transformer.fit_transform(X_counts)
    adata.obsm[dimred_key] = adata.obsm[lsi_key]
    adata.uns["atac_lsi_transformer"] = transformer
    # --------
    # 6. Optional: use LSI as model input X
    # --------
    if use_lsi_as_X:
        X_lsi = np.asarray(adata.obsm[lsi_key], dtype=np.float32)
        if X_lsi.ndim != 2:
            raise ValueError(f"adata.obsm['{lsi_key}'] must be 2D, got shape {X_lsi.shape}.")
        if X_lsi.shape[1] == 0:
            raise ValueError("LSI has 0 dimensions. Cannot use LSI as adata.X.")
        # Preserve important obsm entries such as spatial coordinates.
        old_obsm = {}
        for k in adata.obsm.keys():
            try:
                old_obsm[k] = adata.obsm[k].copy()
            except Exception:
                old_obsm[k] = adata.obsm[k]
        old_uns = adata.uns.copy()
        lsi_var_names = [f"{lsi_var_prefix}_{i + 1}" for i in range(X_lsi.shape[1])]
        lsi_var = pd.DataFrame(index=lsi_var_names)
        lsi_var["highly_variable"] = True
        adata_lsi = sc.AnnData(
            X=X_lsi.copy(),
            obs=adata.obs.copy(),
            var=lsi_var,
        )
        for k, v in old_obsm.items():
            adata_lsi.obsm[k] = v
        adata_lsi.obsm[lsi_key] = X_lsi.copy()
        adata_lsi.obsm[dimred_key] = X_lsi.copy()
        adata_lsi.uns = old_uns
        adata_lsi.uns["atac_model_input"] = "X_lsi"
        adata_lsi.uns["atac_use_lsi_as_X"] = True
        adata_lsi.uns["atac_original_selected_peaks"] = peak_names
        adata_lsi.uns["atac_original_n_selected_peaks"] = len(peak_names)
        # Important:
        # after replacing X by LSI, old peak-count layers are not kept,
        # because layers must have the same shape as adata.X.
        adata_lsi.uns["atac_counts_layer_key"] = None
        adata_lsi.uns["atac_selected_peak_counts_stored"] = False
        adata = adata_lsi
    else:
        adata.uns["atac_model_input"] = "selected_peaks"
        adata.uns["atac_use_lsi_as_X"] = False
        adata.uns["atac_selected_peak_counts_stored"] = True
    # --------
    # 7. Mask and indices
    # --------
    # All PRISM experiments use whole-row modality missingness, so the saved
    # mask remains one boolean value per cell regardless of the representation.
    info = build_and_save_mask(
        adata,
        missing_indices=missing_indices,
        non_missing_indices=non_missing_indices,
        modality="atac",
        mask_key=mask_key,
        data_role=data_role,
        save_modality_mask=save_modality_mask,
    )
    # --------
    # 8. Metadata
    # --------
    adata.uns["atac_selected_peaks"] = peak_names
    adata.uns["atac_n_selected_peaks"] = len(peak_names)
    adata.uns["atac_lsi_key"] = lsi_key
    adata.uns["atac_dimred_key"] = dimred_key
    adata.uns["atac_lsi_dim"] = int(adata.obsm[lsi_key].shape[1])
    adata.uns["atac_lsi_computed_on"] = "non_missing_cells_only" if has_missing else "all_cells"
    adata.uns["atac_preprocess_order"] = (
        "save_counts -> select_variable_peaks -> "
        "TFIDF -> L1_normalize -> log1p_scale -> "
        "LSI/SVD -> drop_first -> z_score"
    )
    info.update(
        {
            "n_selected_peaks": len(peak_names),
            "selected_peaks": peak_names,
            "lsi_key": lsi_key,
            "dimred_key": dimred_key,
            "lsi_dim": int(adata.obsm[lsi_key].shape[1]),
            "model_input": adata.uns["atac_model_input"],
            "use_lsi_as_X": bool(use_lsi_as_X),
        }
    )
    return adata, info
# ============================================================
# 5. Unified interface
# ============================================================
def preprocess_omics(
    adata,
    modality: str,
    **kwargs,
):
    """
    Unified preprocessing interface.
    Supported modalities:
        RNA
        ADT / PROTEIN
        ATAC / EPIGENOME / EPI
        METABOLOMICS / METABOLITE / METABOLITES / MET
    """
    modality_upper = modality.upper()
    if modality_upper == "RNA":
        return preprocess_rna(adata, **kwargs)
    if modality_upper in ["ADT", "PROTEIN"]:
        return preprocess_adt(adata, **kwargs)
    if modality_upper in ["ATAC", "EPIGENOME", "EPI"]:
        return preprocess_atac(adata, **kwargs)
    if modality_upper in ["METABOLOMICS", "METABOLITE", "METABOLITES", "MET"]:
        return preprocess_metabolomics(adata, **kwargs)
    raise ValueError(
        f"Unsupported modality: {modality}. "
        "Supported modalities are: RNA, ADT / PROTEIN, "
        "ATAC / EPIGENOME / EPI, METABOLOMICS / METABOLITE."
    )
