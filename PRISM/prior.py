"""Similarity priors, COVET, and AOT construction for PRISM."""
import os
import numpy as np
import scipy.sparse as sp
import torch
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import scipy.sparse
import scanpy as sc
import sklearn.neighbors
from tqdm import tqdm
# ============================================================
# 1. Single-modality missingness prior
# ============================================================
def prepare_similarity_subset(distance_matrix, non_missing_indices, k_top, device=None, verbose=True):
    """
    Prepare top-k observed target-modality neighbors for each cell.
    Parameters
    ----------
    distance_matrix:
        scipy sparse CSR matrix, torch tensor, or numpy array.
        Sparse mode is recommended for large datasets.
        Only stored non-zero entries are treated as valid distances.
    non_missing_indices:
        Global cell indices with observed target modality.
    k_top:
        Number of observed target-modality neighbors for each cell.
    device:
        Device used to store returned torch tensors.
    Return
    ------
    top_k_indices:
        Local indices inside non_missing_indices, shape [N, k_top].
    non_missing_indices:
        Global observed-cell indices as torch tensor.
    Notes
    -----
    Sparse matrix logic:
        1. only stored distances are considered valid;
        2. explicit zero values are removed;
        3. self-neighbor is always excluded;
        4. neighbors are sorted by distance in ascending order;
        5. if any row has fewer than k_top valid observed neighbors, raise an error.
    """
    import numpy as np
    import scipy.sparse as sp
    import torch
    non_missing_np = np.asarray(non_missing_indices, dtype=np.int64)
    if non_missing_np.size == 0:
        raise ValueError("No non-missing target-modality cells found.")
    if device is None:
        device = distance_matrix.device if isinstance(distance_matrix, torch.Tensor) else torch.device("cpu")
    k_top = int(k_top)
    # =========================================================
    # 1. Sparse CSR mode for large-scale distance matrix
    # =========================================================
    if sp.issparse(distance_matrix):
        D = distance_matrix.tocsr(copy=True)
        # In your sparse prior, zero means missing/empty or self-distance.
        # Remove explicit zeros; implicit zeros are never accessed anyway.
        D.eliminate_zeros()
        if D.shape[0] != D.shape[1]:
            raise ValueError(f"distance_matrix must be square, got {D.shape}.")
        n_obs = D.shape[0]
        if np.any(non_missing_np < 0) or np.any(non_missing_np >= n_obs):
            raise ValueError("non_missing_indices contains out-of-range cell indices.")
        non_missing_mask = np.zeros(n_obs, dtype=bool)
        non_missing_mask[non_missing_np] = True
        global_to_local = np.full(n_obs, -1, dtype=np.int64)
        global_to_local[non_missing_np] = np.arange(non_missing_np.size, dtype=np.int64)
        topk_local = np.full((n_obs, k_top), -1, dtype=np.int64)
        valid_counts = np.zeros(n_obs, dtype=np.int32)
        iterator = range(n_obs)
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Build similarity prior")
            except Exception:
                pass
        for i in iterator:
            start, end = D.indptr[i], D.indptr[i + 1]
            cols = D.indices[start:end]
            vals = D.data[start:end]
            # Valid neighbors must:
            # 1. be target-observed cells;
            # 2. not be itself;
            # 3. have finite distance.
            keep = non_missing_mask[cols] & (cols != i) & np.isfinite(vals)
            cols_keep = cols[keep]
            vals_keep = vals[keep]
            valid_counts[i] = len(cols_keep)
            if len(cols_keep) < k_top:
                continue
            # Sort distances from small to large and select top-k.
            order = np.argsort(vals_keep, kind="mergesort")[:k_top]
            selected_global = cols_keep[order]
            topk_local[i] = global_to_local[selected_global]
        bad_rows = np.where(topk_local.min(axis=1) < 0)[0]
        if bad_rows.size > 0:
            example = bad_rows[:10]
            raise ValueError(
                f"{bad_rows.size} rows have fewer than k_top={k_top} valid observed neighbors. "
                f"Example rows: {example.tolist()}. "
                f"Valid neighbor counts: {valid_counts[example].tolist()}. "
                "Please increase the number of saved neighbors in the prior matrix, "
                "reduce k_top, or check whether too many stored neighbors are target-missing cells."
            )
        top_k_indices = torch.as_tensor(topk_local, device=device, dtype=torch.long)
        non_missing_t = torch.as_tensor(non_missing_np, device=device, dtype=torch.long)
        if verbose:
            print(
                "Similarity prior ready: "
                f"{n_obs} cells, {k_top} observed neighbors per cell."
            )
        return top_k_indices, non_missing_t
    # =========================================================
    # 2. Dense mode for small datasets
    # =========================================================
    if not isinstance(distance_matrix, torch.Tensor):
        distance_matrix = torch.as_tensor(distance_matrix, device=device, dtype=torch.float32)
    else:
        distance_matrix = distance_matrix.to(device=device, dtype=torch.float32)
    non_missing_t = torch.as_tensor(non_missing_np, device=device, dtype=torch.long)
    valid_distances = distance_matrix[:, non_missing_t]
    all_cell_indices = torch.arange(distance_matrix.size(0), device=device, dtype=torch.long).unsqueeze(1)
    self_mask = all_cell_indices == non_missing_t.unsqueeze(0)
    valid_distances = valid_distances.masked_fill(self_mask, float("inf"))
    k_top_eff = min(k_top, valid_distances.size(1))
    top_k_indices = torch.argsort(valid_distances, dim=1)[:, :k_top_eff]
    return top_k_indices, non_missing_t
# ============================================================
# 2. Dual-modality partial missingness prior
# ============================================================
def sparse_topk_global(matrix, candidate_indices, k_top=5, exclude_self=True):
    """
    For each row, find top-k nearest neighbors from candidate_indices in a CSR distance matrix.
    Return global neighbor indices.
    """
    matrix = matrix.tocsr()
    n = matrix.shape[0]
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_mask = np.zeros(n, dtype=bool)
    candidate_mask[candidate_indices] = True
    topk_global = np.full((n, k_top), -1, dtype=np.int64)
    topk_dist = np.full((n, k_top), np.inf, dtype=np.float32)
    for i in range(n):
        start, end = matrix.indptr[i], matrix.indptr[i + 1]
        cols, vals = matrix.indices[start:end], matrix.data[start:end]
        if len(cols) == 0:
            continue
        keep = candidate_mask[cols]
        if exclude_self:
            keep = keep & (cols != i)
        cols_keep, vals_keep = cols[keep], vals[keep]
        if len(cols_keep) == 0:
            continue
        order = np.argsort(vals_keep)[:k_top]
        selected_cols, selected_vals = cols_keep[order], vals_keep[order]
        m = len(selected_cols)
        topk_global[i, :m] = selected_cols
        topk_dist[i, :m] = selected_vals.astype(np.float32)
    return topk_global, topk_dist
def save_aot_matrix(adata, output_path, key="aot_distances", verbose=True):
    """Save sparse AOT distance matrix from adata.obsp[key]."""
    if key not in adata.obsp:
        raise KeyError(f"adata.obsp['{key}'] not found.")
    aot_dist = adata.obsp[key]
    if not sp.isspmatrix(aot_dist):
        raise TypeError(f"adata.obsp['{key}'] must be a scipy sparse matrix.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sp.save_npz(output_path, aot_dist)
    if verbose:
        loaded = sp.load_npz(output_path)
        print(f"Saved AOT prior: {output_path}")
        print("  shape:", loaded.shape, "nnz:", loaded.nnz)
        print("  first 5x5 block:")
        print(loaded[:5, :5].toarray())
    return output_path
def compute_similarity_prior(
    adata_source,
    adata_target,
    prior_path,
    *,
    device="cuda:0",
    k_top=5,
    covet_k_spatial=6,
    covet_gene_num=64,
    covet_use_layer="log",
    covet_use_obsm=None,
    aot_k_env=None,
    aot_chunk_size=None,
    spatial_key="spatial",
    missing_key="missing",
    batch_key=None,
    store_neighbor_index=False,
    evaluate_prior=True,
    compute_structure_metrics=False,
    verbose=True,
    show_prior_metrics=False,
):
    """Load or compute a similarity prior and return its quality metrics.
    ``covet_use_layer`` and ``covet_use_obsm`` select the source
    representation used by COVET.  RNA-based tutorials keep the default log
    layer, while ATAC-based tutorials can use ``X_lsi`` without changing the
    behavior of existing callers. ``aot_k_env`` optionally limits the number
    of AOT environment neighbors retained for each reference cell.
    Set ``aot_chunk_size`` only for datasets that need memory-bounded AOT
    construction; otherwise AOT is computed in one operation. Set
    Set ``show_prior_metrics=False`` to skip prior-metric computation entirely.
    ``evaluate_prior=False`` also skips this diagnostic for backward
    compatibility. ``compute_structure_metrics`` is applied only when prior
    metrics are requested.
    """
    if os.path.exists(prior_path):
        distance_matrix = sp.load_npz(prior_path).tocsr()
        if verbose:
            print("Similarity prior loaded.")
    else:
        batch_key_eff = (
            batch_key
            if batch_key is not None
            else ("batch" if "batch" in adata_source.obs else -1)
        )
        if verbose:
            print("=" * 30)
            print("Step 1: src COVET")
            print("=" * 30)
        adata_source, adata_covet_ref, _ = compute_covet(
            adata_source,
            k_spatial=covet_k_spatial,
            covet_gene_num=covet_gene_num,
            spatial_key=spatial_key,
            batch_key=batch_key_eff,
            use_layer=covet_use_layer,
            use_obsm=covet_use_obsm,
            missing_key=missing_key,
            copy_when_complete=True,
            verbose=verbose,
        )
        if verbose:
            print("\n" + "=" * 30)
            print("Step 2: src AOT")
            print("=" * 30)
        adata_source, adata_covet_ref, _ = compute_aot(
            adata_source,
            adata_covet_ref,
            k_env=aot_k_env,
            device=device,
            use_chunking=aot_chunk_size is not None,
            chunk_size=4096 if aot_chunk_size is None else aot_chunk_size,
            log_every=4096 if aot_chunk_size is None else aot_chunk_size,
            store_neighbor_index=store_neighbor_index,
            verbose=verbose,
        )
        distance_matrix = adata_source.obsp["aot_distances"].tocsr(copy=True)
        os.makedirs(os.path.dirname(os.path.abspath(prior_path)), exist_ok=True)
        sp.save_npz(prior_path, distance_matrix)
        del adata_source.obsp["aot_distances"]
        for key in ("aot_neighbor_indices", "aot_neighbor_distances"):
            adata_source.uns.pop(key, None)
        del adata_covet_ref
    distance_matrix.eliminate_zeros()
    expected_shape = (adata_source.n_obs, adata_source.n_obs)
    if distance_matrix.shape != expected_shape:
        raise ValueError(
            f"Prior shape {distance_matrix.shape} does not match {expected_shape}."
        )
    if not evaluate_prior or not show_prior_metrics:
        return distance_matrix, {}
    from .validation import compute_metrics_each_pair
    rmse, pcc, spcc, cmd, ssim, _ = compute_metrics_each_pair(
        adata_target,
        distance_matrix,
        top_n=k_top,
        compute_structure_metrics=compute_structure_metrics,
    )
    # RMSE is retained as the baseline error in every prior-metric result;
    # CMD and SSIM are appended only when structural metrics are enabled.
    prior_metrics = {
        "RMSE": float(np.nanmean(rmse)),
        "PCC": float(np.nanmean(pcc)),
        "SPCC": float(np.nanmean(spcc)),
    }
    if compute_structure_metrics:
        prior_metrics.update(
            CMD=float(np.nanmean(cmd)),
            SSIM=float(np.nanmean(ssim)),
        )
    if show_prior_metrics:
        metric_text = ", ".join(
            f"{name}={prior_metrics[name]:.4f}"
            for name in ("PCC", "SPCC")
            if name in prior_metrics
        )
        print(f"Similarity prior metrics (PCC/SPCC): {metric_text}")
    return distance_matrix, prior_metrics
def _get_obs_missing_masks(adata_source, adata_target, missing_key="missing", observed_label="1"):
    src_obs = adata_source.obs[missing_key].astype(str).to_numpy() == str(observed_label)
    tgt_obs = adata_target.obs[missing_key].astype(str).to_numpy() == str(observed_label)
    src_missing = ~src_obs
    tgt_missing = ~tgt_obs
    pair_mask = src_obs & tgt_obs
    src_only_mask = src_obs & tgt_missing
    tgt_only_mask = src_missing & tgt_obs
    both_missing_mask = src_missing & tgt_missing
    return {
        "src_obs": src_obs,
        "tgt_obs": tgt_obs,
        "src_missing": src_missing,
        "tgt_missing": tgt_missing,
        "pair_mask": pair_mask,
        "src_only_mask": src_only_mask,
        "tgt_only_mask": tgt_only_mask,
        "both_missing_mask": both_missing_mask,
        "pair_indices": np.flatnonzero(pair_mask),
        "src_only_indices": np.flatnonzero(src_only_mask),
        "tgt_only_indices": np.flatnonzero(tgt_only_mask),
        "both_missing_indices": np.flatnonzero(both_missing_mask),
        "src_obs_indices": np.flatnonzero(src_obs),
        "tgt_obs_indices": np.flatnonzero(tgt_obs),
    }
def _check_candidate_pool_size(mask_info, k_top):
    if len(mask_info["pair_indices"]) <= k_top:
        raise ValueError("Pair region is too small for the requested top-k after excluding self.")
    if len(mask_info["src_obs_indices"]) <= k_top:
        raise ValueError("Source-observed region is too small for the requested top-k after excluding self.")
    if len(mask_info["tgt_obs_indices"]) <= k_top:
        raise ValueError("Target-observed region is too small for the requested top-k after excluding self.")
def _validate_context(context_src, context_tgt, mask_info, n_obs):
    src_obs = mask_info["src_obs"]
    tgt_obs = mask_info["tgt_obs"]
    src_missing = mask_info["src_missing"]
    tgt_missing = mask_info["tgt_missing"]
    pair_mask = mask_info["pair_mask"]
    if np.any(context_src < 0):
        raise ValueError("context_topk_src_global contains -1. Please rebuild prior with enough valid neighbors.")
    if np.any(context_tgt < 0):
        raise ValueError("context_topk_tgt_global contains -1. Please rebuild prior with enough valid neighbors.")
    if np.any(context_src >= n_obs) or np.any(context_tgt >= n_obs):
        raise ValueError("context top-k contains indices out of range.")
    row_idx = np.arange(n_obs)[:, None]
    if np.any(context_src == row_idx):
        raise ValueError("context_topk_src_global contains self-neighbor leakage.")
    if np.any(context_tgt == row_idx):
        raise ValueError("context_topk_tgt_global contains self-neighbor leakage.")
    if not np.all(src_obs[context_src[src_obs]]):
        raise ValueError("For source-observed cells, source context contains source-missing neighbors.")
    if not np.all(pair_mask[context_src[src_missing]]):
        raise ValueError("For source-missing cells, source context must come from pair cells.")
    if not np.all(tgt_obs[context_tgt[tgt_obs]]):
        raise ValueError("For target-observed cells, target context contains target-missing neighbors.")
    if not np.all(pair_mask[context_tgt[tgt_missing]]):
        raise ValueError("For target-missing cells, target context must come from pair cells.")
def _print_region_summary(mask_info):
    print("pair:", len(mask_info["pair_indices"]))
    print("src_only:", len(mask_info["src_only_indices"]))
    print("tgt_only:", len(mask_info["tgt_only_indices"]))
    print("both_missing:", len(mask_info["both_missing_indices"]))
    print("src observed:", len(mask_info["src_obs_indices"]))
    print("tgt observed:", len(mask_info["tgt_obs_indices"]))
def _check_example_rows(adata_source, mask_info, context_src, context_tgt,
                        context_src_dist, context_tgt_dist,
                        context_src_source, context_tgt_source,
                        context_src_pool, context_tgt_pool):
    example_indices = []
    if len(mask_info["pair_indices"]) > 0:
        example_indices.append(mask_info["pair_indices"][0])
    if len(mask_info["src_only_indices"]) > 0:
        example_indices.append(mask_info["src_only_indices"][0])
    if len(mask_info["tgt_only_indices"]) > 0:
        example_indices.append(mask_info["tgt_only_indices"][0])
    for i in example_indices:
        print("\n===== check row", i, adata_source.obs_names[i], "=====")
        print("src observed:", bool(mask_info["src_obs"][i]))
        print("tgt observed:", bool(mask_info["tgt_obs"][i]))
        print("src self missing:", bool(mask_info["src_missing"][i]))
        print("tgt self missing:", bool(mask_info["tgt_missing"][i]))
        print("\nsource context:")
        print("  prior source:", context_src_source[i])
        print("  candidate pool:", context_src_pool[i])
        print("  top-k global:", context_src[i])
        print("  top-k dist:", context_src_dist[i])
        print("  neighbor obs_names:", list(adata_source.obs_names[context_src[i]]))
        print("\ntarget context:")
        print("  prior source:", context_tgt_source[i])
        print("  candidate pool:", context_tgt_pool[i])
        print("  top-k global:", context_tgt[i])
        print("  top-k dist:", context_tgt_dist[i])
        print("  neighbor obs_names:", list(adata_source.obs_names[context_tgt[i]]))
def build_modality_context_prior(
    adata_source,
    adata_target,
    source_distance,
    target_distance,
    k_top=10,
    missing_key="missing",
    observed_label="1",
    allow_both_missing=False,
    verbose=True,
    check_examples=True,
):
    """
    Build an in-memory modality-specific context prior for PRISM-DM.
    Source context:
        source observed -> use source prior, search from source-observed cells.
        source missing  -> use target prior, search from pair cells.
    Target context:
        target observed -> use target prior, search from target-observed cells.
        target missing  -> use source prior, search from pair cells.
    """
    if adata_source.n_obs != adata_target.n_obs:
        raise ValueError("adata_source and adata_target must have the same number of cells.")
    if not np.array_equal(adata_source.obs_names, adata_target.obs_names):
        raise ValueError("Source and target obs_names must have the same order.")
    n_obs = adata_source.n_obs
    prior_src = source_distance.tocsr(copy=True)
    prior_tgt = target_distance.tocsr(copy=True)
    if prior_src.shape != (n_obs, n_obs) or prior_tgt.shape != (n_obs, n_obs):
        raise ValueError("source_distance and target_distance must each have shape [n_obs, n_obs].")
    prior_src.sort_indices()
    prior_tgt.sort_indices()
    mask_info = _get_obs_missing_masks(adata_source, adata_target, missing_key, observed_label)
    if verbose:
        print("source prior:", prior_src.shape, "nnz:", prior_src.nnz)
        print("target prior:", prior_tgt.shape, "nnz:", prior_tgt.nnz)
        _print_region_summary(mask_info)
    if len(mask_info["both_missing_indices"]) > 0 and not allow_both_missing:
        raise ValueError(
            "both_missing cells exist. Current design requires at least one observed modality "
            "to select cross-modality context. Please remove or avoid both-missing cells."
        )
    _check_candidate_pool_size(mask_info, k_top)
    topk_cache = {}
    def get_prior_matrix(prior_name):
        if prior_name == "src":
            return prior_src
        if prior_name == "tgt":
            return prior_tgt
        raise ValueError("prior_name must be 'src' or 'tgt'.")
    def get_topk(prior_name, pool_name, candidate_indices):
        key = (prior_name, pool_name)
        if key not in topk_cache:
            topk_global, topk_dist = sparse_topk_global(
                get_prior_matrix(prior_name),
                candidate_indices=candidate_indices,
                k_top=k_top,
                exclude_self=True,
            )
            topk_cache[key] = {"topk_global": topk_global, "topk_dist": topk_dist}
        return topk_cache[key]
    context_src = np.full((n_obs, k_top), -1, dtype=np.int64)
    context_tgt = np.full((n_obs, k_top), -1, dtype=np.int64)
    context_src_dist = np.full((n_obs, k_top), np.inf, dtype=np.float32)
    context_tgt_dist = np.full((n_obs, k_top), np.inf, dtype=np.float32)
    context_src_source = np.array(["none"] * n_obs, dtype=object)
    context_tgt_source = np.array(["none"] * n_obs, dtype=object)
    context_src_pool = np.array(["none"] * n_obs, dtype=object)
    context_tgt_pool = np.array(["none"] * n_obs, dtype=object)
    src_obs = mask_info["src_obs"]
    tgt_obs = mask_info["tgt_obs"]
    for i in range(n_obs):
        if src_obs[i]:
            item = get_topk("src", "src_obs", mask_info["src_obs_indices"])
            context_src_source[i], context_src_pool[i] = "src", "src_obs"
        else:
            item = get_topk("tgt", "pair", mask_info["pair_indices"])
            context_src_source[i], context_src_pool[i] = "tgt", "pair"
        context_src[i] = item["topk_global"][i]
        context_src_dist[i] = item["topk_dist"][i]
        if tgt_obs[i]:
            item = get_topk("tgt", "tgt_obs", mask_info["tgt_obs_indices"])
            context_tgt_source[i], context_tgt_pool[i] = "tgt", "tgt_obs"
        else:
            item = get_topk("src", "pair", mask_info["pair_indices"])
            context_tgt_source[i], context_tgt_pool[i] = "src", "pair"
        context_tgt[i] = item["topk_global"][i]
        context_tgt_dist[i] = item["topk_dist"][i]
    _validate_context(context_src, context_tgt, mask_info, n_obs)
    if verbose:
        print("source context complete top-k:", int((context_src >= 0).all(axis=1).sum()), "/", n_obs)
        print("target context complete top-k:", int((context_tgt >= 0).all(axis=1).sum()), "/", n_obs)
        print("source context prior source counts:", dict(zip(*np.unique(context_src_source, return_counts=True))))
        print("source context candidate pool counts:", dict(zip(*np.unique(context_src_pool, return_counts=True))))
        print("target context prior source counts:", dict(zip(*np.unique(context_tgt_source, return_counts=True))))
        print("target context candidate pool counts:", dict(zip(*np.unique(context_tgt_pool, return_counts=True))))
    if check_examples:
        _check_example_rows(
            adata_source, mask_info,
            context_src, context_tgt,
            context_src_dist, context_tgt_dist,
            context_src_source, context_tgt_source,
            context_src_pool, context_tgt_pool,
        )
    result = dict(mask_info)
    result.update({
        "context_topk_src_global": context_src,
        "context_topk_tgt_global": context_tgt,
        "context_topk_src_dist": context_src_dist,
        "context_topk_tgt_dist": context_tgt_dist,
        "context_topk_src_source": context_src_source,
        "context_topk_tgt_source": context_tgt_source,
        "context_topk_src_pool": context_src_pool,
        "context_topk_tgt_pool": context_tgt_pool,
    })
    return result
def validate_modality_context_prior(context_prior, n_obs, k_top):
    """
    Validate an in-memory PRISM-DM context prior.
    """
    required_keys = [
        "context_topk_src_global", "context_topk_tgt_global",
        "src_missing", "tgt_missing",
        "src_obs", "tgt_obs", "pair_mask",
    ]
    for key in required_keys:
        if key not in context_prior:
            raise KeyError(f"context_prior must contain '{key}'.")
    context_src_all = np.asarray(context_prior["context_topk_src_global"], dtype=np.int64)
    context_tgt_all = np.asarray(context_prior["context_topk_tgt_global"], dtype=np.int64)
    src_missing = np.asarray(context_prior["src_missing"], dtype=bool)
    tgt_missing = np.asarray(context_prior["tgt_missing"], dtype=bool)
    src_obs = np.asarray(context_prior["src_obs"], dtype=bool)
    tgt_obs = np.asarray(context_prior["tgt_obs"], dtype=bool)
    pair_mask = np.asarray(context_prior["pair_mask"], dtype=bool)
    if context_src_all.shape[0] != n_obs or context_tgt_all.shape[0] != n_obs:
        raise ValueError("context_topk_src_global / context_topk_tgt_global rows must match n_obs.")
    if src_missing.shape[0] != n_obs or tgt_missing.shape[0] != n_obs:
        raise ValueError("src_missing / tgt_missing length must match n_obs.")
    if context_src_all.shape[1] < k_top or context_tgt_all.shape[1] < k_top:
        raise ValueError("Prepared context top-k has fewer columns than requested k_top.")
    context_src = context_src_all[:, :k_top]
    context_tgt = context_tgt_all[:, :k_top]
    both_missing = src_missing & tgt_missing
    if np.any(both_missing):
        raise ValueError("both-missing cells exist. Current context logic requires at least one observed modality.")
    row_idx = np.arange(n_obs)[:, None]
    for name, context in [("context_topk_src_global", context_src), ("context_topk_tgt_global", context_tgt)]:
        if np.any(context < 0):
            raise ValueError(f"{name} contains -1. Please rebuild prior with enough valid neighbors.")
        if np.any(context >= n_obs):
            raise ValueError(f"{name} contains indices out of range.")
        if np.any(context == row_idx):
            raise ValueError(f"{name} contains self-neighbor leakage.")
    for i in range(n_obs):
        if src_obs[i] and not np.all(src_obs[context_src[i]]):
            raise ValueError(f"Row {i}: source observed, but source context contains source-missing neighbors.")
        if (not src_obs[i]) and not np.all(pair_mask[context_src[i]]):
            raise ValueError(f"Row {i}: source missing, but source context is not from pair cells.")
        if tgt_obs[i] and not np.all(tgt_obs[context_tgt[i]]):
            raise ValueError(f"Row {i}: target observed, but target context contains target-missing neighbors.")
        if (not tgt_obs[i]) and not np.all(pair_mask[context_tgt[i]]):
            raise ValueError(f"Row {i}: target missing, but target context is not from pair cells.")
    return context_src, context_tgt, src_missing, tgt_missing
def _to_dense(X):
    """Convert sparse matrix to dense ndarray if needed."""
    return X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)
def batch_knn(data, batch, k):
    """Batch-wise spatial kNN, matching scenvi.utils.batch_knn behavior."""
    kNNGraphIndex = np.zeros(shape=(data.shape[0], k))
    for val in np.unique(batch):
        val_ind = np.where(batch == val)[0]
        batch_knn_graph = sklearn.neighbors.kneighbors_graph(
            data[val_ind],
            n_neighbors=k,
            mode="connectivity",
            n_jobs=-1,
        ).tocoo()
        batch_knn_ind = np.reshape(
            np.asarray(batch_knn_graph.col),
            [data[val_ind].shape[0], k],
        )
        kNNGraphIndex[val_ind] = val_ind[batch_knn_ind]
    return kNNGraphIndex.astype("int")
def calculate_covariance_matrices(
    spatial_data,
    kNN,
    exp_data,
    spatial_key="spatial",
    batch_key=-1,
    batch_size=None,
    verbose=True,
):
    """
    Calculate shifted covariance matrices using scenvi-style COVET logic.
    Notes
    -----
    1. batch_key controls spatial kNN only.
    2. global_mean is computed over all cells in exp_data.
    3. batch_size controls memory chunking only.
    4. denominator is kNN - 1.
    5. diagonal regularization is added after all cells/chunks are computed.
    """
    if batch_key == -1:
        kNNGraph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key],
            n_neighbors=kNN,
            mode="connectivity",
            n_jobs=-1,
        ).tocoo()
        kNNGraphIndex = np.reshape(
            np.asarray(kNNGraph.col),
            [spatial_data.obsm[spatial_key].shape[0], kNN],
        )
    else:
        kNNGraphIndex = batch_knn(
            spatial_data.obsm[spatial_key],
            spatial_data.obs[batch_key],
            kNN,
        )
    global_mean = exp_data.mean(axis=0)
    n_cells = exp_data.shape[0]
    n_features = exp_data.shape[1]
    if batch_size is None or batch_size >= n_cells:
        if verbose:
            print("Calculating covariance matrices for all cells/spots")
        DistanceMatWeighted = (
            global_mean[None, None, :] - exp_data[kNNGraphIndex[np.arange(n_cells)]]
        )
        CovMats = np.matmul(
            DistanceMatWeighted.transpose([0, 2, 1]),
            DistanceMatWeighted,
        ) / (kNN - 1)
    else:
        CovMats = np.zeros((n_cells, n_features, n_features))
        batch_indices = np.array_split(
            np.arange(n_cells),
            np.ceil(n_cells / batch_size),
        )
        for batch_idx in tqdm(
            batch_indices,
            desc="Calculating covariance matrices",
            disable=not verbose,
        ):
            batch_neighbors = kNNGraphIndex[batch_idx]
            batch_distances = global_mean[None, None, :] - exp_data[batch_neighbors]
            batch_covs = np.matmul(
                batch_distances.transpose([0, 2, 1]),
                batch_distances,
            ) / (kNN - 1)
            CovMats[batch_idx] = batch_covs
    reg_term = CovMats.mean() * 0.00001
    identity = np.eye(n_features)[None, :, :]
    CovMats = CovMats + reg_term * identity
    return CovMats
def batch_matrix_sqrt(Mats):
    """Batched symmetric PSD matrix square root."""
    e, v = np.linalg.eigh(Mats)
    e = np.where(e < 0, 0, e)
    e = np.sqrt(e)
    m, n = e.shape
    diag_e = np.zeros((m, n, n), dtype=e.dtype)
    diag_e.reshape(-1, n**2)[..., :: n + 1] = e
    return np.matmul(np.matmul(v, diag_e), v.transpose([0, 2, 1]))
def _select_covet_expression(
    spatial_data,
    g=64,
    genes: Optional[Sequence[str]] = None,
    use_obsm: Optional[str] = None,
    use_layer: Optional[str] = None,
    verbose: bool = True,
):
    """Select expression/features for COVET."""
    genes = [] if genes is None else list(genes)
    if use_obsm is not None:
        if use_obsm not in spatial_data.obsm:
            raise ValueError(f"obsm key {use_obsm!r} not found in spatial_data.obsm")
        if verbose:
            print(
                f"Computing COVET using obsm {use_obsm!r} with "
                f"{spatial_data.obsm[use_obsm].shape[1]} dimensions"
            )
        CovGenes = [
            f"{use_obsm}_{i}" for i in range(spatial_data.obsm[use_obsm].shape[1])
        ]
        exp_data = spatial_data.obsm[use_obsm]
        return exp_data, np.asarray(CovGenes, dtype=str)
    if g == -1 or g >= spatial_data.shape[1]:
        CovGenes = spatial_data.var_names
        if verbose:
            print(f"Computing COVET using all {len(CovGenes)} genes")
    else:
        hvg_genes = None
        if "highly_variable" in spatial_data.var.columns:
            hvg_mask = np.asarray(spatial_data.var["highly_variable"], dtype=bool)
            precomputed_hvgs = np.asarray(spatial_data.var_names[hvg_mask])
            if len(precomputed_hvgs) <= g:
                hvg_genes = precomputed_hvgs
                if verbose:
                    print(f"Using {len(hvg_genes)} pre-calculated highly variable genes for COVET")
            elif "highly_variable_rank" in spatial_data.var.columns:
                # preprocess_omics keeps ranks from its original full-feature HVG pass.
                hvg_ranks = np.asarray(
                    spatial_data.var["highly_variable_rank"],
                    dtype=np.float64,
                )
                ranked_indices = np.flatnonzero(hvg_mask & np.isfinite(hvg_ranks))
                if len(ranked_indices) >= g:
                    top_indices = ranked_indices[
                        np.argsort(hvg_ranks[ranked_indices], kind="stable")[:g]
                    ]
                    selected = np.zeros(spatial_data.n_vars, dtype=bool)
                    selected[top_indices] = True
                    hvg_genes = np.asarray(spatial_data.var_names[selected])
                    if verbose:
                        print(f"Using the top {g} genes from the pre-calculated HVG ranking for COVET")
        if hvg_genes is None:
            if verbose:
                print(f"Identifying top {g} highly variable genes for COVET calculation")
            spatial_data_copy = spatial_data.copy()
            if use_layer is None:
                if "log" in spatial_data_copy.layers:
                    layer = "log"
                elif "log1p" in spatial_data_copy.layers:
                    layer = "log1p"
                elif spatial_data_copy.X.min() < 0:
                    layer = None
                else:
                    spatial_data_copy.layers["log"] = np.log(
                        _to_dense(spatial_data_copy.X) + 1
                    )
                    layer = "log"
            else:
                layer = use_layer
            sc.pp.highly_variable_genes(
                spatial_data_copy,
                n_top_genes=g,
                layer=layer if layer else None,
            )
            hvg_genes = spatial_data_copy.var_names[
                spatial_data_copy.var.highly_variable
            ]
        CovGenes = np.asarray(hvg_genes)
        if len(genes) > 0:
            CovGenes = np.union1d(CovGenes, genes)
            if verbose:
                print(f"Added {len(genes)} user-specified genes to COVET calculation")
        if verbose:
            print(f"Computing COVET using {len(CovGenes)} genes")
    if use_layer is not None:
        if use_layer not in spatial_data.layers:
            raise ValueError(f"Layer {use_layer!r} not found in spatial_data.layers")
        if verbose:
            print(f"Using expression data from layer {use_layer!r}")
        exp_data = _to_dense(spatial_data[:, CovGenes].layers[use_layer])
    else:
        if spatial_data.X.min() < 0:
            if verbose:
                print("Using expression data from X (appears to be log-transformed)")
            exp_data = _to_dense(spatial_data[:, CovGenes].X)
        else:
            if verbose:
                print("Log-transforming expression data from X")
            exp_data = np.log(_to_dense(spatial_data[:, CovGenes].X) + 1)
    return exp_data, np.asarray(CovGenes, dtype=str)
def _compute_covet_base(
    spatial_data,
    k=8,
    g=64,
    genes: Optional[Sequence[str]] = None,
    spatial_key="spatial",
    batch_key: Union[str, int] = "batch",
    batch_size=None,
    use_obsm: Optional[str] = None,
    use_layer: Optional[str] = None,
    store=True,
    verbose=True,
):
    """
    Compute COVET / COVET_SQRT / CovGenes.
    Returns
    -------
    COVET, COVET_SQRT, CovGenes
    If store=True, also writes:
        spatial_data.obsm["COVET"]
        spatial_data.obsm["COVET_SQRT"]
        spatial_data.uns["CovGenes"]
    """
    if isinstance(batch_key, str) and batch_key not in spatial_data.obs.columns:
        batch_key = -1
    exp_data, CovGenes = _select_covet_expression(
        spatial_data,
        g=g,
        genes=genes,
        use_obsm=use_obsm,
        use_layer=use_layer,
        verbose=verbose,
    )
    COVET = calculate_covariance_matrices(
        spatial_data,
        kNN=k,
        exp_data=exp_data,
        spatial_key=spatial_key,
        batch_key=batch_key,
        batch_size=batch_size,
        verbose=verbose,
    )
    if batch_size is None or batch_size >= COVET.shape[0]:
        if verbose:
            print("Computing matrix square root.")
        COVET_SQRT = batch_matrix_sqrt(COVET)
    else:
        n_cells = COVET.shape[0]
        COVET_SQRT = np.zeros_like(COVET)
        batch_indices = np.array_split(
            np.arange(n_cells),
            np.ceil(n_cells / batch_size),
        )
        for batch_idx in tqdm(
            batch_indices,
            desc="Computing matrix square roots",
            disable=not verbose,
        ):
            batch_sqrt = batch_matrix_sqrt(COVET[batch_idx])
            COVET_SQRT[batch_idx] = batch_sqrt
    COVET = COVET.astype("float32")
    COVET_SQRT = COVET_SQRT.astype("float32")
    CovGenes = np.asarray(CovGenes, dtype=str)
    if store:
        spatial_data.obsm["COVET"] = COVET
        spatial_data.obsm["COVET_SQRT"] = COVET_SQRT
        spatial_data.uns["CovGenes"] = CovGenes
    return COVET, COVET_SQRT, CovGenes
def _normalize_device_string(s: str) -> str:
    return s.strip().lower().replace(" ", "")
def _get_torch_device(device=None, *, verbose: bool = False):
    if torch is None:
        raise ImportError("torch is required for backend='torch', but torch is not installed.")
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, int):
        device = f"cuda:{device}"
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return device
        if not torch.cuda.is_available():
            if verbose:
                print("[AOT] CUDA not available; falling back to CPU.")
            return torch.device("cpu")
        idx = 0 if device.index is None else device.index
        n_gpu = torch.cuda.device_count()
        if idx < 0 or idx >= n_gpu:
            if verbose:
                print(
                    f"[AOT] Requested cuda:{idx}, but only {n_gpu} GPU(s) available; "
                    "using cuda:0."
                )
            return torch.device("cuda:0")
        return torch.device(f"cuda:{idx}")
    if isinstance(device, str):
        s = _normalize_device_string(device)
        if s == "cpu":
            return torch.device("cpu")
        if s in {"cuda", "gpu"}:
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            if verbose:
                print("[AOT] CUDA not available; falling back to CPU.")
            return torch.device("cpu")
        if s.startswith("cuda:") or s.startswith("gpu:"):
            try:
                idx = int(s.split(":", 1)[1])
            except Exception as exc:
                raise ValueError(
                    f"Invalid device string: {device!r}. "
                    "Use 'cpu', 'cuda', or 'cuda:N'."
                ) from exc
            if not torch.cuda.is_available():
                if verbose:
                    print("[AOT] CUDA not available; falling back to CPU.")
                return torch.device("cpu")
            n_gpu = torch.cuda.device_count()
            if idx < 0 or idx >= n_gpu:
                if verbose:
                    print(
                        f"[AOT] Requested cuda:{idx}, but only {n_gpu} GPU(s) available; "
                        "using cuda:0."
                    )
                return torch.device("cuda:0")
            return torch.device(f"cuda:{idx}")
        return torch.device(device)
    raise TypeError(f"Unsupported torch device type: {type(device)}")
def covet_sqrt_to_aot_features(
    adata,
    covet_sqrt_key: str = "COVET_SQRT",
    *,
    compact_upper: bool = False,
    store_key: Optional[str] = None,
):
    """
    Convert COVET_SQRT matrices to 2D features for Euclidean/AOT analysis.
    If compact_upper=False:
        flatten the full COVET_SQRT matrix.
    If compact_upper=True:
        use weighted upper triangle. Off-diagonal entries are multiplied by sqrt(2),
        so Euclidean distances match full Frobenius distances.
    """
    if covet_sqrt_key not in adata.obsm:
        raise KeyError(f"adata.obsm[{covet_sqrt_key!r}] not found. Run compute_covet() first.")
    S = np.asarray(adata.obsm[covet_sqrt_key], dtype=np.float32)
    if S.ndim != 3 or S.shape[1] != S.shape[2]:
        raise ValueError(
            f"Expected adata.obsm[{covet_sqrt_key!r}] to have shape (n, g, g); "
            f"got {S.shape}."
        )
    if compact_upper:
        g = S.shape[1]
        tri = np.triu_indices(g)
        F = S[:, tri[0], tri[1]].copy()
        offdiag = tri[0] != tri[1]
        F[:, offdiag] *= np.sqrt(2.0)
        F = F.astype(np.float32, copy=False)
    else:
        F = S.reshape(S.shape[0], -1).astype(np.float32, copy=False)
    if store_key is not None:
        adata.obsm[store_key] = F
    return F
@dataclass
class AOTGraphConfig:
    covet_sqrt_key: str = "COVET_SQRT"
    k_env: int = 30
    store_key: str = "aot_distances"
    compact_upper: bool = False
    store_feature_key: Optional[str] = None
    backend: Literal["sklearn", "torch"] = "sklearn"
    device: Optional[Union[str, int]] = None
    use_chunking: bool = True
    chunk_size: int = 1000
    log_every: int = 1000
    symmetrize: bool = False
    symmetrize_method: Literal["max", "average"] = "max"
    verbose: bool = True
def _effective_k(k_env: int, n: int) -> int:
    if n <= 1:
        raise ValueError("At least two observations are required to build an AOT kNN graph.")
    if k_env < 1:
        raise ValueError("k_env must be >= 1.")
    return min(int(k_env), n - 1)
def _remove_self_from_sklearn_neighbors(idx, dist, row_start, k_eff):
    """Remove self index row-wise after querying k_eff + 1 neighbors."""
    b = idx.shape[0]
    out_idx = np.empty((b, k_eff), dtype=np.int64)
    out_dist = np.empty((b, k_eff), dtype=np.float32)
    for r in range(b):
        global_i = row_start + r
        keep = idx[r] != global_i
        idx_r = idx[r][keep]
        dist_r = dist[r][keep]
        if idx_r.shape[0] < k_eff:
            raise RuntimeError(
                "Could not remove self while keeping k neighbors. "
                "This can happen with malformed kNN output."
            )
        out_idx[r] = idx_r[:k_eff]
        out_dist[r] = dist_r[:k_eff]
    return out_idx, out_dist
def build_aot_knn_graph(adata, cfg: AOTGraphConfig = AOTGraphConfig()):
    """
    Build a sparse AOT kNN graph from COVET_SQRT matrices.
    AOT(i, j) = || sqrt(COVET_i) - sqrt(COVET_j) ||_F^2
    The returned sparse matrix stores squared Euclidean/Frobenius distances.
    """
    F = covet_sqrt_to_aot_features(
        adata,
        covet_sqrt_key=cfg.covet_sqrt_key,
        compact_upper=cfg.compact_upper,
        store_key=cfg.store_feature_key,
    )
    n, d = F.shape
    k_eff = _effective_k(cfg.k_env, n)
    if cfg.verbose:
        rep = "weighted upper triangle" if cfg.compact_upper else "full COVET_SQRT flatten"
        print(f"[AOT] representation={rep}")
        print(f"[AOT] n={n}, d={d}, k_env={k_eff}, backend={cfg.backend}")
        print("[AOT] distances stored are squared Euclidean/Frobenius distances.")
    rows_all = []
    cols_all = []
    data_all = []
    if cfg.backend == "sklearn":
        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=k_eff + 1,
            metric="euclidean",
            algorithm="auto",
        )
        nn.fit(F)
        chunk_size = max(1, int(cfg.chunk_size)) if cfg.use_chunking else n
        processed = 0
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            dist, idx = nn.kneighbors(F[start:end], return_distance=True)
            idx, dist = _remove_self_from_sklearn_neighbors(idx, dist, start, k_eff)
            dist2 = np.square(dist).astype(np.float32, copy=False)
            rows_all.append(np.repeat(np.arange(start, end), k_eff))
            cols_all.append(idx.reshape(-1))
            data_all.append(dist2.reshape(-1))
            processed = end
            if cfg.verbose and (
                processed % max(1, int(cfg.log_every)) == 0 or processed == n
            ):
                print(f"[AOT/sklearn] processed {processed}/{n}")
    elif cfg.backend == "torch":
        if torch is None:
            raise ImportError("torch is required for backend='torch', but torch is not installed.")
        device = _get_torch_device(cfg.device, verbose=cfg.verbose)
        if cfg.verbose:
            print(f"[AOT/torch] device={device}")
        X = torch.as_tensor(F, device=device, dtype=torch.float32)
        x2 = (X * X).sum(dim=1)
        chunk_size = max(1, int(cfg.chunk_size)) if cfg.use_chunking else n
        processed = 0
        with torch.no_grad():
            for start in range(0, n, chunk_size):
                end = min(n, start + chunk_size)
                Q = X[start:end]
                q2 = (Q * Q).sum(dim=1, keepdim=True)
                dist2 = q2 + x2.unsqueeze(0) - 2.0 * (Q @ X.t())
                # Exclude self.
                ar = torch.arange(end - start, device=device)
                dist2[ar, start + ar] = float("inf")
                vals2, idx = torch.topk(
                    dist2,
                    k=k_eff,
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                vals2_np = torch.clamp(vals2, min=0.0).detach().cpu().numpy().astype(np.float32)
                idx_np = idx.detach().cpu().numpy().astype(np.int64)
                rows_all.append(np.repeat(np.arange(start, end), k_eff))
                cols_all.append(idx_np.reshape(-1))
                data_all.append(vals2_np.reshape(-1))
                processed = end
                if cfg.verbose and (
                    processed % max(1, int(cfg.log_every)) == 0 or processed == n
                ):
                    print(f"[AOT/torch] processed {processed}/{n}")
                del dist2, vals2, idx
    else:
        raise ValueError("cfg.backend must be either 'sklearn' or 'torch'.")
    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    data = np.concatenate(data_all)
    D = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    if cfg.symmetrize:
        if cfg.symmetrize_method == "max":
            D = D.maximum(D.T)
        elif cfg.symmetrize_method == "average":
            D = 0.5 * (D + D.T)
        else:
            raise ValueError("symmetrize_method must be 'max' or 'average'.")
    adata.obsp[cfg.store_key] = D
    if cfg.verbose:
        print(f"[AOT] saved adata.obsp[{cfg.store_key!r}] shape={D.shape}, nnz={D.nnz}")
    return adata
def aot_distance_matrix_full(
    adata,
    covet_sqrt_key: str = "COVET_SQRT",
    *,
    compact_upper: bool = False,
    device: Optional[Union[str, int]] = None,
    squared: bool = True,
    verbose: bool = True,
):
    """
    Compute dense full pairwise AOT distance matrix.
    Warning: O(n^2) memory. Use only for small datasets.
    """
    F = covet_sqrt_to_aot_features(
        adata,
        covet_sqrt_key=covet_sqrt_key,
        compact_upper=compact_upper,
        store_key=None,
    )
    if torch is None:
        if verbose:
            print(f"[AOT-full/numpy] n={F.shape[0]}, d={F.shape[1]}")
        x2 = np.sum(F * F, axis=1)
        D2 = x2[:, None] + x2[None, :] - 2.0 * (F @ F.T)
        D2 = np.maximum(D2, 0.0).astype(np.float32)
        return D2 if squared else np.sqrt(D2).astype(np.float32)
    dev = _get_torch_device(device, verbose=verbose)
    if verbose:
        print(f"[AOT-full/torch] n={F.shape[0]}, d={F.shape[1]}, device={dev}")
    with torch.no_grad():
        X = torch.as_tensor(F, device=dev, dtype=torch.float32)
        D = torch.cdist(X, X, p=2)
        if squared:
            D = D.pow(2)
        return D.detach().cpu().numpy().astype(np.float32)
# ============================================================
# 1. Shared missing-aware utilities
# ============================================================
def _get_missing_indices(
    adata,
    missing_key: str = "missing",
    observed_label=1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get missing and non-missing original cell indices.
    Priority:
        1. adata.uns["missing_indices"] and
           adata.uns["non_missing_indices"].
        2. adata.obs[missing_key].
        3. If missing_key does not exist, all cells are treated as observed.
    Convention:
        observed_label = 1 means observed / non-missing.
        Other values are missing.
    """
    if (
        "missing_indices" in adata.uns
        and "non_missing_indices" in adata.uns
    ):
        missing_indices = np.asarray(
            adata.uns["missing_indices"],
            dtype=np.int64,
        )
        non_missing_indices = np.asarray(
            adata.uns["non_missing_indices"],
            dtype=np.int64,
        )
        return missing_indices, non_missing_indices
    if missing_key not in adata.obs.columns:
        missing_indices = np.array([], dtype=np.int64)
        non_missing_indices = np.arange(adata.n_obs, dtype=np.int64)
        return missing_indices, non_missing_indices
    values = adata.obs[missing_key].to_numpy()
    try:
        values_numeric = np.asarray(values, dtype=float)
        observed_mask = values_numeric == float(observed_label)
    except (ValueError, TypeError):
        observed_mask = values.astype(str) == str(observed_label)
    non_missing_indices = np.flatnonzero(observed_mask).astype(np.int64)
    missing_indices = np.flatnonzero(~observed_mask).astype(np.int64)
    return missing_indices, non_missing_indices
def _resolve_batch_key_for_covet(
    adata_ref,
    batch_key,
    k_spatial: int,
    fallback_to_global: bool = True,
):
    """
    Safely resolve batch_key for COVET spatial kNN.
    """
    if batch_key is None or batch_key == -1:
        return -1
    if isinstance(batch_key, str) and batch_key not in adata_ref.obs.columns:
        return -1
    batch_counts = adata_ref.obs[batch_key].value_counts()
    min_batch_size = int(batch_counts.min())
    if min_batch_size < k_spatial:
        msg = (
            f"[COVET] Some batches have fewer cells than k_spatial={k_spatial}. "
            f"Minimum batch size is {min_batch_size}."
        )
        if fallback_to_global:
            print(msg)
            print("[COVET] Falling back to global spatial kNN: batch_key = -1.")
            return -1
        raise ValueError(msg)
    return batch_key
def _remap_subset_graph_to_original(
    D_subset,
    ref_indices,
    n_obs_original: int,
):
    """
    Remap an AOT sparse matrix from subset-index space to original-index space.
    """
    D_subset = D_subset.tocoo()
    ref_indices = np.asarray(ref_indices, dtype=np.int64)
    rows_original = ref_indices[D_subset.row]
    cols_original = ref_indices[D_subset.col]
    D_original = sp.csr_matrix(
        (D_subset.data, (rows_original, cols_original)),
        shape=(n_obs_original, n_obs_original),
        dtype=np.float32,
    )
    return D_original
def _save_aot_neighbor_index_matrix(
    adata,
    distance_key: str = "aot_distances",
    valid_query_indices=None,
    neighbor_indices_key: str = "aot_neighbor_indices",
    neighbor_distances_key: str = "aot_neighbor_distances",
):
    """
    Convert an original-index-space sparse AOT graph into explicit neighbor matrices.
    """
    if distance_key not in adata.obsp:
        raise KeyError(f"adata.obsp[{distance_key!r}] not found.")
    D = adata.obsp[distance_key].tocsr()
    n_obs = adata.n_obs
    if valid_query_indices is None:
        valid_query_indices = np.arange(n_obs, dtype=np.int64)
    else:
        valid_query_indices = np.asarray(valid_query_indices, dtype=np.int64)
    nnz_per_row = D.getnnz(axis=1)
    max_neighbors = (
        int(nnz_per_row[valid_query_indices].max())
        if len(valid_query_indices) > 0
        else 0
    )
    neighbor_indices = np.full(
        (n_obs, max_neighbors),
        fill_value=-1,
        dtype=np.int64,
    )
    neighbor_distances = np.full(
        (n_obs, max_neighbors),
        fill_value=np.inf,
        dtype=np.float32,
    )
    for i in valid_query_indices:
        start = D.indptr[i]
        end = D.indptr[i + 1]
        cols = D.indices[start:end]
        vals = D.data[start:end]
        if len(cols) == 0:
            continue
        order = np.argsort(vals)
        cols = cols[order]
        vals = vals[order]
        k = len(cols)
        neighbor_indices[i, :k] = cols
        neighbor_distances[i, :k] = vals
    adata.uns[neighbor_indices_key] = neighbor_indices
    adata.uns[neighbor_distances_key] = neighbor_distances
    return neighbor_indices, neighbor_distances
# ============================================================
# 2. Unified missing-aware mode
# ============================================================
def compute_covet(
    adata,
    k_spatial: int = 6,
    covet_gene_num: Optional[int] = None,
    spatial_key: str = "spatial",
    batch_key: Union[str, int] = "batch",
    batch_size=None,
    use_layer: Optional[str] = "log",
    use_obsm: Optional[str] = None,
    missing_key: str = "missing",
    observed_label=1,
    fallback_to_global_if_small_batch: bool = True,
    copy_when_complete: bool = False,
    verbose: bool = True,
):
    """
    Compute COVET with missing-aware cell selection.
    If is complete:
        COVET is computed on all cells.
    If has missing cells:
        COVET is computed only on non-missing cells.
    """
    missing_indices, non_missing_indices = _get_missing_indices(
        adata,
        missing_key=missing_key,
        observed_label=observed_label,
    )
    has_missing = len(missing_indices) > 0
    if len(non_missing_indices) < 2:
        raise ValueError(
            f"Only {len(non_missing_indices)} non-missing cells are available. "
            "At least 2 cells are required for COVET/AOT."
        )
    if has_missing:
        adata_covet_ref = adata[non_missing_indices, :].copy()
        ref_indices = non_missing_indices.copy()
    else:
        adata_covet_ref = adata.copy() if copy_when_complete else adata
        ref_indices = np.arange(adata.n_obs, dtype=np.int64)
    adata_covet_ref.obs["original_index"] = ref_indices.astype(np.int64)
    k_spatial_eff = min(int(k_spatial), adata_covet_ref.n_obs)
    if k_spatial_eff < 2:
        raise ValueError(
            f"k_spatial_eff={k_spatial_eff}. "
            "COVET requires at least 2 cells because denominator is k - 1."
        )
    if isinstance(batch_key, str) and batch_key not in adata_covet_ref.obs.columns:
        batch_key_eff = -1
    else:
        batch_key_eff = _resolve_batch_key_for_covet(
            adata_ref=adata_covet_ref,
            batch_key=batch_key,
            k_spatial=k_spatial_eff,
            fallback_to_global=fallback_to_global_if_small_batch,
        )
    covet_gene_num_eff = adata_covet_ref.n_vars if covet_gene_num is None else int(covet_gene_num)
    COVET, COVET_SQRT, CovGenes = _compute_covet_base(
        spatial_data=adata_covet_ref,
        k=k_spatial_eff,
        g=covet_gene_num_eff,
        genes=None,
        spatial_key=spatial_key,
        batch_key=batch_key_eff,
        batch_size=batch_size,
        use_obsm=use_obsm,
        use_layer=use_layer,
        store=True,
        verbose=verbose,
    )
    adata_covet_ref.uns["COVET_genes"] = CovGenes
    adata.uns["covet_mode"] = "auto_missing"
    adata.uns["covet_missing_indices"] = missing_indices
    adata.uns["covet_non_missing_indices"] = non_missing_indices
    adata.uns["covet_ref_indices"] = ref_indices
    adata.uns["covet_has_missing"] = has_missing
    adata.uns["covet_k_spatial"] = k_spatial_eff
    adata.uns["covet_batch_key"] = batch_key_eff
    adata.uns["covet_gene_num"] = len(CovGenes)
    adata.uns["CovGenes"] = CovGenes
    adata.uns["COVET_genes"] = CovGenes
    info = {
        "mode": "auto_missing",
        "has_missing": has_missing,
        "missing_indices": missing_indices,
        "non_missing_indices": non_missing_indices,
        "ref_indices": ref_indices,
        "n_obs_original": adata.n_obs,
        "n_obs_ref": adata_covet_ref.n_obs,
        "k_spatial": k_spatial_eff,
        "batch_key": batch_key_eff,
        "covet_gene_num": len(CovGenes),
        "CovGenes": CovGenes,
    }
    if verbose:
        print("=" * 70)
        print("[COVET | auto missing-aware mode]")
        print("Original n_obs :", adata.n_obs)
        print("Reference n_obs:", adata_covet_ref.n_obs)
        print("has missing    :", has_missing)
        print("missing cells  :", len(missing_indices))
        print("observed cells :", len(non_missing_indices))
        print("k_spatial used     :", k_spatial_eff)
        print("batch_key used     :", batch_key_eff)
        print("COVET shape        :", adata_covet_ref.obsm["COVET"].shape)
        print("COVET_SQRT shape   :", adata_covet_ref.obsm["COVET_SQRT"].shape)
        print("Number of CovGenes :", len(CovGenes))
        print("=" * 70)
    return adata, adata_covet_ref, info
def compute_aot(
    adata,
    adata_covet_ref,
    k_env: Optional[int] = None,
    device="cuda:1",
    covet_sqrt_key: str = "COVET_SQRT",
    aot_ref_key: str = "aot_distances_ref",
    aot_key: str = "aot_distances",
    aot_feature_key: str = "COVET_AOT_FEATURE",
    compact_upper: bool = False,
    backend: str = "torch",
    use_chunking: bool = False,
    chunk_size: int = 3000,
    log_every: int = 3000,
    symmetrize: bool = False,
    symmetrize_method: str = "max",
    store_neighbor_index: bool = True,
    neighbor_indices_key: str = "aot_neighbor_indices",
    neighbor_distances_key: str = "aot_neighbor_distances",
    verbose: bool = True,
):
    """
    Compute AOT from COVET_SQRT with original-index remapping.
    The input adata_covet_ref should be returned by
    compute_covet().
    """
    if covet_sqrt_key not in adata_covet_ref.obsm:
        raise KeyError(
            f"adata_covet_ref.obsm[{covet_sqrt_key!r}] not found. "
            "Run compute_covet() first."
        )
    if "original_index" not in adata_covet_ref.obs.columns:
        raise KeyError(
            "adata_covet_ref.obs['original_index'] not found. "
            "Use the reference returned by compute_covet()."
        )
    ref_indices = np.asarray(
        adata_covet_ref.obs["original_index"].to_numpy(),
        dtype=np.int64,
    )
    n_ref = adata_covet_ref.n_obs
    if n_ref < 2:
        raise ValueError("AOT requires at least 2 reference cells.")
    k_env_eff = n_ref - 1 if k_env is None else min(int(k_env), n_ref - 1)
    if k_env_eff < 1:
        raise ValueError("k_env_eff must be >= 1.")
    adata_covet_ref = build_aot_knn_graph(
        adata_covet_ref,
        AOTGraphConfig(
            covet_sqrt_key=covet_sqrt_key,
            k_env=k_env_eff,
            store_key=aot_ref_key,
            compact_upper=compact_upper,
            store_feature_key=aot_feature_key,
            backend=backend,
            device=device,
            use_chunking=use_chunking,
            chunk_size=chunk_size,
            log_every=log_every,
            symmetrize=symmetrize,
            symmetrize_method=symmetrize_method,
            verbose=verbose,
        ),
    )
    D_ref = adata_covet_ref.obsp[aot_ref_key]
    D_original = _remap_subset_graph_to_original(
        D_subset=D_ref,
        ref_indices=ref_indices,
        n_obs_original=adata.n_obs,
    )
    adata.obsp[aot_key] = D_original
    adata.uns["aot_mode"] = "auto_missing"
    adata.uns["aot_ref_indices"] = ref_indices
    adata.uns["aot_k_env"] = k_env_eff
    adata.uns["aot_key"] = aot_key
    adata.uns["aot_ref_key"] = aot_ref_key
    adata.uns["aot_feature_key"] = aot_feature_key
    if store_neighbor_index:
        neighbor_indices, neighbor_distances = _save_aot_neighbor_index_matrix(
            adata=adata,
            distance_key=aot_key,
            valid_query_indices=ref_indices,
            neighbor_indices_key=neighbor_indices_key,
            neighbor_distances_key=neighbor_distances_key,
        )
    else:
        neighbor_indices = None
        neighbor_distances = None
    info = {
        "mode": "auto_missing",
        "ref_indices": ref_indices,
        "n_obs_original": adata.n_obs,
        "n_obs_ref": n_ref,
        "k_env": k_env_eff,
        "aot_key": aot_key,
        "aot_ref_key": aot_ref_key,
        "aot_feature_key": aot_feature_key,
        "neighbor_indices_key": neighbor_indices_key if store_neighbor_index else None,
        "neighbor_distances_key": neighbor_distances_key if store_neighbor_index else None,
        "neighbor_indices": neighbor_indices,
        "neighbor_distances": neighbor_distances,
    }
    if verbose:
        print("=" * 70)
        print("[AOT | auto missing-aware mode]")
        print("Reference AOT shape      :", adata_covet_ref.obsp[aot_ref_key].shape)
        print("Reference AOT nnz        :", adata_covet_ref.obsp[aot_ref_key].nnz)
        print("Original-space AOT shape :", adata.obsp[aot_key].shape)
        print("Original-space AOT nnz   :", adata.obsp[aot_key].nnz)
        print("k_env used               :", k_env_eff)
        if aot_feature_key in adata_covet_ref.obsm:
            print("AOT feature shape        :", adata_covet_ref.obsm[aot_feature_key].shape)
        if store_neighbor_index:
            print("Neighbor index shape     :", adata.uns[neighbor_indices_key].shape)
            print("Neighbor distance shape  :", adata.uns[neighbor_distances_key].shape)
        print("=" * 70)
    return adata, adata_covet_ref, info
