"""PRISM evaluation, raw-scale imputation, and visualization utilities."""
import copy
import hashlib
import os
import weakref
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors
from .core import (
    align_cluster_to_reference,
    clustering,
    get_domain_label_order,
    register_domain_palette,
)
# Runtime-only cache for repeated domain plotting on an unchanged AnnData.
# Entries hold weak references and therefore disappear after the source object
# is released or after the notebook kernel restarts.
_CLUSTERING_RUNTIME_CACHE = {}
def _copy_adata_without_pairwise_duplication(adata):
    """Copy an AnnData while sharing existing read-only pairwise matrices.
    Domain clustering does not read precomputed priors such as
    ``obsp['aot_distances']``.  Temporarily detaching those matrices avoids a
    costly deep copy, while the returned AnnData still exposes the same values.
    The clustering workflow only adds new ``distances`` and ``connectivities``
    keys and never mutates the shared matrices.
    """
    existing_obsp = {key: value for key, value in adata.obsp.items()}
    if not existing_obsp:
        return adata.copy()
    for key in existing_obsp:
        del adata.obsp[key]
    try:
        copied = adata.copy()
    finally:
        for key, value in existing_obsp.items():
            adata.obsp[key] = value
    for key, value in existing_obsp.items():
        copied.obsp[key] = value
    return copied
def _embedding_signature(embedding):
    """Return a content signature used to invalidate stale runtime caches."""
    values = np.ascontiguousarray(np.asarray(embedding))
    digest = hashlib.blake2b(values.view(np.uint8), digest_size=16).hexdigest()
    return values.shape, values.dtype.str, digest
def _runtime_cache_key(adata, *, emb_key, cluster_key, n_clusters, method, use_pca, n_comps, n_neighbors, seed):
    return (
        emb_key,
        cluster_key,
        int(n_clusters),
        str(method),
        bool(use_pca),
        int(n_comps),
        int(n_neighbors),
        int(seed),
        _embedding_signature(adata.obsm[emb_key]),
    )
def _get_runtime_cluster_cache(adata, cache_key):
    entry = _CLUSTERING_RUNTIME_CACHE.get(id(adata))
    if entry is None or entry["source_ref"]() is not adata:
        return None
    return entry if entry["cache_key"] == cache_key else None
def _store_runtime_cluster_cache(source_adata, cache_key, clustered_adata, *, emb_key, cluster_key, use_pca):
    pca_key = f"{emb_key}_pca"
    cached_obs = {cluster_key: clustered_adata.obs[cluster_key].copy(deep=True)}
    if "mclust" in clustered_adata.obs:
        cached_obs["mclust"] = clustered_adata.obs["mclust"].copy(deep=True)
    _CLUSTERING_RUNTIME_CACHE[id(source_adata)] = {
        "source_ref": weakref.ref(source_adata),
        "cache_key": cache_key,
        "pca": (
            np.asarray(clustered_adata.obsm[pca_key]).copy()
            if use_pca and pca_key in clustered_adata.obsm
            else None
        ),
        "obs": cached_obs,
        "distances": clustered_adata.obsp["distances"].copy(),
        "connectivities": clustered_adata.obsp["connectivities"].copy(),
        "neighbors": copy.deepcopy(clustered_adata.uns["neighbors"]),
        "umap": np.asarray(clustered_adata.obsm["X_umap"]).copy(),
        "umap_uns": copy.deepcopy(clustered_adata.uns.get("umap")),
    }
def _restore_runtime_cluster_cache(adata, cached, *, emb_key, use_pca):
    pca_key = f"{emb_key}_pca"
    if use_pca and cached["pca"] is not None:
        adata.obsm[pca_key] = cached["pca"].copy()
    for key, values in cached["obs"].items():
        adata.obs[key] = values.copy(deep=True)
    adata.obsp["distances"] = cached["distances"].copy()
    adata.obsp["connectivities"] = cached["connectivities"].copy()
    adata.uns["neighbors"] = copy.deepcopy(cached["neighbors"])
    adata.obsm["X_umap"] = cached["umap"].copy()
    if cached["umap_uns"] is not None:
        adata.uns["umap"] = copy.deepcopy(cached["umap_uns"])
# ==========================================================
# General utilities
# ==========================================================
def _fmt4(x):
    """
    Format metric values to 4 decimal places.
    """
    try:
        x = float(x)
        if not np.isfinite(x):
            return "nan"
        return f"{x:.4f}"
    except Exception:
        return "nan"
def _format_metric_dict(metrics):
    """Format a dictionary while preserving four decimal places for metrics."""
    items = []
    for name, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            rendered = _fmt4(value)
        else:
            rendered = repr(value)
        items.append(f"{name!r}: {rendered}")
    return "{" + ", ".join(items) + "}"
def _to_dense_f32(X):
    """Convert sparse/dense matrix to dense float32 numpy array."""
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)
# ==========================================================
# Task 1: domain identification evaluation
# ==========================================================
def evaluate(true_labels, pred_labels):
    ami = adjusted_mutual_info_score(true_labels, pred_labels)
    v_measure = v_measure_score(true_labels, pred_labels)
    mutual_info = mutual_info_score(true_labels, pred_labels)
    homogeneity = homogeneity_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return ami, v_measure, mutual_info, homogeneity, nmi, ari
def run_clustering_eval_plot(
    adata,
    emb_key,
    label_key="final_annot",
    cluster_key=None,
    n_clusters=10,
    method="mclust",
    use_pca=True,
    n_neighbors=10,
    s=None,
    n_comps=20,
    seed=2024,
    align_labels=True,
    aligned_key=None,
    label_order=None,
    dataset_name=None,
    palette=None,
    label_to_color=None,
    cluster_to_label=None,
    cluster_palette=None,
    cluster_label_to_color=None,
):
    """Run clustering and plot domains with stable, reference-aligned colors.
    When a reference annotation is available, ``align_labels=True`` maps the
    unsupervised cluster IDs to reference labels with Hungarian matching. The
    mapped key is used for UMAP/spatial plots, while metrics continue to use
    the original cluster IDs (the metrics are permutation invariant).
    """
    from .core import set_prism_plot_style
    set_prism_plot_style()
    cluster_key = cluster_key or f"{emb_key}_mclust"
    label_order = label_order or get_domain_label_order(dataset_name, label_key)
    cache_key = _runtime_cache_key(
        adata,
        emb_key=emb_key,
        cluster_key=cluster_key,
            n_clusters=n_clusters,
        method=method,
        use_pca=use_pca,
            n_comps=n_comps,
        n_neighbors=n_neighbors,
        seed=seed,
    )
    cached = _get_runtime_cluster_cache(adata, cache_key)
    adata_source = adata
    adata = _copy_adata_without_pairwise_duplication(adata_source)
    if cached is None:
        clustering(
            adata,
            key=emb_key,
            add_key=cluster_key,
            n_clusters=n_clusters,
            method=method,
            use_pca=use_pca,
            n_comps=n_comps,
            random_seed=seed,
        )
        sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=n_neighbors)
        sc.tl.umap(adata)
        _store_runtime_cluster_cache(
            adata_source,
            cache_key,
            adata,
            emb_key=emb_key,
            cluster_key=cluster_key,
            use_pca=use_pca,
        )
    else:
        _restore_runtime_cluster_cache(
            adata,
            cached,
            emb_key=emb_key,
            use_pca=use_pca,
        )
    has_label = label_key in adata.obs.columns
    for key in (cluster_key, label_key):
        if key in adata.obs and isinstance(adata.obs[key].dtype, pd.CategoricalDtype):
            adata.obs[key] = adata.obs[key].cat.remove_unused_categories()
    plot_key = cluster_key
    alignment = None
    if has_label and align_labels:
        # ------------------------------------------------------
        # 1. Align unsupervised clusters to reference domains.
        #
        # This key contains biological names internally, e.g.
        # "germinal center", "lymphoid follicle", ...
        # ------------------------------------------------------
        aligned_label_key = (
            aligned_key
            or f"{cluster_key}_domain"
        )
        alignment, _ = align_cluster_to_reference(
            adata,
            cluster_key,
            label_key,
            aligned_key=aligned_label_key,
            label_order=label_order,
            cluster_to_label=cluster_to_label,
            dataset_name=dataset_name,
            palette=palette,
            label_to_color=label_to_color,
        )
        # ------------------------------------------------------
        # 2. Convert the aligned biological labels into
        #    numeric display labels:
        #
        #       reference label 1 -> "1"
        #       reference label 2 -> "2"
        #       reference label 3 -> "3"
        #       ...
        #
        # Biological names are therefore kept only in
        # Reference Annotation.
        # ------------------------------------------------------
        numeric_plot_key = (
            f"{aligned_label_key}_display"
        )
        aligned_series = adata.obs[
            aligned_label_key
        ]
        # Non-null labels actually present after alignment.
        present_labels = [
            str(x)
            for x in pd.unique(
                aligned_series.dropna().astype(str)
            )
        ]
        # ------------------------------------------------------
        # Determine biological-domain order.
        #
        # Prefer the explicitly supplied DOMAIN_LABEL_ORDER.
        # This guarantees stable:
        #
        #   1 -> first reference domain
        #   2 -> second reference domain
        #   ...
        #
        # across runs and datasets.
        # ------------------------------------------------------
        if label_order is not None:
            ordered_labels = [
                str(x)
                for x in label_order
                if str(x) in present_labels
            ]
        elif isinstance(
            aligned_series.dtype,
            pd.CategoricalDtype,
        ):
            ordered_labels = [
                str(x)
                for x in aligned_series.cat.categories
                if str(x) in present_labels
            ]
        else:
            ordered_labels = (
                present_labels.copy()
            )
        # Safety:
        # keep any aligned labels that were not included
        # in label_order.
        for label in present_labels:
            if label not in ordered_labels:
                ordered_labels.append(label)
        # ------------------------------------------------------
        # Biological label -> numeric display ID
        # ------------------------------------------------------
        label_to_number = {
            label: str(i + 1)
            for i, label
            in enumerate(ordered_labels)
        }
        numeric_categories = [
            str(i + 1)
            for i in range(
                len(ordered_labels)
            )
        ]
        numeric_values = aligned_series.map(
            lambda x: (
                label_to_number.get(
                    str(x),
                    np.nan,
                )
                if pd.notna(x)
                else np.nan
            )
        )
        adata.obs[
            numeric_plot_key
        ] = pd.Categorical(
            numeric_values,
            categories=numeric_categories,
            ordered=True,
        )
        # ------------------------------------------------------
        # 3. Preserve the aligned colors.
        #
        # The legend changes from biological names to 1/2/3/4,
        # but each numeric domain keeps the color corresponding
        # to its matched reference domain.
        # ------------------------------------------------------
        aligned_color_key = (
            f"{aligned_label_key}_colors"
        )
        reference_color_key = (
            f"{label_key}_colors"
        )
        color_map = {}
        # First preference:
        # colors generated for the aligned label key.
        if (
            aligned_color_key in adata.uns
            and isinstance(
                aligned_series.dtype,
                pd.CategoricalDtype,
            )
        ):
            source_categories = [
                str(x)
                for x
                in aligned_series.cat.categories
            ]
            source_colors = list(
                adata.uns[
                    aligned_color_key
                ]
            )
            if (
                len(source_categories)
                == len(source_colors)
            ):
                color_map = dict(
                    zip(
                        source_categories,
                        source_colors,
                    )
                )
        # Second preference:
        # use colors directly from Reference Annotation.
        if (
            len(color_map) == 0
            and reference_color_key in adata.uns
            and isinstance(
                adata.obs[label_key].dtype,
                pd.CategoricalDtype,
            )
        ):
            reference_categories = [
                str(x)
                for x
                in adata.obs[
                    label_key
                ].cat.categories
            ]
            reference_colors = list(
                adata.uns[
                    reference_color_key
                ]
            )
            if (
                len(reference_categories)
                == len(reference_colors)
            ):
                color_map = dict(
                    zip(
                        reference_categories,
                        reference_colors,
                    )
                )
        # Register colors for numeric domains.
        if (
            len(color_map) > 0
            and all(
                label in color_map
                for label in ordered_labels
            )
        ):
            adata.uns[
                f"{numeric_plot_key}_colors"
            ] = [
                color_map[label]
                for label in ordered_labels
            ]
        # ------------------------------------------------------
        # IMPORTANT:
        # UMAP and PRISM Domains use numeric labels from here on.
        # ------------------------------------------------------
        plot_key = numeric_plot_key
    elif has_label:
        # ======================================================
        # Reference labels exist but alignment is disabled.
        #
        # Reference Annotation:
        #     biological names
        #
        # Predicted clusters:
        #     original numeric cluster IDs
        # ======================================================
        register_domain_palette(
            adata,
            label_key,
            label_order=label_order,
            dataset_name=dataset_name,
            palette=palette,
            label_to_color=label_to_color,
        )
        # Convert predicted clusters to clean numeric labels.
        source_cluster = (
            adata.obs[
                cluster_key
            ]
            .astype(str)
        )
        cluster_values = [
            str(x)
            for x in pd.unique(
                source_cluster
            )
        ]
        # Sort numerically when possible.
        try:
            cluster_values = sorted(
                cluster_values,
                key=lambda x: float(x),
            )
        except Exception:
            cluster_values = sorted(
                cluster_values
            )
        numeric_plot_key = (
            f"{cluster_key}_display"
        )
        cluster_to_number = {
            old: str(i + 1)
            for i, old
            in enumerate(cluster_values)
        }
        numeric_categories = [
            str(i + 1)
            for i in range(
                len(cluster_values)
            )
        ]
        adata.obs[
            numeric_plot_key
        ] = pd.Categorical(
            source_cluster.map(
                cluster_to_number
            ),
            categories=numeric_categories,
            ordered=True,
        )
        register_domain_palette(
            adata,
            numeric_plot_key,
        )
        plot_key = numeric_plot_key
    else:
        # ======================================================
        # No reference annotation available.
        #
        # Display predicted domains using the supplied display labels when
        # available, otherwise use the stable cluster1, cluster2, ... order.
        # ======================================================
        source_cluster = (
            adata.obs[
                cluster_key
            ]
            .astype(str)
        )
        cluster_values = [
            str(x)
            for x in pd.unique(
                source_cluster
            )
        ]
        try:
            cluster_values = sorted(
                cluster_values,
                key=lambda x: float(x),
            )
        except Exception:
            cluster_values = sorted(
                cluster_values
            )
        numeric_plot_key = (
            aligned_key
            or f"{cluster_key}_domain"
        )
        display_labels = list(label_order) if label_order is not None else None
        if display_labels is None:
            display_labels = [
                f"cluster{i + 1}"
                for i in range(len(cluster_values))
            ]
        display_labels = [str(label) for label in display_labels]
        if len(display_labels) < len(cluster_values):
            raise ValueError(
                f"label_order has {len(display_labels)} labels for "
                f"{len(cluster_values)} clusters."
            )
        display_labels = display_labels[:len(cluster_values)]
        adata.obs[
            numeric_plot_key
        ] = pd.Categorical(
            source_cluster.map(
                dict(zip(cluster_values, display_labels))
            ),
            categories=display_labels,
            ordered=True,
        )
        register_domain_palette(
            adata,
            numeric_plot_key,
            dataset_name=dataset_name,
            palette=(
                cluster_palette
                if cluster_palette is not None
                else palette
            ),
            label_to_color=(
                cluster_label_to_color
                if cluster_label_to_color
                is not None
                else label_to_color
            ),
        )
        plot_key = numeric_plot_key
    ncols = 3 if has_label else 2
    point_size = (
        float(np.clip(120000.0 / max(adata.n_obs, 1), 4.0, 25.0))
        if s is None
        else float(s)
    )
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(3.5 * ncols, 3.0),
    )
    fig.patch.set_facecolor("white")
    sc.pl.umap(
        adata,
        color=plot_key,
        ax=axes[0],
        title="UMAP",
        s=point_size * 0.8,
        show=False,
    )
    sc.pl.embedding(
        adata,
        basis="spatial",
        color=plot_key,
        ax=axes[1],
        title="PRISM Domains",
        s=point_size,
        show=False,
    )
    metrics = None
    if has_label:
        sc.pl.embedding(
            adata,
            basis="spatial",
            color=label_key,
            ax=axes[2],
            title="Reference Annotation",
            s=point_size,
            show=False,
        )
        valid = ~adata.obs[label_key].isna()
        y_true = adata.obs.loc[valid, label_key]
        y_pred = adata.obs.loc[valid, cluster_key]
        ami, v_measure, mutual_info, homogeneity, nmi, ari = evaluate(y_true, y_pred)
        metrics = {
            "embedding": emb_key,
            "cluster_key": cluster_key,
            "AMI": round(float(ami), 4),
            "V_measure": round(float(v_measure), 4),
            "Mutual_info": round(float(mutual_info), 4),
            "Homogeneity": round(float(homogeneity), 4),
            "NMI": round(float(nmi), 4),
            "ARI": round(float(ari), 4),
        }
        print(f"Domain clustering metrics: {_format_metric_dict(metrics)}")
    plt.tight_layout(w_pad=0.3)
    plt.show()
    return adata, metrics
# ==========================================================
# SSIM utilities
# ==========================================================
def _rasterize_spatial_values(coords, values, max_grid_size=512):
    """Convert spot/cell spatial values to a 2D grid for SSIM."""
    coords = np.asarray(coords, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).ravel()
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords must be [N, 2], got {coords.shape}.")
    if coords.shape[0] != values.shape[0]:
        raise ValueError(f"coords/value length mismatch: {coords.shape[0]} vs {values.shape[0]}.")
    x, y = coords[:, 0], coords[:, 1]
    x_range = float(np.nanmax(x) - np.nanmin(x))
    y_range = float(np.nanmax(y) - np.nanmin(y))
    max_range = max(x_range, y_range, 1e-8)
    scale = (float(max_grid_size) - 1.0) / max_range
    gx = np.rint((x - np.nanmin(x)) * scale).astype(np.int64)
    gy = np.rint((y - np.nanmin(y)) * scale).astype(np.int64)
    h, w = int(gy.max() + 1), int(gx.max() + 1)
    finite = np.isfinite(values)
    fill_value = float(np.nanmin(values[finite])) if np.any(finite) else 0.0
    grid_sum = np.zeros((h, w), dtype=np.float32)
    grid_count = np.zeros((h, w), dtype=np.float32)
    for xi, yi, val in zip(gx, gy, values):
        if np.isfinite(val):
            grid_sum[yi, xi] += val
            grid_count[yi, xi] += 1.0
    grid = np.full((h, w), fill_value, dtype=np.float32)
    valid = grid_count > 0
    grid[valid] = grid_sum[valid] / grid_count[valid]
    return grid
def _auto_ssim_grid_size(n_obs):
    """Automatically choose SSIM rasterization max grid size by number of spots/cells."""
    n_obs = int(n_obs)
    if n_obs < 5000:
        return 256
    if n_obs < 10000:
        return 384
    if n_obs < 50000:
        return 512
    if n_obs < 200000:
        return 768
    return 1024
def _compute_ssim_2d(img_true, img_pred):
    """Compute SSIM for two 2D images."""
    img_true = np.asarray(img_true, dtype=np.float32)
    img_pred = np.asarray(img_pred, dtype=np.float32)
    finite = np.isfinite(img_true) & np.isfinite(img_pred)
    if not np.any(finite):
        return np.nan
    min_v = float(min(np.nanmin(img_true[finite]), np.nanmin(img_pred[finite])))
    max_v = float(max(np.nanmax(img_true[finite]), np.nanmax(img_pred[finite])))
    data_range = max(max_v - min_v, 1e-8)
    if img_true.shape[0] < 3 or img_true.shape[1] < 3:
        return np.nan
    try:
        from skimage.metrics import structural_similarity as skimage_ssim
        min_dim = min(img_true.shape[:2])
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            return np.nan
        return float(skimage_ssim(img_true, img_pred, data_range=data_range, win_size=win_size))
    except Exception:
        x = img_true[finite].astype(np.float64)
        y = img_pred[finite].astype(np.float64)
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        mux, muy = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cov = ((x - mux) * (y - muy)).mean()
        return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux ** 2 + muy ** 2 + c1) * (vx + vy + c2)))
# ==========================================================
# Task 2: prior-quality evaluation
# ==========================================================
def compute_metrics_each_pair(
    adata_omics2,
    distance_matrix,
    top_n=10,
    verbose=False,
    compute_structure_metrics=False,
):
    """
    Evaluate AOT/prior quality.
    For each spot/cell:
        1. Find top_n nearest neighbors from distance_matrix.
        2. Compute RMSE/PCC/SPCC between the spot and each neighbor.
        3. Build a prior-predicted expression matrix by averaging top_n neighbors.
        4. Optionally compute CMD and feature-level spatial SSIM.
    """
    Xp = adata_omics2.X
    protein_data = Xp.toarray() if sp.issparse(Xp) else np.asarray(Xp)
    protein_data = np.asarray(protein_data, dtype=np.float32)
    n_spots, n_features = protein_data.shape
    if distance_matrix.shape[0] != n_spots:
        raise ValueError(
            f"Distance matrix size mismatch: distance_matrix.shape[0]={distance_matrix.shape[0]} "
            f"does not match adata_omics2.n_obs={n_spots}"
        )
    if compute_structure_metrics:
        if "spatial" not in adata_omics2.obsm:
            raise KeyError("SSIM calculation requires adata_omics2.obsm['spatial'].")
        coords = np.asarray(adata_omics2.obsm["spatial"], dtype=np.float32)
    else:
        coords = None
    is_sparse = sp.issparse(distance_matrix)
    rmse_values_all, pcc_values_all, spcc_values_all, cmd_values_all = [], [], [], []
    details = []
    prior_pred_data = np.full_like(protein_data, np.nan, dtype=np.float32)
    for i in range(n_spots):
        if is_sparse:
            row = distance_matrix.getrow(i)
            nbr_idx = row.indices
            nbr_dist = row.data
            mask = nbr_idx != i
            nbr_idx = nbr_idx[mask]
            nbr_dist = nbr_dist[mask]
            if nbr_idx.size == 0:
                similar_spots = np.array([], dtype=int)
                similar_dists = np.array([], dtype=np.float32)
            else:
                order = np.argsort(nbr_dist)
                top = order[:top_n]
                similar_spots = nbr_idx[top]
                similar_dists = nbr_dist[top].astype(np.float32)
        else:
            order = np.argsort(distance_matrix[i, :])
            order = order[order != i]
            similar_spots = order[:top_n]
            similar_dists = distance_matrix[i, similar_spots].astype(np.float32)
        spot_expr = protein_data[i, :]
        rmse_list, pcc_list, spcc_list, cmd_list = [], [], [], []
        if similar_spots.size > 0:
            prior_pred_data[i, :] = np.nanmean(protein_data[similar_spots, :], axis=0)
        else:
            prior_pred_data[i, :] = spot_expr
        for j in similar_spots:
            nbr_expr = protein_data[j, :]
            rmse = np.sqrt(mean_squared_error(spot_expr, nbr_expr))
            rmse_list.append(rmse)
            pcc = np.nan
            spcc_v = np.nan
            try:
                if np.std(spot_expr) > 0 and np.std(nbr_expr) > 0:
                    pcc, _ = pearsonr(spot_expr, nbr_expr)
                    spcc_v, _ = spearmanr(spot_expr, nbr_expr)
            except Exception as e:
                if verbose:
                    print(f"PCC/SPCC failed for spot {i} and neighbor {j}: {e}")
            pcc_list.append(pcc)
            spcc_list.append(spcc_v)
            if compute_structure_metrics:
                denom = np.linalg.norm(spot_expr) * np.linalg.norm(nbr_expr)
                if denom <= 1e-8:
                    cmd = 0.0 if np.linalg.norm(spot_expr - nbr_expr) <= 1e-8 else np.nan
                else:
                    cmd = 1.0 - float(np.dot(spot_expr, nbr_expr) / (denom + 1e-8))
                cmd_list.append(cmd)
        rmse_mean = np.nanmean(rmse_list) if len(rmse_list) else np.nan
        pcc_mean = np.nanmean(pcc_list) if len(pcc_list) else np.nan
        spcc_mean = np.nanmean(spcc_list) if len(spcc_list) else np.nan
        cmd_mean = np.nanmean(cmd_list) if len(cmd_list) else np.nan
        rmse_values_all.append(rmse_mean)
        pcc_values_all.append(pcc_mean)
        spcc_values_all.append(spcc_mean)
        cmd_values_all.append(cmd_mean)
        details.append({
            "spot_index": i,
            "similar_spots": similar_spots,
            "similar_dists": similar_dists,
            "rmse_values": rmse_list,
            "pcc_values": pcc_list,
            "spcc_values": spcc_list,
            "cmd_values": cmd_list,
            "rmse_mean": rmse_mean,
            "pcc_mean": pcc_mean,
            "spcc_mean": spcc_mean,
            "cmd_mean": cmd_mean,
        })
    rmse_values_all = np.asarray(rmse_values_all, dtype=np.float32)
    pcc_values_all = np.asarray(pcc_values_all, dtype=np.float32)
    spcc_values_all = np.asarray(spcc_values_all, dtype=np.float32)
    cmd_values_all = np.asarray(cmd_values_all, dtype=np.float32)
    ssim_values_all = np.full(n_features, np.nan, dtype=np.float32)
    if compute_structure_metrics:
        ssim_grid_size = _auto_ssim_grid_size(n_spots)
        for f in range(n_features):
            img_true = _rasterize_spatial_values(coords, protein_data[:, f], max_grid_size=ssim_grid_size)
            img_pred = _rasterize_spatial_values(coords, prior_pred_data[:, f], max_grid_size=ssim_grid_size)
            ssim_values_all[f] = _compute_ssim_2d(img_true, img_pred)
    return (
        rmse_values_all,
        pcc_values_all,
        spcc_values_all,
        cmd_values_all,
        ssim_values_all,
        details,
    )
# ==========================================================
# Task 2: imputation metrics
# ==========================================================
def evaluate_protein_prediction(
    protein_true,
    protein_pred,
    missing_indices,
    *,
    compute_structure_metrics=False,
    cmd_eps=1e-8,
):
    """
    Calculate imputation metrics on missing cells.
    Metrics:
        PCC, SPCC, MSE
    Optional metric:
        CMD, controlled by compute_structure_metrics=True.
    """
    if protein_true.shape != protein_pred.shape:
        raise ValueError(f"Dimension mismatch: True {protein_true.shape}, Pred {protein_pred.shape}")
    missing_indices = np.asarray(missing_indices, dtype=int)
    y_true = np.asarray(protein_true[missing_indices, :], dtype=np.float32)
    y_pred = np.asarray(protein_pred[missing_indices, :], dtype=np.float32)
    num_features = y_true.shape[1]
    per_feature = {"PCC": [], "SPCC": [], "MSE": []}
    if compute_structure_metrics:
        per_feature["CMD"] = []
    for j in range(num_features):
        yt = y_true[:, j].astype(np.float32)
        yp = y_pred[:, j].astype(np.float32)
        finite = np.isfinite(yt) & np.isfinite(yp)
        if finite.sum() == 0:
            for key in per_feature:
                per_feature[key].append(np.nan)
            continue
        yt = yt[finite]
        yp = yp[finite]
        per_feature["MSE"].append(float(np.mean((yt - yp) ** 2)))
        if "PCC" in per_feature:
            if np.std(yt) == 0 or np.std(yp) == 0:
                per_feature["PCC"].append(0.0)
            else:
                pcc, _ = pearsonr(yt, yp)
                per_feature["PCC"].append(float(pcc))
        if "SPCC" in per_feature:
            if np.std(yt) == 0 or np.std(yp) == 0:
                per_feature["SPCC"].append(0.0)
            else:
                spcc, _ = spearmanr(yt, yp)
                per_feature["SPCC"].append(float(spcc))
        if "CMD" in per_feature:
            denom = np.linalg.norm(yt) * np.linalg.norm(yp)
            if denom <= cmd_eps:
                cmd = 0.0 if np.linalg.norm(yt - yp) <= cmd_eps else np.nan
            else:
                cmd = 1.0 - float(np.dot(yt, yp) / (denom + cmd_eps))
            per_feature["CMD"].append(cmd)
    overall = {}
    for m, vals in per_feature.items():
        overall[m] = float(np.nanmean(np.asarray(vals, dtype=np.float32)))
    return {"overall": overall, "per_protein": per_feature}
def plot_per_protein_correlations(
    imputation_results,
    feature_names,
    *,
    metrics=("PCC", "SPCC"),
    feature_indices=None,
    sort_by_metric=True,
    colors=None,
    figsize=None,
    dpi=300,
):
    """Display per-protein PCC/SPCC bar plots without saving files.
    Parameters
    ----------
    imputation_results
        Result dictionary returned by :func:`prism_eval_and_save`.
    feature_names
        Protein names in the same order used for evaluation.
    metrics
        Correlation metrics to display. Supported values are ``PCC`` and
        ``SPCC``.
    feature_indices
        Optional fixed feature subset. When omitted, all features are shown.
    sort_by_metric
        Sort the selected features by each plotted metric. Set to ``False``
        to preserve the order supplied by ``feature_indices``.
    colors
        Optional mapping from metric name to bar color.
    figsize
        Optional fixed figure size. By default, the width adapts to the
        number of proteins.
    dpi
        Figure resolution used for notebook display.
    Returns
    -------
    dict
        Mapping from metric name to its ``(figure, axes)`` pair.
    """
    try:
        per_protein_metrics = imputation_results["raw"]["per_protein"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "imputation_results must contain ['raw']['per_protein']."
        ) from exc
    feature_names = np.asarray(feature_names, dtype=str).reshape(-1)
    if feature_names.size == 0:
        raise ValueError("feature_names is empty.")
    selected_indices = None
    if feature_indices is not None:
        selected_indices = np.asarray(feature_indices, dtype=np.int64).reshape(-1)
        if selected_indices.size == 0:
            raise ValueError("feature_indices is empty.")
        if np.any(selected_indices < 0) or np.any(selected_indices >= feature_names.size):
            raise IndexError("feature_indices contains an out-of-range feature index.")
        if np.unique(selected_indices).size != selected_indices.size:
            raise ValueError("feature_indices contains duplicate indices.")
    plot_config = {
        "PCC": {
            "color": "#97d7f1",
            "ylabel": "Pearson correlation across missing cells",
        },
        "SPCC": {
            "color": "#59a7f0",
            "ylabel": "Spearman correlation across missing cells",
        },
    }
    metrics = tuple(metrics)
    unsupported = [name for name in metrics if name not in plot_config]
    if unsupported:
        raise ValueError(
            f"Unsupported correlation metric(s): {unsupported}. "
            "Choose from ('PCC', 'SPCC')."
        )
    if colors is not None and not hasattr(colors, "get"):
        raise TypeError("colors must be a mapping from metric name to color.")
    figures = {}
    for metric_name in metrics:
        if metric_name not in per_protein_metrics:
            raise KeyError(
                f"{metric_name} not found in imputation_results"
                "['raw']['per_protein']."
            )
        values = np.asarray(
            per_protein_metrics[metric_name],
            dtype=float,
        ).reshape(-1)
        if values.size != feature_names.size:
            raise ValueError(
                "Per-protein metric count does not match feature_names: "
                f"{values.size} != {feature_names.size}."
            )
        plot_names = feature_names
        plot_values = values
        if selected_indices is not None:
            plot_names = plot_names[selected_indices]
            plot_values = plot_values[selected_indices]
        finite = np.isfinite(plot_values)
        if not np.any(finite):
            raise ValueError(
                f"No finite per-protein {metric_name} values are available."
            )
        plot_names = plot_names[finite]
        plot_values = plot_values[finite]
        if sort_by_metric:
            order = np.argsort(-plot_values, kind="stable")
            plot_names = plot_names[order]
            plot_values = plot_values[order]
        current_figsize = figsize
        if current_figsize is None:
            current_figsize = (max(9.0, 0.38 * len(plot_names)), 4.8)
        config = plot_config[metric_name]
        color = config["color"] if colors is None else colors.get(
            metric_name,
            config["color"],
        )
        fig, ax = plt.subplots(figsize=current_figsize, dpi=dpi)
        bars = ax.bar(plot_names, plot_values, color=color)
        lower_limit = min(0.0, float(np.min(plot_values)) - 0.05)
        ax.set_ylim(lower_limit, 1.05)
        ax.set_xlabel("")
        ax.set_ylabel(config["ylabel"])
        for bar, value in zip(bars, plot_values):
            offset = 0.02 if value >= 0 else -0.02
            vertical_alignment = "bottom" if value >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.4f}",
                ha="center",
                va=vertical_alignment,
                fontsize=8,
                rotation=90,
            )
        ax.tick_params(axis="x", labelsize=9, rotation=45)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        figures[metric_name] = (fig, ax)
        plt.show()
    return figures
def _force_full_frame_on_top(
    ax,
    linewidth=1.2,
    color="black",
    zorder=300,
):
    """
    Draw a complete black frame around the plotting area.
    This follows the visual style used in the PRISM manuscript figures.
    """
    for side in [
        "top",
        "right",
        "bottom",
        "left",
    ]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(linewidth)
        ax.spines[side].set_color(color)
        ax.spines[side].set_zorder(zorder)
    frame = Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        zorder=zorder + 1,
        clip_on=False,
    )
    ax.add_patch(frame)
    return ax
def _resolve_imputation_metric_axis(
    metric,
    values,
    *,
    metric_limits=None,
    metric_steps=None,
    auto_metric_limits=False,
):
    """
    Resolve plotting limits and major-tick spacing.
    Default style follows the PRISM manuscript plotting code:
        PCC  : approximately [-0.05, 1.05]
        SPCC : approximately [-0.05, 1.05]
        MSE  : approximately [-0.01, 0.15]
    For MSE, the upper limit is automatically expanded when necessary,
    so the function remains usable for other modalities.
    """
    values = np.asarray(
        values,
        dtype=float,
    )
    values = values[
        np.isfinite(values)
    ]
    # ----------------------------------------------------------
    # User-specified limits take highest priority.
    # ----------------------------------------------------------
    if (
        metric_limits is not None
        and metric in metric_limits
        and metric_limits[metric] is not None
    ):
        axis_lim = tuple(
            metric_limits[metric]
        )
    elif auto_metric_limits:
        if values.size == 0:
            axis_lim = (0.0, 1.0)
        else:
            lower = float(np.nanmin(values))
            upper = float(np.nanmax(values))
            span = max(upper - lower, 1e-8)
            minimum_padding = 0.02 if metric in ("PCC", "SPCC") else 0.005
            padding = max(span * 0.10, minimum_padding)
            axis_lim = (lower - padding, upper + padding)
    elif metric in [
        "PCC",
        "SPCC",
    ]:
        axis_lim = (
            -0.05,
            1.05,
        )
    elif metric == "MSE":
        if values.size == 0:
            upper = 0.15
        else:
            observed_max = float(
                np.nanmax(values)
            )
            upper = max(
                0.15,
                observed_max * 1.08,
            )
        axis_lim = (
            -0.01,
            upper,
        )
    else:
        if values.size == 0:
            axis_lim = (
                0.0,
                1.0,
            )
        else:
            vmin = float(
                np.nanmin(values)
            )
            vmax = float(
                np.nanmax(values)
            )
            span = max(
                vmax - vmin,
                1e-8,
            )
            axis_lim = (
                vmin - 0.05 * span,
                vmax + 0.05 * span,
            )
    # ----------------------------------------------------------
    # Tick spacing
    # ----------------------------------------------------------
    if (
        metric_steps is not None
        and metric in metric_steps
    ):
        step = metric_steps[
            metric
        ]
    elif metric in [
        "PCC",
        "SPCC",
    ]:
        step = 0.2
    elif (
        metric == "MSE"
        and axis_lim[1] <= 0.16
    ):
        step = 0.03
    else:
        step = None
    return axis_lim, step
def plot_imputation_metric_boxplot(
    imputation_results,
    feature_names,
    *,
    feature_indices=None,
    feature_label="Feature",
    figure_title=None,
    output_suffix=None,
    metrics=("PCC", "SPCC", "MSE"),
    plot_type="raincloud",
    figsize=None,
    dpi=300,
    metric_colors=None,
    metric_limits=None,
    metric_steps=None,
    auto_metric_limits=False,
    negative_floor_for_plot=0.01,
    point_size=8,
    title_fontsize=11,
    metric_label_fontsize=10,
    tick_fontsize=8,
):
    """
    Display feature-wise imputation metrics in a one-row multi-panel figure.
    Default three-panel figure sizes:
        raincloud: (10.5, 3)
        boxplot:   (10, 5)
    Layout
    ------
                RNA imputation (800 HVGs)
             panel      panel      panel
              PCC        SPCC       MSE
    Supported plot types
    --------------------
    raincloud
        Half violin + boxplot + feature-level points.
    boxplot
        Vertical boxplot + feature-level jittered points.
    Notes
    -----
    This function only displays the figure.
    It does not save PNG/PDF files.
    """
    # ==========================================================
    # 1. Validate plot type
    # ==========================================================
    plot_type = str(
        plot_type
    ).strip().lower()
    if plot_type not in [
        "raincloud",
        "boxplot",
    ]:
        raise ValueError(
            "plot_type must be "
            "'raincloud' or 'boxplot'."
        )
    if figsize is None:
        figsize = (
            (10.5, 3)
            if plot_type == "raincloud"
            else (10, 5)
        )
    # ==========================================================
    # 2. Extract feature-level metrics
    # ==========================================================
    try:
        per_feature = (
            imputation_results[
                "raw"
            ][
                "per_protein"
            ]
        )
    except (
        KeyError,
        TypeError,
    ) as exc:
        raise KeyError(
            "imputation_results must contain "
            "['raw']['per_protein']."
        ) from exc
    names = np.asarray(
        feature_names,
        dtype=str,
    ).reshape(-1)
    if names.size == 0:
        raise ValueError(
            "feature_names is empty."
        )
    # ==========================================================
    # 3. Determine feature subset
    #
    # Priority:
    #   explicit feature_indices
    #       >
    #   raw_topk
    #       >
    #   all features
    # ==========================================================
    auto_topk = False
    topk_view = None
    if (
        feature_indices is None
        and "raw_topk" in imputation_results
    ):
        topk_view = (
            imputation_results[
                "raw_topk"
            ]
        )
        feature_indices = (
            topk_view[
                "topk_idx"
            ]
        )
        auto_topk = True
    if feature_indices is None:
        indices = np.arange(
            names.size,
            dtype=int,
        )
    else:
        indices = np.asarray(
            feature_indices,
            dtype=int,
        ).reshape(-1)
    if indices.size == 0:
        raise ValueError(
            "No features were selected "
            "for visualization."
        )
    if (
        np.any(indices < 0)
        or np.any(
            indices >= names.size
        )
    ):
        raise ValueError(
            "feature_indices contains "
            "invalid feature indices."
        )
    n_features_shown = int(
        indices.size
    )
    # ==========================================================
    # 4. Build figure title
    # ==========================================================
    if figure_title is None:
        # ------------------------------------------------------
        # Modality name
        # ------------------------------------------------------
        if output_suffix is not None:
            modality_name = str(
                output_suffix
            )
        else:
            feature_label_lower = (
                str(feature_label)
                .strip()
                .lower()
            )
            if "rna" in feature_label_lower:
                modality_name = "RNA"
            elif (
                "adt" in feature_label_lower
                or "protein" in feature_label_lower
            ):
                modality_name = "ADT"
            elif "metabol" in feature_label_lower:
                modality_name = "Metabolite"
            elif (
                "atac" in feature_label_lower
                or "peak" in feature_label_lower
            ):
                modality_name = "ATAC"
            else:
                modality_name = str(
                    feature_label
                )
        # ------------------------------------------------------
        # Feature description
        # ------------------------------------------------------
        feature_label_lower = (
            str(feature_label)
            .strip()
            .lower()
        )
        is_hvg_topk = (
            auto_topk
            and topk_view is not None
            and str(
                topk_view.get(
                    "rank_by",
                    "",
                )
            ) == "var_order"
            and "rna" in feature_label_lower
        )
        if is_hvg_topk:
            feature_description = (
                f"{n_features_shown} HVGs"
            )
        elif "rna" in feature_label_lower:
            feature_description = (
                f"{n_features_shown} Genes"
            )
        elif (
            "adt" in feature_label_lower
            or "protein" in feature_label_lower
        ):
            feature_description = (
                f"{n_features_shown} Proteins"
            )
        elif "metabol" in feature_label_lower:
            feature_description = (
                f"{n_features_shown} Metabolites"
            )
        elif (
            "atac" in feature_label_lower
            or "peak" in feature_label_lower
        ):
            feature_description = (
                f"{n_features_shown} Peaks"
            )
        else:
            feature_description = (
                f"{n_features_shown} Features"
            )
        figure_title = (
            f"{modality_name} imputation "
            f"({feature_description})"
        )
    # ==========================================================
    # 5. Metric colors
    # ==========================================================
    default_colors = {
        "PCC": "#7678c1",
        "SPCC": "#5eb5cf",
        "MSE": "#de3c3b",
    }
    if metric_colors is None:
        metric_colors = (
            default_colors.copy()
        )
    else:
        merged_colors = (
            default_colors.copy()
        )
        merged_colors.update(
            metric_colors
        )
        metric_colors = (
            merged_colors
        )
    metrics = tuple(
        metrics
    )
    # ==========================================================
    # 6. Collect metric values
    # ==========================================================
    metric_values = {}
    for metric in metrics:
        if metric not in per_feature:
            raise KeyError(
                f"{metric} not found in "
                "imputation_results"
                "['raw']['per_protein']."
            )
        values = np.asarray(
            per_feature[
                metric
            ],
            dtype=float,
        ).reshape(-1)
        if values.size != names.size:
            raise ValueError(
                f"{metric} metric count "
                "does not match feature_names: "
                f"{values.size} != {names.size}."
            )
        values = values[
            indices
        ]
        values = values[
            np.isfinite(values)
        ]
        if values.size == 0:
            raise ValueError(
                f"No finite {metric} "
                "values are available."
            )
        metric_values[
            metric
        ] = values
    # ==========================================================
    # 7. Figure
    #
    # Same size as domain three-panel visualization:
    #     3 × 3.5 = 10.5
    #     height = 3.0
    # ==========================================================
    n_metrics = len(
        metrics
    )
    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )
    axes = axes.ravel()
    fig.patch.set_facecolor(
        "white"
    )
    fig.suptitle(
        figure_title,
        fontsize=title_fontsize,
        y=0.95,
        ha="center",
    )
    # ==========================================================
    # 8. Draw each metric
    # ==========================================================
    for col_idx, (
        ax,
        metric,
    ) in enumerate(
        zip(
            axes,
            metrics,
        )
    ):
        original_values = (
            metric_values[
                metric
            ].copy()
        )
        color = metric_colors.get(
            metric,
            "#777777",
        )
        axis_lim, axis_step = (
            _resolve_imputation_metric_axis(
                metric,
                original_values,
                metric_limits=metric_limits,
                metric_steps=metric_steps,
                auto_metric_limits=auto_metric_limits,
            )
        )
        # ======================================================
        # Raincloud
        # ======================================================
        if plot_type == "raincloud":
            plot_values = (
                original_values.copy()
            )
            if (
                negative_floor_for_plot
                is not None
                and metric in [
                    "PCC",
                    "SPCC",
                ]
            ):
                plot_values = np.where(
                    plot_values < 0,
                    float(
                negative_floor_for_plot
                    ),
                    plot_values,
                )
            # Dummy group only for seaborn.
            # No group name is displayed.
            plot_df = pd.DataFrame(
                {
                    "MetricValue": plot_values,
                    "Group": (
                        [""] * len(plot_values)
                    ),
                }
            )
            # --------------------------------------------------
            # Violin
            # --------------------------------------------------
            sns.violinplot(
                data=plot_df,
                x="MetricValue",
                y="Group",
                orient="h",
                color=color,
                cut=0,
                inner=None,
                density_norm="width",
                linewidth=0,
                width=0.85,
                ax=ax,
                zorder=20,
            )
            xmin, xmax = axis_lim
            # --------------------------------------------------
            # Half-violin mask
            # --------------------------------------------------
            ax.add_patch(
                Rectangle(
                    (
                        xmin,
                        0.0,
                    ),
                    xmax - xmin,
                    0.5,
                    facecolor=(
                        ax.get_facecolor()
                    ),
                    edgecolor="none",
                    zorder=30,
                )
            )
            # --------------------------------------------------
            # Box
            # --------------------------------------------------
            ax.boxplot(
                [plot_values],
                positions=[0.12],
                vert=False,
                widths=0.20,
                patch_artist=True,
                showfliers=False,
                zorder=50,
                boxprops=dict(
                    facecolor=color,
                    alpha=0.60,
                    edgecolor="black",
                    linewidth=1.1,
                ),
                medianprops=dict(
                    color="black",
                    linewidth=1.6,
                ),
                whiskerprops=dict(
                    color="black",
                    linewidth=1.0,
                ),
                capprops=dict(
                    color="black",
                    linewidth=1.0,
                ),
            )
            # --------------------------------------------------
            # Rain points
            # --------------------------------------------------
            rng = np.random.default_rng(
                2024 + col_idx
            )
            y_pts = (
                0.36
                + rng.uniform(
                    -0.055,
                    0.055,
                    size=len(
                        plot_values
                    ),
                )
            )
            ax.scatter(
                plot_values,
                y_pts,
                s=point_size,
                color=color,
                alpha=0.75,
                linewidth=0,
                zorder=55,
            )
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.set_xlim(
                axis_lim
            )
            if axis_step is not None:
                ax.xaxis.set_major_locator(
                    MultipleLocator(
                        axis_step
                    )
                )
            # Metric name only at bottom.
            ax.set_xlabel(
                metric,
                fontsize=(
                    metric_label_fontsize
                ),
            )
            ax.tick_params(
                axis="x",
                labelsize=tick_fontsize,
            )
        # ======================================================
        # Boxplot
        # ======================================================
        else:
            plot_values = (
                original_values.copy()
            )
            ax.boxplot(
                [plot_values],
                positions=[0],
                vert=True,
                widths=0.45,
                patch_artist=True,
                showfliers=False,
                zorder=40,
                boxprops=dict(
                    facecolor=color,
                    alpha=0.75,
                    edgecolor="black",
                    linewidth=1.2,
                ),
                medianprops=dict(
                    color="black",
                    linewidth=1.6,
                ),
                whiskerprops=dict(
                    color="black",
                    linewidth=1.0,
                ),
                capprops=dict(
                    color="black",
                    linewidth=1.0,
                ),
            )
            rng = np.random.default_rng(
                2024 + col_idx
            )
            x_pts = rng.normal(
                loc=0.0,
                scale=0.045,
                size=len(
                    plot_values
                ),
            )
            ax.scatter(
                x_pts,
                plot_values,
                s=max(
                    float(point_size),
                    5.0,
                ),
                color=color,
                edgecolors="black",
                linewidths=0.4,
                alpha=0.85,
                zorder=50,
            )
            ax.set_ylim(
                axis_lim
            )
            if axis_step is not None:
                ax.yaxis.set_major_locator(
                    MultipleLocator(
                        axis_step
                    )
                )
            # Metric name only at bottom.
            ax.set_xticks([0])
            ax.set_xticklabels(
                [metric],
                fontsize=(
                    metric_label_fontsize
                ),
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(
                axis="y",
                labelsize=tick_fontsize,
            )
        # No individual top title.
        ax.set_title("")
        _force_full_frame_on_top(
            ax,
            linewidth=1.2,
            color="black",
        )
    # ==========================================================
    # 9. Compact layout for the shared 10 × 5 tutorial figure.
    # ==========================================================
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.16,
        top=0.87,
        wspace=0.22,
    )
    plt.show()
    return fig, axes
def _normalize_true_pred_separately_01_by_feature(
    true_mat,
    pred_mat,
    fit_indices,
    *,
    clip=True,
    eps=1e-8,
):
    """
    Normalize true and pred matrices separately to [0, 1], feature-wise.
    Used only for metric calculation.
    """
    true_mat = np.asarray(true_mat, dtype=np.float32)
    pred_mat = np.asarray(pred_mat, dtype=np.float32)
    if true_mat.shape != pred_mat.shape:
        raise ValueError(f"true_mat shape {true_mat.shape} != pred_mat shape {pred_mat.shape}")
    fit_indices = np.asarray(fit_indices, dtype=int)
    if fit_indices.size == 0:
        raise ValueError("fit_indices is empty.")
    n_features = true_mat.shape[1]
    true_min = np.zeros(n_features, dtype=np.float32)
    true_max = np.ones(n_features, dtype=np.float32)
    pred_min = np.zeros(n_features, dtype=np.float32)
    pred_max = np.ones(n_features, dtype=np.float32)
    for j in range(n_features):
        xt = true_mat[fit_indices, j]
        xp = pred_mat[fit_indices, j]
        xt = xt[np.isfinite(xt)]
        xp = xp[np.isfinite(xp)]
        if xt.size == 0:
            true_min[j], true_max[j] = 0.0, 1.0
        else:
            true_min[j] = float(np.min(xt))
            true_max[j] = float(np.max(xt))
        if xp.size == 0:
            pred_min[j], pred_max[j] = 0.0, 1.0
        else:
            pred_min[j] = float(np.min(xp))
            pred_max[j] = float(np.max(xp))
        if true_max[j] <= true_min[j]:
            true_max[j] = true_min[j] + 1.0
        if pred_max[j] <= pred_min[j]:
            pred_max[j] = pred_min[j] + 1.0
    true_denom = np.maximum(true_max - true_min, eps)
    pred_denom = np.maximum(pred_max - pred_min, eps)
    true_01 = (true_mat - true_min[None, :]) / true_denom[None, :]
    pred_01 = (pred_mat - pred_min[None, :]) / pred_denom[None, :]
    if clip:
        true_01 = np.clip(true_01, 0.0, 1.0)
        pred_01 = np.clip(pred_01, 0.0, 1.0)
    return true_01.astype(np.float32), pred_01.astype(np.float32)
# ==========================================================
# HVG / Top-K utilities
# ==========================================================
def _bool_var_column_to_numpy(s):
    """Convert adata.var boolean-like column to bool numpy array."""
    if s.dtype == bool:
        return s.to_numpy(dtype=bool)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"]).to_numpy(dtype=bool)
def _get_hvg_candidate_mask(adata):
    """
    Build candidate mask from existing HVG/selected columns.
    If none exists or none is True, all features are candidates.
    """
    candidate_cols = ["selected_feature", "highly_variable", "highly_variable_raw"]
    mask = None
    for col in candidate_cols:
        if col in adata.var.columns:
            m = _bool_var_column_to_numpy(adata.var[col])
            if m.any():
                mask = m if mask is None else (mask | m)
    if mask is None:
        mask = np.ones(adata.n_vars, dtype=bool)
    return mask
def _compute_hvg_order_from_matrix(
    X,
    *,
    candidate_mask=None,
    target_sum=1e4,
    chunk_size=512,
    eps=1e-8,
    score_method="log_norm_var",
):
    """
    Recompute HVG-like ordering from the current raw feature matrix.
    Default score:
        normalize_total(target_sum) + log1p, then feature-wise variance.
    """
    if score_method not in ("log_norm_var", "log_norm_dispersion", "raw_var"):
        raise ValueError("score_method must be 'log_norm_var', 'log_norm_dispersion', or 'raw_var'.")
    n_features = X.shape[1]
    if candidate_mask is None:
        candidate_mask = np.ones(n_features, dtype=bool)
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if candidate_mask.shape[0] != n_features:
        raise ValueError(f"candidate_mask length {candidate_mask.shape[0]} != n_features {n_features}")
    if sp.issparse(X):
        cell_total = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    else:
        cell_total = np.asarray(X, dtype=np.float32).sum(axis=1).astype(np.float32)
    cell_total = np.clip(cell_total, 1.0, None)
    scale = (float(target_sum) / cell_total).astype(np.float32)
    scores = np.full(n_features, -np.inf, dtype=np.float32)
    for start in range(0, n_features, int(chunk_size)):
        end = min(start + int(chunk_size), n_features)
        if sp.issparse(X):
            Xc = X[:, start:end].toarray().astype(np.float32)
        else:
            Xc = np.asarray(X[:, start:end], dtype=np.float32)
        if score_method == "raw_var":
            Z = Xc
        else:
            Z = np.log1p(np.maximum(Xc, 0.0) * scale[:, None]).astype(np.float32)
        mean = np.nanmean(Z, axis=0)
        var = np.nanvar(Z, axis=0)
        if score_method == "log_norm_dispersion":
            chunk_score = var / np.maximum(mean, eps)
        else:
            chunk_score = var
        scores[start:end] = chunk_score.astype(np.float32)
    scores_for_sort = scores.copy()
    scores_for_sort[~candidate_mask] = -np.inf
    candidate_idx = np.where(candidate_mask & np.isfinite(scores_for_sort))[0]
    non_candidate_idx = np.where(~candidate_mask | ~np.isfinite(scores_for_sort))[0]
    candidate_order = candidate_idx[np.argsort(-scores_for_sort[candidate_idx], kind="mergesort")]
    order = np.concatenate([candidate_order, non_candidate_idx]).astype(int)
    return order, scores
def _get_hvg_var_order_indices(
    adata,
    *,
    hvg_matrix=None,
    raw_layer_key="raw_data",
    target_sum=1e4,
    chunk_size=512,
    score_method="log_norm_var",
    verbose=True,
):
    """
    Return feature indices ordered by HVG priority.
    If no rank/score column exists, recompute an internal HVG-like ranking.
    """
    n_features = adata.n_vars
    var = adata.var
    rank_cols = ["highly_variable_rank", "hvg_rank", "hvm_rank", "hv_rank"]
    score_cols = ["dispersions_norm", "variances_norm", "highly_variable_score", "hvg_score"]
    candidate_mask = _get_hvg_candidate_mask(adata)
    for col in rank_cols:
        if col in var.columns:
            rank = pd.to_numeric(var[col], errors="coerce").to_numpy()
            finite = np.isfinite(rank)
            if finite.sum() > 0:
                valid_idx = np.where(finite & candidate_mask)[0]
                invalid_idx = np.setdiff1d(np.arange(n_features), valid_idx)
                valid_order = valid_idx[np.argsort(rank[valid_idx], kind="mergesort")]
                order = np.concatenate([valid_order, invalid_idx]).astype(int)
                if verbose:
                    print(f"Feature ranking: {col}")
                return order
    for col in score_cols:
        if col in var.columns:
            score = pd.to_numeric(var[col], errors="coerce").to_numpy()
            finite = np.isfinite(score)
            if finite.sum() > 0:
                valid_idx = np.where(finite & candidate_mask)[0]
                invalid_idx = np.setdiff1d(np.arange(n_features), valid_idx)
                valid_order = valid_idx[np.argsort(-score[valid_idx], kind="mergesort")]
                order = np.concatenate([valid_order, invalid_idx]).astype(int)
                if verbose:
                    print(f"Feature ranking: {col}")
                return order
    if hvg_matrix is None:
        hvg_matrix = adata.layers[raw_layer_key] if raw_layer_key in adata.layers else adata.X
    order, scores = _compute_hvg_order_from_matrix(
        hvg_matrix,
        candidate_mask=candidate_mask,
        target_sum=target_sum,
        chunk_size=chunk_size,
        score_method=score_method,
    )
    adata.var["prism_hvg_score"] = scores
    adata.var["prism_hvg_rank"] = np.empty(n_features, dtype=np.int64)
    adata.var.iloc[order, adata.var.columns.get_loc("prism_hvg_rank")] = np.arange(n_features, dtype=np.int64)
    if verbose:
        print(f"Feature ranking: internal HVG score ({score_method})")
    return order
def _topk_feature_view(
    metrics_dict,
    var_names,
    *,
    k=800,
    rank_by="var_order",
    adata=None,
    hvg_matrix=None,
    raw_layer_key="raw_data",
    target_sum=1e4,
    hvg_chunk_size=512,
    hvg_score_method="log_norm_var",
):
    """Select top-k features and summarize already-computed per-feature metrics."""
    valid_rank_metrics = ("var_order", "PCC", "SPCC", "MSE", "CMD", "SSIM")
    if rank_by not in valid_rank_metrics:
        raise ValueError(f"rank_by must be one of {valid_rank_metrics}.")
    var_names_arr = np.asarray(var_names, dtype=str)
    n_features = len(var_names_arr)
    k = int(min(k, n_features))
    if k <= 0:
        raise ValueError("k must be > 0.")
    if rank_by == "var_order":
        if adata is None:
            raise ValueError("rank_by='var_order' requires adata.")
        order = _get_hvg_var_order_indices(
            adata,
            hvg_matrix=hvg_matrix,
            raw_layer_key=raw_layer_key,
            target_sum=target_sum,
            chunk_size=hvg_chunk_size,
            score_method=hvg_score_method,
            verbose=False,
        )
        topk_idx = np.asarray(order[:k], dtype=int)
    else:
        if rank_by not in metrics_dict["per_protein"]:
            raise KeyError(
                f"{rank_by} not found in metrics_dict['per_protein']. "
                f"Available metrics: {list(metrics_dict['per_protein'].keys())}"
            )
        scores = np.asarray(metrics_dict["per_protein"][rank_by], dtype=np.float32)
        if rank_by in ("MSE", "CMD"):
            scores_sort = np.where(np.isfinite(scores), scores, np.inf)
            topk_idx = np.argsort(scores_sort, kind="mergesort")[:k]
        else:
            scores_sort = np.where(np.isfinite(scores), scores, -np.inf)
            topk_idx = np.argsort(-scores_sort, kind="mergesort")[:k]
    overall = {}
    for m in ["PCC", "SPCC", "MSE", "CMD", "SSIM"]:
        if m in metrics_dict["per_protein"]:
            vals = np.asarray(metrics_dict["per_protein"][m], dtype=np.float32)
            overall[m] = float(np.nanmean(vals[topk_idx]))
    return {
        "overall": overall,
        "topk_idx": topk_idx,
        "topk_names": list(var_names_arr[topk_idx]),
        "k": k,
        "rank_by": rank_by,
    }
def _save_feature_metric_summary(save_path, eval_prefix, tag, metrics_dict, var_names):
    """
    Save per-feature metrics.
    Metric values are rounded to 4 decimal places.
    """
    data = {
        "feature": np.asarray(var_names, dtype=str),
    }
    for m in ["PCC", "SPCC", "MSE", "CMD", "SSIM"]:
        if m in metrics_dict["per_protein"]:
            vals = np.asarray(metrics_dict["per_protein"][m], dtype=np.float32)
            data[m] = np.round(vals, 4)
    out_csv = os.path.join(save_path, f"{eval_prefix}_{tag}_feature_metrics.csv")
    pd.DataFrame(data).to_csv(out_csv, index=False)
    return out_csv
# ==========================================================
# Simulated missing visualization
# ==========================================================
def _normalize_single_fullmap_01(
    values,
    *,
    clip=True,
    eps=1e-8,
    clip_quantiles=None,
):
    """
    Normalize one full spatial map to [0, 1].
    This is used for visualization only.
    Logic:
        One panel uses its own full-slice raw values to fit the display scaler.
    If clip_quantiles is not None, the scaler is fitted by quantiles,
    e.g. clip_quantiles=(0.01, 0.99).
    """
    values = np.asarray(values, dtype=np.float32).copy()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        if clip_quantiles is None:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        else:
            q_low, q_high = clip_quantiles
            vmin = float(np.quantile(finite, q_low))
            vmax = float(np.quantile(finite, q_high))
            if vmax <= vmin:
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    z = (values - vmin) / max(vmax - vmin, eps)
    if clip:
        z = np.clip(z, 0.0, 1.0)
    return z.astype(np.float32), vmin, vmax
def _normalize_pair_fullmap_01(
    v_true,
    v_pred,
    *,
    clip=True,
    eps=1e-8,
    observed_indices=None,
    clip_quantiles=None,
):
    """
    Panel-wise full-map visualization normalization.
    Function name is kept for backward compatibility.
    New visualization logic:
        True/reference map:
            normalize by its own full-slice values.
        PRISM imputed full map:
            observed/non-missing region = true raw
            missing region = PRISM imputed raw
            normalize by this imputed full map itself.
        Then:
            overwrite the observed/non-missing region of PRISM display
            with the normalized True/reference display values.
    This avoids the visual ambiguity that the observed region in the PRISM
    panel looks slightly different from the True/reference panel only because
    it was normalized together with imputed values.
    """
    v_true = np.asarray(v_true, dtype=np.float32).copy()
    v_pred = np.asarray(v_pred, dtype=np.float32).copy()
    v_true_01, true_vmin, true_vmax = _normalize_single_fullmap_01(
        v_true,
        clip=clip,
        eps=eps,
        clip_quantiles=clip_quantiles,
    )
    v_pred_01, pred_vmin, pred_vmax = _normalize_single_fullmap_01(
        v_pred,
        clip=clip,
        eps=eps,
        clip_quantiles=clip_quantiles,
    )
    if observed_indices is not None:
        obs = np.asarray(observed_indices, dtype=int)
        obs = obs[(obs >= 0) & (obs < len(v_pred_01))]
        v_pred_01[obs] = v_true_01[obs]
    raw_vmin = {
        "true_or_reference": true_vmin,
        "prism_imputed": pred_vmin,
    }
    raw_vmax = {
        "true_or_reference": true_vmax,
        "prism_imputed": pred_vmax,
    }
    return v_true_01.astype(np.float32), v_pred_01.astype(np.float32), raw_vmin, raw_vmax
def _normalize_spatial_raw_display_01(
    v_true,
    v_pred,
    *,
    clip=True,
    eps=1e-8,
    observed_indices=None,
    clip_quantiles=None,
):
    """
    Backward-compatible alias for raw full-map 0/1 visualization normalization.
    Current logic:
        panel-wise normalization, not shared normalization.
    """
    return _normalize_pair_fullmap_01(
        v_true,
        v_pred,
        clip=clip,
        eps=eps,
        observed_indices=observed_indices,
        clip_quantiles=clip_quantiles,
    )
def _resolve_continuous_cmap(cmap_name):
    if cmap_name == "Tropic_7":
        try:
            from palettable.cartocolors.diverging import Tropic_7
            return Tropic_7.mpl_colormap
        except ImportError:
            return plt.get_cmap("viridis")
    return plt.get_cmap(cmap_name)
# ==========================================================
# Real missing visualization utilities
# ==========================================================
def _get_feature_index(adata, feature):
    """Resolve feature index and name."""
    if isinstance(feature, int):
        if feature < 0 or feature >= adata.n_vars:
            raise IndexError(f"feature index {feature} out of range (0..{adata.n_vars - 1}).")
        feat_key = str(adata.var_names[feature])
        return int(feature), feat_key, feat_key
    if isinstance(feature, str):
        if feature in adata.var_names:
            idx = int(np.where(adata.var_names == feature)[0][0])
            feat_key = str(adata.var_names[idx])
            return idx, feat_key, feat_key
        lower_map = {str(v).lower(): i for i, v in enumerate(adata.var_names)}
        key = feature.strip().lower()
        if key in lower_map:
            idx = int(lower_map[key])
            feat_key = str(adata.var_names[idx])
            return idx, feat_key, feat_key
        raise KeyError(
            f"Feature '{feature}' not found in adata.var_names. "
            f"Example names: {list(map(str, adata.var_names[:10]))}"
        )
    raise TypeError("feature must be int or str.")
def _estimate_missing_total_counts_knn(
    coords: np.ndarray,
    total_counts: np.ndarray,
    missing_idx: np.ndarray,
    observed_idx: np.ndarray,
    *,
    k: int = 30,
    eps: float = 1e-6,
):
    """Estimate missing total counts via spatial kNN weighted average among observed cells."""
    coords = np.asarray(coords, dtype=np.float32)
    tc = np.asarray(total_counts, dtype=np.float32)
    obs_coords = coords[observed_idx]
    miss_coords = coords[missing_idx]
    if obs_coords.shape[0] == 0:
        return np.zeros(len(missing_idx), dtype=np.float32)
    k_eff = min(k, obs_coords.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(obs_coords)
    dist, nbr = nn.kneighbors(miss_coords, return_distance=True)
    w = 1.0 / (dist + eps)
    nbr_tc = tc[observed_idx[nbr]]
    est = (w * nbr_tc).sum(axis=1) / np.clip(w.sum(axis=1), a_min=eps, a_max=None)
    return est.astype(np.float32)
def _topk_observed_from_prior_strict(
    prior_matrix,
    query_indices,
    candidate_indices,
    *,
    k: int,
    matrix_type: str = "distance",
    exclude_self: bool = True,
):
    """Strictly select top-k candidate neighbors from a prior matrix."""
    query_indices = np.asarray(query_indices, dtype=np.int64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    k = int(k)
    if k <= 0:
        raise ValueError("k must be > 0.")
    if query_indices.size == 0:
        return np.empty((0, k), dtype=np.int64), np.empty((0, k), dtype=np.float32)
    if candidate_indices.size < k:
        raise ValueError(f"Only {candidate_indices.size} valid observed candidates, but k={k} was requested.")
    if matrix_type not in ("distance", "similarity"):
        raise ValueError("matrix_type must be 'distance' or 'similarity'.")
    if prior_matrix.shape[0] != prior_matrix.shape[1]:
        raise ValueError(f"prior_matrix must be square, got {prior_matrix.shape}.")
    n = prior_matrix.shape[0]
    if np.any(query_indices < 0) or np.any(query_indices >= n):
        raise ValueError("query_indices contains out-of-range indices.")
    if np.any(candidate_indices < 0) or np.any(candidate_indices >= n):
        raise ValueError("candidate_indices contains out-of-range indices.")
    candidate_mask = np.zeros(n, dtype=bool)
    candidate_mask[candidate_indices] = True
    topk_idx = np.empty((query_indices.size, k), dtype=np.int64)
    topk_score = np.empty((query_indices.size, k), dtype=np.float32)
    if sp.issparse(prior_matrix):
        D = prior_matrix.tocsr(copy=True)
        D.eliminate_zeros()
        valid_counts = np.zeros(query_indices.size, dtype=np.int64)
        for r, i in enumerate(query_indices):
            start, end = D.indptr[i], D.indptr[i + 1]
            cols = D.indices[start:end]
            vals = D.data[start:end].astype(np.float32)
            keep = candidate_mask[cols] & np.isfinite(vals)
            if exclude_self:
                keep = keep & (cols != i)
            cols = cols[keep]
            vals = vals[keep]
            valid_counts[r] = cols.size
            if cols.size < k:
                continue
            order = np.argsort(vals, kind="mergesort")[:k] if matrix_type == "distance" else np.argsort(-vals, kind="mergesort")[:k]
            topk_idx[r, :] = cols[order]
            topk_score[r, :] = vals[order]
        bad = np.where(valid_counts < k)[0]
        if bad.size > 0:
            examples = query_indices[bad[:10]]
            counts = valid_counts[bad[:10]]
            raise ValueError(
                f"{bad.size} missing cells have fewer than k={k} valid observed prior neighbors. "
                f"Example cells: {examples.tolist()}, valid counts: {counts.tolist()}. "
                "Please reduce prior_k or save more neighbors in the prior matrix."
            )
        return topk_idx, topk_score
    D = np.asarray(prior_matrix, dtype=np.float32)
    for r, i in enumerate(query_indices):
        vals = D[i, candidate_indices].copy()
        valid = np.isfinite(vals) & (vals > 0) if matrix_type == "distance" else np.isfinite(vals)
        if exclude_self:
            self_pos = np.where(candidate_indices == i)[0]
            if self_pos.size > 0:
                valid[self_pos] = False
        if valid.sum() < k:
            raise ValueError(
                f"Cell {int(i)} has only {int(valid.sum())} valid observed prior neighbors, "
                f"but k={k} was requested. Use sparse CSR prior, reduce prior_k, or save more neighbors."
            )
        cand = candidate_indices[valid]
        score = vals[valid]
        order = np.argsort(score, kind="mergesort")[:k] if matrix_type == "distance" else np.argsort(-score, kind="mergesort")[:k]
        topk_idx[r, :] = cand[order]
        topk_score[r, :] = score[order]
    return topk_idx, topk_score
def _estimate_missing_total_counts_prior(
    *,
    prior_matrix,
    total_counts,
    missing_idx,
    observed_idx,
    k: int = 3,
    matrix_type: str = "distance",
):
    """Estimate raw total counts for real-missing cells using PRISM prior."""
    missing_idx = np.asarray(missing_idx, dtype=np.int64)
    observed_idx = np.asarray(observed_idx, dtype=np.int64)
    tc = np.asarray(total_counts, dtype=np.float32).ravel()
    if missing_idx.size == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty((0, int(k)), dtype=np.int64),
            np.empty((0, int(k)), dtype=np.float32),
        )
    obs_tc = tc[observed_idx]
    valid_obs_mask = np.isfinite(obs_tc) & (obs_tc > 0)
    observed_valid_idx = observed_idx[valid_obs_mask]
    if observed_valid_idx.size < int(k):
        raise ValueError(
            f"Only {observed_valid_idx.size} observed cells have positive raw_total_counts, "
            f"but prior_k={k} was requested."
        )
    topk_idx, topk_score = _topk_observed_from_prior_strict(
        prior_matrix=prior_matrix,
        query_indices=missing_idx,
        candidate_indices=observed_valid_idx,
        k=k,
        matrix_type=matrix_type,
        exclude_self=True,
    )
    neighbor_tc = tc[topk_idx]
    if (not np.isfinite(neighbor_tc).all()) or np.any(neighbor_tc <= 0):
        raise ValueError("Invalid neighbor total_counts found after strict prior selection.")
    estimated_tc = neighbor_tc.mean(axis=1).astype(np.float32)
    return estimated_tc, topk_idx, topk_score
def _make_raw_total_counts_safe(
    adata_aligned,
    missing_indices: np.ndarray,
    observed_indices: np.ndarray,
    *,
    scale_method: str = "prior_tc",
    prior_matrix=None,
    prior_k: int = 5,
    prior_matrix_type: str = "distance",
    knn_k: int = 30,
    knn_eps: float = 1e-6,
    raw_total_key: str = "raw_total_counts",
    raw_layer_key: str = "raw_data",
    store_estimated_key: str = "estimated_raw_total_counts",
    store_neighbor_key: str = "estimated_raw_total_count_neighbors",
    verbose: bool = True,
):
    """Return valid total counts for inverse normalize_total."""
    missing_indices = np.asarray(missing_indices, dtype=np.int64)
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    if raw_total_key in adata_aligned.obs.columns:
        tc = adata_aligned.obs[raw_total_key].to_numpy().astype(np.float32)
    else:
        if raw_layer_key not in adata_aligned.layers:
            raise KeyError(
                f"Need adata.obs['{raw_total_key}'] or adata.layers['{raw_layer_key}'] "
                "to invert prediction to raw scale."
            )
        raw_X = adata_aligned.layers[raw_layer_key]
        tc = np.asarray(raw_X.sum(axis=1)).ravel().astype(np.float32) if sp.issparse(raw_X) else np.asarray(raw_X).sum(axis=1).astype(np.float32)
    if tc.shape[0] != adata_aligned.n_obs:
        raise ValueError(f"total_counts length mismatch: {tc.shape[0]} vs adata.n_obs={adata_aligned.n_obs}.")
    tc_safe = tc.copy()
    miss_bad_mask = (~np.isfinite(tc_safe[missing_indices])) | (tc_safe[missing_indices] <= 0)
    miss_bad = missing_indices[miss_bad_mask]
    if len(miss_bad) == 0:
        adata_aligned.obs[store_estimated_key] = tc_safe.astype(np.float32)
        if verbose:
            print("Total-count estimation: not required")
        return tc_safe
    obs_tc = tc_safe[observed_indices]
    obs_valid = np.isfinite(obs_tc) & (obs_tc > 0)
    if not np.any(obs_valid):
        raise ValueError("No observed cells have positive finite raw_total_counts.")
    observed_valid = observed_indices[obs_valid]
    observed_valid_tc = tc_safe[observed_valid]
    if scale_method == "prior_tc":
        if prior_matrix is None:
            raise ValueError("scale_method='prior_tc' requires prior_matrix.")
        estimated_tc, topk_idx, topk_score = _estimate_missing_total_counts_prior(
            prior_matrix=prior_matrix,
            total_counts=tc_safe,
            missing_idx=miss_bad,
            observed_idx=observed_valid,
            k=prior_k,
            matrix_type=prior_matrix_type,
        )
        tc_safe[miss_bad] = estimated_tc
        neighbor_record = np.full((adata_aligned.n_obs, int(prior_k)), -1, dtype=np.int64)
        neighbor_record[miss_bad, :] = topk_idx
        adata_aligned.obsm[store_neighbor_key] = neighbor_record
    elif scale_method == "median":
        med = float(np.median(observed_valid_tc))
        tc_safe[miss_bad] = med
    elif scale_method == "knn_tc":
        if knn_k is None or int(knn_k) <= 0:
            raise ValueError("scale_method='knn_tc' requires knn_k > 0.")
        if "spatial" not in adata_aligned.obsm:
            raise KeyError("scale_method='knn_tc' requires adata_aligned.obsm['spatial'].")
        estimated_tc = _estimate_missing_total_counts_knn(
            coords=np.asarray(adata_aligned.obsm["spatial"], dtype=np.float32),
            total_counts=tc_safe,
            missing_idx=miss_bad,
            observed_idx=observed_valid,
            k=int(knn_k),
            eps=knn_eps,
        )
        if (not np.isfinite(estimated_tc).all()) or np.any(estimated_tc <= 0):
            raise ValueError("spatial kNN total-count estimation produced invalid values.")
        tc_safe[miss_bad] = estimated_tc
    else:
        raise ValueError("scale_method must be 'prior_tc', 'knn_tc', or 'median'.")
    bad_after = (~np.isfinite(tc_safe[missing_indices])) | (tc_safe[missing_indices] <= 0)
    if np.any(bad_after):
        bad_cells = missing_indices[bad_after][:10]
        raise ValueError(
            f"{int(np.sum(bad_after))} missing cells still have invalid total counts. "
            f"Examples: {bad_cells.tolist()}."
        )
    adata_aligned.obs[store_estimated_key] = tc_safe.astype(np.float32)
    if verbose:
        print(
            f"Total-count estimation: method={scale_method}, "
            f"estimated cells={len(miss_bad)}, observed candidates={len(observed_valid)}"
        )
    return tc_safe
def _prediction_key_and_features(adata, prediction_key: Optional[str] = None):
    """Return a PRISM prediction matrix and its feature names from AnnData."""
    if prediction_key is None:
        for candidate in ("PRISM_tgt_pred", "PRISM_src_pred"):
            if candidate in adata.obsm:
                prediction_key = candidate
                break
    if prediction_key is None or prediction_key not in adata.obsm:
        raise KeyError(
            "No PRISM prediction was found in adata.obsm. "
            "Run PRISM training before raw-scale completion."
        )
    var_key = (
        "PRISM_tgt_pred_var_names"
        if prediction_key == "PRISM_tgt_pred"
        else "PRISM_src_pred_var_names"
    )
    if var_key not in adata.uns:
        raise KeyError(
            f"adata.uns['{var_key}'] is required to identify prediction features."
        )
    prediction = _to_dense_f32(adata.obsm[prediction_key])
    feature_names = pd.Index(np.asarray(adata.uns[var_key]).astype(str))
    if prediction.shape[0] != adata.n_obs:
        raise ValueError(
            f"Prediction rows ({prediction.shape[0]}) do not match adata.n_obs ({adata.n_obs})."
        )
    if prediction.shape[1] != len(feature_names):
        raise ValueError(
            "Prediction columns do not match the stored PRISM feature names."
        )
    if not feature_names.is_unique:
        raise ValueError("PRISM prediction feature names must be unique.")
    available = pd.Index(adata.var_names.astype(str))
    missing_features = feature_names.difference(available)
    if len(missing_features) > 0:
        raise KeyError(
            "Prediction features are not present in adata.var_names: "
            f"{missing_features[:5].tolist()}"
        )
    return prediction, feature_names, prediction_key
def _resolve_missing_indices(
    adata,
    *,
    missing_indices=None,
    missing_key: str = "missing",
    missing_value: str = "0",
):
    if missing_indices is None:
        if missing_key not in adata.obs:
            raise KeyError(
                f"adata.obs['{missing_key}'] is required when missing_indices is not provided."
            )
        missing_indices = np.flatnonzero(
            adata.obs[missing_key].astype(str).to_numpy() == str(missing_value)
        )
    missing_indices = np.asarray(missing_indices, dtype=int)
    if missing_indices.ndim != 1:
        raise ValueError("missing_indices must be a one-dimensional integer array.")
    if np.any(missing_indices < 0) or np.any(missing_indices >= adata.n_obs):
        raise IndexError("missing_indices contains out-of-range observations.")
    return np.unique(missing_indices)
def _raw_matrix_like(template, values):
    values = np.asarray(values, dtype=np.float32)
    return sp.csr_matrix(values) if sp.issparse(template) else values
def materialize_prism_raw_imputation(
    adata,
    *,
    missing_indices=None,
    missing_key: str = "missing",
    missing_value: str = "0",
    prediction_key: Optional[str] = None,
    raw_layer_key: str = "raw_data",
    raw_total_key: str = "raw_total_counts",
    target_sum: float = 1e4,
    scale_method: str = "auto",
    prior_matrix=None,
    prior_k: int = 5,
    prior_matrix_type: str = "distance",
    knn_k: int = 30,
    knn_eps: float = 1e-6,
    output_h5ad: Optional[Union[str, Path]] = None,
    save_raw_missing_csv: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    file_prefix: Optional[str] = None,
):
    """Build a complete raw-scale AnnData directly from the in-memory PRISM output.
    ``.X`` of the returned object contains the observed raw matrix with only
    missing observations replaced by PRISM raw-scale predictions.  The original
    raw matrix remains untouched in ``layers[raw_layer_key]``.
    ``scale_method='stored'`` preserves the recorded preprocessing totals,
    including legitimate zero-total simulated rows.  ``scale_method='auto'``
    uses stored raw totals where available and only estimates missing-row
    totals when they are zero or unavailable.  ``prior_tc`` is used when a
    prior is provided; otherwise the observed median is used as a fallback.
    """
    if raw_layer_key not in adata.layers:
        raise KeyError(f"adata.layers['{raw_layer_key}'] is required for raw completion.")
    if raw_total_key not in adata.obs:
        raise KeyError(f"adata.obs['{raw_total_key}'] is required for raw completion.")
    prediction, feature_names, resolved_prediction_key = _prediction_key_and_features(
        adata, prediction_key
    )
    missing_indices = _resolve_missing_indices(
        adata,
        missing_indices=missing_indices,
        missing_key=missing_key,
        missing_value=missing_value,
    )
    # Restrict to exactly the modeled features, in prediction-column order.
    result = adata[:, feature_names].copy()
    # The prior graph is not needed in the portable completion result.
    result.obsp.clear()
    observed_indices = np.setdiff1d(
        np.arange(result.n_obs, dtype=int), missing_indices, assume_unique=True
    )
    observed_total_counts = result.obs[raw_total_key].to_numpy(dtype=np.float32)
    n_valid_observed_totals = int(np.sum(
        np.isfinite(observed_total_counts[observed_indices])
        & (observed_total_counts[observed_indices] > 0)
    ))
    if prior_matrix is not None and n_valid_observed_totals == 0:
        raise ValueError(
            "No observed cells have positive stored raw_total_counts for prior-based "
            "raw-scale completion."
        )
    effective_prior_k = min(int(prior_k), n_valid_observed_totals)
    source_raw = result.layers[raw_layer_key]
    raw_observed = _to_dense_f32(source_raw)
    if scale_method == "stored":
        total_counts = result.obs[raw_total_key].to_numpy(dtype=np.float32)
        invalid_missing = missing_indices[
            ~np.isfinite(total_counts[missing_indices]) | (total_counts[missing_indices] < 0)
        ]
        if len(invalid_missing) > 0:
            raise ValueError(
                "Missing observations have invalid stored raw_total_counts. "
                "Use scale_method='auto' for real missing data."
            )
        total_counts = np.maximum(total_counts, 0.0)
    elif scale_method == "prior_tc":
        total_counts = _make_raw_total_counts_safe(
            result,
            missing_indices=missing_indices,
            observed_indices=observed_indices,
            scale_method="prior_tc",
            prior_matrix=prior_matrix,
            prior_k=effective_prior_k,
            prior_matrix_type=prior_matrix_type,
            knn_k=knn_k,
            knn_eps=knn_eps,
            raw_total_key=raw_total_key,
            raw_layer_key=raw_layer_key,
            store_estimated_key="estimated_raw_total_counts",
            verbose=False,
        )
    elif scale_method == "auto":
        # Keep all valid stored totals unchanged.  This is important for
        # simulated incomplete data, where the original count scale is known.
        # Real registration gaps commonly have zero totals and require an
        # estimate before predictions can be returned to raw scale.
        total_counts = _make_raw_total_counts_safe(
            result,
            missing_indices=missing_indices,
            observed_indices=observed_indices,
            scale_method="prior_tc" if prior_matrix is not None else "median",
            prior_matrix=prior_matrix,
            prior_k=effective_prior_k,
            prior_matrix_type=prior_matrix_type,
            knn_k=knn_k,
            knn_eps=knn_eps,
            raw_total_key=raw_total_key,
            raw_layer_key=raw_layer_key,
            store_estimated_key="estimated_raw_total_counts",
            verbose=False,
        )
    else:
        raise ValueError("scale_method must be 'auto', 'stored', or 'prior_tc'.")
    prediction_raw = np.expm1(prediction).astype(np.float32)
    prediction_raw *= (total_counts / float(target_sum))[:, None]
    prediction_raw = np.maximum(prediction_raw, 0.0)
    completed_raw = raw_observed.copy()
    completed_raw[missing_indices, :] = prediction_raw[missing_indices, :]
    result.X = _raw_matrix_like(source_raw, completed_raw)
    result.obs["PRISM_imputed"] = False
    result.obs.iloc[missing_indices, result.obs.columns.get_loc("PRISM_imputed")] = True
    result.obs["imputation_total_counts"] = np.asarray(total_counts, dtype=np.float32)
    result.uns["PRISM_raw_imputation"] = {
        "prediction_key": resolved_prediction_key,
        "prediction_scale": "normalize_total(target_sum=1e4) + log1p",
        "target_sum": float(target_sum),
        "raw_layer_key": raw_layer_key,
        "raw_total_key": raw_total_key,
        "total_count_method": scale_method,
        "prior_k_used": int(effective_prior_k) if prior_matrix is not None else None,
        "n_imputed": int(len(missing_indices)),
        "model_feature_count": int(result.n_vars),
    }
    if save_raw_missing_csv:
        if output_dir is None or file_prefix is None:
            raise ValueError(
                "output_dir and file_prefix are required when save_raw_missing_csv=True."
            )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_pred_path = output_dir / f"{file_prefix}_raw_pred.csv"
        pd.DataFrame(
            prediction_raw[missing_indices, :],
            index=result.obs_names[missing_indices],
            columns=result.var_names,
        ).to_csv(raw_pred_path)
        result.uns["PRISM_raw_imputation"]["raw_pred_missing_csv"] = str(raw_pred_path)
    if output_h5ad is not None:
        output_h5ad = Path(output_h5ad)
        output_h5ad.parent.mkdir(parents=True, exist_ok=True)
        result.uns["PRISM_raw_imputation"]["output_h5ad"] = str(output_h5ad)
        result.write_h5ad(output_h5ad)
    return result
def prism_eval_and_save(
    *,
    truth_adata,
    adata,
    save_path: Union[str, Path],
    first_name: str,
    missing_indices,
    prediction_key: Optional[str] = None,
    save_files: bool = False,
    compute_structure_metrics: bool = False,
    normalize_raw_for_metrics: bool = True,
    raw_eval_clip: bool = True,
    raw_layer_key: str = "raw_data",
    topk_features: Optional[int] = None,
    topk_rank_by: str = "var_order",
    topk_only: bool = True,
    save_topk_summary: bool = False,
    hvg_score_method: str = "log_norm_var",
    hvg_chunk_size: int = 512,
    output_suffix: Optional[str] = None,
    verbose: bool = True,
):
    """Complete, evaluate, and optionally save an in-memory PRISM result.
    ``adata`` is the trained PRISM AnnData.  Since this API evaluates a
    simulated experiment with known ground truth, it always restores the
    stored pre-masking totals internally.  The complete raw-scale object is
    returned as ``results['imputed_adata']`` for plotting in the same notebook.
    Setting ``save_files=True`` writes the complete ``.h5ad`` plus raw-scale
    CSV and metric tables; ``False`` keeps this workflow entirely in memory.
    """
    eval_prefix = first_name if output_suffix is None else f"{first_name}_{output_suffix}"
    output_dir = Path(save_path)
    if save_files:
        output_dir.mkdir(parents=True, exist_ok=True)
    imputed_adata = materialize_prism_raw_imputation(
        adata,
        missing_indices=missing_indices,
        prediction_key=prediction_key,
        scale_method="stored",
        output_h5ad=(
            output_dir / f"{eval_prefix}_raw_imputed.h5ad"
            if save_files else None
        ),
    )
    missing_indices = _resolve_missing_indices(
        imputed_adata, missing_indices=missing_indices
    )
    if truth_adata.n_obs != imputed_adata.n_obs:
        raise ValueError("truth_adata and imputed_adata must have the same observations.")
    if not truth_adata.obs_names.equals(imputed_adata.obs_names):
        raise ValueError("truth_adata and imputed_adata observation names must match exactly.")
    if raw_layer_key not in truth_adata.layers:
        raise KeyError(f"truth_adata.layers['{raw_layer_key}'] is required for evaluation.")
    model_features = pd.Index(imputed_adata.var_names.astype(str))
    truth_features = pd.Index(truth_adata.var_names.astype(str))
    if len(model_features.difference(truth_features)) > 0:
        raise KeyError("Imputed features are not all present in truth_adata.var_names.")
    truth_raw = _to_dense_f32(truth_adata[:, model_features].layers[raw_layer_key])
    imputed_raw = _to_dense_f32(imputed_adata.X)
    if truth_raw.shape != imputed_raw.shape:
        raise ValueError("Truth and imputed raw matrices must have identical shapes.")
    raw_true_metric = truth_raw[missing_indices, :]
    raw_pred_metric = imputed_raw[missing_indices, :]
    if normalize_raw_for_metrics:
        raw_true_eval, raw_pred_eval = _normalize_true_pred_separately_01_by_feature(
            raw_true_metric,
            raw_pred_metric,
            fit_indices=np.arange(len(missing_indices), dtype=int),
            clip=raw_eval_clip,
        )
    else:
        raw_true_eval, raw_pred_eval = raw_true_metric, raw_pred_metric
    metrics_raw = evaluate_protein_prediction(
        raw_true_eval,
        raw_pred_eval,
        np.arange(len(missing_indices), dtype=int),
        compute_structure_metrics=compute_structure_metrics,
    )
    results = {
        "raw": metrics_raw,
        "saved": {},
        "imputed_adata": imputed_adata,
        "prediction_source": "in-memory raw-scale AnnData",
        "eval_prefix": eval_prefix,
        "metric_logic": "feature-wise across missing cells, then averaged across features",
    }
    if topk_features is not None and int(topk_features) > 0:
        k = min(int(topk_features), imputed_adata.n_vars)
        topk_view = _topk_feature_view(
            metrics_raw,
            model_features,
            k=k,
            rank_by=topk_rank_by,
            adata=truth_adata[:, model_features],
            hvg_matrix=truth_raw,
            raw_layer_key=raw_layer_key,
            hvg_chunk_size=hvg_chunk_size,
            hvg_score_method=hvg_score_method,
        )
        results["raw_topk"] = topk_view
        results["raw_all_feature_overall"] = metrics_raw["overall"].copy()
        if topk_only:
            metrics_raw["overall"] = topk_view["overall"].copy()
    if save_files:
        h5ad_path = output_dir / f"{eval_prefix}_raw_imputed.h5ad"
        results["saved"]["raw_imputed_h5ad"] = str(h5ad_path)
        missing_names = imputed_adata.obs_names[missing_indices]
        raw_true_path = output_dir / f"{eval_prefix}_unreg_raw_true.csv"
        raw_pred_path = output_dir / f"{eval_prefix}_unreg_raw_pred.csv"
        pd.DataFrame(raw_true_metric, index=missing_names, columns=model_features).to_csv(raw_true_path)
        pd.DataFrame(raw_pred_metric, index=missing_names, columns=model_features).to_csv(raw_pred_path)
        results["saved"]["raw_true_missing"] = str(raw_true_path)
        results["saved"]["raw_pred_missing"] = str(raw_pred_path)
        metric_path = _save_feature_metric_summary(
            str(output_dir), eval_prefix, "raw", metrics_raw, model_features
        )
        results["saved"]["raw_feature_metrics"] = metric_path
        if save_topk_summary and "raw_topk" in results:
            idx = np.asarray(results["raw_topk"]["topk_idx"], dtype=int)
            data = {
                "feature": np.asarray(model_features, dtype=str)[idx],
                "topk_rank_by": results["raw_topk"]["rank_by"],
            }
            for metric_name in ("PCC", "SPCC", "MSE", "CMD", "SSIM"):
                if metric_name in metrics_raw["per_protein"]:
                    data[metric_name] = np.round(
                        np.asarray(metrics_raw["per_protein"][metric_name], dtype=np.float32)[idx], 4
                    )
            topk_path = output_dir / (
                f"{eval_prefix}_topk_raw_{results['raw_topk']['rank_by']}_"
                f"k{results['raw_topk']['k']}.csv"
            )
            pd.DataFrame(data).to_csv(topk_path, index=False)
            results["saved"]["topk_raw_summary"] = str(topk_path)
    if verbose:
        per_feature = results["raw"]["per_protein"]
        selected_indices = np.arange(imputed_adata.n_vars, dtype=int)
        if topk_only and "raw_topk" in results:
            selected_indices = np.asarray(results["raw_topk"]["topk_idx"], dtype=int)
        metric_summary = {}
        for metric in ("PCC", "SPCC", "MSE"):
            if metric not in per_feature:
                continue
            values = np.asarray(per_feature[metric], dtype=np.float32)[selected_indices]
            metric_summary[metric] = (
                float(np.nanmean(values)),
                float(np.nanstd(values)),
            )
        print(
            "Imputation metrics (mean +/- s.d.): "
            + ", ".join(
                f"{metric}={mean:.4f} +/- {std:.4f}"
                for metric, (mean, std) in metric_summary.items()
            )
        )
        print(f"Missing observations: {len(missing_indices)}")
    return results
def _temporary_obs_key(adata, base: str):
    key = base
    suffix = 1
    while key in adata.obs:
        key = f"{base}_{suffix}"
        suffix += 1
    return key
def plot_prism_imputation_spatial(
    *,
    imputation_results,
    split1_indices,
    feature: Union[int, str],
    point_size: Optional[float] = None,
    show_missing_only: bool = False,
    highlight_missing: bool = False,
    highlight_color: str = "red",
    highlight_linewidth: float = 0.2,
    raw01_clip: bool = True,
    cbar_ticksize: Optional[int] = 10,
    figsize=(8, 3),
    dpi: int = 300,
    cmap_name: str = "viridis",
    title_true: Optional[str] = None,
    title_pred: Optional[str] = None,
    clip_quantiles=(0.01, 0.99),
):
    """Display simulated truth and completion from ``prism_eval_and_save``."""
    try:
        adata_imputed_raw = imputation_results["imputed_adata"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "imputation_results must be returned by prism_eval_and_save."
        ) from exc
    if "spatial" not in adata_imputed_raw.obsm:
        raise KeyError("adata_imputed_raw.obsm['spatial'] not found.")
    if "raw_data" not in adata_imputed_raw.layers:
        raise KeyError("adata_imputed_raw.layers['raw_data'] not found.")
    missing_indices = _resolve_missing_indices(
        adata_imputed_raw, missing_indices=split1_indices
    )
    all_indices = np.arange(adata_imputed_raw.n_obs, dtype=int)
    observed_indices = np.setdiff1d(all_indices, missing_indices, assume_unique=True)
    feature_index, feature_name, _ = _get_feature_index(adata_imputed_raw, feature)
    raw_true = _to_dense_f32(adata_imputed_raw.layers["raw_data"])[:, feature_index]
    raw_pred = _to_dense_f32(adata_imputed_raw.X)[:, feature_index]
    true_plot, pred_plot, _, _ = _normalize_spatial_raw_display_01(
        raw_true,
        raw_pred,
        clip=raw01_clip,
        observed_indices=observed_indices,
        clip_quantiles=clip_quantiles,
    )
    true_key = _temporary_obs_key(adata_imputed_raw, "_prism_truth_plot")
    pred_key = _temporary_obs_key(adata_imputed_raw, "_prism_imputed_plot")
    adata_imputed_raw.obs[true_key] = true_plot
    adata_imputed_raw.obs[pred_key] = pred_plot
    adata_plot = (
        adata_imputed_raw[missing_indices].copy()
        if show_missing_only
        else adata_imputed_raw
    )
    cmap = _resolve_continuous_cmap(cmap_name)
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    try:
        for axis, key, title in (
            (axes[0], true_key, title_true or f"Ground Truth {feature_name}"),
            (axes[1], pred_key, title_pred or f"PRISM Prediction {feature_name}"),
        ):
            kwargs = {
                "adata": adata_plot,
                "basis": "spatial",
                "color": key,
                "title": title,
                "ax": axis,
                "show": False,
                "cmap": cmap,
                "vmin": 0.0,
                "vmax": 1.0,
                "colorbar_loc": "right",
            }
            if point_size is not None:
                kwargs["size"] = float(point_size)
            sc.pl.embedding(**kwargs)
            axis.set_xlabel("spatial1")
            axis.set_ylabel("spatial2")
        if highlight_missing and not show_missing_only:
            coords = np.asarray(adata_imputed_raw.obsm["spatial"])
            marker_size = float(point_size) if point_size is not None else 10.0
            for axis in axes:
                axis.scatter(
                    coords[missing_indices, 0],
                    coords[missing_indices, 1],
                    facecolors="none",
                    edgecolors=highlight_color,
                    s=marker_size,
                    linewidths=highlight_linewidth,
                    zorder=10,
                )
        if cbar_ticksize is not None:
            for axis in fig.axes:
                if axis not in set(axes.tolist()):
                    axis.tick_params(labelsize=int(cbar_ticksize))
        fig.subplots_adjust(wspace=0.35)
    finally:
        del adata_imputed_raw.obs[true_key]
        del adata_imputed_raw.obs[pred_key]
    return fig, axes
def plot_task2_real_three_panel(
    *,
    adata_aligned,
    adata_unaligned_raw,
    feature: Union[int, str],
    missing_indices=None,
    prediction_key: Optional[str] = None,
    prior_matrix=None,
    prior_k: int = 5,
    prior_matrix_type: str = "distance",
    save_files: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    file_prefix: Optional[str] = None,
    clip_quantiles=(0.01, 0.99),
    point_size: Optional[float] = None,
    alpha: float = 0.9,
    show_missing_only: bool = False,
    figsize=(12, 3),
    dpi: int = 300,
    cmap_name: str = "viridis",
    cbar_ticksize: int = 10,
    missing_key: str = "missing",
    missing_value: str = "0",
    observed_value: str = "1",
):
    """Complete and display reference, real-missing data, and PRISM output.
    The raw-scale result stays internal to this plotting call.  When
    ``save_files=True``, ``output_dir`` and ``file_prefix`` select the two
    retained artifacts: ``*_raw_pred.csv`` and ``*_raw_imputed.h5ad``.
    """
    if save_files and (output_dir is None or file_prefix is None):
        raise ValueError(
            "output_dir and file_prefix are required when save_files=True."
        )
    output_dir_path = Path(output_dir) if output_dir is not None else None
    adata_imputed_raw = materialize_prism_raw_imputation(
        adata_aligned,
        missing_indices=missing_indices,
        prediction_key=prediction_key,
        # Real registration gaps do not contain their true raw total counts.
        scale_method="auto",
        prior_matrix=prior_matrix,
        prior_k=prior_k,
        prior_matrix_type=prior_matrix_type,
        output_h5ad=(
            output_dir_path / f"{file_prefix}_raw_imputed.h5ad"
            if save_files else None
        ),
        save_raw_missing_csv=save_files,
        output_dir=output_dir_path,
        file_prefix=file_prefix,
    )
    if "spatial" not in adata_imputed_raw.obsm or "spatial" not in adata_unaligned_raw.obsm:
        raise KeyError("Both AnnData objects must contain obsm['spatial'].")
    if "raw_data" not in adata_imputed_raw.layers:
        raise KeyError("adata_imputed_raw.layers['raw_data'] not found.")
    missing_indices = _resolve_missing_indices(
        adata_imputed_raw,
        missing_key=missing_key,
        missing_value=missing_value,
    )
    observed_indices = np.flatnonzero(
        adata_imputed_raw.obs[missing_key].astype(str).to_numpy() == str(observed_value)
    )
    if len(observed_indices) == 0:
        raise ValueError("No observed locations were found for the requested missing convention.")
    imputed_idx, imputed_feature, feature_label = _get_feature_index(adata_imputed_raw, feature)
    reference_idx, _, _ = _get_feature_index(adata_unaligned_raw, imputed_feature)
    reference_values = _to_dense_f32(adata_unaligned_raw.X)[:, reference_idx]
    aligned_values = _to_dense_f32(adata_imputed_raw.layers["raw_data"])[:, imputed_idx]
    completed_values = _to_dense_f32(adata_imputed_raw.X)[:, imputed_idx]
    if not np.allclose(
        completed_values[observed_indices], aligned_values[observed_indices], atol=1e-6
    ):
        raise ValueError("Observed locations changed in the completed raw AnnData object.")
    reference_plot, _, _ = _normalize_single_fullmap_01(
        reference_values, clip=True, clip_quantiles=clip_quantiles
    )
    aligned_display = aligned_values.copy()
    aligned_display[missing_indices] = np.nan
    aligned_plot, _, _ = _normalize_single_fullmap_01(
        aligned_display, clip=True, clip_quantiles=clip_quantiles
    )
    completed_plot, _, _ = _normalize_single_fullmap_01(
        completed_values, clip=True, clip_quantiles=clip_quantiles
    )
    completed_plot[observed_indices] = aligned_plot[observed_indices]
    reference_key = _temporary_obs_key(adata_unaligned_raw, "_prism_reference_plot")
    aligned_key = _temporary_obs_key(adata_imputed_raw, "_prism_aligned_plot")
    completed_key = _temporary_obs_key(adata_imputed_raw, "_prism_completed_plot")
    adata_unaligned_raw.obs[reference_key] = reference_plot
    adata_imputed_raw.obs[aligned_key] = aligned_plot
    adata_imputed_raw.obs[completed_key] = completed_plot
    aligned_plot_adata = (
        adata_imputed_raw[missing_indices].copy()
        if show_missing_only
        else adata_imputed_raw
    )
    cmap = _resolve_continuous_cmap(cmap_name)
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    try:
        entries = (
            (adata_unaligned_raw, reference_key, f"Unaligned {feature_label}", axes[0], None),
            (aligned_plot_adata, aligned_key, f"Aligned {feature_label}", axes[1], "#B8B8B8"),
            (aligned_plot_adata, completed_key, f"PRISM Prediction {feature_label}", axes[2], None),
        )
        for plot_adata, key, title, axis, na_color in entries:
            kwargs = {
                "adata": plot_adata,
                "basis": "spatial",
                "color": key,
                "title": title,
                "ax": axis,
                "show": False,
                "cmap": cmap,
                "vmin": 0.0,
                "vmax": 1.0,
                "colorbar_loc": "right",
                "alpha": alpha,
            }
            if na_color is not None:
                kwargs["na_color"] = na_color
            if point_size is not None:
                kwargs["size"] = float(point_size)
            sc.pl.embedding(**kwargs)
            axis.set_xlabel("spatial1")
            axis.set_ylabel("spatial2")
        for axis in fig.axes:
            if axis not in set(axes.tolist()):
                axis.tick_params(labelsize=int(cbar_ticksize))
        fig.subplots_adjust(wspace=0.35)
    finally:
        del adata_unaligned_raw.obs[reference_key]
        del adata_imputed_raw.obs[aligned_key]
        del adata_imputed_raw.obs[completed_key]
    return fig, axes
