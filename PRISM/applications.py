"""SMA registration, human PD enrichment, and application visualizations.
The SMA assay measures Visium RNA and MALDI-MSI metabolomics on the same
section but at different spatial resolutions.  These helpers apply MAGPIE
landmarks, retain reliable one-to-one RNA-MSI pairs, and encode the remaining
RNA locations as MSI-missing before PRISM integration.
"""
from __future__ import annotations
import heapq
from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
SMA_PAPER_METABOLITES = {
    "Dopamine": 421.19,
    "Dopamine_DD": 674.28,
    "GABA": 371.18,
    "GABA_H2O": 353.17,
    "Tocopherol": 698.49,
    "Norepinephrine_DD": 690.27,
    "DOPAC_DD": 698.24,
    "3-MT": 435.21,
    "Serotonin": 444.21,
    "Taurine": 393.13,
    "Histidine": 423.18,
}
_SMA_TAGS = {"A1", "B1", "C1"}
def load_visium_rna_with_spatial(
    h5_path,
    tissue_positions_csv,
    *,
    swap_xy=True,
):
    """Read a 10x matrix and attach tissue-position coordinates in CSV order."""
    import scanpy as sc
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    columns = ["barcode", "in_tissue", "array_row", "array_col", "x", "y"]
    positions = pd.read_csv(tissue_positions_csv, header=None, names=columns)
    positions["barcode"] = positions["barcode"].astype(str).str.replace(
        " ", "-", regex=False
    )
    positions = positions.set_index("barcode")
    common = positions.index[positions.index.isin(adata.obs_names)]
    if common.empty:
        raise ValueError("No overlapping barcodes between the 10x matrix and positions CSV.")
    adata = adata[common].copy()
    spatial = positions.loc[common, ["x", "y"]].to_numpy(dtype=float)
    if swap_xy:
        spatial = spatial[:, [1, 0]].copy()
    adata.obsm["spatial"] = spatial
    return adata
def attach_sma_metadata(
    adata,
    csv_path,
    *,
    barcode_col,
    value_col,
    obs_key,
):
    """Attach one barcode-indexed SMA annotation column to ``adata.obs``."""
    metadata = pd.read_csv(csv_path)
    metadata[barcode_col] = metadata[barcode_col].astype(str).str.replace(
        " ", "-", regex=False
    )
    values = metadata.set_index(barcode_col).reindex(adata.obs_names)[value_col]
    adata.obs[obs_key] = values.astype("category")
def build_sma_new_label(region_series, dopamine_series, *, tag):
    """Build slice-specific dopamine and brain-region labels for SMA sections.
    A1, B1, and C1 use distinct rules in the original SMA annotations.  The
    explicit branches below preserve those rules while exposing one interface
    for all three sections.
    """
    tag = str(tag).upper()
    if tag not in _SMA_TAGS:
        raise ValueError(f"tag must be one of {sorted(_SMA_TAGS)}, got {tag!r}.")
    labels = pd.DataFrame(
        {
            "region": region_series.astype(str),
            "dopamine": dopamine_series.astype(str),
        },
        index=region_series.index,
    )
    labels["region"] = labels["region"].replace("nan", "N/A")
    labels["dopamine"] = (
        labels["dopamine"].str.replace(" ", "_", regex=False).replace("nan", "N/A")
    )
    def label_a1(region, dopamine):
        if region == "Cd":
            if dopamine == "dopamine_Cd":
                return "Cd_dopamine"
            if dopamine == "not_dopamine_Cd":
                return "Cd_not_dopamine"
            if dopamine in ("N/A", "CI"):
                return np.nan
            return "Cd_Unknown"
        if region == "CI":
            if dopamine in ("CI", "N/A", "not_dopamine_Cd"):
                return "CI"
            if dopamine in ("Cd", "dopamine_Cd"):
                return np.nan
            return "CI_Unknown"
        if region == "N/A":
            return np.nan if dopamine in (
                "dopamine_Cd", "not_dopamine_Cd", "N/A", "CI"
            ) else "Unknown"
        return ""
    def label_b1(region, dopamine):
        if region == "Cd":
            if dopamine == "dopamine_Cd":
                return "Cd_dopamine"
            if dopamine in ("not_dopamine_Cd", "CI"):
                return "Cd_not_dopamine"
            if dopamine == "N/A":
                return np.nan
            return "Cd_Unknown"
        if region == "CI":
            if dopamine in ("CI", "N/A", "not_dopamine_Cd"):
                return "CI"
            if dopamine in ("Cd", "dopamine_Cd"):
                return np.nan
            return "CI_Unknown"
        if region == "N/A":
            return np.nan if dopamine in (
                "dopamine_Cd", "not_dopamine_Cd", "N/A", "CI"
            ) else "Unknown"
        return ""
    def label_c1(region, dopamine):
        if region == "Cd":
            if dopamine == "dopamine_Cd":
                return "Cd_dopamine"
            if dopamine in ("not_dopamine_Cd", "CI"):
                return "Cd_not_dopamine"
            if dopamine == "N/A":
                return np.nan
            return "Cd_Unknown"
        if region == "CI":
            if dopamine in ("CI", "not_dopamine_Cd"):
                return "CI"
            if dopamine in ("N/A", "Cd", "dopamine_Cd"):
                return np.nan
            return "CI_Unknown"
        if region == "ACB":
            if dopamine == "dopamine_Cd":
                return "ACB_dopamine"
            if dopamine in ("CI", "not_dopamine_Cd"):
                return "ACB_not_dopamine"
            if dopamine in ("N/A", "Cd"):
                return np.nan
            return "ACB_Unknown"
        if region == "unk":
            return np.nan
        if region == "N/A":
            return np.nan if dopamine in (
                "dopamine_Cd", "not_dopamine_Cd", "N/A", "CI"
            ) else "Unknown"
        return ""
    label_fn = {"A1": label_a1, "B1": label_b1, "C1": label_c1}[tag]
    result = pd.Series(
        [label_fn(region, dopamine) for region, dopamine in labels.itertuples(index=False)],
        index=labels.index,
        name="new_label",
    )
    return result
def prepare_sma_rna(
    h5_path,
    tissue_positions_csv,
    region_csv,
    dopamine_csv,
    *,
    tag,
    swap_xy=True,
):
    """Load one SMA RNA section and attach region, dopamine, and ``new_label``."""
    adata = load_visium_rna_with_spatial(
        h5_path,
        tissue_positions_csv,
        swap_xy=swap_xy,
    )
    attach_sma_metadata(
        adata,
        region_csv,
        barcode_col="Barcode",
        value_col="RegionLoupe",
        obs_key="region",
    )
    attach_sma_metadata(
        adata,
        dopamine_csv,
        barcode_col="Barcode",
        value_col="dopamine",
        obs_key="dopamine",
    )
    adata.obs["new_label"] = pd.Categorical(
        build_sma_new_label(adata.obs["region"], adata.obs["dopamine"], tag=tag)
    )
    return adata
def sma_label_table(adata):
    """Return the barcode-indexed SMA annotation table saved beside RNA data."""
    return pd.DataFrame(
        {
            "region": adata.obs["region"].astype(str),
            "dopamine": adata.obs["dopamine"].astype(str),
            "new_label": adata.obs["new_label"].astype(str),
        },
        index=adata.obs_names,
    )
def estimate_magpie_affine(landmark_csv):
    """Estimate the MAGPIE landmark affine transform from MSI to Visium pixels."""
    import cv2
    landmarks = pd.read_csv(landmark_csv)
    source = landmarks[["Y_left", "X_left"]].to_numpy(dtype=float)
    target = landmarks[["X_right", "Y_right"]].to_numpy(dtype=float)
    affine, inliers = cv2.estimateAffine2D(source, target)
    if affine is None:
        raise ValueError("MAGPIE landmarks did not yield a valid affine transform.")
    return affine, inliers
def transform_coordinates_affine(coordinates, affine):
    """Apply a 2D affine transform to an ``(n, 2)`` coordinate array."""
    coordinates = np.asarray(coordinates, dtype=float)
    homogeneous = np.column_stack([coordinates, np.ones(coordinates.shape[0])])
    return homogeneous @ np.asarray(affine, dtype=float).T
def apply_magpie_landmarks_to_msi(
    msi,
    landmark_csv,
    msi_metadata_csv,
    *,
    library_id=None,
):
    """Transform MSI pixels with MAGPIE landmarks and store Visium-scale coordinates.
    ``obsm['spatial']`` receives coordinates divided by Visium's
    ``tissue_hires_scalef`` so they are in the same coordinate system as the
    RNA AnnData spatial coordinates.
    """
    metadata = pd.read_csv(msi_metadata_csv, index_col=0)
    metadata.index = metadata.index.astype(str)
    coordinates = metadata.reindex(msi.obs_names)[["x", "y"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("MSI metadata indices do not align with MSI obs_names.")
    affine, inliers = estimate_magpie_affine(landmark_csv)
    transformed_pixels = transform_coordinates_affine(coordinates, affine)
    spatial_info = msi.uns["spatial"]
    if library_id is None:
        library_id = next(iter(spatial_info))
    scale = spatial_info[library_id]["scalefactors"]["tissue_hires_scalef"]
    msi.obsm["spatial"] = transformed_pixels / float(scale)
    return transformed_pixels, affine, inliers
def greedy_spatial_matching(
    reference_spatial,
    query_spatial,
    *,
    max_neighbors=5,
    distance_threshold=240.0,
):
    """Greedily build unique MSI-to-RNA matches within a distance threshold."""
    reference_spatial = np.asarray(reference_spatial, dtype=float)
    query_spatial = np.asarray(query_spatial, dtype=float)
    n_neighbors = min(int(max_neighbors), reference_spatial.shape[0])
    if n_neighbors < 1:
        raise ValueError("reference_spatial must contain at least one location.")
    distances, neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(
        reference_spatial
    ).kneighbors(query_spatial)
    candidates = []
    for query_index in range(query_spatial.shape[0]):
        for rank in range(n_neighbors):
            distance = float(distances[query_index, rank])
            if distance <= distance_threshold:
                heapq.heappush(
                    candidates,
                    (distance, query_index, int(neighbors[query_index, rank])),
                )
    mapping = np.full(query_spatial.shape[0], -1, dtype=int)
    matched_distance = np.full(query_spatial.shape[0], np.nan, dtype=np.float32)
    used_reference = set()
    while candidates:
        distance, query_index, reference_index = heapq.heappop(candidates)
        if mapping[query_index] == -1 and reference_index not in used_reference:
            mapping[query_index] = reference_index
            matched_distance[query_index] = distance
            used_reference.add(reference_index)
    return mapping, matched_distance
def spatial_mapping_table(reference, query, mapping, matched_distance):
    """Create the C1-compatible RNA-MSI mapping table."""
    mapping = np.asarray(mapping, dtype=int)
    return pd.DataFrame(
        {
            "omics2_index": np.arange(query.n_obs, dtype=int),
            "omics2_barcode": query.obs_names.to_numpy(),
            "omics1_index": mapping,
            "omics1_barcode": np.where(
                mapping >= 0,
                reference.obs_names[mapping].to_numpy(),
                "",
            ),
            "distance": matched_distance,
        }
    )
def build_aligned_msi(reference, msi, mapping, matched_distance):
    """Build the matched-only MSI AnnData indexed by its paired RNA barcodes."""
    matched = np.asarray(mapping, dtype=int) >= 0
    matched_msi = msi[matched].copy()
    reference_index = np.asarray(mapping, dtype=int)[matched]
    aligned = AnnData(
        X=matched_msi.X.copy(),
        obs=matched_msi.obs.copy(),
        var=matched_msi.var.copy(),
    )
    aligned.obs_names = reference.obs_names[reference_index].to_numpy()
    aligned.obs["original_omics2_barcode"] = matched_msi.obs_names.to_numpy()
    aligned.obs["match_distance"] = np.asarray(matched_distance)[matched].astype(
        np.float32
    )
    aligned.obsm["spatial"] = reference.obsm["spatial"][reference_index].copy()
    return aligned
def build_full_msi_with_missing(
    reference,
    aligned_msi,
    *,
    missing_key="missing",
    label_key="new_label",
):
    """Project matched MSI onto all RNA spots and zero unregistered locations."""
    reference_order = reference.obs_names.to_numpy()
    aligned_set = set(aligned_msi.obs_names)
    missing_barcodes = [barcode for barcode in reference_order if barcode not in aligned_set]
    missing = AnnData(
        X=np.zeros((len(missing_barcodes), aligned_msi.n_vars), dtype=np.float32),
        obs=pd.DataFrame(index=missing_barcodes),
        var=aligned_msi.var.copy(),
        obsm={"spatial": reference[missing_barcodes].obsm["spatial"].copy()},
    )
    observed = aligned_msi.copy()
    observed.obs[missing_key] = "1"
    missing.obs[missing_key] = "0"
    full = ad.concat([observed, missing], join="outer", axis=0)[reference_order].copy()
    for key in ("region", "dopamine", "new_label"):
        if key in reference.obs:
            full.obs[key] = reference.obs[key].copy()
    if label_key in full.obs:
        full.obs.loc[full.obs[label_key].isna(), missing_key] = "0"
    missing_rows = (full.obs[missing_key].astype(str) == "0").to_numpy()
    if sp.issparse(full.X):
        matrix = full.X.tolil(copy=True)
        matrix[missing_rows, :] = 0
        full.X = matrix.tocsr().astype(np.float32)
    else:
        matrix = np.asarray(full.X, dtype=np.float32)
        matrix[missing_rows, :] = 0
        full.X = matrix
    return full
def write_sma_h5ad(adata, path):
    """Write a float32 CSR AnnData object using the SMA output convention."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = sp.csr_matrix(np.asarray(adata.X, dtype=np.float32))
    adata.write_h5ad(path)
def annotate_sma_paper_metabolites(
    adata,
    *,
    metabolites=None,
    max_abs_error=0.02,
    rename_var_names=True,
):
    """Annotate each SMA paper metabolite to its closest unused m/z feature."""
    annotated = adata.copy()
    annotated.var_names_make_unique()
    mz_original = annotated.var_names.astype(str)
    mz_values = pd.to_numeric(mz_original, errors="coerce").to_numpy(dtype=float)
    metabolites = SMA_PAPER_METABOLITES if metabolites is None else metabolites
    annotated.var["mz_original"] = mz_original
    annotated.var["mz"] = mz_values
    annotated.var["paper_metabolite"] = ""
    annotated.var["paper_target_mz"] = np.nan
    annotated.var["mz_error"] = np.nan
    annotated.var["ppm_error"] = np.nan
    annotated.var["is_paper_annotated"] = False
    used_indices = set()
    renamed = {}
    records = []
    valid_indices = np.flatnonzero(np.isfinite(mz_values))
    for metabolite, target_mz in metabolites.items():
        candidates = np.asarray(
            [index for index in valid_indices if index not in used_indices], dtype=int
        )
        if candidates.size == 0:
            records.append(
                {
                    "metabolite": metabolite,
                    "target_mz": target_mz,
                    "matched_mz": np.nan,
                    "mz_error": np.nan,
                    "ppm_error": np.nan,
                    "matched": False,
                }
            )
            continue
        feature_index = candidates[np.argmin(np.abs(mz_values[candidates] - target_mz))]
        matched_mz = mz_values[feature_index]
        error = matched_mz - target_mz
        ppm_error = error / target_mz * 1e6
        matched = abs(error) <= max_abs_error
        records.append(
            {
                "metabolite": metabolite,
                "target_mz": target_mz,
                "matched_mz": matched_mz,
                "mz_error": error,
                "ppm_error": ppm_error,
                "matched": matched,
            }
        )
        if matched:
            used_indices.add(feature_index)
            old_name = str(mz_original[feature_index])
            renamed[old_name] = metabolite
            annotated.var.iloc[
                feature_index, annotated.var.columns.get_loc("paper_metabolite")
            ] = metabolite
            annotated.var.iloc[
                feature_index, annotated.var.columns.get_loc("paper_target_mz")
            ] = target_mz
            annotated.var.iloc[
                feature_index, annotated.var.columns.get_loc("mz_error")
            ] = error
            annotated.var.iloc[
                feature_index, annotated.var.columns.get_loc("ppm_error")
            ] = ppm_error
            annotated.var.iloc[
                feature_index, annotated.var.columns.get_loc("is_paper_annotated")
            ] = True
    feature_names = [renamed.get(str(mz), str(mz)) for mz in mz_original]
    annotated.var["feature_name_paper"] = feature_names
    if rename_var_names:
        annotated.var_names = pd.Index(feature_names)
        annotated.var_names_make_unique()
    return annotated, pd.DataFrame(records)
def plot_magpie_alignment_preview(
    image_path,
    transformed_pixels,
    msi_intensities,
    *,
    feature="674.28592",
    point_size=5,
    alpha=0.5,
):
    """Display the MAGPIE-transformed MSI signal over the Visium H&E image."""
    import matplotlib.pyplot as plt
    from PIL import Image
    image = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(np.asarray(image))
    ax.scatter(
        transformed_pixels[:, 0],
        transformed_pixels[:, 1],
        c=msi_intensities[feature],
        s=point_size,
        alpha=alpha,
    )
    ax.set_aspect(1)
    ax.axis("off")
    plt.show()
    return fig, ax
def plot_matching_summary_bar(
    mapping,
    *,
    dpi=300,
    figsize=(6, 5),
    palette=None,
    font_family="Arial",
    font_size=15,
):
    """Display the number of retained and unregistered RNA-MSI pairs."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    mapping = np.asarray(mapping, dtype=int)
    matched_count = int(np.sum(mapping != -1))
    unmatched_count = int(np.sum(mapping == -1))
    total_cells = int(mapping.size)
    if palette is None:
        palette = {"Matched": "#5A6E8C", "Unmatched": "#9268AD"}
    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = font_size
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    categories = ["Matched", "Unmatched"]
    counts = [matched_count, unmatched_count]
    bars = ax.bar(categories, counts, color=[palette[name] for name in categories], width=0.6, linewidth=1, alpha=0.7, edgecolor="black")
    y_offset = max(1, int(0.01 * max(counts)))
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset, f"{count}", ha="center", va="bottom", fontsize=font_size + 1)
    ax.set_ylabel("Number of Cells")
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="x", colors="black", width=1, length=5, direction="out")
    ax.tick_params(axis="y", colors="black", width=1, length=5, direction="out")
    ax.grid(False)
    handles = [Patch(facecolor=palette["Matched"], edgecolor="none"), Patch(facecolor=palette["Unmatched"], edgecolor="none")]
    ax.legend(handles, ["Matched", "Unmatched"], loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, handlelength=1.8, columnspacing=1.2)
    plt.show()
    if total_cells > 0:
        print(f"Total cells: {total_cells}")
        print(f"Matched cells: {matched_count} ({matched_count / total_cells * 100:.1f}%)")
        print(f"Unmatched cells: {unmatched_count} ({unmatched_count / total_cells * 100:.1f}%)")
def plot_spatial_overlay(
    omics1_spatial,
    omics2_spatial,
    *,
    mapping=None,
    show_match_lines=False,
    label1="RNA",
    label2="MSI",
    color1="blue",
    color2="red",
    s=5,
    dpi=300,
    figsize=(10, 10),
    font_family="Arial",
    font_size=18,
    margin=1000.0,
    legend_marker_size=20,
    legend_bbox=(0.385, 1.10),
    demo_ex_y=1.02,
    demo_ex_x=0.61,
    demo_s=80,
    demo_line_color="gray",
    demo_linewidth=1.2,
    demo_text="Matched",
    demo_text_dx=0.02,
):
    """Plot RNA and MSI coordinates, with optional white lines for matched pairs."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    omics1_spatial = np.asarray(omics1_spatial, dtype=float)
    omics2_spatial = np.asarray(omics2_spatial, dtype=float)
    plt.rcParams.update({"font.family": font_family, "font.size": font_size, "xtick.labelsize": font_size, "ytick.labelsize": font_size})
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    matched_segments = None
    if show_match_lines and mapping is not None:
        mapping = np.asarray(mapping, dtype=int)
        query_indices = np.arange(min(mapping.size, omics2_spatial.shape[0]))
        reference_indices = mapping[: query_indices.size]
        matched = (reference_indices >= 0) & (reference_indices < omics1_spatial.shape[0])
        matched_segments = np.stack(
            (omics1_spatial[reference_indices[matched]], omics2_spatial[query_indices[matched]]), axis=1
        )
    ax.scatter(omics1_spatial[:, 0], omics1_spatial[:, 1], c=color1, s=s, label=label1, zorder=2)
    ax.scatter(omics2_spatial[:, 0], omics2_spatial[:, 1], c=color2, s=s, label=label2, zorder=2)
    if matched_segments is not None and matched_segments.size:
        ax.add_collection(LineCollection(matched_segments, colors="white", linewidths=0.7, alpha=1.0, zorder=3))
    ax.legend(loc="upper center", fontsize=font_size, markerscale=legend_marker_size / 5, ncol=2, columnspacing=1.5, labelspacing=0.1, borderpad=1.5, handletextpad=-0.1, bbox_to_anchor=legend_bbox, frameon=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", bottom=False, left=False, top=False, right=False, labelbottom=False, labelleft=False)
    fig.canvas.draw()
    radius_points = np.sqrt(demo_s / np.pi)
    radius_pixels = radius_points * fig.dpi / 72.0
    dx_axes = radius_pixels / ax.get_window_extent().width
    x_left, x_right = demo_ex_x - dx_axes / 2, demo_ex_x + dx_axes / 2
    ax.plot([x_left, x_right], [demo_ex_y, demo_ex_y], transform=ax.transAxes, color=demo_line_color, linewidth=demo_linewidth, clip_on=False)
    ax.scatter([x_left], [demo_ex_y], transform=ax.transAxes, c=color1, s=demo_s, clip_on=False, zorder=3)
    ax.scatter([x_right], [demo_ex_y], transform=ax.transAxes, c=color2, s=demo_s, clip_on=False, zorder=3)
    ax.text(x_right + demo_text_dx, demo_ex_y, demo_text, transform=ax.transAxes, va="center", ha="left", fontsize=font_size)
    ax.set_xlim(float(np.min(omics1_spatial[:, 0])) - margin, float(np.max(omics1_spatial[:, 0])) + margin)
    ax.set_ylim(float(np.min(omics1_spatial[:, 1])) - margin, float(np.max(omics1_spatial[:, 1])) + margin)
    ax.set_aspect("equal")
    plt.show()
    return fig, ax
def load_imputed_dopamine_predictions(prediction_paths, slice_ids, feature_name="Dopamine_DD"):
    """Load one PRISM prediction matrix per slice and return the selected metabolite."""
    frames = []
    for slice_id in slice_ids:
        prediction = pd.read_csv(prediction_paths[slice_id], index_col=0)
        prediction.index = pd.Index(
            [f"{slice_id}|{spot_id}" for spot_id in prediction.index.astype(str)]
        )
        frames.append(prediction.loc[:, [feature_name]].assign(slice_id=slice_id))
    dopamine_df = pd.concat(frames, axis=0)
    dopamine_df[feature_name] = pd.to_numeric(dopamine_df[feature_name], errors="coerce")
    dopamine_df = dopamine_df.dropna(subset=[feature_name])
    dopamine_df[f"log10_{feature_name}"] = np.log10(dopamine_df[feature_name] + 1)
    return dopamine_df
def assign_dopamine_status(
    dopamine_df,
    feature_name="Dopamine_DD",
    threshold_method="GMM_midpoint",
    random_seed=2024,
):
    """Split spots into response and silent groups from the imputed Dopamine_DD value."""
    log_feature = f"log10_{feature_name}"
    log_values = dopamine_df[log_feature].to_numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=random_seed).fit(log_values)
    gmm_midpoint_log = float(np.mean(np.sort(gmm.means_.ravel())))
    threshold_candidates = {
        "median": float(dopamine_df[feature_name].median()),
        "q75": float(dopamine_df[feature_name].quantile(0.75)),
        "q80": float(dopamine_df[feature_name].quantile(0.80)),
        "GMM_midpoint": float(10**gmm_midpoint_log - 1),
    }
    threshold_raw = threshold_candidates[threshold_method]
    labeled_df = dopamine_df.copy()
    labeled_df["Dopamine_status"] = np.where(
        labeled_df[feature_name] >= threshold_raw,
        "Dopamine_response",
        "Dopamine_silent",
    )
    threshold_table = pd.DataFrame(
        {
            "method": threshold_candidates.keys(),
            "threshold_raw": threshold_candidates.values(),
        }
    )
    return labeled_df, threshold_table, threshold_raw
def plot_dopamine_distribution(dopamine_df, threshold_raw, feature_name="Dopamine_DD"):
    """Display slice-level Dopamine_DD distributions and the selected threshold."""
    log_feature = f"log10_{feature_name}"
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    sns.violinplot(
        data=dopamine_df,
        x="slice_id",
        y=log_feature,
        inner=None,
        cut=0,
        linewidth=0.8,
        ax=axes[0],
    )
    sns.boxplot(
        data=dopamine_df,
        x="slice_id",
        y=log_feature,
        width=0.18,
        showfliers=False,
        boxprops={"facecolor": "white", "zorder": 3},
        medianprops={"color": "black", "linewidth": 1.0},
        linewidth=0.8,
        ax=axes[0],
    )
    axes[0].set(
        xlabel="",
        ylabel="log10(Dopamine-DD + 1)",
        title="Dopamine-DD distribution",
    )
    sns.histplot(
        data=dopamine_df,
        x=log_feature,
        hue="slice_id",
        bins=40,
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[1],
    )
    axes[1].axvline(np.log10(threshold_raw + 1), color="black", linestyle="--", linewidth=1.2)
    axes[1].set(
        xlabel="log10(Dopamine-DD + 1)",
        ylabel="Density",
        title="GMM midpoint threshold",
    )
    fig.tight_layout()
    return fig, axes
def run_imputed_dopamine_deg(
    rna_paths,
    slice_ids,
    dopamine_df,
    output_dir,
    gene_list_paths,
    top_k=100,
    padj_cutoff=0.05,
):
    """Run imputed-only Wilcoxon DEG and save selected significant gene outputs."""
    rna_slices = []
    for slice_id in slice_ids:
        adata = sc.read_h5ad(rna_paths[slice_id])
        adata.obs_names = pd.Index(
            [f"{slice_id}|{spot_id}" for spot_id in adata.obs_names.astype(str)]
        )
        rna_slices.append(adata)
    adata_rna = sc.concat(rna_slices, join="inner", merge="same", index_unique=None)
    common_spots = adata_rna.obs_names.intersection(dopamine_df.index)
    adata_deg = adata_rna[common_spots].copy()
    adata_deg.obs = adata_deg.obs.join(dopamine_df[["Dopamine_status"]])
    sc.pp.normalize_total(adata_deg, target_sum=1e4)
    sc.pp.log1p(adata_deg)
    sc.pp.filter_genes(adata_deg, min_cells=max(10, int(0.005 * adata_deg.n_obs)))
    sc.tl.rank_genes_groups(
        adata_deg,
        groupby="Dopamine_status",
        method="wilcoxon",
        use_raw=False,
    )
    output_dir = Path(output_dir)
    deg_results = {}
    significant_results = {}
    top_gene_sets = {}
    for group_name, gene_list_path in gene_list_paths.items():
        deg_table = sc.get.rank_genes_groups_df(adata_deg, group=group_name).rename(
            columns={
                "names": "gene",
                "logfoldchanges": "logFC",
                "pvals": "pval",
                "pvals_adj": "padj",
            }
        )
        deg_table["neglog10_padj"] = -np.log10(deg_table["padj"].clip(lower=1e-300))
        deg_table["score"] = deg_table["logFC"].abs() * deg_table["neglog10_padj"]
        deg_table = deg_table.sort_values(["padj", "score"], ascending=[True, False]).reset_index(drop=True)
        significant_up = deg_table[(deg_table["padj"] < padj_cutoff) & (deg_table["logFC"] > 0)].copy()
        top_genes = (
            significant_up.sort_values("score", ascending=False)
            .head(top_k)["gene"]
            .astype(str)
            .tolist()
        )
        deg_results[group_name] = deg_table
        significant_results[group_name] = significant_up
        top_gene_sets[group_name] = top_genes
        significant_up.to_csv(output_dir / f"DEG_{group_name}_significant_up.csv", index=False)
        Path(gene_list_path).write_text("\n".join(top_genes) + "\n")
    return deg_results, significant_results, top_gene_sets
def plot_pd_human_volcano(
    deg_table,
    logfc_cutoff=0.25,
    padj_cutoff=0.05,
    visual_logfc_cutoff=1.0,
    visual_neglog10_fdr_cutoff=5.0,
):
    """Plot the Dopamine-response volcano chart with the tutorial display settings."""
    y_breaks = np.array([0, 5, 10, 50, 150, 250, 350], dtype=float)
    y_positions = np.arange(len(y_breaks), dtype=float)
    y_cap = y_breaks[-1]
    volcano_df = deg_table.copy()
    volcano_df["category"] = "Not significant"
    volcano_df.loc[
        (volcano_df["padj"] < padj_cutoff) & (volcano_df["logFC"] >= logfc_cutoff),
        "category",
    ] = "Dopamine response enriched"
    volcano_df.loc[
        (volcano_df["padj"] < padj_cutoff) & (volcano_df["logFC"] <= -logfc_cutoff),
        "category",
    ] = "Dopamine silent enriched"
    volcano_df["neglog10_padj_raw"] = volcano_df["neglog10_padj"].astype(float)
    volcano_df["is_y_capped"] = volcano_df["neglog10_padj_raw"] > y_cap
    volcano_df["y_plot"] = np.interp(
        np.clip(volcano_df["neglog10_padj_raw"], y_breaks[0], y_cap),
        y_breaks,
        y_positions,
    )
    plt.rcParams.update({"font.family": "Arial", "font.size": 12})
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    silent_mask = volcano_df["category"].eq("Dopamine silent enriched")
    response_mask = volcano_df["category"].eq("Dopamine response enriched")
    colored_silent = (
        silent_mask
        & (volcano_df["logFC"] <= -visual_logfc_cutoff)
        & (volcano_df["neglog10_padj_raw"] >= visual_neglog10_fdr_cutoff)
    )
    colored_response = (
        response_mask
        & (volcano_df["logFC"] >= visual_logfc_cutoff)
        & (volcano_df["neglog10_padj_raw"] >= visual_neglog10_fdr_cutoff)
    )
    grey_mask = ~(colored_silent | colored_response)
    volcano_df["is_colored"] = colored_silent | colored_response
    capped_mask = volcano_df["is_y_capped"]
    scatter_specs = (
        (grey_mask, "#C7C7C7", 0.35, "Other genes", 1),
        (colored_silent, "#4575B4", 0.78, "Silent-enriched", 2),
        (colored_response, "#D73027", 0.80, "Response-enriched", 2),
    )
    for mask, color, alpha, label, zorder in scatter_specs:
        ax.scatter(
            volcano_df.loc[mask, "logFC"],
            volcano_df.loc[mask, "y_plot"],
            s=22,
            c=color,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            label=label,
            zorder=zorder,
        )
    for mask, color, alpha, zorder in (
        (capped_mask & grey_mask, "#C7C7C7", 0.45, 3),
        (capped_mask & colored_silent, "#4575B4", 0.95, 4),
        (capped_mask & colored_response, "#D73027", 0.95, 4),
    ):
        ax.scatter(
            volcano_df.loc[mask, "logFC"],
            volcano_df.loc[mask, "y_plot"],
            s=24,
            c=color,
            marker="^",
            alpha=alpha,
            linewidths=0,
            zorder=zorder,
        )
    visual_y_cutoff_plot = np.interp(
        visual_neglog10_fdr_cutoff,
        y_breaks,
        y_positions,
    )
    ax.axvline(0, color="#333333", linewidth=0.7, zorder=0)
    ax.axvline(visual_logfc_cutoff, color="#8A8A8A", linestyle="--", linewidth=0.7, zorder=0)
    ax.axvline(-visual_logfc_cutoff, color="#8A8A8A", linestyle="--", linewidth=0.7, zorder=0)
    ax.axhline(visual_y_cutoff_plot, color="#8A8A8A", linestyle="--", linewidth=0.7, zorder=0)
    label_order = ["NCDN", "SNAP25", "SCG2", "MAG", "MBP", "PLP1"]
    label_categories = {
        "NCDN": "Dopamine response enriched",
        "SNAP25": "Dopamine response enriched",
        "SCG2": "Dopamine response enriched",
        "MAG": "Dopamine silent enriched",
        "MBP": "Dopamine silent enriched",
        "PLP1": "Dopamine silent enriched",
    }
    label_offsets = {
        "NCDN": (0.06, 0.10),
        "SNAP25": (0.06, 0.10),
        "SCG2": (0.06, 0.10),
        "MAG": (-0.06, -0.40),
        "MBP": (-0.06, -0.16),
        "PLP1": (-0.06, 0.08),
    }
    label_df = volcano_df.assign(gene_upper=volcano_df["gene"].astype(str).str.upper())
    label_df = label_df[
        label_df["gene_upper"].isin(label_order)
        & label_df["is_colored"]
        & label_df["category"].eq(label_df["gene_upper"].map(label_categories))
    ].drop_duplicates("gene_upper")
    label_df["label_order"] = pd.Categorical(label_df["gene_upper"], label_order, ordered=True)
    for _, row in label_df.sort_values("label_order").iterrows():
        x_offset, y_offset = label_offsets[row["gene_upper"]]
        ax.text(
            row["logFC"] + x_offset,
            min(row["y_plot"] + y_offset, y_positions[-1] - 0.18),
            row["gene_upper"],
            fontsize=8,
            color="black",
            ha="left" if row["logFC"] >= 0 else "right",
            va="bottom",
            clip_on=True,
            zorder=5,
        )
    x_limit = max(np.nanpercentile(np.abs(volcano_df["logFC"]), 99.5), visual_logfc_cutoff * 3.35) * 1.15
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(y_positions[0], y_positions[-1] + 0.18)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(int(value)) for value in y_breaks])
    ax.set_xlabel("log2 fold change\nDopamine response vs silent", fontsize=11)
    ax.set_ylabel("-log10 adjusted p-value\n(compressed scale)", fontsize=11)
    ax.set_title("Dopamine response vs silent", fontsize=12, pad=7)
    if capped_mask.any():
        ax.text(-x_limit * 0.98, y_positions[-1] + 0.05, f"▲ > {int(y_cap)}", fontsize=8, ha="left", va="bottom")
    ax.text(visual_logfc_cutoff + 0.03, 0.08, "logFC = 1.0", fontsize=7.5, color="#666666", rotation=90)
    ax.text(-visual_logfc_cutoff - 0.03, 0.08, "logFC = -1.0", fontsize=7.5, color="#666666", ha="right", rotation=90)
    ax.text(-x_limit * 0.98, visual_y_cutoff_plot + 0.05, "-log10(FDR) = 5", fontsize=7.5, color="#666666")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", handletextpad=0.2, borderpad=0.2, labelspacing=0.35, markerscale=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9.5, width=0.8, length=3)
    fig.tight_layout()
    return fig, ax
