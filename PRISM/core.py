"""Core PRISM utilities for graphs, clustering, domains, and missingness."""
import os
import random
from collections.abc import Mapping
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data
# ============================================================
# 1. Reproducibility and array utilities
# ============================================================
def set_seed(seed=2024):
    """Set random seeds for Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
def select_best_device():
    """Select the CUDA device with the most currently free memory, or CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_index = 0
    best_free = -1
    for index in range(torch.cuda.device_count()):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(index)
        except Exception:
            free_bytes = 0
        if free_bytes > best_free:
            best_index = index
            best_free = free_bytes
    return torch.device(f"cuda:{best_index}")
def set_prism_plot_style():
    """Apply the compact tutorial plotting style used by PRISM visualizations."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "savefig.dpi": 300,
            "savefig.transparent": False,
        }
    )
def to_dense(X):
    """Convert sparse/dense matrix to dense numpy array."""
    return X.toarray() if sp.issparse(X) else np.asarray(X)
def pca(adata, use_reps=None, n_comps=10, random_state=2024):
    """
    Deterministic PCA wrapper.
    Parameters
    ----------
    adata:
        AnnData object.
    use_reps:
        If not None, use adata.obsm[use_reps].
        Otherwise use adata.X.
    n_comps:
        Number of PCA components.
    """
    X = np.asarray(adata.obsm[use_reps]) if use_reps is not None else to_dense(adata.X)
    n_comps = min(int(n_comps), X.shape[0] - 1, X.shape[1])
    if n_comps <= 0:
        return np.empty((X.shape[0], 0), dtype=np.float32)
    return PCA(n_components=n_comps, svd_solver="full", random_state=random_state).fit_transform(X).astype(np.float32)
def load_checkpoint_state(model_path, device):
    """Load checkpoint saved by train_PRISM or train_PRISM_DM."""
    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        loss = checkpoint.get("loss", np.nan)
    else:
        state_dict = checkpoint
        loss = np.nan
    if isinstance(loss, torch.Tensor):
        loss = loss.detach().cpu().item()
    return state_dict, float(loss) if np.isscalar(loss) else loss
# ============================================================
# 2. Spatial graph and PyG conversion
# ============================================================
def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model="Radius", verbose=True):
    """
    Construct spatial neighbor graph and save it to adata.uns['Spatial_Net'].
    model:
        "Radius": neighbors within rad_cutoff.
        "KNN": k nearest neighbors.
    """
    if model not in ["Radius", "KNN"]:
        raise ValueError("model must be 'Radius' or 'KNN'.")
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] not found.")
    if verbose:
        print("------Calculating spatial graph...")
    coor = pd.DataFrame(adata.obsm["spatial"], index=adata.obs.index)
    coor.columns = ["imagerow", "imagecol"]
    if model == "Radius":
        if rad_cutoff is None:
            raise ValueError("rad_cutoff is required when model='Radius'.")
        nbrs = NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        knn_list = [pd.DataFrame(zip([i] * len(indices[i]), indices[i], distances[i])) for i in range(len(indices))]
    else:
        if k_cutoff is None:
            raise ValueError("k_cutoff is required when model='KNN'.")
        nbrs = NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        knn_list = [pd.DataFrame(zip([i] * indices.shape[1], indices[i], distances[i])) for i in range(indices.shape[0])]
    spatial_net = pd.concat(knn_list, ignore_index=True)
    spatial_net.columns = ["Cell1", "Cell2", "Distance"]
    spatial_net = spatial_net.loc[spatial_net["Distance"] > 0].copy()
    id_to_cell = dict(zip(range(coor.shape[0]), np.asarray(coor.index)))
    spatial_net["Cell1"] = spatial_net["Cell1"].map(id_to_cell)
    spatial_net["Cell2"] = spatial_net["Cell2"].map(id_to_cell)
    adata.uns["Spatial_Net"] = spatial_net
    if verbose:
        print(f"The graph contains {spatial_net.shape[0]} edges, {adata.n_obs} cells.")
        print(f"{spatial_net.shape[0] / adata.n_obs:.4f} neighbors per cell on average.")
def Stats_Spatial_Net(adata):
    """Plot neighbor-count distribution of adata.uns['Spatial_Net']."""
    if "Spatial_Net" not in adata.uns:
        raise KeyError("adata.uns['Spatial_Net'] not found. Please run Cal_Spatial_Net first.")
    num_edge = adata.uns["Spatial_Net"]["Cell1"].shape[0]
    mean_edge = num_edge / adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns["Spatial_Net"]["Cell1"])) / adata.shape[0]
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.bar(plot_df.index, plot_df)
    ax.set_ylabel("Percentage")
    ax.set_xlabel("")
    ax.set_title(f"Number of Neighbors (Mean={mean_edge:.2f})")
    plt.show()
def Transfer_pytorch_Data(adata):
    """Convert AnnData with adata.uns['Spatial_Net'] to PyG Data."""
    if "Spatial_Net" not in adata.uns:
        raise KeyError("adata.uns['Spatial_Net'] not found. Please run Cal_Spatial_Net first.")
    graph_df = adata.uns["Spatial_Net"].copy()
    cells = np.asarray(adata.obs_names)
    cell_to_id = dict(zip(cells, range(cells.shape[0])))
    graph_df["Cell1"] = graph_df["Cell1"].map(cell_to_id)
    graph_df["Cell2"] = graph_df["Cell2"].map(cell_to_id)
    graph = sp.coo_matrix(
        (np.ones(graph_df.shape[0]), (graph_df["Cell1"], graph_df["Cell2"])),
        shape=(adata.n_obs, adata.n_obs),
    )
    graph = graph + sp.eye(graph.shape[0])
    edge_list = np.nonzero(graph)
    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if not X.flags.c_contiguous:
        X = np.ascontiguousarray(X)
    return Data(edge_index=torch.LongTensor(np.array([edge_list[0], edge_list[1]])), x=torch.from_numpy(X))
def rewire_spatial_net(adata, noise_ratio=0.0, seed=2024, spatial_net_key="Spatial_Net", spatial_key="spatial"):
    """Randomly replace a fraction of spatial-graph destinations while preserving edge count."""
    noise_ratio = float(noise_ratio)
    if noise_ratio == 0.0:
        return adata
    if spatial_net_key not in adata.uns:
        raise KeyError(f"adata.uns['{spatial_net_key}'] not found. Run Cal_Spatial_Net first.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] not found.")
    graph = adata.uns[spatial_net_key].copy()
    required = {"Cell1", "Cell2"}
    if not required.issubset(graph.columns):
        raise ValueError(f"adata.uns['{spatial_net_key}'] must contain Cell1 and Cell2 columns.")
    cell_ids = np.asarray(adata.obs_names.astype(str))
    graph["Cell1"] = graph["Cell1"].astype(str)
    graph["Cell2"] = graph["Cell2"].astype(str)
    if "Distance" in graph.columns:
        graph["Distance"] = graph["Distance"].astype(float)
    rng = np.random.default_rng(int(seed))
    neighbors = {}
    for src, dst in zip(graph["Cell1"], graph["Cell2"]):
        neighbors.setdefault(src, set()).add(dst)
    obs_pos = {name: i for i, name in enumerate(cell_ids)}
    for row_i in np.flatnonzero(rng.random(graph.shape[0]) < noise_ratio):
        src = graph.iloc[row_i]["Cell1"]
        old_dst = graph.iloc[row_i]["Cell2"]
        candidates = [x for x in cell_ids if x != src and x not in (neighbors.get(src, set()) - {old_dst})]
        if not candidates:
            candidates = [x for x in cell_ids if x != src]
        if not candidates:
            continue
        new_dst = str(rng.choice(candidates))
        graph.iat[row_i, graph.columns.get_loc("Cell2")] = new_dst
        neighbors.setdefault(src, set()).discard(old_dst)
        neighbors[src].add(new_dst)
        if "Distance" in graph.columns:
            p1 = np.asarray(adata.obsm[spatial_key][obs_pos[src]], dtype=float)
            p2 = np.asarray(adata.obsm[spatial_key][obs_pos[new_dst]], dtype=float)
            graph.iat[row_i, graph.columns.get_loc("Distance")] = float(np.linalg.norm(p1 - p2))
    adata.uns[spatial_net_key] = graph
    return adata
# ============================================================
# 3. Embedding post-processing
# ============================================================
def add_prism_interaction_pca(
    adata_source,
    adata_target,
    base_key="PRISM_emb_base",
    out_key="PRISM_emb",
    src_dim=32,
    tgt_dim=32,
    n_components=8,
    random_state=2024,
):
    """
    Compute final PRISM embedding after training.
    base_emb = concat(tf_src, tf_tgt)
    interaction_i = outer(tf_src_i, tf_tgt_i).flatten()
    PRISM_emb = concat(tf_src, tf_tgt, PCA(interaction))
    """
    if base_key not in adata_target.obsm:
        raise KeyError(f"{base_key} not found in adata_target.obsm. Please run train_PRISM first.")
    base_emb = np.asarray(adata_target.obsm[base_key], dtype=np.float32)
    if src_dim + tgt_dim != base_emb.shape[1]:
        raise ValueError(
            f"src_dim + tgt_dim must equal base embedding dim. "
            f"Got {src_dim}+{tgt_dim}, base_dim={base_emb.shape[1]}."
        )
    tf_src = base_emb[:, :src_dim]
    tf_tgt = base_emb[:, src_dim:src_dim + tgt_dim]
    interaction = np.einsum("ni,nj->nij", tf_src, tf_tgt).reshape(base_emb.shape[0], -1)
    n_components_eff = min(int(n_components), interaction.shape[0], interaction.shape[1])
    int_pca = PCA(n_components=n_components_eff, svd_solver="full", random_state=random_state).fit_transform(interaction)
    prism_emb = np.concatenate([tf_src, tf_tgt, int_pca], axis=1).astype(np.float32)
    adata_source.obsm[out_key] = prism_emb
    adata_target.obsm[out_key] = prism_emb
    adata_source.obsm[f"{out_key}_interaction_pca"] = int_pca.astype(np.float32)
    adata_target.obsm[f"{out_key}_interaction_pca"] = int_pca.astype(np.float32)
    return adata_source, adata_target
def apply_interaction_embedding(
    adata_source,
    adata_target,
    *,
    enabled=False,
    base_key="PRISM_emb_base",
    out_key="PRISM_emb",
    src_dim=32,
    tgt_dim=32,
    n_components=16,
    random_state=2024,
):
    """Create the optional interaction-PCA embedding or keep the base embedding."""
    if enabled:
        return add_prism_interaction_pca(
            adata_source,
            adata_target,
            base_key=base_key,
            out_key=out_key,
            src_dim=src_dim,
            tgt_dim=tgt_dim,
            n_components=n_components,
            random_state=random_state,
        )
    if base_key not in adata_source.obsm:
        raise KeyError(f"{base_key} not found in adata_source.obsm. Please run train_PRISM first.")
    adata_source.obsm[out_key] = np.asarray(adata_source.obsm[base_key], dtype=np.float32)
    adata_target.obsm[out_key] = np.asarray(adata_target.obsm[base_key], dtype=np.float32)
    return adata_source, adata_target
# ============================================================
# 4. Clustering
# ============================================================
def mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb", random_seed=2024):
    """Clustering using R mclust."""
    set_seed(random_seed)
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()
    robjects.r["set.seed"](int(random_seed))
    rmclust = robjects.r["Mclust"]
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(np.asarray(adata.obsm[used_obsm])), int(num_cluster), modelNames)
    labels = np.asarray(res[-2]).astype(int)
    adata.obs["mclust"] = pd.Categorical(labels)
    return adata
def search_res(adata, n_clusters, method="leiden", use_rep="emb", start=0.1, end=3.0, increment=0.01, random_seed=2024):
    """Search Leiden/Louvain resolution matching the target cluster number."""
    print("Searching resolution...")
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep, random_state=random_seed)
    for res in sorted(np.arange(start, end, increment), reverse=True):
        if method == "leiden":
            sc.tl.leiden(adata, random_state=random_seed, resolution=res)
            count_unique = adata.obs["leiden"].nunique()
        elif method == "louvain":
            sc.tl.louvain(adata, random_state=random_seed, resolution=res)
            count_unique = adata.obs["louvain"].nunique()
        else:
            raise ValueError("method must be 'leiden' or 'louvain'.")
        print(f"resolution={res}, cluster number={count_unique}")
        if count_unique == n_clusters:
            return res
    raise RuntimeError("Resolution is not found. Please try bigger range or smaller step.")
def clustering(
    adata,
    n_clusters=7,
    key="emb",
    add_key="PRISM",
    method="mclust",
    start=0.1,
    end=3.0,
    increment=0.01,
    use_pca=False,
    n_comps=20,
    random_seed=2024,
):
    """
    Cluster cells based on adata.obsm[key].
    For PRISM_emb_base / PRISM_emb, use_pca=False is recommended because PRISM
    already outputs low-dimensional embeddings and extra PCA may introduce instability.
    """
    if key not in adata.obsm:
        raise KeyError(f"adata.obsm['{key}'] not found.")
    set_seed(random_seed)
    used_key = key
    if use_pca:
        used_key = f"{key}_pca"
        adata.obsm[used_key] = pca(adata, use_reps=key, n_comps=n_comps, random_state=random_seed)
    if method == "mclust":
        mclust_R(adata, used_obsm=used_key, num_cluster=n_clusters, random_seed=random_seed)
        adata.obs[add_key] = adata.obs["mclust"]
    elif method == "leiden":
        res = search_res(adata, n_clusters, method=method, use_rep=used_key, start=start, end=end, increment=increment, random_seed=random_seed)
        sc.tl.leiden(adata, random_state=random_seed, resolution=res)
        adata.obs[add_key] = adata.obs["leiden"]
    elif method == "louvain":
        res = search_res(adata, n_clusters, method=method, use_rep=used_key, start=start, end=end, increment=increment, random_seed=random_seed)
        sc.tl.louvain(adata, random_state=random_seed, resolution=res)
        adata.obs[add_key] = adata.obs["louvain"]
    else:
        raise ValueError("method must be one of ['mclust', 'leiden', 'louvain'].")
    return adata
_FALLBACK_PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#11A333",
    "#8c564b",
    "#e377c2",
    "#9268AD",
    "#00C2A0",
    "#d62728",
    "#8C9093",
    "#BCBE32",
    "#17becf",
    "#8CD0FF",
    "#ffbb78",
    "#98df8a",
    "#ad494a",
    "#0AA6D8",
)
# Keep biologically meaningful colors stable across tutorials and data sets.
_KNOWN_DOMAIN_PRESETS = {
    "tonsil": {
        "final_annot": {
            "connective & epithelial tissue": "#8c564b",
            "germinal center": "#ff7f0e",
            "lymphoid follicle": "#1f77b4",
            "tonsillar parenchyma": "#11A333",
        },
    },
    "lymph": {
        "final_annot": {
            "capsule": "#e377c2",
            "cortex": "#1f77b4",
            "follicle": "#00C2A0",
            "hilum": "#8c564b",
            "medulla cords": "#d62728",
            "medulla sinuses": "#ff7f0e",
            "medulla vessels": "#8C9093",
            "pericapsular adipose tissue": "#9268AD",
            "subcapsular sinus": "#BCBE32",
            "trabeculae": "#0AA6D8",
        },
    },
    "mouse_thymus": {
        "final_annot": {
            "1-Medulla (SP T, mTEC, DC)": "#e377c2",
            "2-Corticomedullary junction (CMJ)": "#ff7f0e",
            "3-Inner cortex region 1 (DN T, DP T, cTEC)": "#00C2A0",
            "4-Middle cortex region 2 (DN T, DP T, cTEC)": "#1f77b4",
            "5-Outer cortex region 3 (DN T, DP T, cTEC)": "#9268AD",
            "6-Connective tissue capsule (fibroblast)": "#8C9093",
            "7-Subcapsular zone (DN T)": "#d62728",
            "8-Connective tissue capsule (fibroblast, RBC, myeloid)": "#8c564b",
        },
    },
    "emb_mouse_e13.5": {
        "Combined_Clusters_annotation": {
            "Basal_plate_of_hindbrain": "#1f77b4",
            "Cartilage_1": "#ff7f0e",
            "Cartilage_2": "#11A333",
            "Cartilage_3": "#8CD0FF",
            "Cartilage_4": "#9268AD",
            "DPallm": "#8c564b",
            "DPallv": "#e377c2",
            "Diencephalon_and_hindbrain": "#BCBE32",
            "Mesenchyme": "#17becf",
            "Midbrain": "#8C9093",
            "Muscle": "#ffbb78",
            "Primary_brain_1": "#98df8a",
        },
    },
    "emb_mouse_e15.5": {
        "Combined_Clusters_annotation": {
            "Basal_plate_of_hindbrain": "#9268AD",
            "Cartilage_2": "#ff7f0e",
            "Cartilage_3": "#d62728",
            "DPallm": "#98df8a",
            "DPallv": "#ffbb78",
            "Diencephalon_and_hindbrain": "#17becf",
            "Mesenchyme": "#e377c2",
            "Midbrain": "#00C2A0",
            "Muscle": "#BCBE32",
            "Primary_brain_1": "#8C9093",
            "Subpallium_2": "#8c564b",
            "Thalamus": "#1f77b4",
        },
    },
    "emb_mouse_e18.5": {
        "Combined_Clusters_annotation": {
            "Basal_plate_of_hindbrain": "#9268AD",
            "Cartilage_1": "#BCBE32",
            "Cartilage_2": "#e377c2",
            "Cartilage_3": "#d62728",
            "Cartilage_4": "#ff7f0e",
            "DPallm": "#98df8a",
            "DPallv": "#ffbb78",
            "Diencephalon_and_hindbrain": "#17becf",
            "Mesenchyme": "#8CD0FF",
            "Midbrain": "#8C9093",
            "Muscle": "#ad494a",
            "Subpallium_1": "#8c564b",
            "Subpallium_2": "#aa40fc",
            "Thalamus": "#1f77b4",
        },
    },
    "pd_human_brain_a1": {
        "new_label": {
            "CI": "#1f77b4",
            "Cd_dopamine": "#ff7f0e",
            "Cd_not_dopamine": "#11A333",
        },
    },
    "pd_human_brain_b1": {
        "new_label": {
            "CI": "#1f77b4",
            "Cd_dopamine": "#ff7f0e",
            "Cd_not_dopamine": "#11A333",
        },
    },
    "pd_human_brain_c1": {
        "new_label": {
            "ACB_dopamine": "#8c564b",
            "ACB_not_dopamine": "#17becf",
            "CI": "#1f77b4",
            "Cd_dopamine": "#ff7f0e",
            "Cd_not_dopamine": "#11A333",
        },
    },
    # P22 has no biological reference annotation in the domain-vs workflow;
    # these labels are stable display names for the PRISM/reference clusters.
    "p22_mouse_brain": {
        "__default__": {
            **{
                f"PRISM_cluster{i}": color
                for i, color in enumerate(
                    (
                        "#1f77b4",
                        "#ff7f0e",
                        "#00C2A0",
                        "#d62728",
                        "#9268AD",
                        "#8c564b",
                        "#e377c2",
                        "#BCBE32",
                        "#17becf",
                        "#9edae5",
                        "#ffbb78",
                        "#98df8a",
                    ),
                    start=1,
                )
            },
            **{
                f"cluster{i}": color
                for i, color in enumerate(
                    (
                        "#1f77b4",
                        "#ff7f0e",
                        "#00C2A0",
                        "#d62728",
                        "#9268AD",
                        "#8c564b",
                        "#e377c2",
                        "#BCBE32",
                        "#17becf",
                        "#9edae5",
                        "#ffbb78",
                        "#98df8a",
                    ),
                    start=1,
                )
            },
        },
    },
    # COAD cell-level domain notebooks use spatial_cluster 0-4 and preserve
    # this explicitly curated order instead of Scanpy's category defaults.
    "coad": {
        "__default__": {
            **{
                str(i): color
                for i, color in enumerate(
                    ("#1f78b4", "#ffbb78", "#ff7f0e", "#9268AD", "#11A333")
                )
            },
            **{
                f"cluster{i + 1}": color
                for i, color in enumerate(
                    ("#1f77b4", "#ff7f0e", "#11A333", "#d62728", "#9268AD")
                )
            },
        },
    },
}
# Reuse the same curated labels when a tutorial stores its reference
# annotations under a workflow-specific key rather than the original key.
_KNOWN_DOMAIN_PRESETS["mouse_thymus"]["reference_domain"] = dict(
    _KNOWN_DOMAIN_PRESETS["mouse_thymus"]["final_annot"]
)
_KNOWN_DOMAIN_PRESETS["coad"]["spatial_cluster"] = dict(
    zip("01234", ("#1f78b4", "#ffbb78", "#ff7f0e", "#9268AD", "#11A333"))
)
# P22 has no biological reference domain. Its plotted PRISM domains are kept
# as 1, 2, ... with the same curated colors used by the original tutorial.
_P22_NUMERIC_DOMAIN_COLORS = {
    str(i): color
    for i, color in enumerate(
        (
            "#1f77b4", "#ff7f0e", "#00C2A0", "#d62728", "#9268AD", "#8c564b",
            "#e377c2", "#BCBE32", "#17becf", "#9edae5", "#ffbb78", "#98df8a",
        ),
        start=1,
    )
}
_KNOWN_DOMAIN_PRESETS["p22_mouse_brain"]["__default__"] = {
    **_P22_NUMERIC_DOMAIN_COLORS,
    **_KNOWN_DOMAIN_PRESETS["p22_mouse_brain"]["__default__"],
}
_KNOWN_DOMAIN_PRESETS["p22_mouse_brain"]["reference_domain"] = dict(
    _P22_NUMERIC_DOMAIN_COLORS
)
def _as_label_list(labels):
    if labels is None:
        return None
    values = list(labels)
    if len(values) != len(set(values)):
        raise ValueError("label_order must not contain duplicate labels.")
    return values
def _infer_dataset_name(adata):
    for key in ("dataset_name", "dataset", "sample", "sample_id"):
        value = adata.uns.get(key)
        if isinstance(value, str):
            return value
    return None
def _resolve_preset_name(dataset_name):
    if dataset_name is None:
        return None
    normalized = str(dataset_name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _KNOWN_DOMAIN_PRESETS:
        return normalized
    aliases = {
        "mouse_brain_s1": "emb_mouse_e13.5",
        "mouse_brain_s2": "emb_mouse_e15.5",
        "mouse_brain_s3": "emb_mouse_e18.5",
        "embryonic_mouse_brain_s1": "emb_mouse_e13.5",
        "embryonic_mouse_brain_s2": "emb_mouse_e15.5",
        "embryonic_mouse_brain_s3": "emb_mouse_e18.5",
        "embryonic_mouse_brain_slice1": "emb_mouse_e13.5",
        "embryonic_mouse_brain_slice2": "emb_mouse_e15.5",
        "embryonic_mouse_brain_slice3": "emb_mouse_e18.5",
        "pd_brain": "pd_human_brain_c1",
        "pd_human_brain": "pd_human_brain_c1",
        "p22": "p22_mouse_brain",
        "p22_mouse": "p22_mouse_brain",
        "coad_cell": "coad",
        "coad_cell_data": "coad",
    }
    if normalized in aliases:
        return aliases[normalized]
    if "tonsil" in normalized:
        return "tonsil"
    if "thymus" in normalized:
        return "mouse_thymus"
    if "pd" in normalized and "brain" in normalized:
        for slice_id in ("a1", "b1", "c1"):
            if slice_id in normalized:
                return f"pd_human_brain_{slice_id}"
        return "pd_human_brain_c1"
    if "p22" in normalized:
        return "p22_mouse_brain"
    if "mouse" in normalized and ("brain" in normalized or "emb" in normalized):
        for suffix, preset_name in (
            ("e13.5", "emb_mouse_e13.5"),
            ("e13_5", "emb_mouse_e13.5"),
            ("s1", "emb_mouse_e13.5"),
            ("slice1", "emb_mouse_e13.5"),
            ("e15.5", "emb_mouse_e15.5"),
            ("e15_5", "emb_mouse_e15.5"),
            ("s2", "emb_mouse_e15.5"),
            ("slice2", "emb_mouse_e15.5"),
            ("e18.5", "emb_mouse_e18.5"),
            ("e18_5", "emb_mouse_e18.5"),
            ("s3", "emb_mouse_e18.5"),
            ("slice3", "emb_mouse_e18.5"),
        ):
            if suffix in normalized:
                return preset_name
    if "coad" in normalized:
        return "coad"
    if "lymph" in normalized:
        return "lymph"
    return None
def _infer_preset_from_labels(labels, label_key):
    """Infer a preset only when its labels unambiguously cover all categories."""
    label_set = set(labels)
    label_set_strings = {str(label) for label in labels}
    candidates = []
    for preset_name, by_key in _KNOWN_DOMAIN_PRESETS.items():
        preset = by_key.get(label_key, by_key.get("__default__", {}))
        preset_strings = {str(label) for label in preset}
        if label_set and (
            label_set.issubset(preset)
            or label_set_strings.issubset(preset_strings)
        ):
            candidates.append((preset_name, preset))
    if len(candidates) == 1:
        return candidates[0][0]
    return None
def get_domain_label_order(dataset_name, label_key=None):
    """Return the curated display order for a known dataset, if available."""
    preset_name = _resolve_preset_name(dataset_name)
    if preset_name is None:
        return None
    by_key = _KNOWN_DOMAIN_PRESETS.get(preset_name, {})
    labels = by_key.get(label_key, by_key.get("__default__", {}))
    return list(labels) or None
def get_domain_palette(
    labels,
    *,
    label_key=None,
    dataset_name=None,
    palette=None,
    label_to_color=None,
):
    """Return a stable ``{label: color}`` mapping for discrete domain labels.
    Explicit ``label_to_color`` takes priority, then a supplied ``palette``, then a
    known dataset preset, and finally a deterministic generic palette.  This lets
    new data sets work without Scanpy assigning colors from category order.
    """
    labels = _as_label_list(labels) or []
    if len(labels) == 0:
        return {}
    explicit = {} if label_to_color is None else dict(label_to_color)
    if palette is not None:
        if isinstance(palette, Mapping):
            explicit = {**dict(palette), **explicit}
        else:
            palette = list(palette)
            if len(palette) < len(labels):
                raise ValueError(
                    f"palette has {len(palette)} colors for {len(labels)} labels."
                )
            explicit = {
                **{label: color for label, color in zip(labels, palette)},
                **explicit,
            }
    label_strings = {str(label) for label in labels}
    explicit_strings = {str(label): color for label, color in explicit.items()}
    unknown = {
        label
        for label in explicit
        if label not in labels and str(label) not in label_strings
    }
    if unknown:
        raise ValueError(
            "palette/label_to_color contains labels absent from label_order: "
            + ", ".join(map(str, sorted(unknown, key=str)))
        )
    preset_name = _resolve_preset_name(dataset_name)
    if preset_name is None:
        preset_name = _infer_preset_from_labels(labels, label_key)
    by_key = _KNOWN_DOMAIN_PRESETS.get(preset_name, {})
    preset = by_key.get(label_key, by_key.get("__default__", {}))
    preset_strings = {str(label): color for label, color in preset.items()}
    colors = {}
    fallback_i = 0
    for label in labels:
        if label in explicit:
            colors[label] = explicit[label]
        elif str(label) in explicit_strings:
            colors[label] = explicit_strings[str(label)]
        elif label in preset:
            colors[label] = preset[label]
        elif str(label) in preset_strings:
            colors[label] = preset_strings[str(label)]
        else:
            while (
                fallback_i < len(_FALLBACK_PALETTE)
                and _FALLBACK_PALETTE[fallback_i] in colors.values()
            ):
                fallback_i += 1
            # More categories than bundled colors: cycle deterministically
            # instead of looping forever while seeking a unique color.
            colors[label] = _FALLBACK_PALETTE[fallback_i % len(_FALLBACK_PALETTE)]
            fallback_i += 1
    return colors
def register_domain_palette(
    adata,
    label_key,
    *,
    label_order=None,
    dataset_name=None,
    palette=None,
    label_to_color=None,
    ordered=False,
):
    """Categorize labels and register Scanpy colors in ``adata.uns``.
    The order in ``label_order`` controls both category ordering and the order of
    colors written to ``adata.uns[f'{label_key}_colors']``.
    """
    if label_key not in adata.obs:
        raise KeyError(f"adata.obs['{label_key}'] not found.")
    labels = _as_label_list(label_order)
    if labels is None:
        values = adata.obs[label_key]
        if isinstance(values.dtype, pd.CategoricalDtype):
            labels = list(values.cat.categories)
        else:
            labels = list(pd.unique(values.dropna()))
    observed = set(adata.obs[label_key].dropna())
    labels_for_check = set(labels)
    if not labels_for_check.issuperset(observed):
        # Compare through strings so integer/object labels are accepted while
        # still producing a useful error for genuinely omitted categories.
        observed_as_str = {str(value) for value in observed}
        labels_as_str = {str(value) for value in labels}
        unknown = observed_as_str.difference(labels_as_str)
    else:
        unknown = observed.difference(labels_for_check)
    if unknown:
        raise ValueError(
            f"label_order does not include observed '{label_key}' labels: "
            + ", ".join(map(str, sorted(unknown, key=str)))
        )
    dataset_name = dataset_name or _infer_dataset_name(adata)
    colors = get_domain_palette(
        labels,
        label_key=label_key,
        dataset_name=dataset_name,
        palette=palette,
        label_to_color=label_to_color,
    )
    adata.obs[label_key] = pd.Categorical(
        adata.obs[label_key], categories=labels, ordered=ordered
    )
    adata.uns[f"{label_key}_colors"] = [colors[label] for label in labels]
    return colors
def _cluster_sort_key(value):
    """Sort numeric cluster IDs numerically and other IDs lexicographically."""
    text = str(value)
    try:
        return 0, float(text)
    except (TypeError, ValueError):
        return 1, text
def relabel_clusters_for_display(
    adata,
    cluster_key,
    *,
    aligned_key=None,
    label_order=None,
    cluster_to_label=None,
    palette=None,
    label_to_color=None,
    dataset_name=None,
    ordered=False,
):
    """Give clusters stable display labels when no reference annotation exists.
    With no explicit mapping, numeric cluster IDs are sorted numerically and
    mapped to ``cluster1``, ``cluster2``, ... (or to ``label_order``).  This is
    the reusable form of the P22/COAD notebook visualization logic.
    """
    if cluster_key not in adata.obs:
        raise KeyError(f"adata.obs['{cluster_key}'] not found.")
    aligned_key = aligned_key or f"{cluster_key}_display"
    observed = adata.obs[cluster_key].dropna().astype(str)
    clusters = sorted(pd.unique(observed), key=_cluster_sort_key)
    if len(clusters) == 0:
        raise ValueError(f"adata.obs['{cluster_key}'] contains no cluster labels.")
    if cluster_to_label is None:
        labels = _as_label_list(label_order)
        if labels is None:
            labels = [f"cluster{i + 1}" for i in range(len(clusters))]
        labels = [str(label) for label in labels]
        if len(labels) < len(clusters):
            raise ValueError(
                f"label_order has {len(labels)} labels for {len(clusters)} clusters."
            )
        mapping = dict(zip(clusters, labels))
    else:
        mapping = {
            str(cluster): str(label)
            for cluster, label in dict(cluster_to_label).items()
        }
        missing = set(clusters).difference(mapping)
        if missing:
            raise ValueError(
                "cluster_to_label is missing observed clusters: "
                + ", ".join(map(str, sorted(missing, key=_cluster_sort_key)))
            )
        labels = _as_label_list(label_order)
        if labels is None:
            labels = list(dict.fromkeys(mapping[cluster] for cluster in clusters))
        labels = [str(label) for label in labels]
        invalid = set(mapping.values()).difference(labels)
        if invalid:
            raise ValueError(
                "cluster_to_label maps to labels absent from label_order: "
                + ", ".join(map(str, sorted(invalid, key=str)))
            )
    adata.obs[aligned_key] = pd.Categorical(
        adata.obs[cluster_key].astype(str).map(mapping),
        categories=labels,
        ordered=ordered,
    )
    colors = register_domain_palette(
        adata,
        aligned_key,
        label_order=labels,
        dataset_name=dataset_name,
        palette=palette,
        label_to_color=label_to_color,
        ordered=ordered,
    )
    return mapping, colors
def align_cluster_to_reference(
    adata,
    cluster_key,
    reference_key,
    *,
    aligned_key=None,
    label_order=None,
    cluster_to_label=None,
    palette=None,
    label_to_color=None,
    dataset_name=None,
    ordered=False,
):
    """Align cluster IDs to reference labels and register a shared color mapping.
    If ``cluster_to_label`` is not supplied, Hungarian matching maximizes the
    overlap between clusters and reference labels.  Extra clusters are assigned
    by their reference-label majority vote.  The aligned categorical key always
    receives the same Scanpy palette as the reference labels.
    """
    for key in (cluster_key, reference_key):
        if key not in adata.obs:
            raise KeyError(f"adata.obs['{key}'] not found.")
    aligned_key = aligned_key or f"{cluster_key}_aligned"
    labels = _as_label_list(label_order)
    if labels is None:
        reference = adata.obs[reference_key]
        if isinstance(reference.dtype, pd.CategoricalDtype):
            labels = list(reference.cat.categories)
        else:
            labels = list(pd.unique(reference.dropna()))
    if len(labels) == 0:
        raise ValueError("No reference labels are available for alignment.")
    # Scanpy categories and crosstab columns must use one comparable type.  The
    # notebook workflows use string labels, and normalizing here also keeps
    # numeric/object labels deterministic for callers outside the notebooks.
    labels = [str(label) for label in labels]
    adata.obs[reference_key] = adata.obs[reference_key].astype("string")
    observed_reference = {
        str(value) for value in adata.obs[reference_key].dropna()
    }
    unknown_reference = observed_reference.difference(labels)
    if unknown_reference:
        raise ValueError(
            f"label_order does not include observed '{reference_key}' labels: "
            + ", ".join(map(str, sorted(unknown_reference, key=str)))
        )
    obs = adata.obs[[cluster_key, reference_key]].dropna().copy()
    if obs.empty:
        raise ValueError(
            f"No cells have both '{cluster_key}' and '{reference_key}' labels."
        )
    obs[cluster_key] = obs[cluster_key].astype(str)
    obs[reference_key] = obs[reference_key].astype(str)
    if cluster_to_label is None:
        contingency = pd.crosstab(obs[cluster_key], obs[reference_key])
        contingency = contingency.reindex(columns=labels, fill_value=0)
        clusters = list(contingency.index)
        row_ind, col_ind = linear_sum_assignment(-contingency.to_numpy())
        mapping = {
            contingency.index[row]: contingency.columns[col]
            for row, col in zip(row_ind, col_ind)
        }
        for cluster in clusters:
            if cluster not in mapping:
                mapping[cluster] = contingency.loc[cluster].idxmax()
    else:
        clusters = list(pd.unique(obs[cluster_key]))
        mapping = {
            str(cluster): str(label)
            for cluster, label in dict(cluster_to_label).items()
        }
        invalid = set(mapping.values()).difference(labels)
        if invalid:
            raise ValueError(
                "cluster_to_label maps to labels absent from label_order: "
                + ", ".join(map(str, sorted(invalid, key=str)))
            )
        missing = set(clusters).difference(mapping)
        if missing:
            raise ValueError(
                "cluster_to_label is missing observed clusters: "
                + ", ".join(map(str, sorted(missing, key=str)))
            )
    colors = register_domain_palette(
        adata,
        reference_key,
        label_order=labels,
        dataset_name=dataset_name,
        palette=palette,
        label_to_color=label_to_color,
        ordered=ordered,
    )
    adata.obs[aligned_key] = pd.Categorical(
        adata.obs[cluster_key].astype(str).map(mapping),
        categories=labels,
        ordered=ordered,
    )
    adata.uns[f"{aligned_key}_colors"] = [colors[label] for label in labels]
    return mapping, colors
def align_cluster_to_truth(
    adata,
    cluster_key,
    true_key,
    aligned_key,
    label_order=None,
    **kwargs,
):
    """Notebook-compatible wrapper around :func:`align_cluster_to_reference`.
    It returns only the cluster-to-label mapping, as the original evaluation
    notebooks did.  The shared colors are registered in ``adata.uns``.
    """
    mapping, _ = align_cluster_to_reference(
        adata,
        cluster_key,
        true_key,
        aligned_key=aligned_key,
        label_order=label_order,
        **kwargs,
    )
    return mapping
def known_domain_presets():
    """Return a copy of the bundled dataset-specific label/color presets."""
    return {
        dataset: {key: dict(colors) for key, colors in by_key.items()}
        for dataset, by_key in _KNOWN_DOMAIN_PRESETS.items()
    }
def simulate_celltype_missing(
    adata,
    *,
    annotation_key="annotation",
    missing_celltypes=None,
    missing_key="missing",
    observed_label=1,
    missing_label=0,
    inplace=False,
):
    """Label cells of selected annotations as missing in one modality.
    The expression matrix is left unchanged; training uses ``missing_key`` to
    mask the selected rows.  The returned indices are in AnnData row order.
    """
    if missing_celltypes is None:
        raise ValueError("missing_celltypes must contain at least one label.")
    if isinstance(missing_celltypes, str):
        missing_celltypes = [missing_celltypes]
    ad = adata if inplace else adata.copy()
    labels = ad.obs[annotation_key].astype(str)
    selected = labels.isin([str(x) for x in missing_celltypes]).to_numpy()
    missing_indices = np.flatnonzero(selected).astype(np.int64)
    observed_indices = np.flatnonzero(~selected).astype(np.int64)
    ad.obs[missing_key] = np.where(selected, missing_label, observed_label).astype(int)
    ad.uns[f"{missing_key}_celltype_missing_info"] = {
        "annotation_key": str(annotation_key),
        "missing_celltypes": [str(x) for x in missing_celltypes],
        "missing_key": str(missing_key),
        "n_missing": int(missing_indices.size),
        "n_observed": int(observed_indices.size),
    }
    ad.uns[f"{missing_key}_missing_indices"] = missing_indices
    ad.uns[f"{missing_key}_observed_indices"] = observed_indices
    return ad, missing_indices, observed_indices
def simulate_missing_sliding(
    adata,
    spatial_key: str = "spatial",
    direction: str = "h",
    missing_width: float = 0.5,
    step_ratio: float = 0.1,
    step_id: int = 1,
    label_key: str = "missing",
    lock_at_end: bool = True,
    plot: bool = True,
    figsize=(6, 5),
    point_size: float = 10.0,
    # Only used for sliding-window random missingness
    window_direction: str = "H",
    missing_ratio_in_window=None,
    random_seed=None,
    return_window_indices: bool = False,
):
    """
    Simulate missing region.
    Supported modes
    ---------------
    H / horizontal:
        Select a contiguous missing window after sorting by x coordinate.
    V / vertical:
        Select a contiguous missing window after sorting by y coordinate.
    R / random:
        Randomly select missing cells from the whole tissue.
    RW / random-window:
        First select a sliding spatial window, then randomly mask a proportion
        of cells within that window.
    Parameters
    ----------
    missing_width:
        For H/V:
            fraction of all cells in the continuous missing window.
        For R:
            fraction of all cells randomly selected as missing.
        For RW:
            fraction of all cells included in the sliding window.
    missing_ratio_in_window:
        Only used for RW.
        Fraction of cells inside the selected sliding window to be marked missing.
        Example: missing_width=0.5 and missing_ratio_in_window=0.5 means:
        select a window containing 50% of all cells, then randomly mask 50% of
        cells within that window. Overall missing ratio is about 25%.
    random_seed:
        Only used for R/RW.
        If None, step_id is used as the seed for backward compatibility.
    return_window_indices:
        Only meaningful for RW.
        If True, return (missing_indices, window_indices).
        If False, return missing_indices, same as the original function.
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] not found.")
    coords = np.asarray(adata.obsm[spatial_key])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] must be a (n_obs, 2) array.")
    # Keep the same coordinate convention as the original function
    x_coords = coords[:, 1]
    y_coords = coords[:, 0]
    if not (0.0 < missing_width <= 1.0):
        raise ValueError("missing_width must be in (0, 1].")
    if not (0.0 <= step_ratio <= 1.0):
        raise ValueError("step_ratio must be in [0, 1].")
    if step_id < 0:
        raise ValueError("step_id must be >= 0.")
    N = int(adata.shape[0])
    direction_clean = str(direction).strip().upper()
    # Use step_id as seed by default, preserving the old R-mode behavior
    if random_seed is None:
        random_seed = step_id
    window_indices = None
    # =========================
    # H: horizontal continuous missing
    # =========================
    if direction_clean in ["H", "HORIZONTAL"]:
        window = int(N * missing_width)
        window = max(1, min(window, N))
        sorted_indices = np.argsort(x_coords)
        step_size = int(N * step_ratio)
        step_size = max(1, step_size)
        start = step_id * step_size
        end = start + window
        if end > N:
            if lock_at_end:
                end = N
                start = max(0, N - window)
            else:
                end = N
        split1_indices = sorted_indices[start:end].astype(int)
    # =========================
    # V: vertical continuous missing
    # =========================
    elif direction_clean in ["V", "VERTICAL"]:
        window = int(N * missing_width)
        window = max(1, min(window, N))
        sorted_indices = np.argsort(y_coords)
        step_size = int(N * step_ratio)
        step_size = max(1, step_size)
        start = step_id * step_size
        end = start + window
        if end > N:
            if lock_at_end:
                end = N
                start = max(0, N - window)
            else:
                end = N
        split1_indices = sorted_indices[start:end].astype(int)
    # =========================
    # R: global random missing
    # =========================
    elif direction_clean in ["R", "RAND", "RANDOM"]:
        n_missing = int(N * missing_width)
        n_missing = max(1, min(n_missing, N))
        rng = np.random.default_rng(seed=random_seed)
        split1_indices = rng.choice(
            N,
            size=n_missing,
            replace=False,
        ).astype(int)
    # =========================
    # RW: random missing within a sliding window
    # =========================
    elif direction_clean in [
        "RW",
        "RANDOM_WINDOW",
        "WINDOW_RANDOM",
        "SLIDING_RANDOM",
        "SLIDING_WINDOW_RANDOM",
    ]:
        if missing_ratio_in_window is None:
            raise ValueError(
                "missing_ratio_in_window must be provided when direction='RW'."
            )
        if not (0.0 < float(missing_ratio_in_window) <= 1.0):
            raise ValueError("missing_ratio_in_window must be in (0, 1].")
        window_direction_clean = str(window_direction).strip().upper()
        if window_direction_clean in ["H", "HORIZONTAL"]:
            sorted_indices = np.argsort(x_coords)
        elif window_direction_clean in ["V", "VERTICAL"]:
            sorted_indices = np.argsort(y_coords)
        else:
            raise ValueError(
                "window_direction must be 'H' or 'V' when direction='RW'. "
                f"Got: {window_direction}"
            )
        # Select sliding window
        window = int(N * missing_width)
        window = max(1, min(window, N))
        step_size = int(N * step_ratio)
        step_size = max(1, step_size)
        start = step_id * step_size
        end = start + window
        if end > N:
            if lock_at_end:
                end = N
                start = max(0, N - window)
            else:
                end = N
        window_indices = sorted_indices[start:end].astype(int)
        # Randomly mask cells inside this window
        n_missing = int(len(window_indices) * float(missing_ratio_in_window))
        n_missing = max(1, min(n_missing, len(window_indices)))
        rng = np.random.default_rng(seed=random_seed)
        split1_indices = rng.choice(
            window_indices,
            size=n_missing,
            replace=False,
        ).astype(int)
    else:
        raise ValueError(
            "direction must be 'H', 'V', 'R', or 'RW'. "
            f"Got: '{direction}'"
        )
    # ------------------
    # Label missing and non-missing data
    # ------------------
    adata.obs[label_key] = "1"
    adata.obs.iloc[
        split1_indices,
        adata.obs.columns.get_loc(label_key),
    ] = "0"
    # ------------------
    # Visualize
    # Keep the same style as the original simulate_missing_sliding()
    # ------------------
    if plot:
        labels = adata.obs[label_key].astype(str).to_numpy()
        miss_mask = labels == "0"
        obs_mask = labels == "1"
        plt.figure(figsize=figsize)
        plt.scatter(
            y_coords[miss_mask],
            x_coords[miss_mask],
            c="gray",
            s=point_size,
            alpha=0.7,
            label="Simulation missing (label=0)",
        )
        plt.scatter(
            y_coords[obs_mask],
            x_coords[obs_mask],
            c="blue",
            s=point_size,
            alpha=0.7,
            label="Non-missing (label=1)",
        )
        if direction_clean == "RW":
            title_tag = f"RW-{window_direction_clean}"
        else:
            title_tag = direction_clean
        plt.title(f"Simulated Data Visualization ({title_tag})")
        plt.xlabel("Y Coordinate")
        plt.ylabel("X Coordinate")
        legend = plt.legend(bbox_to_anchor=(1, 0.6))
        for handle in getattr(legend, "legend_handles", getattr(legend, "legendHandles", [])):
            if hasattr(handle, "set_sizes"):
                handle.set_sizes([10.0])
        plt.show()
    if direction_clean == "RW":
        print("Total cells:", N)
        print("Window cells:", len(window_indices))
        print("Missing cells in window:", len(split1_indices))
        print("Overall missing ratio:", len(split1_indices) / N)
        print("Missing ratio within window:", len(split1_indices) / len(window_indices))
        if return_window_indices:
            return split1_indices, window_indices
    return split1_indices
def show_real_missing(adata, *, spatial_key="spatial", label_key="missing", missing_value="0", observed_value="1",
                      normalize_to_str=True, plot=True, figsize=(6, 5), s=10, alpha=0.7,
                      legend_loc=(1, 0.6), legend_marker_size=10, title="Real Missing Visualization"):
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] not found.")
    if label_key not in adata.obs.columns:
        raise KeyError(f"adata.obs['{label_key}'] not found.")
    coords = np.asarray(adata.obsm[spatial_key])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] must be a (n_obs, 2) array.")
    x_coords, y_coords = coords[:, 1], coords[:, 0]
    labels = adata.obs[label_key].to_numpy()
    if normalize_to_str:
        labels = labels.astype(str)
        miss_mask = labels == str(missing_value)
        obs_mask = labels == str(observed_value) if observed_value is not None else ~miss_mask
    else:
        miss_mask = labels == missing_value
        obs_mask = labels == observed_value if observed_value is not None else ~miss_mask
    missing_indices = np.flatnonzero(miss_mask)
    observed_indices = np.flatnonzero(obs_mask)
    if plot:
        plt.figure(figsize=figsize)
        plt.scatter(y_coords[missing_indices], x_coords[missing_indices], c="gray", s=s, alpha=alpha,
                    label=f"Missing ({label_key}={missing_value})")
        plt.scatter(y_coords[observed_indices], x_coords[observed_indices], c="blue", s=s, alpha=alpha,
                    label=f"Observed ({label_key}={observed_value})" if observed_value is not None else "Observed")
        plt.title(title)
        plt.xlabel("Y Coordinate")
        plt.ylabel("X Coordinate")
        legend = plt.legend(bbox_to_anchor=legend_loc)
        for handle in getattr(legend, "legend_handles", getattr(legend, "legendHandles", [])):
            if hasattr(handle, "set_sizes"):
                handle.set_sizes([legend_marker_size])
        plt.show()
    return missing_indices, observed_indices
