"""Training routines for single- and dual-modality PRISM workflows."""
import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .models import PRISM, PRISM_DM
from .core import (
    Transfer_pytorch_Data,
    load_checkpoint_state,
    apply_interaction_embedding,
    rewire_spatial_net,
)
from .prior import prepare_similarity_subset, validate_modality_context_prior
def get_hv_mask(adata):
    """Return highly-variable feature mask if available; otherwise use all features."""
    if "highly_variable" in adata.var.columns:
        return np.asarray(adata.var["highly_variable"].values, dtype=bool)
    return np.ones(adata.n_vars, dtype=bool)
def prepare_target_mask(
    adata_target,
    target_shape,
    device,
    mask_key="target_mask",
    fallback_mask_keys=None,
):
    """
    Prepare the target modality row mask.
    Preferred:
        adata_target.obsm["target_mask"]
    Fallback keys:
        protein_mask / adt_mask / rna_mask / atac_mask / metabolomics_mask
    """
    if fallback_mask_keys is None:
        fallback_mask_keys = ["target_mask", "protein_mask", "adt_mask", "rna_mask", "atac_mask", "metabolomics_mask"]
    candidate_keys = []
    if mask_key is not None:
        candidate_keys.append(mask_key)
    if "target_mask_key" in adata_target.uns:
        candidate_keys.append(str(adata_target.uns["target_mask_key"]))
    if "mask_key" in adata_target.uns:
        candidate_keys.append(str(adata_target.uns["mask_key"]))
    for k in fallback_mask_keys:
        if k not in candidate_keys:
            candidate_keys.append(k)
    selected_key = None
    for k in candidate_keys:
        if k in adata_target.obsm:
            selected_key = k
            break
    if selected_key is None:
        raise KeyError(
            "No valid target mask found. Expected one of: "
            f"{candidate_keys}. Please run preprocess_omics(..., data_role='target')."
        )
    mask = np.asarray(adata_target.obsm[selected_key])
    if mask.ndim != 1:
        raise ValueError(f"{selected_key} must be a 1D row mask, got shape {mask.shape}.")
    if mask.shape[0] != target_shape[0]:
        raise ValueError(f"{selected_key} rows mismatch: {mask.shape[0]} vs {target_shape[0]}.")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{selected_key} must contain binary row-mask values.")
    return torch.as_tensor(mask.astype(bool), device=device, dtype=torch.bool)
def train_PRISM(adata_source, adata_target, distance_matrix, k_top=5, hidden_dims=None, n_epochs=1000, lr=0.001,
                gradient_clipping=5.0, weight_decay=0.0001, verbose=True, random_seed=2024,
                save_loss=True, save_reconstruction=False, output_dir="Default", file_prefix="Default",
                device=torch.device("cuda:0"), patience=50, min_epochs=200, center_drop_rate=0.1,
                load_model_path=None, target_mask_key="target_mask",
                save_result_files=False,
                interaction_pca=False, interaction_pca_components=16,
                noise=0.0, spatial_key="spatial"):
    """
    Train PRISM.
    Main output:
        adata.obsm["PRISM_emb_base"] = concat(tf_src, tf_tgt)
        adata.obsm["PRISM_emb"] = base embedding or base + interaction PCA
    """
    hidden_dims = [512, 32] if hidden_dims is None else hidden_dims
    os.makedirs(output_dir, exist_ok=True)
    # 1. Reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    # 2. Copy data and prepare distance matrix
    adata_source, adata_target = adata_source.copy(), adata_target.copy()
    if isinstance(load_model_path, (bool, np.bool_)):
        load_model_path = os.path.join(output_dir, f"{file_prefix}_model.pth") if load_model_path else None
    # Optionally perturb the physical spatial graphs before PyG conversion.
    noise = float(noise)
    if noise > 0.0:
        adata_source = rewire_spatial_net(adata_source, noise_ratio=noise, seed=random_seed, spatial_key=spatial_key)
        adata_target = rewire_spatial_net(adata_target, noise_ratio=noise, seed=random_seed, spatial_key=spatial_key)
    # Large prior matrices should stay as scipy sparse CSR on CPU.
    # Only top_k_indices and non_missing_indices will be moved to GPU.
    if sp.issparse(distance_matrix):
        distance_matrix = distance_matrix.tocsr(copy=True)
        distance_matrix.eliminate_zeros()
    elif isinstance(distance_matrix, torch.Tensor):
        distance_matrix = distance_matrix.to(device=device, dtype=torch.float32)
    else:
        distance_matrix = torch.as_tensor(distance_matrix, device=device, dtype=torch.float32)
    # 3. Prepare missing indicators
    adata_source.X = sp.csr_matrix(adata_source.X)
    adata_target.X = sp.csr_matrix(adata_target.X)
    if "missing" not in adata_target.obs.columns:
        raise KeyError("adata_target.obs['missing'] is required.")
    missing_flag = adata_target.obs["missing"].astype(str).values
    missing_indices = np.where(missing_flag == "0")[0]
    non_missing_indices_np = np.where(missing_flag == "1")[0]
    if len(non_missing_indices_np) == 0:
        raise ValueError("No cells with observed target modality were found.")
    adata_target.X[missing_indices, :] = 0
    if sp.issparse(adata_target.X):
        adata_target.X.eliminate_zeros()
    # 4. Select highly-variable features
    src_var_mask = get_hv_mask(adata_source)
    tgt_var_mask = get_hv_mask(adata_target)
    adata_src_vars = adata_source[:, src_var_mask].copy()
    adata_tgt_vars = adata_target[:, tgt_var_mask].copy()
    source_data = Transfer_pytorch_Data(adata_src_vars)
    target_data = Transfer_pytorch_Data(adata_tgt_vars)
    source_data = source_data.to(device)
    target_data = target_data.to(device)
    if distance_matrix.shape[0] != target_data.x.size(0) or distance_matrix.shape[1] != target_data.x.size(0):
        raise ValueError(
            f"distance_matrix must be [n_cells, n_cells]. "
            f"Got distance_matrix={distance_matrix.shape}, n_cells={target_data.x.size(0)}."
        )
    mask_target = prepare_target_mask(adata_target, tuple(target_data.x.shape), device, mask_key=target_mask_key)
    target_labels = target_data.x
    # 5. Top-k non-missing neighbors
    top_k_indices, non_missing_indices = prepare_similarity_subset(
        distance_matrix, non_missing_indices_np, k_top,
        device=device, verbose=verbose,
    )
    # 6. Model
    model = PRISM(
        src_hidden_dims=[source_data.x.shape[1]] + hidden_dims,
        tgt_hidden_dims=[target_data.x.shape[1]] + hidden_dims,
    ).to(device)
    # 7. Cell-level masks
    n_cells = target_data.x.size(0)
    true_missing_mask = torch.zeros(n_cells, device=device, dtype=torch.bool)
    true_missing_mask[torch.as_tensor(missing_indices, device=device, dtype=torch.long)] = True
    observed_cell_mask = torch.zeros(n_cells, device=device, dtype=torch.bool)
    observed_cell_mask[non_missing_indices] = True
    # 8. Train or load pretrained weights
    if load_model_path is None:
        w_params = torch.nn.Parameter(torch.ones(4, device=device))
        optimizer = torch.optim.Adam(
            [{"params": model.parameters()}, {"params": [w_params], "lr": lr * 0.1}],
            lr=lr,
            weight_decay=weight_decay,
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.5, patience=20, verbose=verbose
        # )
        scheduler_kwargs = dict(
            mode="min",
            factor=0.5,
            patience=20,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **scheduler_kwargs,
        )
        best_loss, best_model_state, stop_counter = float("inf"), None, 0
        pbar = tqdm(
            range(1, n_epochs + 1),
            desc=f"PRISM [{file_prefix}]",
            disable=not verbose,
        )
        for epoch in pbar:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            random_center_mask = (torch.rand(n_cells, device=device) < center_drop_rate) & observed_cell_mask
            center_missing_mask = true_missing_mask | random_center_mask
            s_rec, t_rec, p_emb, s_pre, t_pre = model(
                source_data.x,
                source_data.edge_index,
                target_data.x,
                target_data.edge_index,
                top_k_indices=top_k_indices,
                non_missing_indices=non_missing_indices,
                center_missing_mask=center_missing_mask,
            )
            l_s_rec = F.mse_loss(s_rec, source_data.x)
            l_s_pre = F.mse_loss(s_pre, source_data.x)
            mask_denom = (mask_target.sum() * target_labels.size(1)).to(dtype=target_labels.dtype).clamp_min(1e-8)
            target_row_mask = mask_target.unsqueeze(1)
            l_t_rec = (F.mse_loss(t_rec, target_labels, reduction="none") * target_row_mask).sum() / mask_denom
            l_t_pre = (F.mse_loss(t_pre, target_labels, reduction="none") * target_row_mask).sum() / mask_denom
            norm_w = torch.softmax(w_params, dim=0)
            total_loss = norm_w[0] * l_s_rec + norm_w[1] * l_t_rec + norm_w[2] * l_s_pre + norm_w[3] * l_t_pre
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            scheduler.step(total_loss.item())
            current_loss = total_loss.item()
            l_t_pre_value = l_t_pre.item()
            del s_rec, t_rec, p_emb, s_pre, t_pre
            del l_s_rec, l_t_rec, l_s_pre, l_t_pre, total_loss, norm_w, mask_denom
            del random_center_mask, center_missing_mask
            if verbose:
                pbar.set_postfix({"loss": f"{current_loss:.4f}", "tgt_pre": f"{l_t_pre_value:.4f}"})
            if current_loss < best_loss:
                best_loss = current_loss
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stop_counter = 0
            elif epoch >= min_epochs:
                stop_counter += 1
            if stop_counter >= patience:
                if verbose:
                    print(f"\n>>> Early stopping at epoch {epoch}.")
                break
        if best_model_state is None:
            raise RuntimeError("Training finished without a valid best model state.")
        model_path = os.path.join(output_dir, f"{file_prefix}_model.pth")
        torch.save({"state_dict": best_model_state, "loss": best_loss}, model_path)
    else:
        best_model_state, best_loss = load_checkpoint_state(load_model_path, device)
        if verbose:
            print(f"Loaded pretrained PRISM model: {load_model_path}")
    if load_model_path is None:
        model.zero_grad(set_to_none=True)
        del optimizer, scheduler, w_params
    del mask_target, target_labels, true_missing_mask, observed_cell_mask
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    # 10. Deterministic inference: all center ADT tokens are masked.
    model.load_state_dict(best_model_state, strict=True)
    model.eval()
    with torch.no_grad():
        inference_center_mask = torch.ones(n_cells, device=device, dtype=torch.bool)
        s_rec, t_rec, p_emb, s_pre, t_pre = model(
            source_data.x,
            source_data.edge_index,
            target_data.x,
            target_data.edge_index,
            top_k_indices=top_k_indices,
            non_missing_indices=non_missing_indices,
            center_missing_mask=inference_center_mask,
        )
    base_emb_np = p_emb.detach().cpu().numpy()
    adata_source.obsm["PRISM_emb_base"] = base_emb_np
    adata_target.obsm["PRISM_emb_base"] = base_emb_np
    # Build the final embedding before exporting it so the CSV always matches
    # adata.obsm["PRISM_emb"].
    adata_source, adata_target = apply_interaction_embedding(
        adata_source,
        adata_target,
        enabled=interaction_pca,
        src_dim=hidden_dims[-1],
        tgt_dim=hidden_dims[-1],
        n_components=interaction_pca_components,
        random_state=random_seed,
    )
    _save_single_results(
        adata_source, adata_target,
        p_emb.detach(), s_pre.detach(), t_pre.detach(),
        best_loss, save_loss, save_reconstruction,
        output_dir, file_prefix, adata_tgt_vars.var_names,
        save_files=save_result_files and load_model_path is None,
        base_emb_np=base_emb_np,
        base_emb_preassigned=True,
    )
    if verbose:
        if load_model_path is None:
            print(f"PRISM training and inference complete: {file_prefix}")
        else:
            print(f"PRISM inference complete: {file_prefix}")
    return adata_source, adata_target
def _save_single_results(adata_src, adata_tgt, p_emb, s_pre, t_pre, loss, save_loss, save_recon,
                  output_dir, file_prefix, target_var_names, save_files=True,
                  base_emb_np=None, base_emb_preassigned=False):
    """
    Save embedding and prediction.
    PRISM_emb_base:
        concat(tf_src, tf_tgt)
    PRISM_emb:
        final embedding, with optional interaction PCA already applied.
    """
    if base_emb_np is None:
        base_emb_np = p_emb.detach().cpu().numpy()
    src_pred_np = s_pre.detach().cpu().numpy()
    tgt_pred_np = t_pre.detach().cpu().numpy()
    if not base_emb_preassigned:
        adata_src.obsm["PRISM_emb_base"] = base_emb_np
        adata_tgt.obsm["PRISM_emb_base"] = base_emb_np
    if "PRISM_emb" not in adata_src.obsm:
        adata_src.obsm["PRISM_emb"] = base_emb_np
    if "PRISM_emb" not in adata_tgt.obsm:
        adata_tgt.obsm["PRISM_emb"] = base_emb_np
    final_emb_np = np.asarray(adata_tgt.obsm["PRISM_emb"], dtype=np.float32)
    adata_src.obsm["PRISM_src_pred"] = src_pred_np
    adata_tgt.obsm["PRISM_tgt_pred"] = tgt_pred_np
    adata_tgt.uns["PRISM_tgt_pred_var_names"] = np.asarray(target_var_names).astype(str)
    if save_files:
        target_csv = os.path.join(output_dir, f"{file_prefix}_pre.csv")
        embed_csv = os.path.join(output_dir, f"{file_prefix}_emb.csv")
        pd.DataFrame(
            tgt_pred_np,
            index=adata_tgt.obs_names,
            columns=target_var_names,
        ).to_csv(target_csv)
        pd.DataFrame(
            final_emb_np,
            index=adata_tgt.obs_names,
        ).to_csv(embed_csv)
    if save_loss:
        adata_src.uns["PRISM_loss"] = float(loss)
        adata_tgt.uns["PRISM_loss"] = float(loss)
    if save_recon:
        if src_pred_np.shape == adata_src.X.shape:
            adata_src.layers["PRISM_src_pred"] = src_pred_np.clip(min=0)
        else:
            adata_src.obsm["PRISM_src_pred"] = src_pred_np
        if tgt_pred_np.shape == adata_tgt.X.shape:
            adata_tgt.layers["PRISM_tgt_pred"] = tgt_pred_np.clip(min=0)
        else:
            adata_tgt.obsm["PRISM_tgt_pred"] = tgt_pred_np
def build_feature_mask(adata_full, adata_used, self_missing_np, device, mask_key=None, fallback_mask_keys=None):
    """
    Build a row-level mask aligned with adata_used.
    Preferred:
        source modality -> source_mask
        target modality -> target_mask
    Fallback keys:
        rna_mask / protein_mask / adt_mask / atac_mask / metabolomics_mask
    If no mask is found:
        build row-level mask: observed rows = 1, missing rows = 0.
    """
    n_obs = adata_used.n_obs
    if fallback_mask_keys is None:
        fallback_mask_keys = ["source_mask", "target_mask", "rna_mask", "protein_mask", "adt_mask", "atac_mask", "metabolomics_mask"]
    candidate_keys = []
    if mask_key is not None:
        candidate_keys.append(mask_key)
    for uns_key in ["source_mask_key", "target_mask_key", "mask_key"]:
        if uns_key in adata_full.uns:
            k = str(adata_full.uns[uns_key])
            if k not in candidate_keys:
                candidate_keys.append(k)
    for k in fallback_mask_keys:
        if k not in candidate_keys:
            candidate_keys.append(k)
    selected_key = None
    for k in candidate_keys:
        if k in adata_full.obsm:
            selected_key = k
            break
    if selected_key is not None:
        mask_np = np.asarray(adata_full.obsm[selected_key])
        if mask_np.ndim != 1:
            raise ValueError(f"{selected_key} must be a 1D row mask, got shape {mask_np.shape}.")
        if mask_np.shape[0] != n_obs:
            raise ValueError(f"{selected_key} rows mismatch: {mask_np.shape[0]} vs {n_obs}.")
        if not np.all((mask_np == 0) | (mask_np == 1)):
            raise ValueError(f"{selected_key} must contain binary row-mask values.")
        return torch.as_tensor(mask_np.astype(bool) & (~self_missing_np), device=device, dtype=torch.bool)
    else:
        # Safe fallback for modality-level missingness.
        return torch.as_tensor(~self_missing_np, device=device, dtype=torch.bool)
def masked_mse_loss(pred, target, mask):
    """Masked MSE loss for whole-row modality missingness."""
    if mask.ndim != 1:
        raise ValueError(f"mask must be a 1D row mask, got shape {tuple(mask.shape)}.")
    denom = (mask.sum() * pred.size(1)).to(dtype=pred.dtype).clamp_min(1.0)
    mask = mask.unsqueeze(1)
    return (F.mse_loss(pred, target, reduction="none") * mask).sum() / denom
def train_PRISM_DM(adata_source, adata_target, context_prior, k_top=5, hidden_dims=None, n_epochs=1000, lr=0.001,
                   gradient_clipping=5.0, weight_decay=0.0001, verbose=True, random_seed=2024,
                   save_loss=True, save_reconstruction=False, output_dir="Default", file_prefix="Default",
                   device=torch.device("cuda:0"), patience=50, min_epochs=200, source_mask_key="source_mask",
                   target_mask_key="target_mask", center_drop_rate=0.2,
                   load_model_path=None, save_result_files=False,
                   interaction_pca=True, interaction_pca_components=16,
                   noise=0.0, spatial_key="spatial"):
    """
    Train or load PRISM-DM for dual-modality partial missingness.
    If load_model_path is None:
        train model and save best checkpoint.
    If load_model_path is not None:
        skip training, load checkpoint, and run deterministic inference.
    Training token logic:
        true missing center token -> learnable modality token
        observed center token     -> encoded latent
        randomly masked observed  -> learnable modality token
    Final inference:
        all source and target center tokens are masked to avoid missing-pattern leakage.
    """
    hidden_dims = [512, 32] if hidden_dims is None else hidden_dims
    os.makedirs(output_dir, exist_ok=True)
    # 1. Reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    # 2. Copy data and validate the modality-specific context prior
    adata_source, adata_target = adata_source.copy(), adata_target.copy()
    if isinstance(load_model_path, (bool, np.bool_)):
        load_model_path = os.path.join(output_dir, f"{file_prefix}_model.pth") if load_model_path else None
    # Optionally perturb the physical spatial graphs before PyG conversion.
    noise = float(noise)
    if noise > 0.0:
        adata_source = rewire_spatial_net(adata_source, noise_ratio=noise, seed=random_seed, spatial_key=spatial_key)
        adata_target = rewire_spatial_net(adata_target, noise_ratio=noise, seed=random_seed, spatial_key=spatial_key)
    if adata_source.n_obs != adata_target.n_obs:
        raise ValueError("adata_source and adata_target must have the same number of cells.")
    context_src_np, context_tgt_np, src_missing_np, tgt_missing_np = validate_modality_context_prior(
        context_prior, adata_source.n_obs, k_top
    )
    context_topk_src_global = torch.as_tensor(context_src_np, device=device, dtype=torch.long)
    context_topk_tgt_global = torch.as_tensor(context_tgt_np, device=device, dtype=torch.long)
    src_self_missing = torch.as_tensor(src_missing_np, device=device, dtype=torch.bool)
    tgt_self_missing = torch.as_tensor(tgt_missing_np, device=device, dtype=torch.bool)
    # 3. Zero truly missing rows in raw inputs
    adata_source.X = sp.csr_matrix(adata_source.X)
    adata_target.X = sp.csr_matrix(adata_target.X)
    adata_source.X[np.flatnonzero(src_missing_np), :] = 0
    adata_target.X[np.flatnonzero(tgt_missing_np), :] = 0
    if sp.issparse(adata_source.X):
        adata_source.X.eliminate_zeros()
    if sp.issparse(adata_target.X):
        adata_target.X.eliminate_zeros()
    # 4. Select highly-variable features if available
    adata_src_vars = adata_source[:, adata_source.var["highly_variable"]].copy() if "highly_variable" in adata_source.var.columns else adata_source.copy()
    adata_tgt_vars = adata_target[:, adata_target.var["highly_variable"]].copy() if "highly_variable" in adata_target.var.columns else adata_target.copy()
    source_data = Transfer_pytorch_Data(adata_src_vars)
    target_data = Transfer_pytorch_Data(adata_tgt_vars)
    source_data = source_data.to(device)
    target_data = target_data.to(device)
    # 5. Build loss masks aligned with selected features
    mask_source = build_feature_mask(adata_source, adata_src_vars, src_missing_np, device, source_mask_key,
                                     fallback_mask_keys=["source_mask", "rna_mask", "adt_mask", "protein_mask", "atac_mask", "metabolomics_mask"],)
    mask_target = build_feature_mask(adata_target, adata_tgt_vars, tgt_missing_np, device, target_mask_key,
                                     fallback_mask_keys=["target_mask", "rna_mask", "adt_mask", "protein_mask", "atac_mask", "metabolomics_mask"],)
    n_cells = source_data.x.size(0)
    src_true_missing_mask = src_self_missing
    tgt_true_missing_mask = tgt_self_missing
    src_observed_mask = ~src_true_missing_mask
    tgt_observed_mask = ~tgt_true_missing_mask
    # 6. Initialize model
    model = PRISM_DM(
        src_hidden_dims=[source_data.x.shape[1]] + hidden_dims,
        tgt_hidden_dims=[target_data.x.shape[1]] + hidden_dims,
    ).to(device)
    # 7. Train or load pretrained weights
    if load_model_path is None:
        w_params = torch.nn.Parameter(torch.ones(4, device=device))
        optimizer = torch.optim.Adam(
            [{"params": model.parameters()}, {"params": [w_params], "lr": lr * 0.1}],
            lr=lr,
            weight_decay=weight_decay,
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.5, patience=20, verbose=verbose
        # )
        scheduler_kwargs = dict(
            mode="min",
            factor=0.5,
            patience=20,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **scheduler_kwargs,
        )
        best_loss, best_model_state, stop_counter = float("inf"), None, 0
        pbar = tqdm(
            range(1, n_epochs + 1),
            desc=f"PRISM-DM [{file_prefix}]",
            disable=not verbose,
        )
        for epoch in pbar:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            src_random_mask = (torch.rand(n_cells, device=device) < center_drop_rate) & src_observed_mask
            tgt_random_mask = (torch.rand(n_cells, device=device) < center_drop_rate) & tgt_observed_mask
            src_center_missing_mask = src_true_missing_mask | src_random_mask
            tgt_center_missing_mask = tgt_true_missing_mask | tgt_random_mask
            s_rec, t_rec, p_emb, s_pre, t_pre = model(
                source_data.x,
                source_data.edge_index,
                target_data.x,
                target_data.edge_index,
                context_topk_src_global=context_topk_src_global,
                context_topk_tgt_global=context_topk_tgt_global,
                src_center_missing_mask=src_center_missing_mask,
                tgt_center_missing_mask=tgt_center_missing_mask,
            )
            l_s_rec = masked_mse_loss(s_rec, source_data.x, mask_source)
            l_t_rec = masked_mse_loss(t_rec, target_data.x, mask_target)
            l_s_pre = masked_mse_loss(s_pre, source_data.x, mask_source)
            l_t_pre = masked_mse_loss(t_pre, target_data.x, mask_target)
            norm_w = torch.softmax(w_params, dim=0)
            total_loss = norm_w[0] * l_s_rec + norm_w[1] * l_t_rec + norm_w[2] * l_s_pre + norm_w[3] * l_t_pre
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            scheduler.step(total_loss.item())
            current_loss = total_loss.item()
            l_s_pre_value = l_s_pre.item()
            l_t_pre_value = l_t_pre.item()
            del s_rec, t_rec, p_emb, s_pre, t_pre
            del l_s_rec, l_t_rec, l_s_pre, l_t_pre, total_loss, norm_w
            del src_random_mask, tgt_random_mask, src_center_missing_mask, tgt_center_missing_mask
            if verbose:
                pbar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "src_pre": f"{l_s_pre_value:.4f}",
                    "tgt_pre": f"{l_t_pre_value:.4f}",
                })
            if current_loss < best_loss:
                best_loss = current_loss
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stop_counter = 0
            elif epoch >= min_epochs:
                stop_counter += 1
            if stop_counter >= patience:
                if verbose:
                    print(f"\n>>> Early stopping at epoch {epoch}.")
                break
        if best_model_state is None:
            raise RuntimeError("Training finished without a valid best model state.")
        model_path = os.path.join(output_dir, f"{file_prefix}_model.pth")
        torch.save({"state_dict": best_model_state, "loss": best_loss}, model_path)
        if verbose:
            print(f"Best model saved: {model_path}")
    else:
        best_model_state, best_loss = load_checkpoint_state(load_model_path, device)
        if verbose:
            print(f"Loaded pretrained model: {load_model_path}")
            print(f"Loaded checkpoint loss: {best_loss}")
    if load_model_path is None:
        model.zero_grad(set_to_none=True)
        del optimizer, scheduler, w_params
    del mask_source, mask_target, src_self_missing, tgt_self_missing
    del src_true_missing_mask, tgt_true_missing_mask, src_observed_mask, tgt_observed_mask
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    # 8. Deterministic inference: mask all source and target center tokens
    model.load_state_dict(best_model_state, strict=True)
    model.eval()
    with torch.no_grad():
        inference_src_mask = torch.ones(n_cells, device=device, dtype=torch.bool)
        inference_tgt_mask = torch.ones(n_cells, device=device, dtype=torch.bool)
        s_rec, t_rec, p_emb, s_pre, t_pre = model(
            source_data.x,
            source_data.edge_index,
            target_data.x,
            target_data.edge_index,
            context_topk_src_global=context_topk_src_global,
            context_topk_tgt_global=context_topk_tgt_global,
            src_center_missing_mask=inference_src_mask,
            tgt_center_missing_mask=inference_tgt_mask,
        )
    base_emb_np = p_emb.detach().cpu().numpy()
    adata_source.obsm["PRISM_emb_base"] = base_emb_np
    adata_target.obsm["PRISM_emb_base"] = base_emb_np
    # Build the final embedding inside the trainer, matching train_PRISM.
    adata_source, adata_target = apply_interaction_embedding(
        adata_source,
        adata_target,
        enabled=interaction_pca,
        src_dim=hidden_dims[-1],
        tgt_dim=hidden_dims[-1],
        n_components=interaction_pca_components,
        random_state=random_seed,
    )
    _save_dual_results(
        adata_source, adata_target,
        adata_src_vars.var_names, adata_tgt_vars.var_names,
        p_emb.detach(), s_pre.detach(), t_pre.detach(),
        best_loss, save_loss, save_reconstruction,
        output_dir, file_prefix,
        save_files=save_result_files and load_model_path is None,
        base_emb_np=base_emb_np,
        base_emb_preassigned=True,
    )
    if verbose:
        if load_model_path is None:
            print(f"Finished training and deterministic inference: {file_prefix}")
        else:
            print(f"Finished pretrained deterministic inference: {file_prefix}")
            print(f"Loaded model: {load_model_path}")
    return adata_source, adata_target
def _save_dual_results(adata_src, adata_tgt, src_var_names, tgt_var_names, p_emb, s_pre, t_pre,
                 loss, save_loss, save_recon, output_dir, file_prefix, save_files=True,
                 base_emb_np=None, base_emb_preassigned=False):
    """Save embedding and source / target predictions."""
    if base_emb_np is None:
        base_emb_np = p_emb.detach().cpu().numpy()
    src_pred_np = s_pre.detach().cpu().numpy()
    tgt_pred_np = t_pre.detach().cpu().numpy()
    # Keep the base representation for in-memory compatibility, but export the
    # final PRISM_emb representation under one stable filename.
    if not base_emb_preassigned:
        adata_src.obsm["PRISM_emb_base"] = base_emb_np
        adata_tgt.obsm["PRISM_emb_base"] = base_emb_np
    if "PRISM_emb" not in adata_src.obsm:
        adata_src.obsm["PRISM_emb"] = base_emb_np
    if "PRISM_emb" not in adata_tgt.obsm:
        adata_tgt.obsm["PRISM_emb"] = base_emb_np
    final_emb_np = np.asarray(adata_tgt.obsm["PRISM_emb"], dtype=np.float32)
    # Save predictions in obsm by default because selected feature dimensions may differ from original X.
    adata_src.obsm["PRISM_src_pred"] = src_pred_np
    adata_tgt.obsm["PRISM_tgt_pred"] = tgt_pred_np
    adata_src.uns["PRISM_src_pred_var_names"] = np.asarray(src_var_names).astype(str)
    adata_tgt.uns["PRISM_tgt_pred_var_names"] = np.asarray(tgt_var_names).astype(str)
    if save_files:
        source_csv = os.path.join(output_dir, f"{file_prefix}_src_pre.csv")
        target_csv = os.path.join(output_dir, f"{file_prefix}_tgt_pre.csv")
        embed_csv = os.path.join(output_dir, f"{file_prefix}_emb.csv")
        pd.DataFrame(
            src_pred_np,
            index=adata_src.obs_names,
            columns=src_var_names,
        ).to_csv(source_csv)
        pd.DataFrame(
            tgt_pred_np,
            index=adata_tgt.obs_names,
            columns=tgt_var_names,
        ).to_csv(target_csv)
        pd.DataFrame(
            final_emb_np,
            index=adata_tgt.obs_names,
        ).to_csv(embed_csv)
        if verbose:
            print(f"Results for '{file_prefix}' saved:")
            print(f"  source prediction: {source_csv}")
            print(f"  target prediction: {target_csv}")
            print(f"  embedding:          {embed_csv}")
    if save_loss:
        adata_src.uns["PRISM_loss"] = float(loss)
        adata_tgt.uns["PRISM_loss"] = float(loss)
    if save_recon:
        if src_pred_np.shape == adata_src.X.shape:
            adata_src.layers["PRISM_src_pred"] = src_pred_np.clip(min=0)
        if tgt_pred_np.shape == adata_tgt.X.shape:
            adata_tgt.layers["PRISM_tgt_pred"] = tgt_pred_np.clip(min=0)
