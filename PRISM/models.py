"""Neural models for single- and dual-modality PRISM training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat_conv import GATConv

BATCH_SIZE = 64
DROPOUT = 0.1
HEADS = 8

class TransformerEncoder(nn.Module):
    """Self-attention encoder for center and context tokens."""
    def __init__(self, embedding_dim, nhead, mlp_dim):
        super().__init__()
        if embedding_dim % nhead != 0:
            raise ValueError(f"embedding_dim={embedding_dim} must be divisible by nhead={nhead}.")
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, embedding_dim))
    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_output)
        return self.norm2(x + self.mlp(x))
class PRISM(nn.Module):
    """PRISM model for target-modality missingness."""
    def __init__(self, src_hidden_dims, tgt_hidden_dims):
        super().__init__()
        self.batch_size = BATCH_SIZE
        src_in_dim, src_num_hidden, src_out_dim = src_hidden_dims
        tgt_in_dim, tgt_num_hidden, tgt_out_dim = tgt_hidden_dims
        self.en_src_conv1 = GATConv(src_in_dim, src_num_hidden, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.en_src_conv2 = GATConv(src_num_hidden, src_out_dim, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.de_src_conv1 = GATConv(src_out_dim, src_num_hidden, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.de_src_conv2 = GATConv(src_num_hidden, src_in_dim, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.en_tgt_conv1 = GATConv(tgt_in_dim, tgt_num_hidden, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.en_tgt_conv2 = GATConv(tgt_num_hidden, tgt_out_dim, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.de_tgt_conv1 = GATConv(tgt_out_dim, tgt_num_hidden, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.de_tgt_conv2 = GATConv(tgt_num_hidden, tgt_in_dim, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.token = nn.Parameter(torch.empty(tgt_out_dim))
        nn.init.normal_(self.token, mean=0.0, std=0.2)
        fusion_dim = src_out_dim + tgt_out_dim
        self.transformer = TransformerEncoder(embedding_dim=fusion_dim, nhead=HEADS, mlp_dim=fusion_dim)
        self.proj_src = nn.Sequential(nn.Linear(fusion_dim, src_num_hidden), nn.ReLU(), nn.Linear(src_num_hidden, src_out_dim))
        self.proj_tgt = nn.Sequential(nn.Linear(fusion_dim, tgt_num_hidden), nn.ReLU(), nn.Linear(tgt_num_hidden, tgt_out_dim))
        self.de_src_pre1, self.de_src_pre2 = self.de_src_conv1, self.de_src_conv2
        self.de_tgt_pre1, self.de_tgt_pre2 = self.de_tgt_conv1, self.de_tgt_conv2
    def _tie_source_decoder_weights(self):
        self.de_src_conv1.lin_src.data = self.en_src_conv2.lin_src.transpose(0, 1)
        self.de_src_conv1.lin_dst.data = self.en_src_conv2.lin_dst.transpose(0, 1)
        self.de_src_conv2.lin_src.data = self.en_src_conv1.lin_src.transpose(0, 1)
        self.de_src_conv2.lin_dst.data = self.en_src_conv1.lin_dst.transpose(0, 1)
    def _tie_target_decoder_weights(self):
        self.de_tgt_conv1.lin_src.data = self.en_tgt_conv2.lin_src.transpose(0, 1)
        self.de_tgt_conv1.lin_dst.data = self.en_tgt_conv2.lin_dst.transpose(0, 1)
        self.de_tgt_conv2.lin_src.data = self.en_tgt_conv1.lin_src.transpose(0, 1)
        self.de_tgt_conv2.lin_dst.data = self.en_tgt_conv1.lin_dst.transpose(0, 1)
    def forward(self, src_x, src_edge, tgt_x, tgt_edge, top_k_indices, non_missing_indices, center_missing_mask=None):
        device = src_x.device
        top_k_indices = top_k_indices.to(device=device, dtype=torch.long)
        non_missing_indices = non_missing_indices.to(device=device, dtype=torch.long)
        s1 = F.elu(self.en_src_conv1(src_x, src_edge))
        s2 = self.en_src_conv2(s1, src_edge, attention=False)
        self._tie_source_decoder_weights()
        s_recon_h = F.elu(self.de_src_conv1(s2, src_edge, attention=True, tied_attention=self.en_src_conv1.attentions))
        s_recon = self.de_src_conv2(s_recon_h, src_edge, attention=False)
        t1 = F.elu(self.en_tgt_conv1(tgt_x, tgt_edge))
        t2 = self.en_tgt_conv2(t1, tgt_edge, attention=False)
        self._tie_target_decoder_weights()
        t_recon_h = F.elu(self.de_tgt_conv1(t2, tgt_edge, attention=True, tied_attention=self.en_tgt_conv1.attentions))
        t_recon = self.de_tgt_conv2(t_recon_h, tgt_edge, attention=False)
        neighbor_idx = non_missing_indices[top_k_indices]
        s2_tensor = torch.cat([s2.unsqueeze(1), s2[neighbor_idx]], dim=1)
        if center_missing_mask is None:
            center_missing_mask = torch.zeros(src_x.size(0), dtype=torch.bool, device=device)
        else:
            center_missing_mask = center_missing_mask.to(device=device, dtype=torch.bool)
        if center_missing_mask.numel() != src_x.size(0):
            raise ValueError(f"center_missing_mask length mismatch: expected {src_x.size(0)}, got {center_missing_mask.numel()}.")
        mask_token = self.token.view(1, -1).expand(t2.size(0), -1)
        t2_center = torch.where(center_missing_mask.unsqueeze(1), mask_token, t2)
        t2_tensor = torch.cat([t2_center.unsqueeze(1), t2[neighbor_idx]], dim=1)
        fusion_input = torch.cat([s2_tensor, t2_tensor], dim=-1)
        outputs = [self.transformer(fusion_input[start:start + self.batch_size]) for start in range(0, fusion_input.size(0), self.batch_size)]
        fused_center = torch.cat(outputs, dim=0)[:, 0, :]
        tf_src = self.proj_src(fused_center)
        tf_tgt = self.proj_tgt(fused_center)
        p_emb = torch.cat([tf_src, tf_tgt], dim=-1)
        src_pred_h = F.elu(self.de_src_pre1(tf_src, src_edge, attention=True, tied_attention=self.en_src_conv1.attentions))
        src_pred = self.de_src_pre2(src_pred_h, src_edge, attention=False)
        tgt_pred_h = F.elu(self.de_tgt_pre1(tf_tgt, tgt_edge))
        tgt_pred = self.de_tgt_pre2(tgt_pred_h, tgt_edge, attention=False)
        return s_recon, t_recon, p_emb, src_pred, tgt_pred
class PRISM_DM(nn.Module):
    """PRISM model for dual-modality partial missingness."""
    def __init__(self, src_hidden_dims, tgt_hidden_dims):
        super().__init__()
        self.batch_size = BATCH_SIZE
        src_in_dim, src_num_hidden, src_out_dim = src_hidden_dims
        tgt_in_dim, tgt_num_hidden, tgt_out_dim = tgt_hidden_dims
        self.en_src_conv1 = GATConv(src_in_dim, src_num_hidden, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.en_src_conv2 = GATConv(src_num_hidden, src_out_dim, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.de_src_conv1 = GATConv(src_out_dim, src_num_hidden, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.de_src_conv2 = GATConv(src_num_hidden, src_in_dim, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.en_tgt_conv1 = GATConv(tgt_in_dim, tgt_num_hidden, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.en_tgt_conv2 = GATConv(tgt_num_hidden, tgt_out_dim, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.de_tgt_conv1 = GATConv(tgt_out_dim, tgt_num_hidden, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.de_tgt_conv2 = GATConv(tgt_num_hidden, tgt_in_dim, heads=1, concat=False, dropout=DROPOUT, add_self_loops=False, bias=False)
        self.src_token = nn.Parameter(torch.empty(src_out_dim))
        self.tgt_token = nn.Parameter(torch.empty(tgt_out_dim))
        nn.init.normal_(self.src_token, mean=0.0, std=0.2)
        nn.init.normal_(self.tgt_token, mean=0.0, std=0.2)
        fusion_dim = src_out_dim + tgt_out_dim
        self.transformer = TransformerEncoder(embedding_dim=fusion_dim, nhead=HEADS, mlp_dim=fusion_dim)
        self.proj_src = nn.Sequential(nn.Linear(fusion_dim, src_num_hidden), nn.ReLU(), nn.Linear(src_num_hidden, src_out_dim))
        self.proj_tgt = nn.Sequential(nn.Linear(fusion_dim, tgt_num_hidden), nn.ReLU(), nn.Linear(tgt_num_hidden, tgt_out_dim))
        self.de_src_pre1, self.de_src_pre2 = self.de_src_conv1, self.de_src_conv2
        self.de_tgt_pre1, self.de_tgt_pre2 = self.de_tgt_conv1, self.de_tgt_conv2
    def _tie_source_decoder_weights(self):
        self.de_src_conv1.lin_src.data = self.en_src_conv2.lin_src.transpose(0, 1)
        self.de_src_conv1.lin_dst.data = self.en_src_conv2.lin_dst.transpose(0, 1)
        self.de_src_conv2.lin_src.data = self.en_src_conv1.lin_src.transpose(0, 1)
        self.de_src_conv2.lin_dst.data = self.en_src_conv1.lin_dst.transpose(0, 1)
    def _tie_target_decoder_weights(self):
        self.de_tgt_conv1.lin_src.data = self.en_tgt_conv2.lin_src.transpose(0, 1)
        self.de_tgt_conv1.lin_dst.data = self.en_tgt_conv2.lin_dst.transpose(0, 1)
        self.de_tgt_conv2.lin_src.data = self.en_tgt_conv1.lin_src.transpose(0, 1)
        self.de_tgt_conv2.lin_dst.data = self.en_tgt_conv1.lin_dst.transpose(0, 1)
    def forward(self, src_x, src_edge, tgt_x, tgt_edge, context_topk_src_global, context_topk_tgt_global=None, src_center_missing_mask=None, tgt_center_missing_mask=None):
        s1 = F.elu(self.en_src_conv1(src_x, src_edge))
        s2 = self.en_src_conv2(s1, src_edge, attention=False)
        self._tie_source_decoder_weights()
        s_recon_h = F.elu(self.de_src_conv1(s2, src_edge, attention=True, tied_attention=self.en_src_conv1.attentions))
        s_recon = self.de_src_conv2(s_recon_h, src_edge, attention=False)
        t1 = F.elu(self.en_tgt_conv1(tgt_x, tgt_edge))
        t2 = self.en_tgt_conv2(t1, tgt_edge, attention=False)
        self._tie_target_decoder_weights()
        t_recon_h = F.elu(self.de_tgt_conv1(t2, tgt_edge, attention=True, tied_attention=self.en_tgt_conv1.attentions))
        t_recon = self.de_tgt_conv2(t_recon_h, tgt_edge, attention=False)
        device = src_x.device
        n_cells = src_x.size(0)
        context_topk_src_global = context_topk_src_global.to(device=device, dtype=torch.long)
        if context_topk_tgt_global is None:
            context_topk_tgt_global = context_topk_src_global
        else:
            context_topk_tgt_global = context_topk_tgt_global.to(device=device, dtype=torch.long)
        if context_topk_src_global.shape[0] != n_cells:
            raise ValueError(f"context_topk_src_global rows must equal n_cells: {context_topk_src_global.shape[0]} vs {n_cells}.")
        if context_topk_tgt_global.shape[0] != n_cells:
            raise ValueError(f"context_topk_tgt_global rows must equal n_cells: {context_topk_tgt_global.shape[0]} vs {n_cells}.")
        if torch.any(context_topk_src_global < 0):
            raise ValueError("context_topk_src_global contains -1. Some cells do not have valid source top-k neighbors.")
        if torch.any(context_topk_tgt_global < 0):
            raise ValueError("context_topk_tgt_global contains -1. Some cells do not have valid target top-k neighbors.")
        src_neighbors = s2[context_topk_src_global]
        tgt_neighbors = t2[context_topk_tgt_global]
        if src_center_missing_mask is None:
            src_center_missing_mask = torch.zeros(n_cells, dtype=torch.bool, device=device)
        else:
            src_center_missing_mask = src_center_missing_mask.to(device=device, dtype=torch.bool)
        if tgt_center_missing_mask is None:
            tgt_center_missing_mask = torch.zeros(n_cells, dtype=torch.bool, device=device)
        else:
            tgt_center_missing_mask = tgt_center_missing_mask.to(device=device, dtype=torch.bool)
        if src_center_missing_mask.numel() != n_cells:
            raise ValueError(f"src_center_missing_mask length mismatch: expected {n_cells}, got {src_center_missing_mask.numel()}.")
        if tgt_center_missing_mask.numel() != n_cells:
            raise ValueError(f"tgt_center_missing_mask length mismatch: expected {n_cells}, got {tgt_center_missing_mask.numel()}.")
        src_token = self.src_token.view(1, -1).expand(n_cells, -1)
        tgt_token = self.tgt_token.view(1, -1).expand(n_cells, -1)
        src_center = torch.where(src_center_missing_mask.unsqueeze(1), src_token, s2)
        tgt_center = torch.where(tgt_center_missing_mask.unsqueeze(1), tgt_token, t2)
        s2_tensor = torch.cat([src_center.unsqueeze(1), src_neighbors], dim=1)
        t2_tensor = torch.cat([tgt_center.unsqueeze(1), tgt_neighbors], dim=1)
        fusion_input = torch.cat([s2_tensor, t2_tensor], dim=-1)
        outputs = [self.transformer(fusion_input[start:start + self.batch_size]) for start in range(0, fusion_input.size(0), self.batch_size)]
        fused_center = torch.cat(outputs, dim=0)[:, 0, :]
        tf_src = self.proj_src(fused_center)
        tf_tgt = self.proj_tgt(fused_center)
        p_emb = torch.cat([tf_src, tf_tgt], dim=-1)
        src_pred_h = F.elu(self.de_src_pre1(tf_src, src_edge, attention=True, tied_attention=self.en_src_conv1.attentions))
        src_pred = self.de_src_pre2(src_pred_h, src_edge, attention=False)
        tgt_pred_h = F.elu(self.de_tgt_pre1(tf_tgt, tgt_edge))
        tgt_pred = self.de_tgt_pre2(tgt_pred_h, tgt_edge, attention=False)
        return s_recon, t_recon, p_emb, src_pred, tgt_pred
