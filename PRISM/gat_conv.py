"""Graph-attention layer used by the PRISM neural models."""
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, OptTensor)
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
class _SparseEdgeAggregation(torch.autograd.Function):
    """Sparse weighted aggregation without a dense adjacency gradient."""
    @staticmethod
    def forward(ctx, x, edge_weight, src_index, dst_index,
                num_src, num_dst, edge_chunk_size):
        indices = torch.stack([dst_index, src_index], dim=0)
        adjacency = torch.sparse_coo_tensor(
            indices,
            edge_weight,
            size=(num_dst, num_src),
            dtype=edge_weight.dtype,
            device=edge_weight.device,
        ).coalesce()
        ctx.save_for_backward(x, edge_weight, src_index, dst_index)
        ctx.num_src = int(num_src)
        ctx.num_dst = int(num_dst)
        ctx.edge_chunk_size = int(edge_chunk_size)
        return torch.sparse.mm(adjacency, x)
    @staticmethod
    def backward(ctx, grad_output):
        x, edge_weight, src_index, dst_index = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x = None
        if ctx.needs_input_grad[0]:
            transpose_indices = torch.stack([src_index, dst_index], dim=0)
            adjacency_t = torch.sparse_coo_tensor(
                transpose_indices,
                edge_weight,
                size=(ctx.num_src, ctx.num_dst),
                dtype=edge_weight.dtype,
                device=edge_weight.device,
            ).coalesce()
            grad_x = torch.sparse.mm(adjacency_t, grad_output)
        grad_edge_weight = None
        if ctx.needs_input_grad[1]:
            # dL/dw_e = <dL/dY[target_e], X[source_e]>.  Chunking keeps
            # the gathered workspace bounded instead of materializing E x C.
            grad_edge_weight = torch.empty_like(edge_weight)
            chunk_size = max(ctx.edge_chunk_size, 1)
            for start in range(0, edge_weight.numel(), chunk_size):
                end = min(start + chunk_size, edge_weight.numel())
                src_chunk = src_index[start:end]
                dst_chunk = dst_index[start:end]
                src_features = x.index_select(0, src_chunk)
                dst_gradient = grad_output.index_select(0, dst_chunk)
                grad_edge_weight[start:end] = torch.bmm(
                    dst_gradient.unsqueeze(1),
                    src_features.unsqueeze(2),
                ).reshape(-1)
        return grad_x, grad_edge_weight, None, None, None, None, None
class GATConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True,
                 sparse_aggregation: bool = True,
                 edge_gradient_chunk_size: int = 65536, **kwargs):
        # Set the aggregation method to 'add' by default
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)
        # Initialize input/output channels and hyperparameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        # Keep the original message-passing path available for regression tests.
        self.sparse_aggregation = bool(sparse_aggregation)
        self.edge_gradient_chunk_size = int(edge_gradient_chunk_size)
        if self.edge_gradient_chunk_size <= 0:
            raise ValueError('edge_gradient_chunk_size must be positive.')
        # Linear transformation for source and destination nodes
        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src  # Tied weights for destination nodes
        # Attention parameters initialization
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))  # Source node attention
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))  # Destination node attention
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)
        # Initialize alpha values and attention weights
        self._alpha = None
        self.attentions = None
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, attention=True, tied_attention=None):
        # Define the dimensions for heads and output channels
        H, C = self.heads, self.out_channels
        # Apply the linear transformation to the source and destination node features
        if isinstance(x, Tensor):
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:
            x_src, x_dst = x
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)
        # Set x as a tuple for source and destination node features
        x = (x_src, x_dst)
        # If attention is not needed, return the mean of the source features
        if not attention:
            return x[0].mean(dim=1)
        # If tied attention is not provided, calculate attention scores
        if tied_attention is None:
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention  # Use provided tied attention scores
        # Add self-loops to the graph if required
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = min(x_src.size(0), x_dst.size(0) if x_dst is not None else x_src.size(0))
                edge_index, _ = remove_self_loops(edge_index)  # Remove self loops if present
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # Add self loops
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)  # Set diagonal for sparse matrix
        # For dense edge_index, aggregate dynamic attention weights with sparse
        # matrix multiplication instead of materializing E x hidden messages.
        # SparseTensor inputs retain the original PyG path for compatibility.
        if self.sparse_aggregation and isinstance(edge_index, Tensor):
            out = self._sparse_propagate(edge_index, x, alpha, size=size)
        else:
            out = self.propagate(edge_index, x=x, alpha=alpha, size=size)
        # Clear alpha to avoid leakage
        alpha = self._alpha
        self._alpha = None
        # Concatenate the output across heads or average them
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        # If attention weights are required, return them along with the output
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out
    def _sparse_propagate(self, edge_index: Tensor, x, alpha, size: Size = None):
        """GAT aggregation with scalar edge weights and sparse matmul.
        The attention score and destination-wise softmax are identical to
        ``message`` below.  The sparse matrix is created only after softmax,
        so duplicate edges are not merged before attention normalization.
        """
        x_src, x_dst = x
        alpha_src, alpha_dst = alpha
        src_index, dst_index = edge_index[0], edge_index[1]
        if size is None:
            num_src = x_src.size(0)
            num_dst = x_dst.size(0) if x_dst is not None else x_src.size(0)
        else:
            num_src = size[0] if size[0] is not None else x_src.size(0)
            if x_dst is None:
                num_dst = size[1] if size[1] is not None else x_src.size(0)
            else:
                num_dst = size[1] if size[1] is not None else x_dst.size(0)
        n_edges = edge_index.size(1)
        score = torch.zeros(
            (n_edges, self.heads),
            dtype=x_src.dtype,
            device=x_src.device,
        )
        if alpha_src is not None:
            score = score + alpha_src[src_index]
        if alpha_dst is not None:
            score = score + alpha_dst[dst_index]
        score = torch.sigmoid(score)
        score = softmax(score, dst_index, None, num_dst)
        # Keep the pre-dropout attention for the existing tied/debug API.
        self._alpha = score
        edge_weight = F.dropout(score, p=self.dropout, training=self.training)
        # One sparse matrix per head preserves the existing [N, H, C] layout.
        outputs = []
        for head in range(self.heads):
            outputs.append(
                _SparseEdgeAggregation.apply(
                    x_src[:, head, :],
                    edge_weight[:, head],
                    src_index,
                    dst_index,
                    num_src,
                    num_dst,
                    self.edge_gradient_chunk_size,
                ).unsqueeze(1)
            )
        return torch.cat(outputs, dim=1)
    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        # Combine attention scores for source and destination nodes
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = torch.sigmoid(alpha)  # Apply sigmoid activation to attention scores
        alpha = softmax(alpha, index, ptr, size_i)  # Apply softmax normalization
        # Store attention scores for debugging
        self._alpha = alpha
        # Dropout the attention scores for regularization
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # Return the weighted sum of neighbor features
        return x_j * alpha.unsqueeze(-1)
    def __repr__(self):
        # Return a string representation of the class
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
