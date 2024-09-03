import torch
import torch.nn as nn
# from torch_geometric.nn import  GATConv, GATv2Conv
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from torch_geometric.nn import BatchNorm, LayerNorm
# from torch_geometric.nn import GCNConv

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor, torch_sparse
from torch_geometric.utils import (add_self_loops, is_torch_sparse_tensor,
                                   remove_self_loops, softmax)
from torch_geometric.utils.sparse import set_sparse_value


class GATv2Conv(MessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the
    static attention problem of the standard
    :class:`~torch_geometric.conv.GATConv` layer.
    Since the linear layers in the standard GAT are applied right after each
    other, the ranking of attended nodes is unconditioned on the query node.
    In contrast, in :class:`GATv2`, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j \, \Vert \, \mathbf{e}_{i,j}]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k \, \Vert \, \mathbf{e}_{i,k}]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops
            (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduc99438e operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor],
                edge_index: Adj,
                edge_weight: OptTensor = None,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_weight = remove_self_loops(
                    edge_index, edge_weight)
                edge_index, edge_weight = add_self_loops(
                    edge_index, edge_weight, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_weight=edge_weight,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_weight: OptTensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)

        # # Sum to 1 by neighborhood
        # print('Before')
        # print('alpha', alpha.sum(dim=0))
        # print('edge_wt', edge_weight.sum(dim=0))

        alpha = alpha*edge_weight.to(torch.float32)
        
        # Setting both data driven and bio driven attention to same scale
        N = maybe_num_nodes(index, size_i)
        out_sum = scatter(alpha, index, 0, dim_size=N, reduce='sum') + 1e-16
        out_sum = out_sum.index_select(0, index)
        alpha_new=alpha/out_sum

        # for n in torch.unique(index):
        #     idx = torch.argwhere(index==n)
        #     alpha[idx] = alpha[idx] / alpha[idx].sum(dim=0)
        # print('After')
        # print('alpha', alpha.sum(dim=0))
        # print('edge_wt', edge_weight.sum(dim=0))

        self._alpha = alpha_new
        alpha = F.dropout(alpha_new, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class FNN(nn.Module):
    """ Class for Feed Forward Network"""

    def __init__(self, input_dim, hidden_dim_array, dropout_prob):
        super(FNN, self).__init__()
        self.layers = nn.ModuleList()
        self.hidden_layers = len(hidden_dim_array)
        for idx in range(self.hidden_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim_array[idx]))
            if self.hidden_layers >= 1 and idx < (self.hidden_layers - 1):
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_prob))
            input_dim = hidden_dim_array[idx]

    def forward(self, inp):
        for layer in self.layers:
            inp = layer(inp)
        return inp


class KG_GNN(torch.nn.Module):
    def __init__(self, input_dim, num_gat_nodes, num_attn_heads, num_fc_nodes, num_classes,
                 dropout, need_layer_norm, need_batch_norm, need_attn_concat, is_training=True):
        super(KG_GNN, self).__init__()

        # GATConv section
        self.num_gat_layers = len(num_gat_nodes)
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.GAT_layers = nn.ModuleList()

        self.batch_norms, self.layer_norms = None, None

        if need_batch_norm:
            self.batch_norms = torch.nn.ModuleList()
        if need_layer_norm:
            self.layer_norms = torch.nn.ModuleList()

        out_dim = None
        add_self_loop = True
        for idx in range(self.num_gat_layers):
            out_dim = num_gat_nodes[idx]
            if need_attn_concat:
                out_dim = out_dim * self.num_attn_heads
            
            if not is_training:
                add_self_loop = False

            self.GAT_layers.append(GATv2Conv(input_dim, out_dim, heads=self.num_attn_heads,
                                             concat=need_attn_concat, add_self_loops=add_self_loop))

            if need_batch_norm:
                self.batch_norms.append(BatchNorm(out_dim))

            if need_layer_norm:
                self.layer_norms.append(LayerNorm(out_dim))

            input_dim = num_gat_nodes[idx]

        self.ffn = FNN(out_dim, num_fc_nodes, dropout)
        self.pred = nn.Linear(num_fc_nodes[-1], num_classes)

    def forward(self, x, edge_index, batch, edge_weight: OptTensor = None,
                edge_attr: OptTensor = None, return_attention_weights=True):

        # 1. Run GAT layer
        for i in range(len(self.GAT_layers)):
            ret_attn_flag = None
            if i == (len(self.GAT_layers)-1):
                ret_attn_flag = True

            x = self.GAT_layers[i](x, edge_index, edge_weight,
                                   return_attention_weights=ret_attn_flag)

            if i != len(self.GAT_layers)-1:
                if self.layer_norms is not None:
                    x = self.layer_norms[i](x)
                    x = F.leaky_relu(x)
                if self.batch_norms is not None:
                    x = self.batch_norms[i](x)
                if self.batch_norms is None and self.layer_norms is None:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    x = x.relu()

        # 2. Get attention and embeddings
        attn_scores = x[1]
        x = global_mean_pool(x[0], batch)  # [batch_size, hidden_channels]
        graph_embedding = x

        # 3. MLP layers
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ffn(x)

        # 4. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pred(x)

        return x, attn_scores, graph_embedding
