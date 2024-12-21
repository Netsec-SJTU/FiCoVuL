from abc import abstractmethod
import torch
from torch import nn
from FiCoVuL.models.layers.MLP import MLP4Layer


class NodesToGraphRepresentation(nn.Module):
    def __init__(self, graph_representation_size: int):
        super().__init__()
        self._graph_representation_size = graph_representation_size

    @abstractmethod
    def forward(self, inputs):
        pass


class WeightedSumGraphRepresentation(NodesToGraphRepresentation):
    graph_dim = 0

    def __init__(self,
                 graph_representation_size: int,
                 num_heads: int,
                 num_in_features: int,
                 scoring_mlp_bias: bool = True,
                 transformation_mlp_bias: bool = True,
                 dropout_prob: float = 0.2,
                 # One of {"softmax", "sigmoid"}
                 weighting_fun: str = "softmax"):
        super().__init__(graph_representation_size)
        assert (graph_representation_size % num_heads == 0), \
            f"Number of heads {num_heads} needs to divide final representation size {graph_representation_size}!"
        assert weighting_fun.lower() in {"softmax", "sigmoid"}, \
            f"Weighting function {weighting_fun} unknown, {{'softmax', 'sigmoid'}} supported."

        self._graph_representation_size = graph_representation_size
        self._num_heads = num_heads
        self._scoring_mlp = MLP4Layer(num_in_features, num_heads, output_activate='Mish', bias=scoring_mlp_bias,
                                      dropout_prob=dropout_prob)
        self._score_fn = nn.Softmax(dim=0) if weighting_fun.lower() == "softmax" else nn.Sigmoid()
        self._transformation_mlp = MLP4Layer(num_in_features, graph_representation_size, output_activate='Mish',
                                             bias=transformation_mlp_bias, dropout_prob=dropout_prob)

    def forward(self, inputs):
        """

        :param inputs:
            node_embeddings: torch.Tensor (N, FIN)
            node_to_graph_map: torch.Tensor (N, )
        :return:
        """
        node_embeddings, node_to_graph_map = inputs
        num_graphs = node_to_graph_map.max().item() + 1

        # (1) compute weights for each node/head pair:
        scores = self._scoring_mlp(node_embeddings)  # (N, NH)
        if isinstance(self._score_fn, nn.Softmax):
            scores = scores - torch.max(scores, dim=0, keepdim=True).values
        weights = self.softmax_per_graph(scores, node_to_graph_map, num_graphs)  # (N, NH)

        # (2) compute representations for each node/head pair:
        # shape: (N, FG) where FG - graph_representation_size
        nodes_repr = self._transformation_mlp(node_embeddings)
        # Shape: (N, NH, GD//H)
        nodes_repr = nodes_repr.view(-1, self._num_heads, self._graph_representation_size // self._num_heads)

        # (3) weight representations and aggregate by graph:
        # Shape: (N, NH, 1)
        weights = weights.unsqueeze(-1)
        # Shape: (N, NH, 1) * (N, NH, GD//H) -> (N, NH, GD//H)
        weighted_node_reprs = weights * nodes_repr
        # shape: (N, FG)
        weighted_node_reprs = weighted_node_reprs.view(-1, self._graph_representation_size)

        # Shape (G, FG)
        graphs_repr = self.sum_per_graph(weighted_node_reprs, node_to_graph_map, num_graphs)

        return graphs_repr

    def softmax_per_graph(self, features, node_to_graph_map, num_graphs):
        features_per_graph = self.sum_per_graph(features, node_to_graph_map, num_graphs)
        return features / features_per_graph.index_select(self.graph_dim, node_to_graph_map)

    def sum_per_graph(self, features, node_to_graph_map, num_graphs):
        # features: (N, F)
        size = list(features.shape)  # convert to list otherwise assignment is not possible
        size[self.graph_dim] = num_graphs  # torch.Size([G, F])
        graphs_features = torch.zeros(size, dtype=features.dtype, device=features.device)
        # Shape: (N, 1) -> (N, F)
        node_to_graph_map_broadcasted = self.explicit_broadcast(node_to_graph_map, features)
        # (G, FG)
        graphs_features.scatter_add_(self.graph_dim, node_to_graph_map_broadcasted, features)
        # print("graphs_features=", graphs_features)
        return graphs_features

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)
