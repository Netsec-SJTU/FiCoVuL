from abc import abstractmethod

import torch
from torch import nn
from FiCoVuL.models.layers.nodes_to_graph_representation import WeightedSumGraphRepresentation
from FiCoVuL.models.layers.MLP import MLP4Layer


class GraphGlobalExchange(torch.nn.Module):
    """Update node representations based on graph-global information."""

    node_dim = 0

    def __init__(self, num_graph_features: int, num_node_features: int, weighting_fun: str = "softmax", num_heads: int = 4, dropout_prob: float = 0.1):
        super().__init__()

        self._num_graph_features = num_graph_features
        self._num_node_features = num_node_features
        self._weighting_fun = weighting_fun
        self._num_heads = num_heads
        self._dropout_prob = dropout_prob

        self._node_to_graph_representation_layer = WeightedSumGraphRepresentation(
            graph_representation_size=num_graph_features,
            num_heads=num_heads,
            num_in_features=num_node_features,
            dropout_prob=dropout_prob,
            weighting_fun=weighting_fun,
        )
        self._dropout = nn.Dropout(p=dropout_prob)

    @abstractmethod
    def forward(self, inputs):
        pass

    def _compute_per_node_graph_representations(self, inputs):
        _, node_to_graph_map = inputs
        cur_graph_representations = self._node_to_graph_representation_layer(inputs)  # Shape [G, hidden_dim]
        graph_representations_per_node_lifted = cur_graph_representations.index_select(
            self.node_dim, node_to_graph_map)  # Shape [V, hidden_dim]
        graph_representations_per_node_lifted = self._dropout(graph_representations_per_node_lifted)

        return graph_representations_per_node_lifted


class GraphGlobalGRUExchange(GraphGlobalExchange):
    def __init__(self, num_graph_features: int, num_node_features: int, num_in_heads: int = 1,
                 weighting_fun: str = "softmax", dropout_prob: float = 0.1):
        super().__init__(num_graph_features, num_node_features//num_in_heads, weighting_fun, 4, dropout_prob)

        self._gru_cell = nn.GRUCell(num_graph_features, num_node_features)

    def forward(self, inputs):
        # [N, NH, FN]
        node_embeddings, node_to_graph_map = inputs
        per_node_graph_representations = self._compute_per_node_graph_representations((node_embeddings.mean(dim=1),
                                                                                       node_to_graph_map))
        cur_node_representations = self._gru_cell(
            input=per_node_graph_representations,                  # (N, FG) where FG - num of graph features
            hx=node_embeddings.view(node_embeddings.shape[0], -1)  # (N, FN) where FG - num of node features
        )  # (N, H) - the next hidden state

        return cur_node_representations.view(*node_embeddings.shape)


class GraphGlobalMLPExchange(GraphGlobalExchange):
    def __init__(self, num_graph_features: int, num_node_features: int, weighting_fun: str = "softmax",
                 num_heads: int = 4, dropout_rate: float = 0.1):
        """Initialise the layer."""
        super().__init__(num_graph_features, num_node_features, weighting_fun, num_heads, dropout_rate)

        self._mlp = MLP4Layer(num_graph_features + num_node_features, num_node_features)

    def forward(self, inputs):
        per_node_graph_representations = self._compute_per_node_graph_representations(inputs)
        cur_node_representations = self._mlp(
            torch.cat([per_node_graph_representations, inputs.node_embeddings], dim=-1),
        )

        return cur_node_representations
