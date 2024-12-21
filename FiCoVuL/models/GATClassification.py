import torch
from torch import nn
from FiCoVuL.models.layers.embedding_layer import JointEmbeddingLayer
from FiCoVuL.models.layers.gat_layer import GATLayer
from FiCoVuL.models.layers.nodes_to_graph_representation import WeightedSumGraphRepresentation
from FiCoVuL.models.layers.MLP import MLP4Layer
from FiCoVuL.dataset.data_preprocess import PreprocessMethod, DatasetName


class GATBinaryClassification(nn.Module):
    def __init__(self, type_map, lexical_map, word_map, method: PreprocessMethod, device,
                 num_of_gat_layers, num_heads_per_gat_layer, num_features_per_gat_layer, add_skip_connection=True,
                 enable_global_exchange=False, bias=True, num_of_relations=3, dropout=0.6, log_attention_weights=False,
                 graph_representation_size=128, num_heads_in_graph_representation=8, dataset: DatasetName = None):
        super().__init__()
        assert num_of_gat_layers == len(num_heads_per_gat_layer) == len(
            num_features_per_gat_layer) - 1, f'Enter valid arch params.'

        num_heads_per_gat_layer = [1] + num_heads_per_gat_layer  # trick - so that I can nicely create GAT layers below

        self.embedding = JointEmbeddingLayer(type_map, lexical_map, word_map, num_features_per_gat_layer[0],
                                             method=method, device=device, dataset=dataset)
        gat_layers = []  # collect GAT layers
        for i in range(num_of_gat_layers):
            # output size: num_heads_per_gat_layer[i + 1]*num_features_per_gat_layer[i + 1]
            layer = GATLayer(
                # consequence of concatenation
                num_in_features=num_features_per_gat_layer[i] * num_heads_per_gat_layer[i],
                num_out_features=num_features_per_gat_layer[i + 1],
                num_of_heads=num_heads_per_gat_layer[i + 1],
                num_of_relations=num_of_relations,
                concat=True if i < num_of_gat_layers - 1 else False,
                # last GAT layer does mean avg, the others do concat
                output_activation=nn.ELU() if i < num_of_gat_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                enable_global_exchange=enable_global_exchange,
                # enable_global_exchange=True if i % 2 == 0 else False,
                enable_LN=True,  # 别改
                num_graph_features=graph_representation_size,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)
        self.gats = nn.Sequential(*gat_layers)  # (N, FOUT)
        # self.LN1 = nn.LayerNorm(num_features_per_gat_layer[-1])
        self.graph_repr = WeightedSumGraphRepresentation(graph_representation_size, num_heads_in_graph_representation,
                                                         num_features_per_gat_layer[-1])  # (G, FG)
        self.LN2 = nn.LayerNorm(graph_representation_size)
        self.graph_classifier = MLP4Layer(graph_representation_size, 1, dropout_prob=dropout)  # (G, 1)

    def get_node_graph_repr(self, data):
        in_nodes_words, edge_index, node_to_graph_map, _ = data  # unpack data
        nodes_embeddings = self.embedding(in_nodes_words)  # (N, FIN)
        del in_nodes_words
        nodes_features, _, _ = self.gats((nodes_embeddings, edge_index, node_to_graph_map))
        # nodes_features = self.LN1(nodes_features)
        graphs_representation = self.graph_repr((nodes_features, node_to_graph_map))
        graphs_representation = self.LN2(graphs_representation)

        return nodes_features, graphs_representation

    # data is just a (in_nodes_features, edge_index) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        _, graphs_representation = self.get_node_graph_repr(data)
        output = self.graph_classifier(graphs_representation)

        return output


class GATDualClassification(nn.Module):
    def __init__(self, type_map, lexical_map, word_map, method: PreprocessMethod, device,
                 num_of_gat_layers, num_heads_per_gat_layer, num_features_per_gat_layer, add_skip_connection=True,
                 enable_global_exchange=False, bias=True, num_of_relations=3, dropout=0.6, log_attention_weights=False,
                 graph_representation_size=128, num_heads_in_graph_representation=8, dataset: DatasetName = None,
                 concat_gat_outputs=False):
        super().__init__()
        assert num_of_gat_layers == len(num_heads_per_gat_layer) == len(
            num_features_per_gat_layer) - 1, f'Enter valid arch params.'

        num_heads_per_gat_layer = [1] + num_heads_per_gat_layer  # trick - so that I can nicely create GAT layers below

        self.embedding = JointEmbeddingLayer(type_map, lexical_map, word_map, num_features_per_gat_layer[0],
                                             method=method, device=device, dataset=dataset)
        self.gat_layers = []  # collect GAT layers
        for i in range(num_of_gat_layers):
            # output size: num_heads_per_gat_layer[i + 1]*num_features_per_gat_layer[i + 1]
            layer = GATLayer(
                # consequence of concatenation
                num_in_features=num_features_per_gat_layer[i] * num_heads_per_gat_layer[i],
                num_out_features=num_features_per_gat_layer[i + 1],
                num_of_heads=num_heads_per_gat_layer[i + 1],
                num_of_relations=num_of_relations,
                concat=True if i < num_of_gat_layers - 1 else False,
                # last GAT layer does mean avg, the others do concat
                output_activation=nn.ELU() if i < num_of_gat_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                enable_global_exchange=enable_global_exchange,
                # enable_global_exchange=True if i % 2 == 0 else False,
                enable_LN=True,  # 别改
                num_graph_features=graph_representation_size,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            self.gat_layers.append(layer)
        self.LN1 = nn.LayerNorm(num_features_per_gat_layer[-1])
        self.graph_repr = WeightedSumGraphRepresentation(graph_representation_size, num_heads_in_graph_representation,
                                                         num_features_per_gat_layer[-1])  # (G, FG)
        self.LN2 = nn.LayerNorm(graph_representation_size)
        self.graph_classifier = MLP4Layer(graph_representation_size, 1, dropout_prob=dropout)  # (G, 1)

        self.num_of_gat_layers = num_of_gat_layers
        self.num_heads_per_gat_layer = num_heads_per_gat_layer
        self.num_features_per_gat_layer = num_features_per_gat_layer
        self.concat_gat_outputs = concat_gat_outputs
        # TODO：改成自己给出各层参数
        if concat_gat_outputs:
            self.node_classifier = MLP4Layer(sum(num_features_per_gat_layer), 1, dropout_prob=dropout)
        else:
            self.node_classifier = MLP4Layer(num_features_per_gat_layer[-1], 1, dropout_prob=dropout)

    def get_node_graph_repr(self, data):
        in_nodes_words, edge_index, node_to_graph_map = data  # unpack data
        nodes_features = self.embedding(in_nodes_words)  # (N, FIN)
        del in_nodes_words
        concat_nodes_embeddings = []
        for i, gat in enumerate(self.gat_layers):
            if self.concat_gat_outputs:
                concat_nodes_embeddings.append(nodes_features.view(-1, self.num_heads_per_gat_layer[i], self.num_features_per_gat_layer[i]).mean(dim=1))
            nodes_features, _, _ = gat((nodes_features, edge_index, node_to_graph_map))
        if self.concat_gat_outputs:
            concat_nodes_embeddings.append(nodes_features)
        # nodes_features = self.LN1(nodes_features)
        graphs_representation = self.graph_repr((nodes_features, node_to_graph_map))
        graphs_representation = self.LN2(graphs_representation)

        nodes_features = torch.concat(concat_nodes_embeddings, dim=1) if self.concat_gat_outputs else nodes_features

        return nodes_features, graphs_representation

    def forward(self, data):
        # node_representation 可以是每一层的 concat
        nodes_representation, graphs_representation = self.get_node_graph_repr(data)
        goutput = self.graph_classifier(graphs_representation)
        noutput = self.node_classifier(nodes_representation)

        return goutput, noutput

    def to(self, *args, **kwargs):
        for x in self.gat_layers:
            x.to(*args, **kwargs)
        return super().to(*args, **kwargs)
