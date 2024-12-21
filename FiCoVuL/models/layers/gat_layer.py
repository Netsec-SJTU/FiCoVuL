import torch
import torch.nn as nn
from FiCoVuL.models.layers.graph_global_exchange import GraphGlobalGRUExchange
from FiCoVuL.models.layers.MLP import MLP3Layer


class AttentionedAggregationLayer(torch.nn.Module):
    relation_dim = -2

    def __init__(self, num_in_features_per_relation, num_of_heads, num_of_relations, num_out_features, bias=False):
        super().__init__()

        self.num_in_features_per_relation = num_in_features_per_relation
        self.num_of_heads = num_of_heads
        self.num_of_relations = num_of_relations
        self.num_out_features = num_out_features

        # self.score_fn = nn.Parameter(torch.Tensor(num_of_heads, 1, num_in_features_per_relation))
        self.score_fn = MLP3Layer(num_in_features=num_in_features_per_relation, num_out_features=1, dropout_prob=0.)
        if num_in_features_per_relation != num_out_features:
            self.linear_proj = nn.Linear(num_in_features_per_relation, num_out_features, bias=bias)
        else:
            self.register_parameter('linear_proj', None)
        # self.activation_fn = nn.ReLU()
        self.activation_fn = nn.Mish()
        self.weighting_fun = nn.Softmax(dim=self.relation_dim)
        self.init_params()

    def init_params(self):
        if self.linear_proj is not None:
            nn.init.xavier_uniform_(self.linear_proj.weight)
            if self.linear_proj.bias is not None:
                nn.init.zeros_(self.linear_proj.bias)
        # nn.init.xavier_uniform_(self.score_fn)

    def forward(self, data):
        # data shape: ([...,], NH, NR, FIN) , output shape: (1, FOUT)
        # in this specific case input shape: (N, NH, NR, FOUT)
        assert self.num_of_heads == data.shape[-3], \
            f"num of head {self.num_of_heads} must match input data shape[-3] {data.shape[-3]}"

        '''
        # (NH, 1, FIN) -> ([...,] NH, 1, FIN)
        score_fn_broadcasted = self.explicit_broadcast(self.score_fn, data)
        # ([...,] NH, NR, FIN) * ([...,] NH, 1, FIN) -> ([...,], NH, NR, FIN) -> ([...,], NH, NR, 1)
        scores = (data * score_fn_broadcasted).sum(-1, keepdim=True)
        '''
        # ([...,], NH, NR, FIN) -> ([...,], NH, NR, 1)
        scores = self.score_fn(data)
        scores = self.activation_fn(scores)
        attentions_per_relation = self.weighting_fun(scores)  # ([...,], NH, NR, 1)
        # ([...,], NH, NR, FIN) * ([...,], NH, NR, 1) -> ([...,], NH, NR, FIN)
        features_weighted = data * attentions_per_relation
        out_features = features_weighted.sum(self.relation_dim)  # ([...,], NH, FIN) due to automatic squeeze by sum()
        if self.num_in_features_per_relation != self.num_out_features:
            # ([...,] FIN) @ (FIN, FOUT) -> ([...,] FOUT)
            out_features = self.linear_proj(features_weighted)
        return out_features

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(0)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)


class GATLayer(torch.nn.Module):
    # We'll use these constants in many functions so just extracting them here as member fields
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index
    rel_type_dim = 2  # position of relation in edge index

    # These may change in the inductive setting - leaving it like this for now (not future proof)
    nodes_dim = 0  # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1  # attention head dim
    relation_dim = 2  # different relations

    def __init__(self, num_in_features, num_out_features, num_of_heads, num_of_relations,
                 concat=True, output_activation=nn.ELU(), dropout_prob=0.2,
                 add_skip_connection=True, enable_global_exchange=False, enable_LN=True, num_graph_features=None, bias=True,
                 log_attention_weights=False):
        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.num_of_relations = num_of_relations
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.enable_global_exchange = enable_global_exchange

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_relations&num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_of_relations * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the "additive" scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.
        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_of_relations, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_of_relations, num_out_features))

        # 聚合不同的类型信息
        self.aggregate_relations = AttentionedAggregationLayer(num_out_features, num_of_heads, num_of_relations,
                                                               num_out_features, bias=False)

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        if enable_global_exchange:
            assert num_graph_features is not None, f"global exchange need specific num_graph_features"
            self.global_exchange = GraphGlobalGRUExchange(num_graph_features=num_graph_features,
                                                          num_node_features=num_of_heads*num_out_features,
                                                          num_in_heads=num_of_heads)
        else:
            self.register_parameter('global_exchange', None)

        if enable_LN:
            if concat:
                self.LN = nn.LayerNorm(num_of_heads * num_out_features)
            else:
                self.LN = nn.LayerNorm(num_out_features)
        else:
            self.register_parameter('LN', None)

        #
        # End of trainable weights
        #

        self.score_activation = nn.LeakyReLU(0.2)
        self.output_activation = output_activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = []  # for later visualization purposes, I cache the weights here

        self.init_params()

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #

        in_nodes_features, edge_index, node_to_graph_map = data  # unpack data
        # in_nodes_features: (N, FIN), edge_index: (3, E)
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 3, f'Expected edge index with shape=(3,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all the input node features (as mentioned in the paper)
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*NR*FOUT) -> (N, NH, NR, FOUT)
        # where NH - number of heads, NR - num of relations, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(
            -1, self.num_of_heads, self.num_of_relations, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, NR, FOUT) * (1, NH, NR, FOUT) -> (N, NH, NR, 1) -> (N, NH, NR)
        # because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        out_nodes_features_list = []
        for rel_i in range(self.num_of_relations):
            mask = edge_index[self.rel_type_dim] == rel_i
            edge_index_rel_i = edge_index[:, mask]
            not_exists_rel_i = edge_index_rel_i.size().count(0) > 0  # 判断是否全为 False
            if not_exists_rel_i:
                out_nodes_features_list.append(torch.zeros_like(nodes_features_proj[:, :, rel_i, :]))
                continue
            # 选择需要的 relation 对应的 score , shape: (N, NH)
            scores_source_rel_i = torch.index_select(scores_source, dim=self.relation_dim,
                                                     index=torch.tensor(rel_i, dtype=torch.long,
                                                                        device=scores_source.device)
                                                     ).squeeze(-1)
            scores_target_rel_i = torch.index_select(scores_target, dim=self.relation_dim,
                                                     index=torch.tensor(rel_i, dtype=torch.long,
                                                                        device=scores_target.device)
                                                     ).squeeze(-1)

            # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
            # the possible combinations of scores we just prepare those that will actually be used and those are defined
            # by the edge index.
            # scores shape = (E_rel_i, NH), nodes_features_proj_rel_i_lifted shape = (E_rel_i, NH, FOUT),
            # E_rel_i - number of edges with relation i in the graph
            nodes_features_proj_rel_i = nodes_features_proj.index_select(self.relation_dim,
                                                                         torch.tensor(rel_i, dtype=torch.long,
                                                                                      device=nodes_features_proj.device)
                                                                         ).squeeze(self.relation_dim)
            scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(
                scores_source_rel_i, scores_target_rel_i, nodes_features_proj_rel_i, edge_index_rel_i)
            scores_per_edge = self.score_activation(scores_source_lifted + scores_target_lifted)
            del scores_source_lifted, scores_target_lifted

            # shape = (E_rel_i, NH, 1)
            attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index_rel_i[self.trg_nodes_dim],
                                                                  num_of_nodes)
            # Add stochasticity to neighborhood aggregation
            attentions_per_edge = self.dropout(attentions_per_edge)

            #
            # Step 3: Neighborhood aggregation
            #

            # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
            # shape = (E_rel_i, NH, FOUT) * (E_rel_i, NH, 1) -> (E_rel_i, NH, FOUT), 1 gets broadcast into FOUT
            nodes_features_proj_lifted *= attentions_per_edge

            # This part sums up weighted and projected neighborhood feature vectors for every target node
            # shape = (N, NH, FOUT)
            out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted, edge_index_rel_i,
                                                          in_nodes_features, num_of_nodes)
            del nodes_features_proj_lifted

            out_nodes_features_list.append(out_nodes_features)

            if self.log_attention_weights:  # potentially log for later visualization
                self.attention_weights.append(attentions_per_edge)

        # shape: (N, NH, NR, FOUT)
        out_nodes_features = torch.stack(out_nodes_features_list, dim=self.relation_dim)
        del out_nodes_features_list

        # 先 aggregate 后 skip_connection

        #
        # Step 4: aggregate relational features
        #

        # output shape: (N, NH, FOUT)
        out_nodes_features = self.aggregate_relations(out_nodes_features)

        #
        # Step 5: Residual/skip connections, concat and bias
        #

        # output shape: (N, NH*FOUT) or (N, FOUT) for last layer
        out_nodes_features = self.skip_global_concat_bias(in_nodes_features, node_to_graph_map, out_nodes_features)

        if self.LN is not None:
            out_nodes_features = self.LN(out_nodes_features)

        return out_nodes_features, edge_index, node_to_graph_map

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and its (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result, but it improves the numerical stability,
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        try:
            scores_per_edge = scores_per_edge - scores_per_edge.max(dim=0).values
        except Exception as e:
            print(scores_per_edge)
            print(scores_per_edge.shape)
            raise e
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index,
                                                                                num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim],
                                                        nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_global_concat_bias(self, in_nodes_features, node_to_graph_map, out_nodes_features):
        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads,
                                                                             self.num_out_features)

        # 先 global exchange 再 concat
        if self.enable_global_exchange:
            out_nodes_features = self.global_exchange((out_nodes_features, node_to_graph_map))

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)  # TODO

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.output_activation is None else self.output_activation(out_nodes_features)
