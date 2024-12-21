import torch
import torch.nn as nn
import math
from itertools import chain
from ordered_set import OrderedSet
from FiCoVuL.dataset.data_preprocess import PreprocessMethod, Word2VecWrapper, DatasetName


class TypeEmbedding(nn.Embedding):
    def __init__(self, n_classes, embed_size=512):
        super().__init__(n_classes, embed_size)


class LexicalEmbedding(nn.Embedding):
    def __init__(self, lex_size, embed_size=512):
        super().__init__(lex_size, embed_size)


class WordEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size)


class PositionalEmbedding(nn.Module):
    tokens_dim = 0

    def __init__(self, num_features, max_len=512, device='cpu'):
        super().__init__()

        self.max_len = max_len
        self.num_features = num_features

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, num_features, device=device).float()  # (L_MAX, FOUT)
        pe.require_grad = False

        position = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        div_term = (torch.arange(0, num_features, 2, device=device).float() * -(math.log(10000.0) / num_features)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
        self.proportion = torch.ones(1, device=device)

        self.device = device

    def forward(self, positions):
        assert positions.dim() == 1
        # assert positions.max() < self.max_len, f"PositionalEmbedding cache is not big enough!"

        if positions.max() < self.max_len:
            return self.proportion * self.pe.index_select(self.tokens_dim, positions)  # (L_MAX, FOUT) -> (N^\prime, FOUT)
        else:
            pe = torch.zeros((len(positions), self.num_features), dtype=torch.float,
                             device=self.proportion.device, requires_grad=False)
            position = positions.view(-1, 1)
            div_term = (torch.arange(0, self.num_features, 2).float() * -(math.log(10000.0) / self.num_features)).exp().to(position.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            return (self.proportion * pe).to(self.device)


class AttentionedSumLayer(torch.nn.Module):
    tokens_dim = 0

    def __init__(self, num_features, num_scores):
        super().__init__()

        self._num_scores = num_scores

        self._linear_proj = nn.Linear(num_features, num_scores, bias=True)
        self._scoring_fn = nn.Parameter(torch.Tensor(num_scores, 1))
        # self._activation_fn = nn.LeakyReLU(0.2)
        self._activation_fn = nn.Mish()

        self.init_params()

    def init_params(self):
        nn.init.xavier_normal_(self._scoring_fn)
        nn.init.xavier_normal_(self._linear_proj.weight)
        nn.init.zeros_(self._linear_proj.bias)

    def forward(self, data, tokens_to_node_map):
        """

        :param data: (N^\prime, FOUT)
        :param tokens_to_node_map: (N^\prime, ) indicating each token belongs to which node
        :return:
        """
        # shape: (N^\prime, FOUT) @ (FOUT, FS) -> (N^\prime, FS) where FS - num of scores
        scores_per_token = self._linear_proj(data)
        # (N^\prime, FS) @ (FS, 1) -> (N^\prime, 1)
        scores_per_token = scores_per_token @ self._scoring_fn
        scores_per_token /= math.sqrt(self._num_scores)
        scores_per_token = self._activation_fn(scores_per_token)
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_token = scores_per_token - scores_per_token.max()
        exp_scores_per_token = scores_per_token.exp()  # softmax
        exp_scores_per_token = torch.ones_like(exp_scores_per_token, dtype=exp_scores_per_token.dtype, device=exp_scores_per_token.device)

        num_nodes = tokens_to_node_map.max() + 1
        # (N, 1)
        exp_scores_sum_per_node = torch.zeros((num_nodes, 1), device=exp_scores_per_token.device)
        exp_scores_sum_per_node.scatter_add_(self.tokens_dim, tokens_to_node_map.view(-1, 1), exp_scores_per_token)
        # (N, 1) -> (N^\prime, 1)
        exp_scores_sum_per_node_lifted = exp_scores_sum_per_node.index_select(self.tokens_dim, tokens_to_node_map)
        # softmax
        weights = exp_scores_per_token / exp_scores_sum_per_node_lifted  # (N^\prime, 1)

        data = data * weights
        size = list(data.shape)
        size[0] = num_nodes

        nodes_feature = torch.zeros(size, device=data.device)
        nodes_feature.scatter_add_(self.tokens_dim, tokens_to_node_map.view(-1, 1).expand_as(data), data)

        return nodes_feature


class JointEmbeddingLayer(nn.Module):
    def __init__(self, type_map, lexical_map, word_map, num_out_features, method: PreprocessMethod, device,
                 dataset: DatasetName = None):
        """
        输入的句子的分词格式（简化后 word->id->str(id)），得到每个 node 的特征向量
        输入格式：[[...], [...], ...] 每个元素不等长
        输出格式：Tensor of N x FOUT
        Args:
            type_map:
            lexical_map:
            word_map:
            num_out_features:
            method:
            device:
        """
        super(JointEmbeddingLayer, self).__init__()

        assert method == PreprocessMethod.RAW or method == PreprocessMethod.RAW_WO_RENAME or (method != PreprocessMethod.RAW and dataset is not None), \
            "[Param Missing] param `dataset` should be given when method is not raw!"
        assert dataset is None or dataset in DatasetName, "[Param Error] `dataset` is not in defined list!"

        type_map.update(["<UNK>"])  # add token <UNK>
        lexical_map.update(["<EMPTY>", "<UNK>"])  # add token <EMPTY> and <UNK>
        word_map.update(["<EMPTY>", "<UNK>"])  # add token <EMPTY> and <UNK>
        self.type_map = type_map
        self.lexical_map = lexical_map
        self.word_map = word_map
        self.num_of_types = len(type_map)
        self.num_of_lexicals = len(lexical_map)
        self.num_of_words = len(word_map)

        self.num_out_features = num_out_features

        # layers
        self.type_embedding = TypeEmbedding(n_classes=self.num_of_types, embed_size=self.num_out_features)
        self.method = method
        if method == PreprocessMethod.RAW or method == PreprocessMethod.RAW_WO_RENAME:
            # 自嵌入具有 empty 的特征向量
            self.lexical_empty_index = self.lexical_map.index("<EMPTY>")
            self.word_empty_index = self.word_map.index("<EMPTY>")
            self.lexical_embedding = LexicalEmbedding(lex_size=self.num_of_lexicals, embed_size=self.num_out_features)
            self.word_embedding = WordEmbedding(vocab_size=self.num_of_words, embed_size=self.num_out_features)
        elif method == PreprocessMethod.WORD2VEC:
            # 预训练的 WV 模型只具有 <UNK> 的特征向量
            self.lexical_empty_index = self.lexical_map.index("<UNK>")
            self.word_empty_index = self.word_map.index("<UNK>")
            self.lexical_embedding = Word2VecWrapper(dataset, f"lexical_{self.num_out_features}.pkl")
            self.word_embedding = Word2VecWrapper(dataset, f"value_{self.num_out_features}.pkl")
        else:
            raise Exception('[JointEmbeddingLayer] Method have not been implemented!')
        self.positional_embedding = PositionalEmbedding(num_features=self.num_out_features, max_len=1024, device=device)

        self.LN_inner = nn.LayerNorm(self.num_out_features)

        self.aggregation = AttentionedSumLayer(num_features=self.num_out_features, num_scores=self.num_out_features // 4)
        self.LN_out = nn.LayerNorm(self.num_out_features)

        # device
        self.device = device

    def forward(self, in_nodes_features):
        """
        先对 lexical 和 word 进行 embedding，然后对句子进行 aggregation，然后再连接 type 的 one-hot 编码
        :param in_nodes_features: List of 3 List of unfixed-length LongTensor
        :return:  fixed-dim FloatTensor
        """
        node_type_features, node_lexicals_features, node_words_features = in_nodes_features

        node_type_features = torch.tensor(node_type_features, dtype=torch.long, device=self.device)
        type_embeddings = self.type_embedding(node_type_features)

        if self.method == PreprocessMethod.RAW or self.method == PreprocessMethod.RAW_WO_RENAME or \
                self.method == PreprocessMethod.WORD2VEC and str(self.lexical_empty_index) in self.lexical_embedding.model.wv.index_to_key and str(self.word_empty_index) in self.word_embedding.model.wv.index_to_key:
            # 将 <EMPTY> 标签放置于空的 sentence 处
            node_lexical_features = list(
                map(lambda x: x if len(x) != 0 else torch.tensor([self.lexical_empty_index], dtype=torch.long, device=x.device),
                    node_lexicals_features))
            node_value_features = list(
                map(lambda x: x if len(x) != 0 else torch.tensor([self.word_empty_index], dtype=torch.long, device=x.device),
                    node_words_features))

            # 获取 node feature
            flattened_lexicals = torch.cat(node_lexical_features, dim=0).to(self.device)
            flattened_values = torch.cat(node_value_features, dim=0).to(self.device)
            tokens_to_node_map = torch.cat(
                list(map(lambda index_tokens: torch.tensor([index_tokens[0]] * torch.numel(index_tokens[1]),
                                                           dtype=torch.long, device=self.device),
                         enumerate(node_value_features))))  # (N^\prime, )
            positions = torch.cat(list(map(lambda x: torch.arange(torch.numel(x), device=self.device),
                                           node_value_features))).to(self.device)       # (N^\prime, )
            flattened_lexicals_embeddings = self.lexical_embedding(flattened_lexicals).to(self.device)  # (N^\prime, FOUT)
            flattened_values_embeddings = self.word_embedding(flattened_values).to(self.device)         # (N^\prime, FOUT)
        elif str(self.word_empty_index) in self.word_embedding.model.wv.index_to_key:
            # 将 <EMPTY> 标签放置于空的 sentence 处
            lexicals_embeddings = list(map(lambda x: self.lexical_embedding(x) if torch.numel(x) != 0 else torch.randn((1, self.num_out_features)),
                                           node_lexicals_features))
            flattened_lexicals_embeddings = torch.cat(lexicals_embeddings, dim=0).to(self.device)
            node_value_features = list(map(lambda x: x.tolist() if len(x) != 0 else [self.word_empty_index], node_words_features))
            flattened_values = torch.tensor(list(chain(*node_value_features)), dtype=torch.long)
            flattened_values_embeddings = self.word_embedding(flattened_values).to(self.device)  # (N^\prime, FOUT)
            tokens_to_node_map = torch.cat(
                list(map(lambda index_tokens: torch.tensor([index_tokens[0]] * len(index_tokens[1]),
                                                           dtype=torch.long, device=self.device),
                         enumerate(node_value_features))))  # (N^\prime, )
            positions = torch.cat(list(map(lambda x: torch.arange(len(x), device=self.device),
                                           node_value_features))).to(self.device)  # (N^\prime, )
        else:
            raise Exception("NOT implemented.")
        flattened_positional_embedding = self.positional_embedding(positions).to(self.device)       # (N^\prime, FOUT)
        flattened_embeddings = (flattened_lexicals_embeddings + flattened_values_embeddings + flattened_positional_embedding) / 3
        # flattened_embeddings = (flattened_values_embeddings + flattened_positional_embedding) / 2
        node_features = self.aggregation(flattened_embeddings, tokens_to_node_map)

        node_features = (type_embeddings + node_features) / 2
        node_features = self.LN_out(node_features)

        return node_features.to(self.device)

    def get_indexes(self):
        pass


if __name__ == "__main__":
    virtual_map = OrderedSet(range(10))
    e = JointEmbeddingLayer(virtual_map, virtual_map, virtual_map, 256, PreprocessMethod.RAW, torch.device('cpu'))
    types = torch.tensor([5, 2, 3], dtype=torch.long)
    lexicals = [torch.tensor([0, 0, 1], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
                torch.tensor([0, 2, 2, 0, 1, 2], dtype=torch.long)]
    values = [torch.tensor([0, 1, 2], dtype=torch.long),
              torch.tensor([], dtype=torch.long),
              torch.tensor([4, 2, 2, 0, 1, 3], dtype=torch.long)]
    result = e((types, lexicals, values))
    print(result)
    print(result.shape)
