import os
import json
import tqdm
import pickle
from ordered_set import OrderedSet
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.decomposition import PCA

def json_read(path):
    with open(path, 'r') as f:
        content = json.load(f)
    return content

def json_save(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

def pickle_read(path):
    with open(path, 'rb') as f:
        content = pickle.load(f)
    return content

def data_preprocess():
    dataset_path = r'/home/huanghongjun/FiCoVuL/FiCoVuL/data/datasets/CroVul_RAW'
    MINE_NUM_RELATIONS = 4
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    for split in ['train', 'valid', 'test']:
        print(split)
        graphs = json_read(os.path.join(dataset_path, f"{split}.json"))
        node_value_dict = pickle_read(os.path.join(dataset_path, f"word_value_dict.pkl"))
        node_features_list = []
        edge_index_list = []
        graph_label_list = []
        for graph_id in tqdm.tqdm(graphs["graph_id_index"], desc=f"Loading {split} graphs"):
            graph = graphs["graphs"][graph_id]
            nodes = list(range(len(graph["node_types"])))
            # print("nodes=", nodes)
            edge_index = graph["edges"]
            edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1).contiguous()
            if True:
                edges_reversed = edge_index.index_select(0, torch.LongTensor([1, 0, 2])) + \
                                 torch.tensor([[0], [0], [MINE_NUM_RELATIONS]], dtype=torch.long)
                edge_index = torch.cat([edge_index, edges_reversed], dim=1)
            if True:
                nodes_index = torch.tensor(nodes, dtype=torch.long).repeat(1, MINE_NUM_RELATIONS)
                relations_index = torch.tensor(range(MINE_NUM_RELATIONS),
                                               dtype=torch.long).view(1, -1).repeat_interleave(len(nodes), dim=1)
                assert nodes_index.shape[1] == relations_index.shape[1]
                self_loop_index = torch.cat([nodes_index, nodes_index, relations_index], dim=0)
                edge_index = torch.cat([edge_index, self_loop_index], dim=1).tolist()
            edge_index_list.append(edge_index)

            node_value_features = list(map(lambda x: node_value_dict[x], graph["node_value_features"]))
            node_value_features = list(map(lambda x: ''.join(x).replace('<UNK>', 'UNK'), node_value_features))
            # print('\n'.join(node_value_features))
            node_value_features = list(map(lambda x: [tokenizer.cls_token]+[tokenizer.sep_token]+tokenizer.tokenize(x)+[tokenizer.eos_token], node_value_features))
            node_value_features = list(map(tokenizer.convert_tokens_to_ids, node_value_features))
            node_value_features = list(map(lambda x: model(torch.tensor(x)[None,:])[0].squeeze(0).detach().numpy(), node_value_features))
            # 将向量视为样本，执行PCA
            node_value_features = list(map(lambda x: PCA(n_components=1).fit_transform(x.T).T if x.shape[0] != 1 else x, node_value_features))
            node_value_features = np.concatenate(node_value_features, axis=0)
            # print(node_value_features.shape)
            node_features_list.append(node_value_features.tolist())

            # shape = (1,), BCEWithLogitsLoss doesn't require long/int64 so saving some memory by using float32
            graph_label_list.append(float(graph["label"]))
        json_save(node_features_list, os.path.join('/home/huanghongjun/FiCoVuL/FiCoVuL/data/datasets/CroVul_PRECB', 'features.json'))
        json_save(edge_index_list, os.path.join('/home/huanghongjun/FiCoVuL/FiCoVuL/data/datasets/CroVul_PRECB', 'edges.json'))
        json_save(graph_label_list, os.path.join('/home/huanghongjun/FiCoVuL/FiCoVuL/data/datasets/CroVul_PRECB', 'labels.json'))

data_preprocess()