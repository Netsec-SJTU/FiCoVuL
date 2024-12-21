from sklearn.metrics import f1_score

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import shutil
import time
import enum
import git
import re  # regex
import argparse
import tqdm
import random
import math
import numpy as np
import json5
from copy import deepcopy

from dataset.data_preprocess import DATA_DIR_PATH
from dataset.data_loader import DatasetName, PreprocessMethod, load_graph_data, MINE_NUM_RELATIONS, load_map
from models.GATBinaryClassification import GATBinaryClassification


config = {
  "task_name": "",
  "dataset_name": DatasetName.CroVul,
  "preprocess": {
    "method": PreprocessMethod.RAW,
    "proportion": [8, 1, 1],
    "origin_data_path_pattern": "/home/huanghongjun/FiCoVuL/FiCoVuL/data/datasets/CroVul_ORIG/*.c",
  },
  "model": {
    "num_of_gat_layers": 4,
    "num_heads_per_gat_layer": [8, 4, 4, 6],
    "num_features_per_gat_layer": [128, 32, 64, 64, 64],
    "graph_representation_size": 512,
    "num_heads_in_graph_representation": 4,
    "add_self_loop": True,
    "add_reverse_edge": True,
    "add_skip_connection": True,
    "bias": True,
    "dropout": 0.1,
  },
  "run": {
    "manual_seed": None,
    "num_of_epochs": 200,
    "batch_size": 128,
    "loader_batch_size": 128,
    "lr": 0.0001,
    "weight_decay": 1e-4,
    "patience_period": 32,
    "classfication_threshold": 0.5,
    "enable_tensorboard": True,
    "console_log_freq": 5,
    "checkpoint_freq": 1,
    "load_model": "/home/huanghongjun/FiCoVuL/FiCoVuL/data/models/gat_CroVul_crovul_graph_focal_045_2_000000.pth",
    "test_only": False,
    "force_cpu": False,
  },
  "force_refresh": False,
}

from dataset.data_preprocess import *

preprocess_config = config

if True:
    filenames = glob.glob(rf"/home/huanghongjun/FiCoVuL/FiCoVuL/data/datasets/CroVul/*.json")
    type_map = pickle_read("/home/huanghongjun/FiCoVuL/FiCoVuL/data/datasets/CroVul_RAW/node_type_dict.pkl")
    lexical_map = pickle_read("/home/huanghongjun/FiCoVuL/FiCoVuL/data/datasets/CroVul_RAW/word_lexical_dict.pkl")
    value_map = pickle_read("/home/huanghongjun/FiCoVuL/FiCoVuL/data/datasets/CroVul_RAW/word_value_dict.pkl")

    total_filenames = []
    for filename in filenames:
        data = json_read(filename)
        len_nodes = len(data['nodes'])
        # 数据清洗：删除具有 5000 及以上个节点的图样本
        if len_nodes > 5000:
            continue
        total_filenames.append(filename)
    filenames = total_filenames

    # 匹配处理完的文件和源文件，即 preprocess 模块中的 result 和 enhanced
    # result -> uid -> enhanced
    result2uid = dict(map(lambda path: (path, '-'.join(path.split('/')[-1].split('.')[0].split('-')[:-1])), filenames))
    uid2enhanced = dict(map(lambda path: (path.split('/')[-1].split('.')[0], path),
                            glob.iglob(preprocess_config['preprocess']['origin_data_path_pattern'], recursive=True)))
    processed2orig = dict(map(lambda item: (item[0], uid2enhanced[item[1]]), result2uid.items()))

    most_commons = collect_function_names(preprocess_config['preprocess']['origin_data_path_pattern'], 'cpp')

    # 统一用 cpp 处理
    node_type_map = OrderedSet()  # 将字符串转化为数字，减少存储量
    # 先生成完整的字典
    min_count = 20
    lexical_counter = Counter()
    value_counter = Counter()
    for filename in tqdm.tqdm(glob.iglob(preprocess_config['preprocess']['origin_data_path_pattern'], recursive=True),
                              desc="Generating word dict"):
        with open(filename, 'r') as fp:
            code = fp.read()
        cn = CodeNormalizer('cpp', code, most_commons)
        _lexical, _value = cn.get_all_tokens()
        lexical_counter.update(_lexical)
        value_counter.update(_value)
    # joern 自己加入的一个代码，特殊处理
    value_counter.update(["RET"])

    total_graphs = {
        "n_graphs": len(filenames),
        "graph_id_index": [],
        "graphs": {}
    }

    print("Generating dataset...")
    cnt = 0
    for filename in tqdm.tqdm(filenames):
        with open(processed2orig[filename], 'r') as fp:
            orig_code = fp.read()
        cn = CodeNormalizer('cpp', orig_code, most_commons)
        split_lines = orig_code.split('\n')

        data = json_read(filename)
        graph_index_id = filename.split('/')[-1]

        # 先生成str版本的数据集
        sorted_data = sorted(data["nodes"].items(), key=lambda item: int(item[0]))
        # node type
        sorted_type = list(map(lambda item: item[1]["type"], sorted_data))
        sorted_ndoes_lines = list(map(lambda item: int(item[1]["line"]), sorted_data))
        _type = list(map(type_map.index, sorted_type))

        # node lexical & value
        def extract_lexical_and_value(item):  # 传入 sorted_data: (id, data)
            if len(item[1]["code"]) == 0:
                return [], []
            if item[1]["type"] == "METHOD_RETURN" and item[1]["code"] == 'RET':
                return ["Token.Keyword"], ["RET"]
            # position: line 从 1 开始，column 从 0 开始
            _pos = (int(item[1]["line"]), int(item[1]["column"]))
            if _pos == (-1, -1):
                if item[1]["type"] == "METHOD":
                    return ["Token.Name.Function"], [
                        item[1]["code"] if item[1]["code"] not in cn.renamer.keys() else cn.renamer[
                            item[1]["code"]]]
                elif item[1]["type"] == "LOCAL":
                    return ["Token.Name"], [
                        item[1]["code"] if item[1]["code"] not in cn.renamer.keys() else cn.renamer[
                            item[1]["code"]]]
                else:
                    print(filename)
                    print(item[1])
                    raise Exception()
            sstart = len('\n'.join(split_lines[:_pos[0] - 1])) + 1 + _pos[1] if _pos[0] != 1 else len(
                '\n'.join(split_lines[:_pos[0] - 1])) + _pos[1]
            ssend = sstart + len(item[1]["code"])
            _lexicals, _values = cn.localize_tokens(sstart, ssend)
            return _lexicals, _values

        _lexical, _value = zip(*map(extract_lexical_and_value, sorted_data))
        # 去除一些错误的 sample
        if False in map(lambda _v: _v in value_counter.keys(), chain(*_value)):
            cnt += 1
            continue
        converted_lexical = list(map(lambda lexicals: list(map(
            lambda _l: lexical_map.index(_l) if _l in lexical_map else lexical_map.index('<UNK>'), lexicals)),
                                     _lexical))
        converted_value = list(map(lambda values: list(map(
            lambda _v: value_map.index(_v) if _v in value_map else value_map.index('<UNK>'), values)), _value))
        # links
        _edges = data["edges"]

        total_graphs["graph_id_index"].append(graph_index_id)
        total_graphs["graphs"][graph_index_id] = {
            "node_types": _type,
            "node_lexical_features": converted_lexical,
            "node_value_features": converted_value,
            "node_lines": sorted_ndoes_lines,
            "edges": _edges,
            "lines": data["properties"]["lines"],
        }
        total_graphs["graphs"][graph_index_id].update(data["properties"])

import os
import torch
from torch.utils.data import DataLoader, Dataset

MINE_NUM_RELATIONS = 4


class GraphDataLoader(DataLoader):
    """
    When dealing with batches it's always a good idea to inherit from PyTorch's provided classes (Dataset/DataLoader).

    """
    def __init__(self, node_features_list, node_labels_list, edge_index_list, graph_list, batch_size=1, shuffle=False):
        graph_dataset = GraphDataset(node_features_list, node_labels_list, edge_index_list, graph_list)
        # We need to specify a custom collate function, it doesn't work with the default one
        super().__init__(graph_dataset, batch_size, shuffle, collate_fn=graph_collate_fn)


class GraphDataset(Dataset):
    """
    This one just fetches a single graph from the split when GraphDataLoader "asks" it

    """
    def __init__(self, node_features_list, node_labels_list, edge_index_list, graph_list):
        self.node_features_list = node_features_list
        self.node_labels_list = node_labels_list
        self.edge_index_list = edge_index_list
        self.graph_list = graph_list

    # 2 interface functions that need to be defined are len and getitem so that DataLoader can do it's magic
    def __len__(self):
        return len(self.edge_index_list)

    def __getitem__(self, idx):  # we just fetch a single graph
        return self.node_features_list[idx], self.node_labels_list[idx], self.edge_index_list[idx], self.graph_list[idx]


def graph_collate_fn(batch):
    """
    The main idea here is to take multiple graphs from dataset as defined by the batch size
    and merge them into a single graph with multiple connected components.

    It's important to adjust the node ids in edge indices such that they form a consecutive range. Otherwise,
    the scatter functions in the implementation 3 will fail.

    :param batch: contains a list of edge_index, node_features, graph_labels_index tuples (as provided by the GraphDataset)
    """

    edge_index_list = []
    node_type_list = []
    node_lexical_list = []
    node_value_list = []
    graph_label_list = []
    num_nodes_seen = 0
    num_graph_seen = 0
    node_to_graph_map_list = []
    graph_list = []

    for features_labels_edge_index_tuple in batch:
        # Just collect these into separate lists
        node_features = features_labels_edge_index_tuple[0]
        assert len(node_features[0]) == len(node_features[1]) == len(node_features[2])
        node_type_list.extend(node_features[0])
        node_lexical_list.extend(node_features[1])
        node_value_list.extend(node_features[2])
        num_nodes = len(node_features[0])
        graph_label_list.append(features_labels_edge_index_tuple[1])
        edge_index = features_labels_edge_index_tuple[2]  # all of the components are in the [0, N] range
        # very important! translate the range of this component
        edge_index_list.append(edge_index + torch.tensor([[num_nodes_seen], [num_nodes_seen], [0]], dtype=torch.long))
        node_to_graph_map_list = node_to_graph_map_list + [num_graph_seen] * num_nodes
        graph_list.append(features_labels_edge_index_tuple[3])

        num_nodes_seen += num_nodes  # update the number of nodes we've seen so far
        num_graph_seen += 1

    # Merge the graphs into a single graph with multiple connected components
    '''nll
    graph_labels_index = torch.cat(graph_label_list, 0).view(-1, 1)
    size = list(graph_labels_index.shape)
    size[1] = MINE_NUM_CLASSES
    graph_labels = torch.zeros(size, dtype=torch.float)
    graph_labels.scatter_add_(1, graph_labels_index.long(),
                              torch.tensor([1], dtype=torch.float).expand_as(graph_labels_index))
    '''
    graph_labels = torch.cat(graph_label_list)
    edge_index = torch.cat(edge_index_list, 1)
    node_to_graph_map = torch.tensor(node_to_graph_map_list, dtype=torch.long)

    return [node_type_list, node_lexical_list, node_value_list], graph_labels, edge_index, node_to_graph_map, graph_list


def load_graph_data(training_config, device):
    global total_graphs
    edge_index_list = []
    node_features_list = []
    graph_label_list = []
    graph_list = []
    if True:
        for graph_id in tqdm.tqdm(total_graphs["graph_id_index"]):
            graph = total_graphs["graphs"][graph_id]
            # shape = (3, E) - where E is the number of edges in the graph
            # Note: leaving the tensors on CPU I'll load them to GPU in the training loop on-the-fly as VRAM
            # is a scarcer resource than CPU's RAM and the whole dataset can't fit during the training.
            edge_index = torch.tensor(graph["edges"], dtype=torch.long).transpose(0, 1).contiguous()
            nodes = list(range(len(graph["node_types"])))
            if training_config['model']["add_reverse_edge"]:
                edges_reversed = edge_index.index_select(0, torch.LongTensor([1, 0, 2])) + \
                                 torch.tensor([[0], [0], [MINE_NUM_RELATIONS]], dtype=torch.long)
                edge_index = torch.cat([edge_index, edges_reversed], dim=1)
            if training_config['model']["add_self_loop"]:
                nodes_index = torch.tensor(nodes, dtype=torch.long).repeat(1, MINE_NUM_RELATIONS)
                relations_index = torch.tensor(range(MINE_NUM_RELATIONS),
                                               dtype=torch.long).view(1, -1).repeat_interleave(len(nodes), dim=1)
                assert nodes_index.shape[1] == relations_index.shape[1]
                self_loop_index = torch.cat([nodes_index, nodes_index, relations_index], dim=0)
                edge_index = torch.cat([edge_index, self_loop_index], dim=1)
            edge_index_list.append(edge_index)
            # shape = (N, list[longTensor]) - where N is the number of nodes in the graph
            node_features_list.append([
                list(map(lambda _t: torch.tensor(_t, dtype=torch.long), graph["node_types"])),
                list(map(lambda tokens: torch.tensor(tokens, dtype=torch.long), graph["node_lexical_features"])),
                list(map(lambda tokens: torch.tensor(tokens, dtype=torch.long), graph["node_value_features"]))
            ])
            # shape = (N, 121), BCEWithLogitsLoss doesn't require long/int64 so saving some memory by using float32
            graph_label_list.append(
                torch.tensor([graph["label"]] if isinstance(graph["label"], int) else graph["label"],
                             dtype=torch.float))
            graph_list.append(graph)
        # Prepare graph data loaders
        data_loader = GraphDataLoader(
            node_features_list, graph_label_list, edge_index_list, graph_list,
            batch_size=training_config['run']['loader_batch_size'],
            # I implement batch during training, where loss is added, so leave it for now
            shuffle=False
        )
        return data_loader
data_loader = load_graph_data(config, "cpu")

device = torch.device("cpu")
last_training_state = torch.load(config['run']['load_model'], device)
print("last test: ", last_training_state['perf'])

type_map, lexical_map, word_map = load_map(last_training_state)

# Step 2: prepare the model
gat = GATBinaryClassificationRandom(
    type_map=type_map,
    lexical_map=lexical_map,
    word_map=word_map,
    method=config['preprocess']['method'],
    device=device,
    num_of_gat_layers=config['model']['num_of_gat_layers'],
    num_heads_per_gat_layer=config['model']['num_heads_per_gat_layer'],
    num_features_per_gat_layer=config['model']['num_features_per_gat_layer'],
    add_skip_connection=config['model']['add_skip_connection'],
    bias=config['model']['bias'],
    num_of_relations=MINE_NUM_RELATIONS * 2 if config['model']['add_reverse_edge'] else MINE_NUM_RELATIONS,
    dropout=config['model']['dropout'],
    log_attention_weights=False,  # no need to store attentions, used only in playground.py for visualizations
    graph_representation_size=config['model']['graph_representation_size'],
    num_heads_in_graph_representation=config['model']['num_heads_in_graph_representation'],
    dataset=config['dataset_name']
)
gat.load_state_dict(last_training_state['state_dict'])

import random
bar = tqdm.tqdm(data_loader)
max_mini_batch = len(bar)

dataset0 = np.array([[]])
dataset1 = np.array([[]])

for mini_batch_idx, (node_features, gt_graph_labels, edge_index, node_to_graph_map, graphs) in enumerate(bar):
    with torch.no_grad():
        nodes_embeddings = gat.embedding(node_features)  # (N, FIN)
        nodes_features, _, _ = gat.gats((nodes_embeddings, edge_index, node_to_graph_map))

    cur_graph_idx = -1
    for graph in graphs:
        cur_graph_idx += 1
        lines = set(graph['lines'])
        node_lines = list(map(int, graph['node_lines']))
        nodes_interested_idx = list(filter(lambda item: item[1] in lines, map(lambda item: (item[0], node_lines[item[0]], len(item[1])), enumerate(graph['node_value_features']))))
        node_result = []
        for l in lines:
            nodes = list(filter(lambda item: item[1] == l, nodes_interested_idx))
            if len(nodes) != 0:
                node = max(nodes, key=lambda item: item[2])
                node_result.append(node)
        if len(node_result) == 0:
            continue
        idxs = list(map(lambda item: item[0], node_result))

        label = graph['label']
        labels = [int(label)] * len(idxs)

        _nodes_features = nodes_features[node_to_graph_map == cur_graph_idx]
        interested_nodes_features = torch.index_select(_nodes_features, 0, torch.LongTensor(idxs)).numpy()
        labels = np.array(labels, dtype=np.float64).reshape((-1, 1))
        
        if dataset0.size == 0:
            dataset0 = interested_nodes_features
            dataset1 = labels
        else:
            dataset0 = np.concatenate([dataset0, interested_nodes_features], 0)
            dataset1 = np.concatenate([dataset1, labels], 0)

    bar.set_postfix({"size": dataset0.shape[0]})

np.save("/home/huanghongjun/FiCoVuL/FiCoVuL/data/datasets/node_level/crovul_data1016.npy", dataset0)
np.save("/home/huanghongjun/FiCoVuL/FiCoVuL/data/datasets/node_level/crovul_label1016.npy", dataset1)

