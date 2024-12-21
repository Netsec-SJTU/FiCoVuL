import os
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from FiCoVuL.dataset.data_preprocess import PreprocessMethod, DatasetName, DATASET_DIR_PATH, preprocess, json_read, pickle_read

MINE_RELATIONS_OF_INTEREST = [0, 1, 2, 3]
# MINE_RELATIONS_OF_INTEREST = list(range(12))  # Joern2
MINE_RALATIONS_INDEX_MAP = dict(map(lambda x: (x[1], x[0]), enumerate(MINE_RELATIONS_OF_INTEREST)))
MINE_NUM_RELATIONS = len(MINE_RELATIONS_OF_INTEREST)


class GraphDataLoader(DataLoader):
    """
    When dealing with batches it's always a good idea to inherit from PyTorch's provided classes (Dataset/DataLoader).

    """
    def __init__(self, node_features_list, graph_labels_list, edge_index_list, rpst_labels_list, batch_size=1, shuffle=False):
        graph_dataset = GraphDataset(node_features_list, graph_labels_list, edge_index_list, rpst_labels_list)
        # We need to specify a custom collate function, it doesn't work with the default one
        super().__init__(graph_dataset, batch_size, shuffle, collate_fn=graph_collate_fn)


class GraphDataset(Dataset):
    """
    This one just fetches a single graph from the split when GraphDataLoader "asks" it

    """
    def __init__(self, node_features_list, node_labels_list, edge_index_list, rpst_labels_list):
        self.node_features_list = node_features_list
        self.node_labels_list = node_labels_list
        self.edge_index_list = edge_index_list
        self.rpst_labels_list = rpst_labels_list

    # 2 interface functions that need to be defined are len and getitem so that DataLoader can do it's magic
    def __len__(self):
        return len(self.edge_index_list)

    def __getitem__(self, idx):  # we just fetch a single graph
        return (self.node_features_list[idx], self.node_labels_list[idx],
                self.edge_index_list[idx], self.rpst_labels_list[idx])


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
    rpst_label_list = []
    num_nodes_seen = 0
    num_graph_seen = 0
    node_to_graph_map_list = []

    for features_labels_edge_index_rpst_labels_tuple in batch:
        # Just collect these into separate lists
        node_features = features_labels_edge_index_rpst_labels_tuple[0]
        assert len(node_features[0]) == len(node_features[1]) == len(node_features[2])
        node_type_list.extend(node_features[0])
        node_lexical_list.extend(node_features[1])
        node_value_list.extend(node_features[2])
        num_nodes = len(node_features[0])
        graph_label_list.append(features_labels_edge_index_rpst_labels_tuple[1])
        edge_index = features_labels_edge_index_rpst_labels_tuple[2]  # all of the components are in the [0, N] range
        # very important! translate the range of this component
        edge_index_list.append(edge_index + torch.tensor([[num_nodes_seen], [num_nodes_seen], [0]], dtype=torch.long))
        node_to_graph_map_list = node_to_graph_map_list + [num_graph_seen] * num_nodes
        rpst_label_list.append(features_labels_edge_index_rpst_labels_tuple[3])

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
    graph_labels = torch.cat(graph_label_list).unsqueeze(-1)
    edge_index = torch.cat(edge_index_list, 1)
    node_to_graph_map = torch.tensor(node_to_graph_map_list, dtype=torch.long)
    rpst_labels = torch.cat(rpst_label_list, dim=0).unsqueeze(-1)

    return [node_type_list, node_lexical_list, node_value_list], graph_labels, edge_index, node_to_graph_map, rpst_labels


def load_graph_data(training_config, device):
    global MINE_RALATIONS_INDEX_MAP

    dataset_name = training_config['dataset_name']

    if training_config['preprocess']['method'] == PreprocessMethod.RAW_WO_RENAME:
        preprocess_data_path = f"{os.path.join(DATASET_DIR_PATH, f'{dataset_name.name}_{PreprocessMethod.RAW_WO_RENAME.name}')}"
    else:
        preprocess_data_path = f"{os.path.join(DATASET_DIR_PATH, f'{dataset_name.name}_{PreprocessMethod.RAW.name}')}"
    # preprocess for the first time
    preprocess(training_config)

    # Small optimization "trick" since we only need test in the playground.py
    test_only = training_config['run']['test_only']
    splits = ['test'] if test_only else ['train', 'valid', 'test']
    dataloaders = [None, None] if test_only else []
    for split in splits:
        # Collect train/val/test graphs here
        edge_index_list = []
        node_features_list = []
        graph_label_list = []
        rpst_nodes_label_list = []  # 0: for benign nodes, 1: for vulnerable nodes, -1 for not rpst nodes
        print(f"Loading {os.path.join(preprocess_data_path, f'{split}.json')}... ", end="")
        graphs = json_read(path=os.path.join(preprocess_data_path, f"{split}.json"))
        print("Done!")
        for graph_id in tqdm.tqdm(graphs["graph_id_index"], desc=f"Loading {split} graphs"):
            graph = graphs["graphs"][graph_id]

            nodes = list(range(len(graph["node_types"])))

            # shape = (3, E) - where E is the number of edges in the graph
            # Note: leaving the tensors on CPU I'll load them to GPU in the training loop on-the-fly as VRAM
            # is a scarcer resource than CPU's RAM and the whole dataset can't fit during the training.
            # (inNode, outNode, type)
            edge_index = list(map(lambda e: [e[0], e[1], MINE_RALATIONS_INDEX_MAP[e[2]]], filter(lambda e: e[2] in MINE_RALATIONS_INDEX_MAP, graph["edges"])))
            edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1).contiguous()
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

            # shape = (1,), BCEWithLogitsLoss doesn't require long/int64 so saving some memory by using float32
            graph_label_list.append(
                torch.tensor([graph["label"]] if isinstance(graph["label"], int) else graph["label"],
                             dtype=torch.float))

            # shape = (N,)
            rpst_nodes_pos_idx = torch.tensor(graph["rpst_nodes"]["1"], dtype=torch.long)
            rpst_nodes_neg_idx = torch.tensor(graph["rpst_nodes"]["0"], dtype=torch.long)
            rpst_nodes_label = torch.tensor([-1.] * len(nodes), dtype=torch.float)  # (N, )
            rpst_nodes_label[rpst_nodes_pos_idx] = 1.
            rpst_nodes_label[rpst_nodes_neg_idx] = 0.
            rpst_nodes_label_list.append(rpst_nodes_label)
        del graphs

        # Prepare graph data loaders
        data_loader = GraphDataLoader(
            node_features_list, graph_label_list, edge_index_list, rpst_nodes_label_list,
            batch_size=training_config['run']['loader_batch_size'],
            # I implement batch during training, where loss is added, so leave it for now
            shuffle=True if split == "train" else False
        )
        dataloaders.append(data_loader)
    return dataloaders


def load_map(config):
    dataset_path = os.path.join(DATASET_DIR_PATH, f"{config['dataset_name'].name}_{PreprocessMethod.RAW.name}")
    type_map = pickle_read(os.path.join(dataset_path, 'node_type_dict.pkl'))
    lexical_map = pickle_read(os.path.join(dataset_path, 'word_lexical_dict.pkl'))
    word_map = pickle_read(os.path.join(dataset_path, 'word_value_dict.pkl'))
    return type_map, lexical_map, word_map


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

    config = {
        'dataset_name': DatasetName.CroVul,
        'preprocess': {
            'method': PreprocessMethod.RAW,
        },
        'model': {
            'loader_batch_size': 2,
            'add_reverse_edge': True,
            'add_self_loop': True,
        },
        'run': {
            'test_only': True  # small optimization for loading test graphs only, we won't use it here
        },
        'force_refresh': False,
    }

    data_loader_train, data_loader_val, data_loader_test = load_graph_data(config, device)
    # Let's fetch a single batch from the train graph data loader
    node_features, graph_label, edge_index, node_to_graph_map = next(iter(data_loader_test))

    print('*' * 20)
    print("Num of nodes: ", len(node_features[1]))
    print("graph_label.shape: ", graph_label.shape, "\t, graph_label.dtype: ", graph_label.dtype)
    print("edge_index.shape: ", edge_index.shape, "\t, edge_index.dtype: ", edge_index.dtype)
    print("node_to_graph_map.shape: ", node_to_graph_map.shape, "\t, node_to_graph_map.dtype: ", node_to_graph_map.dtype)
