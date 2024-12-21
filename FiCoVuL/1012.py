import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
import os
import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler, SGD
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import shutil
import time
import enum
import git
import re  # regex
import argparse
import random
import math
import numpy as np
import json5
from copy import deepcopy

from dataset.data_preprocess import DATA_DIR_PATH, DatasetName, PreprocessMethod
from dataset.data_loader import DatasetName, PreprocessMethod, load_graph_data, MINE_NUM_RELATIONS, load_map
from models.GATClassification import GATBinaryClassification
from models.layers.MLP import MLP4Layer


def gen_data():
    global folder
    load_model = "data/others/gat_CroVul_crovul_graph_focal_045_best.pth"  # 1012 use this
    # load_model = "data/others/gat_CroVul_crovul_graph_focal_045_2_best.pth"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    last_training_state = torch.load(load_model, map_location=device)
    print("last test: ", last_training_state['perf'])

    type_map, lexical_map, word_map = load_map(last_training_state)

    # Step 2: prepare the model
    gat = GATBinaryClassification(
        type_map=type_map,
        lexical_map=lexical_map,
        word_map=word_map,
        method=last_training_state['preprocess']['method'],
        device=device,
        num_of_gat_layers=last_training_state['model']['num_of_gat_layers'],
        num_heads_per_gat_layer=last_training_state['model']['num_heads_per_gat_layer'],
        num_features_per_gat_layer=last_training_state['model']['num_features_per_gat_layer'],
        add_skip_connection=last_training_state['model']['add_skip_connection'],
        bias=last_training_state['model']['bias'],
        num_of_relations=MINE_NUM_RELATIONS * 2 if last_training_state['model']['add_reverse_edge'] else MINE_NUM_RELATIONS,
        dropout=last_training_state['model']['dropout'],
        log_attention_weights=False,  # no need to store attentions, used only in playground.py for visualizations
        graph_representation_size=last_training_state['model']['graph_representation_size'],
        num_heads_in_graph_representation=last_training_state['model']['num_heads_in_graph_representation'],
        dataset=last_training_state['dataset_name']
    )
    gat.load_state_dict(last_training_state['state_dict'])
    gat.to(device)
    gat.eval()

    last_training_state['run']['loader_batch_size'] = 1
    data_loader_train, data_loader_valid, data_loader_test = load_graph_data(last_training_state, device)

    cnt_false = 0
    features = []
    labels = []
    for split, data_loader in zip(['train', 'valid', 'test'], [data_loader_train, data_loader_valid, data_loader_test]):
        cnt = 0
        bar = tqdm.tqdm(data_loader)
        max_mini_batch = len(bar)
        for mini_batch_idx, (node_features, gt_graph_labels, edge_index, node_to_graph_map, rpst_labels) in enumerate(bar):
            if gt_graph_labels.item() == 0 or node_to_graph_map.numel() == 1:
                continue
            torch.cuda.empty_cache()
            # Push the batch onto GPU - note PPI is to big to load the whole dataset into a normal GPU
            # it takes almost 8 GBs of VRAM to train it on a GPU
            edge_index = edge_index.to(device)
            node_to_graph_map = node_to_graph_map.to(device)
            # We will later put node_features to device as it is a list not a tensor now
            # node_features = node_features.to(device)
            gt_graph_labels = gt_graph_labels.to(device)
            rpst_labels = rpst_labels.to(device)
            
            with torch.no_grad():
                nodes_features, graphs_representation = gat.get_node_graph_repr((node_features, edge_index, node_to_graph_map, None))
                if split == 'test':
                    output = gat.graph_classifier(graphs_representation)
                    if torch.sigmoid(output) < 0.5:
                        cnt_false += 1
                        continue

            indices = torch.nonzero(rpst_labels.squeeze(-1) != -1., as_tuple=False).squeeze(-1).cpu().tolist()
            if len(indices) == 0:
                continue
            filter_nodes_features = nodes_features[indices]
            filter_rpst_labels = rpst_labels[indices, 0]
            
            print(filter_nodes_features.cpu().numpy().shape)
            print(filter_rpst_labels.cpu().numpy().shape)
            exit(0)
            
            features.append(filter_nodes_features.cpu().numpy())
            labels.append(filter_rpst_labels.cpu().numpy())
            
            cnt += 1
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(features.shape)
    print(labels.shape)
    print(cnt_false)

gen_data()