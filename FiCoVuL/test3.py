import math
import os
import torch
import tqdm
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import numpy as np

from FiCoVuL.dataset.data_loader import load_graph_data, load_map, MINE_NUM_RELATIONS
from FiCoVuL.models.GATClassification import GATBinaryClassification
from train3 import load_config

if __name__ == '__main__':
    config = load_config()
    config['run']['loader_batch_size'] = 1
    device = torch.device("cuda" if torch.cuda.is_available() and not config['run']['force_cpu'] else "cpu")

    # Initialize variables for tracking results
    total_graph_gts = []
    total_graph_preds = []
    total_node_preds = []
    total_node_gts = []

    _, _, data_loader = load_graph_data(config, device)
    last_training_state = torch.load(config['run']['load_model'], device)
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
        num_of_relations=MINE_NUM_RELATIONS * 2 if last_training_state['model'][
            'add_reverse_edge'] else MINE_NUM_RELATIONS,
        dropout=last_training_state['model']['dropout'],
        log_attention_weights=False,  # no need to store attentions, used only in playground.py for visualizations
        graph_representation_size=last_training_state['model']['graph_representation_size'],
        num_heads_in_graph_representation=last_training_state['model']['num_heads_in_graph_representation'],
        dataset=last_training_state['dataset_name']
    )
    gat.load_state_dict(last_training_state['state_dict'])
    gat = gat.to(device)
    gat.eval()
    bar = tqdm.tqdm(data_loader, desc=f"TEST-node")
    max_mini_batch = len(bar)
    for mini_batch_idx, (node_features, gt_graph_labels, edge_index, node_to_graph_map, rpst_labels) in enumerate(bar):
        torch.cuda.empty_cache()
        edge_index = edge_index.to(device)
        node_to_graph_map = node_to_graph_map.to(device)
        gt_graph_labels = gt_graph_labels.to(device)
        label = gt_graph_labels.item()
        rpst_labels = rpst_labels.to(device)

        graph_data = (node_features, edge_index, node_to_graph_map)

        graph_unnormalized_scores, node_unnormalized_scores = gat(graph_data)

        label_theshold = -math.log(1 / config['run']['classfication_threshold'] - 1)
        graph_pred = (graph_unnormalized_scores.squeeze() > label_theshold).int().cpu()
        total_graph_gts.extend(gt_graph_labels.squeeze(-1).int().cpu().tolist())
        total_graph_preds.extend(graph_pred.tolist())
        
        filter_graph_indices = torch.nonzero(gt_graph_labels.squeeze() == 1., as_tuple=False).squeeze()
        node_indices = torch.isin(node_to_graph_map, filter_graph_indices)
        node_indices = set(torch.nonzero(node_indices, as_tuple=False).squeeze().cpu().tolist())
        rpst_indices = set(torch.nonzero(rpst_labels.squeeze() != -1., as_tuple=False).squeeze().cpu().tolist())
        indices = list(node_indices & rpst_indices)
        if len(indices) > 0 and gt_graph_labels.item() == 1:
            filter_node_unnormalized_scores = node_unnormalized_scores[indices, 0].unsqueeze(-1)
            filter_rpst_labels = rpst_labels[indices, 0].unsqueeze(-1)
            node_pred = (filter_node_unnormalized_scores.squeeze(-1) > label_theshold).int().cpu()
            total_node_preds.extend(node_pred.tolist())
            total_node_gts.extend(filter_rpst_labels.squeeze(-1).int().cpu().tolist())

    # Calculate evaluation metrics
    if len(total_graph_gts) > 0:
        Gprecision = precision_score(total_graph_gts, total_graph_preds, average='binary')
        Grecall = recall_score(total_graph_gts, total_graph_preds, average='binary')
        Gaccuracy = accuracy_score(total_graph_gts, total_graph_preds)
        Gf1 = f1_score(total_graph_gts, total_graph_preds, average='binary')
    else:
        Gprecision = Grecall = Gaccuracy = Gf1 = 0.0

    if len(total_node_gts) > 0:
        Nprecision = precision_score(total_node_gts, total_node_preds, average='binary')
        Nrecall = recall_score(total_node_gts, total_node_preds, average='binary')
        Naccuracy = accuracy_score(total_node_gts, total_node_preds)
        Nf1 = f1_score(total_node_gts, total_node_preds, average='binary')
    else:
        Nprecision = Nrecall = Naccuracy = Nf1 = 0.0

    # Calculate weighted metrics (same as train3.py)
    alpha = 0.7
    precision = alpha * Gprecision + (1 - alpha) * Nprecision
    recall = alpha * Grecall + (1 - alpha) * Nrecall
    accuracy = alpha * Gaccuracy + (1 - alpha) * Naccuracy
    micro_f1 = alpha * Gf1 + (1 - alpha) * Nf1

    # Create performance dictionary
    perf = {
        "Gprecision": Gprecision,
        "Grecall": Grecall,
        "Gaccuracy": Gaccuracy,
        "Gf1": Gf1,
        "Nprecision": Nprecision,
        "Nrecall": Nrecall,
        "Naccuracy": Naccuracy,
        "Nf1": Nf1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "micro_f1": micro_f1,
        "alpha": alpha,
    }

    # Print results
    print('*' * 50)
    print(f'Test Results:')
    print(f'Graph-level - Precision: {Gprecision:.3f}, Recall: {Grecall:.3f}, Accuracy: {Gaccuracy:.3f}, F1: {Gf1:.3f}')
    print(f'Node-level   - Precision: {Nprecision:.3f}, Recall: {Nrecall:.3f}, Accuracy: {Naccuracy:.3f}, F1: {Nf1:.3f}')
    print(f'Weighted     - Precision: {precision:.3f}, Recall: {recall:.3f}, Accuracy: {accuracy:.3f}, F1: {micro_f1:.3f}')
    print('*' * 50)

    # Create results directory
    results_dir = os.path.join(os.getcwd(), 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save results to JSON file
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(config['run']['load_model']).replace('.pth', '')
    result_filename = f"test_results_{model_name}_{timestamp}.json"
    result_path = os.path.join(results_dir, result_filename)

    with open(result_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'dataset_name': last_training_state['dataset_name'].name,
            'timestamp': timestamp,
            'perf': perf,
            'config': config
        }, f, indent=2)


