from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import math
import shutil
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.getcwd(), '..'))
import time
import torch
from torch import nn
import glob
from collections import Counter
from itertools import chain
from pathlib import Path
import tqdm
from FiCoVuL.train import LoopPhase, typist
from FiCoVuL.dataset.data_loader import load_graph_data, load_map, MINE_NUM_RELATIONS
from FiCoVuL.dataset.data_preprocess import DATASET_DIR_PATH, DatasetName, PreprocessMethod, CodeNormalizer, json_read, json_dump
from FiCoVuL.models.GATClassification import GATBinaryClassification


def calculate_recall_at_fpr(predicted_labels, true_labels, fpr_threshold):
    # 计算预测概率
    predicted_probabilities = torch.sigmoid(predicted_labels)

    # 根据概率和阈值获取预测标签
    predicted_thresholded_labels = (predicted_probabilities >= fpr_threshold).int()

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_thresholded_labels)
    print(cm)
    # 提取混淆矩阵中的 TP 和 FN
    tp = cm[1, 1]
    fn = cm[1, 0]

    # 计算召回率
    recall = tp / (tp + fn)

    return recall


def get_main_loop(config, gat, loss_func, optimizer, patience_period, time_start):

    device = next(gat.parameters()).device  # fetch the device info from the model instead of passing it as a param

    def main_loop(phase, data_loader, epoch=0):
        global BEST_VAL_MICRO_F1, BEST_VAL_LOSS, PATIENCE_CNT, writer

        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        if phase == LoopPhase.TRAIN:
            gat.train()
        else:
            gat.eval()

        # Iterate over batches of graph data (2 graphs per batch was used in the original paper for the PPI dataset)
        # We merge them into a single graph with 2 connected components, that's the main idea. After that
        # the implementation #3 is agnostic to the fact that those are multiple and not a single graph!
        bar = tqdm.tqdm(data_loader, desc=phase.name)
        max_mini_batch = len(bar)
        total_loss_list = []
        total_scores = []
        total_preds = []
        total_gts = []
        last_update = 0  # for debug
        for mini_batch_idx, (node_features, gt_graph_labels, edge_index, node_to_graph_map, rpst_labels) in enumerate(bar):
            torch.cuda.empty_cache()
            # Push the batch onto GPU - note PPI is to big to load the whole dataset into a normal GPU
            # it takes almost 8 GBs of VRAM to train it on a GPU
            edge_index = edge_index.to(device)
            node_to_graph_map = node_to_graph_map.to(device)
            # We will later put node_features to device as it is a list not a tensor now
            # node_features = node_features.to(device)
            gt_graph_labels = gt_graph_labels.to(device)
            rpst_labels = rpst_labels.to(device)

            # I pack data into tuples because GAT uses nn.Sequential which expects this format
            graph_data = (node_features, edge_index, node_to_graph_map, rpst_labels)

            graph_unnormalized_scores = gat(graph_data)
            loss = loss_func(graph_unnormalized_scores, gt_graph_labels)
            total_loss_list.append(loss.item())

            mini_batch_size = int(config['run']['batch_size']/config['run']['loader_batch_size'])
            is_show_this_time = (mini_batch_idx+1) % mini_batch_size == 0 or (mini_batch_idx+1) == max_mini_batch
            if phase == LoopPhase.TRAIN:
                if mini_batch_idx % mini_batch_size == 0:
                    # clean the trainable weights gradients in the computational graph (.grad fields)
                    optimizer.zero_grad()
                loss.backward()  # compute the gradients for every trainable weight in the computational graph
                if is_show_this_time:
                    optimizer.step()  # apply the gradients to weights
                    last_update = mini_batch_idx

            # Calculate the main metric - micro F1, check out this link for what micro-F1 exactly is:
            # https://www.kaggle.com/enforcer007/what-is-micro-averaged-f1-score

            # Convert unnormalized scores into predictions. Explanation:
            # If the unnormalized score is the largest that means that its sigmoid is the largest too because it's just
            # binary classification now
            total_scores.extend(graph_unnormalized_scores.squeeze(-1).cpu().tolist())
            label_theshold = -math.log(1/(config['run']['classfication_threshold']-1e-9)-1)
            total_preds.extend((graph_unnormalized_scores.squeeze(-1) > label_theshold).int().cpu().tolist())
            total_gts.extend(gt_graph_labels.squeeze(-1).int().cpu().tolist())
            if is_show_this_time:
                # 一个 batch_size 的 loss 和
                batch_mean_loss = sum(total_loss_list[mini_batch_idx+1-mini_batch_size:]) / mini_batch_size
                batch_micro_f1 = f1_score(
                    total_gts[(mini_batch_idx + 1) * config['run']['loader_batch_size'] - config['run']['batch_size']:],
                    total_preds[(mini_batch_idx + 1) * config['run']['loader_batch_size'] - config['run']['batch_size']:],
                    average='binary')
                bar.set_postfix({"last_update": last_update, "loss": batch_mean_loss, f"{phase.name}_f1": batch_micro_f1})

                # Logging: Log metrics
                if config['run']['enable_tensorboard']:
                    steps_per_epoch = math.ceil(len(data_loader) / mini_batch_size)
                    cur_step = math.ceil((mini_batch_idx + 1) / mini_batch_size)
                    global_step = steps_per_epoch * epoch + cur_step
                    if phase == LoopPhase.TRAIN:
                        writer.add_scalar('training_loss', batch_mean_loss, global_step)
                        writer.add_scalar('training_micro_f1', batch_micro_f1, global_step)
                    elif phase == LoopPhase.VAL:
                        writer.add_scalar('val_loss', batch_mean_loss, global_step)
                        writer.add_scalar('val_micro_f1', batch_micro_f1, global_step)

        # calculate average metrics
        mean_loss = np.array(total_loss_list).mean()
        precision = precision_score(total_gts, total_preds, average='binary')
        recall = recall_score(total_gts, total_preds, average='binary')
        accuracy = accuracy_score(total_gts, total_preds)
        mean_micro_f1 = f1_score(total_gts, total_preds, average='binary')
        perf = {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "micro_f1": mean_micro_f1,
        }

        if phase == LoopPhase.TRAIN:
            # Log to console
            if config['run']['console_log_freq'] is not None and (epoch + 1) % config['run']['console_log_freq'] == 0:
                print(f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] |'
                      f' epoch={epoch} | train mean-loss={mean_loss:.4f} | train precision={precision:.3f} |'
                      f' train recall={recall:.3f} | train accuracy={accuracy:.3f} | train mean-micro-F1={mean_micro_f1:.3f}.')
            # Save model checkpoint
            if config['run']['checkpoint_freq'] is not None and (epoch + 1) % config['run']['checkpoint_freq'] == 0:
                ckpt_model_name = f'gat_{config["dataset_name"].name}_ckpt_{config["task_name"]}_epoch_{epoch + 1}.pth'
                torch.save(typist(config, perf, gat.state_dict()), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))
        elif phase == LoopPhase.VAL:
            # Log to console
            if config['run']['console_log_freq'] is not None and (epoch + 1) % config['run']['console_log_freq'] == 0:
                print(f'GAT validation: time elapsed= {(time.time() - time_start):.2f} [s] |'
                      f' epoch={epoch} | val mean-loss={mean_loss:.4f} | val precision={precision:.3f} |'
                      f' val recall={recall:.3f}  | val accuracy={accuracy:.3f} | val mean-micro-F1={mean_micro_f1:.3f}')

            # The "patience" logic - should we break out from the training loop? If either validation micro-F1
            # keeps going up or the val loss keeps going down we won't stop
            if mean_micro_f1 >= BEST_VAL_MICRO_F1 or (mean_micro_f1 == BEST_VAL_MICRO_F1 and mean_loss < BEST_VAL_LOSS):
                # save the best model for test
                BEST_VAL_LOSS = mean_loss  # and the minimal loss
                ckpt_model_name = f'gat_{config["dataset_name"].name}_{config["task_name"]}_best.pth'
                perf = {
                    "precision": precision,
                    "recall": recall,
                    "accuracy": accuracy,
                    "micro_f1": mean_micro_f1,
                }
                torch.save(typist(config, perf, gat.state_dict()), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))
                BEST_VAL_MICRO_F1 = max(mean_micro_f1, BEST_VAL_MICRO_F1)  # keep track of the best validation micro_f1 so far
                PATIENCE_CNT = 0  # reset the counter every time we encounter new best micro_f1
                print(f"Reset PATIENCE.")
            else:
                PATIENCE_CNT += 1  # otherwise keep counting
                print(f"{patience_period - PATIENCE_CNT} PATIENCE remains.")

            if PATIENCE_CNT >= patience_period:
                raise Exception('Stopping the training, the universe has no more patience for this training.')
        else:
            for fpr_threshold in [0.05, 0.1]:
                recall = calculate_recall_at_fpr(torch.tensor(total_scores), torch.tensor(total_gts), fpr_threshold)
                print(f"Recall at {fpr_threshold} FPR: {recall}")
            # in the case of test phase we just report back the test micro_f1
            return precision, recall, accuracy, mean_micro_f1

    return main_loop  # return the decorated function


if __name__ == '__main__':
    dataset = DatasetName.WILDONLY  # 19:1
    orig_dataset = DatasetName.CroVulNew
    model_path = "gat_CroVulNew_crovulnew_graph_focal_000000.pth"
    model_path = os.path.join("data/models/binaries", model_path)

    save_dir = os.path.join(DATASET_DIR_PATH, f"{dataset.name}_{PreprocessMethod.RAW.name}")
    Path(save_dir).mkdir(exist_ok=True)  # 不需要删除旧文件，处理完后会覆盖

    dataset_path = os.path.join(DATASET_DIR_PATH, dataset.name)
    orig_dataset_path = os.path.join(DATASET_DIR_PATH, orig_dataset.name)
    # assert os.path.exists(dataset_path), f"[ERROR] {dataset_path} is invalid."
    # origin_data_path_pattern = f"{dataset_path}_ORIG/*.c"
    # if not os.path.exists(os.path.join(save_dir, 'test.json')):
    #     print(f"Generating raw dataset of {dataset.name} ...")
    #     filenames = list(glob.glob(rf"{dataset_path}/*.json"))
    #     # JOERN1
    #     result2uid = dict(map(lambda path: (path, '-'.join(path.split('/')[-1].split('.')[0].split('-')[:-1])), filenames))
    #     # JOERN2
    #     # result2uid = dict(map(lambda path: (path, path.split('/')[-1].split('.')[0]), filenames))
    #     uid2enhanced = dict(map(lambda path: (path.split('/')[-1].split('.')[0], path),
    #                             glob.iglob(origin_data_path_pattern, recursive=True)))
    #     processed2orig = dict(map(lambda item: (item[0], uid2enhanced[item[1]]), result2uid.items()))
    #     del result2uid, uid2enhanced
    #
    #     with open(f'{orig_dataset_path}_RAW/most_common_function_names.txt', 'r') as f:
    #         most_commons = f.readlines()
    #     node_type_map, lexical_map, value_map = load_map({'dataset_name': orig_dataset})
    #     lexical_counter = Counter()
    #     value_counter = Counter()
    #     for filename in tqdm.tqdm(glob.iglob(origin_data_path_pattern, recursive=True),
    #                               desc="Generating word dict"):
    #         with open(filename, 'r') as fp:
    #             code = fp.read()
    #         cn = CodeNormalizer('cpp', code, most_commons)
    #         _lexical, _value = cn.get_all_tokens()
    #         lexical_counter.update(_lexical)
    #         value_counter.update(_value)
    #     value_counter.update(["RET"])
    #
    #     print("Generating dataset...")
    #     cnt = 0
    #     graph = {
    #         "n_graphs": len(filenames),
    #         "graph_id_index": [],
    #         "graphs": {}
    #     }
    #     for filename in tqdm.tqdm(filenames):
    #         with open(processed2orig[filename], 'r') as fp:
    #             orig_code = fp.read()
    #         cn = CodeNormalizer('cpp', orig_code, most_commons)
    #         split_lines = orig_code.split('\n')
    #
    #         data = json_read(filename)
    #         graph_index_id = filename.split('/')[-1]
    #
    #         # 先生成str版本的数据集
    #         sorted_data = sorted(data["nodes"].items(), key=lambda item: int(item[0]))
    #         # node type
    #         sorted_type = list(map(lambda item: item[1]["type"], sorted_data))
    #         _type = list(map(node_type_map.index, sorted_type))
    #
    #         # node lexical & value
    #         def extract_lexical_and_value(item):  # 传入 sorted_data: (id, data)
    #             if len(item[1]["code"]) == 0:
    #                 return [], []
    #             if item[1]["type"] == "METHOD_RETURN" and item[1]["code"] == 'RET':
    #                 return ["Token.Keyword"], ["RET"]
    #             # position: line 从 1 开始，column 从 0 开始
    #             _pos = (int(item[1]["line"]), int(item[1]["column"]))
    #             if _pos == (-1, -1):
    #                 if item[1]["type"] == "METHOD":
    #                     return ["Token.Name.Function"], [
    #                         item[1]["code"] if item[1]["code"] not in cn.renamer.keys() else cn.renamer[
    #                             item[1]["code"]]]
    #                 elif item[1]["type"] == "LOCAL":
    #                     return ["Token.Name"], [
    #                         item[1]["code"] if item[1]["code"] not in cn.renamer.keys() else cn.renamer[
    #                             item[1]["code"]]]
    #                 else:
    #                     print(filename)
    #                     print(item[1])
    #                     raise Exception()
    #             sstart = len('\n'.join(split_lines[:_pos[0] - 1])) + 1 + _pos[1] if _pos[0] != 1 else len(
    #                 '\n'.join(split_lines[:_pos[0] - 1])) + _pos[1]
    #             ssend = sstart + len(item[1]["code"])
    #             _lexicals, _values = cn.localize_tokens(sstart, ssend)
    #             return _lexicals, _values
    #
    #
    #         _lexical, _value = zip(*map(extract_lexical_and_value, sorted_data))
    #         # 去除一些错误的 sample
    #         if False in map(lambda _v: _v in value_counter.keys(), chain(*_value)):
    #             cnt += 1
    #             continue
    #         converted_lexical = list(map(lambda lexicals: list(map(
    #             lambda _l: lexical_map.index(_l) if _l in lexical_map else lexical_map.index('<UNK>'), lexicals)),
    #                                      _lexical))
    #         converted_value = list(map(lambda values: list(map(
    #             lambda _v: value_map.index(_v) if _v in value_map else value_map.index('<UNK>'), values)), _value))
    #         # rpst_nodes
    #         _lines = list(map(lambda item: int(item[1]["line"]) - 1, sorted_data))  # joern 从 1 开始
    #         _lines_code_length = list(map(lambda item: len(item[1]["code"]), sorted_data))
    #         max_values = {}
    #         result_indexes = []
    #         for i, (v1, v2) in enumerate(zip(_lines, _lines_code_length)):
    #             # TODO: 第二个脚本中是 0
    #             if v1 < 0:  # RET 不管了，应该不会有小于三个字符的函数名吧
    #                 continue
    #             if v1 not in max_values:
    #                 max_values[v1] = (v2, i)
    #             else:
    #                 if v2 > max_values[v1][0]:
    #                     max_values[v1] = (v2, i)
    #         for v1 in max_values:
    #             result_indexes.append(max_values[v1][1])
    #
    #         interested_lines = data["properties"]["lines"]
    #         interested_nodes_indexes = list(
    #             map(lambda item: item[0], filter(lambda _l: _l[1] in interested_lines, enumerate(_lines))))
    #
    #         # links
    #         _edges = data["edges"]
    #
    #         graph["graph_id_index"].append(graph_index_id)
    #         graph["graphs"][graph_index_id] = {
    #             "node_types": _type,
    #             "node_lexical_features": converted_lexical,
    #             "node_value_features": converted_value,
    #             "edges": _edges,
    #             "rpst_nodes": {
    #                 0: list(set(result_indexes) - set(interested_nodes_indexes)),
    #                 1: list(set(result_indexes) & set(interested_nodes_indexes))
    #             },
    #             "class": data["properties"]["class"],
    #             "label": data["properties"]["label"],
    #         }
    #
    #     json_dump(obj=graph, path=f"{save_dir}/test.json")
    #     print(f"Skip {cnt} samples due to failure.")
    #     shutil.copyfile(f"{orig_dataset_path}_RAW/most_common_function_names.txt", os.path.join(save_dir, "most_common_function_names.txt"))
    #     shutil.copyfile(f"{orig_dataset_path}_RAW/node_type_dict.pkl", os.path.join(save_dir, "node_type_dict.pkl"))
    #     shutil.copyfile(f"{orig_dataset_path}_RAW/word_lexical_dict.pkl", os.path.join(save_dir, "word_lexical_dict.pkl"))
    #     shutil.copyfile(f"{orig_dataset_path}_RAW/word_value_dict.pkl", os.path.join(save_dir, "word_value_dict.pkl"))
    #     with open(os.path.join(save_dir, "train.json"), 'w') as f:
    #         f.write('{"n_graphs": 0, "graph_id_index": [], "graphs": {}}')
    #     with open(os.path.join(save_dir, "valid.json"), 'w') as f:
    #         f.write('{"n_graphs": 0, "graph_id_index": [], "graphs": {}}')
    #     print(f"Raw dataset of {dataset.name} has been generated!")
    #     sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() and not bin['run']['force_cpu'] else "cpu")
    # Step 2: prepare the model
    last_training_state = torch.load(model_path, map_location=device)
    print("last test: ", last_training_state['perf'])

    last_training_state['task_name'] = f"test_{dataset.name}"
    last_training_state['dataset_name'] = dataset
    last_training_state['run']['test_only'] = True
    last_training_state['run']['load_model'] = model_path
    last_training_state['run']['classfication_threshold'] = 0.5

    _, _, data_loader_test = load_graph_data(last_training_state, device=device)
    # # Let's fetch a single batch from the train graph data loader
    # node_features, graph_label, edge_index, node_to_graph_map, rpst_nodes_label = next(iter(data_loader_test))
    # print('*' * 20)
    # print("Num of nodes: ", len(node_features[1]))
    # print("graph_label.shape: ", graph_label.shape, "\t, graph_label.dtype: ", graph_label.dtype)
    # print("edge_index.shape: ", edge_index.shape, "\t, edge_index.dtype: ", edge_index.dtype)
    # print("node_to_graph_map.shape: ", node_to_graph_map.shape, "\t, node_to_graph_map.dtype: ",
    #       node_to_graph_map.dtype)
    # print("rpst_nodes_label.shape: ", rpst_nodes_label.shape, "\t, rpst_nodes_label.dtype: ", rpst_nodes_label.dtype)
    # exit(0)

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

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_main_loop(
        last_training_state,
        gat,
        loss_fn,
        None,
        None,
        time.time())
    precision, recall, accuracy, micro_f1 = main_loop(phase=LoopPhase.TEST, data_loader=data_loader_test)

    print('*' * 50)
    print(
        f'Test precision = {precision:.3f}, recall = {recall:.3f}, accuracy = {accuracy:.3f}, micro-F1 = {micro_f1:.3f}')


"""WILD
Num of nodes:  29397
graph_label.shape:  torch.Size([71, 1])         , graph_label.dtype:  torch.float32
edge_index.shape:  torch.Size([3, 315912])      , edge_index.dtype:  torch.int64
node_to_graph_map.shape:  torch.Size([29397])   , node_to_graph_map.dtype:  torch.int64
rpst_nodes_label.shape:  torch.Size([29397, 1])         , rpst_nodes_label.dtype:  torch.float32

Generating raw dataset of WILD ...
Generating word dict: 106it [00:01, 98.29it/s]
Generating dataset...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 72/72 [00:17<00:00,  4.19it/s]
/Users/huanghongjun/FiCoVuL/FiCoVuL/data/datasets/WILD_RAW/test.json saved!
Skip 1 samples due to failure.
Raw dataset of WILD has been generated!
dict_keys(['task_name', 'dataset_name', 'preprocess', 'model', 'run', 'force_refresh', 'commit_hash', 'perf', 'state_dict'])
Loading /Users/huanghongjun/FiCoVuL/FiCoVuL/data/datasets/WILD_RAW/test.json...
last test:  {'precision': 0.7164276401564538, 'recall': 0.9825659365221279, 'accuracy': 0.792749658002736, 'micro_f1': 0.8286522148916118}
TEST: 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:24<00:00, 24.68s/it, last_update=0, loss=0.548, TEST_f1=0.956]
**************************************************
Test precision = 0.985, recall = 0.929, accuracy = 0.915, micro-F1 = 0.956
"""
