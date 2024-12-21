# -*- coding: UTF-8 -*-
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

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
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

from dataset.data_preprocess import DATA_DIR_PATH
from dataset.data_loader import DatasetName, PreprocessMethod, load_graph_data, MINE_NUM_RELATIONS, load_map
from models.GATClassification import GATBinaryClassification


# Global vars used for early stopping. After some number of epochs (as defined by the patience_period var) without
# any improvement on the validation dataset (measured via micro-F1 metric), we'll break out from the training loop.
BEST_VAL_MICRO_F1 = 0
BEST_VAL_LOSS = 0
PATIENCE_CNT = 0

BINARIES_PATH = os.path.join(os.getcwd(), 'data', 'models', 'binaries')
CHECKPOINTS_PATH = os.path.join(os.getcwd(), 'data', 'models', 'checkpoints')

# Make sure these exist as the rest of the code assumes it
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)


# 3 different model training/eval phases used in train.py
class LoopPhase(enum.Enum):
    TRAIN = 0,
    VAL = 1,
    TEST = 2


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLoss()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def load_config():
    def json5_read(path):
        with open(path, 'r') as fp:
            data = json5.load(fp)
        return data

    config = json5_read(os.path.join(os.getcwd(), 'configs/config.json5'))
    return config


def typist(config, perf, state_dict):
    record = config.copy()
    record["commit_hash"] = git.Repo(search_parent_directories=True).head.object.hexsha
    record['perf'] = perf
    record['state_dict'] = deepcopy(state_dict)

    return record


def print_model_metadata(training_state):
    header = f'\n{"*" * 5} Model training metadata: {"*" * 5}'
    print(header)

    for key, value in training_state.items():
        print(f'{key}: {str(value)}')
    print(f'{"*" * len(header)}\n')


# This one makes sure we don't overwrite the valuable model binaries (feel free to ignore - not crucial to GAT method)
def get_available_binary_name(name: str):
    prefix = f'gat_{name}'

    def valid_binary_name(binary_name):
        # First time you see raw f-string? Don't worry the only trick is to double the brackets.
        pattern = re.compile(rf'{prefix}_[0-9]{{6}}\.pth')
        return re.fullmatch(pattern, binary_name) is not None

    # Just list the existing binaries so that we don't overwrite them but write to a new one
    valid_binary_names = list(filter(valid_binary_name, os.listdir(BINARIES_PATH)))
    if len(valid_binary_names) > 0:
        last_binary_name = sorted(valid_binary_names)[-1]
        new_suffix = int(last_binary_name.split('.')[0][-6:]) + 1  # increment by 1
        return f'{prefix}_{str(new_suffix).zfill(6)}.pth'
    else:
        return f'{prefix}_000000.pth'


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def standardize_config(config: dict):
    """
    dict 会原地修改
    """
    if config['run']['manual_seed'] is None:
        config['run']['manual_seed'] = random.randint(0, 2147483647)
    assert config['dataset_name'] in [el.name for el in DatasetName]
    config['dataset_name'] = eval(f"DatasetName.{config['dataset_name']}")
    assert config['preprocess']['method'] in [el.name for el in PreprocessMethod]
    config['preprocess']['method'] = eval(f"PreprocessMethod.{config['preprocess']['method']}")
    assert config['run']['batch_size'] % config['run']['loader_batch_size'] == 0, \
        "loader_batch_size represents dataloader batch_size, and batch_size represents how many instances before " \
        "we step forward, so batch_size % loader_batch_size needs to be 0"


def train_gat(config):
    """
    Very similar to Cora's training script. The main differences are:
    1. Using dataloaders since we're dealing with an inductive setting - multiple graphs per batch
    2. Doing multi-class classification (BCEWithLogitsLoss) and reporting micro-F1 instead of accuracy
    3. Model architecture and hyperparams are a bit different (as reported in the GAT paper)
    """
    global BEST_VAL_MICRO_F1, BEST_VAL_LOSS

    # Checking whether you have a strong GPU.
    # I've added the option to force the use of CPU even though you have a GPU on your system (but it's too weak).
    device = torch.device("cuda" if torch.cuda.is_available() and not config['run']['force_cpu'] else "cpu")

    # Step 1: prepare the data loaders
    data_loader_train, data_loader_val, data_loader_test = load_graph_data(config, device)

    # Step 2: prepare the model
    type_map, lexical_map, word_map = load_map(config)
    gat = GATBinaryClassification(
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
    if config['run']['load_model'] is not None:
        print(f"Trying to load parameters from saved model at {config['load_model']}...")
        data = torch.load(config['run']['load_model'], device)
        gat.load_state_dict(data['state_dict'])
        print(f"Finish loading params. Performance: ")
        print(data['perf'])
    gat = gat.to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(100/107)).to(device)
    # loss_fn = FocalLoss(alpha=0.75).to(device)
    optimizer = Adam(gat.parameters(), lr=config['run']['lr'], weight_decay=config['run']['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, config['run']['num_of_epochs'], eta_min=0, last_epoch=-1)

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_main_loop(
        config,
        gat,
        loss_fn,
        optimizer,
        config['run']['patience_period'],
        time.time())

    BEST_VAL_MICRO_F1, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]  # reset vars used for early stopping

    # Step 4: Start the training procedure
    bar = tqdm.trange(config['run']['num_of_epochs'], desc="epoch")
    for epoch in bar:
        print('=' * 15 + f" * EPOCH \t{epoch:4}\t" + '=' * 15)
        # Training loop
        main_loop(phase=LoopPhase.TRAIN, data_loader=data_loader_train, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            try:
                main_loop(phase=LoopPhase.VAL, data_loader=data_loader_val, epoch=epoch)
            except Exception as e:  # "patience has run out" exception :O
                print(str(e))
                break  # break out from the training loop
            precision, recall, accuracy, micro_f1 = main_loop(phase=LoopPhase.TEST, data_loader=data_loader_test)

            print('*' * 50)
            print(
                f'EPOCH {epoch} TEST precision = {precision:.3f}, recall = {recall:.3f}, accuracy = {accuracy:.3f}, micro-F1 = {micro_f1:.3f}')
            print('*' * 50)

        # Learning rate decay
        scheduler.step()
        print(f"EPOCH {epoch} learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print('=' * 50)

    # Step 5: Potentially test your model
    # Don't overfit to the test dataset - only when you've fine-tuned your model on the validation dataset should you
    # report your final loss and micro-F1 on the test dataset. Friends don't let friends overfit to the test data. <3
    gat.load_state_dict(torch.load(
            os.path.join(CHECKPOINTS_PATH, f'gat_{config["dataset_name"].name}_{config["task_name"]}_best.pth'))['state_dict'], device)
    precision, recall, accuracy, micro_f1 = main_loop(phase=LoopPhase.TEST, data_loader=data_loader_test)

    print('*' * 50)
    print(
        f'FINAL Test precision = {precision:.3f}, recall = {recall:.3f}, accuracy = {accuracy:.3f}, micro-F1 = {micro_f1:.3f}')
    print('*' * 50)

    # Save the latest GAT in the binaries directory
    print("""model saved at """
          f"""{os.path.join(BINARIES_PATH, get_available_binary_name(config['dataset_name'].name+'_'+config['task_name']))}""")
    perf = {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "micro_f1": micro_f1,
    }
    torch.save(typist(config, perf, gat.state_dict()),
               os.path.join(BINARIES_PATH, get_available_binary_name(config['dataset_name'].name+'_'+config["task_name"])))


def test_gat(config):
    # if test only
    assert config['run']['load_model'] is not None, "model should be specified for test"

    # Checking whether you have a strong GPU.
    # I've added the option to force the use of CPU even though you have a GPU on your system (but it's too weak).
    device = torch.device("cuda" if torch.cuda.is_available() and not config['run']['force_cpu'] else "cpu")

    # Step 1: prepare the data loaders
    _, _, data_loader_test = load_graph_data(config, device)

    # Step 2: prepare the model
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
        num_of_relations=MINE_NUM_RELATIONS * 2 if last_training_state['model']['add_reverse_edge'] else MINE_NUM_RELATIONS,
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
    print(f'Test precision = {precision:.3f}, recall = {recall:.3f}, accuracy = {accuracy:.3f}, micro-F1 = {micro_f1:.3f}')


# Simple decorator function so that I don't have to pass arguments that don't change from epoch to epoch
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
            # in the case of test phase we just report back the test micro_f1
            return precision, recall, accuracy, mean_micro_f1

    return main_loop  # return the decorated function


if __name__ == "__main__":
    # Train the graph attention network (GAT)
    parser = argparse.ArgumentParser()
    # Task name
    parser.add_argument("--task_name", type=str, help="task name to distinguish train file",
                        default='default_task')
    args = parser.parse_args()

    config = load_config()
    config['task_name'] = args.task_name
    standardize_config(config)
    setup_seed(config['run']['manual_seed'])

    # (tensorboard) writer will output to ./runs/ directory by default
    tensorboard_log_path = os.path.join(DATA_DIR_PATH, f"runs/{config['task_name']}")
    if os.path.exists(tensorboard_log_path):
        shutil.rmtree(tensorboard_log_path)
    writer = SummaryWriter(tensorboard_log_path)

    if config['run']['enable_tensorboard']:
        print(f"***Logging output to {writer.log_dir} ***")
    print(f"======Training config======\n{config}\n===========================")

    if not config['run']['test_only']:
        train_gat(config)
    else:
        test_gat(config)
