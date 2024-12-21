import os
from pathlib import Path
import glob
import enum
import random
import sys
import regex
import torch
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from ordered_set import OrderedSet
import json
import pickle
import tqdm
from collections import Counter
from itertools import chain
from gensim.models import word2vec, Word2Vec

# We'll be dumping and reading the datum from this directory
DATA_DIR_PATH = os.path.abspath(os.path.join(os.path.join(__file__, '..', '..'), 'data'))
# path of datasets
DATASET_DIR_PATH = os.path.join(DATA_DIR_PATH, 'datasets')

# Supported datasets
class DatasetName(enum.Enum):
    CroVul = 0
    FUNDED = 1
    MINE = 2
    Simple = 999


# Supported preprocessing method
class PreprocessMethod(enum.Enum):
    RAW = 0
    WORD2VEC = 1
    CODE2VEC = 2
    RAW_WO_RENAME = 10


def json_read(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data


def json_dump(obj, path):
    with open(path, 'w') as fp:
        json.dump(obj, fp)


if True:
    # create save dir
    save_dir = os.path.join(DATASET_DIR_PATH, f"{DatasetName.CroVul.name}_{PreprocessMethod.RAW.name}")
    Path(save_dir).mkdir(exist_ok=True)  # 不需要删除旧文件，处理完后会覆盖

    dataset_path = os.path.join(DATASET_DIR_PATH, DatasetName.CroVul.name)
    assert os.path.exists(dataset_path), f"[ERROR] {dataset_path} is invalid."

    print(f"Generating raw dataset of {DatasetName.CroVul.name} ...")

    filenames = glob.glob(r"/Users/huanghongjun/FiCoVuL/preprocess/data/enhanced/CroVul/CVEfixes/*/*.json")
    cache = {}
    for filename in filenames:
        id = filename.split('/')[-1].split(".")[0]
        content = json_read(filename)
        cache[id] = content

    for filename in glob.glob(r"/Users/huanghongjun/FiCoVuL/FiCoVuL/data/datasets/CroVul/*.json"):
        id = '-'.join(filename.split('/')[-1].split('-')[:-1])
        if id in cache.keys():
            content = json_read(filename)
            content["properties"]["lines"] = cache[id]["roi"]
            content["properties"]["class"] = cache[id]["class"]
            json_dump(content, filename)
