import glob
import json
from collections import Counter
from tqdm import tqdm
from sklearn.manifold import TSNE


def json_read(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data


