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

# FUNDED 中发布的数据集格式
FUNDED_FORMAT = r"^(?P<edge>(?:\(\d+,\d+,\d+\)\n)+)-{35}\n(?P<feature>(?:\(.*\n)+)\^{35}\n(?P<label>\d+)$"


# Supported datasets
class DatasetName(enum.Enum):
    CroVul = 0
    FUNDED = 1
    MINE = 2
    CroVul2 = 3
    CroVulNew = 4
    WILD0 = 9
    WILD = 10
    WILD2 = 11
    WILD3 = 12
    SIMPLE = 999


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
    print(f"{path} saved!")


def pickle_read(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data


def pickle_dump(obj, path):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp)
    print(f"{path} saved!")


class CodeSpliter:
    class Language(enum.Enum):
        """
        Currently supported language
        """
        C = 0
        CPP = 1

    def __init__(self, lan: str):
        """
        确定语言，初始化词法分析器
        """
        assert lan.lower() in [el.name.lower() for el in CodeSpliter.Language]  # 目前只支持 C, CPP
        self.language = lan
        self.lexer = get_lexer_by_name(lan)  # 初始化词法解析器

    @staticmethod
    def _remove_comment(tks):
        return list(filter(lambda tk: tk[0] not in Token.Comment, tks))

    def get_tokens(self, code):
        """
        返回格式 [(Token.Keyword, 'if'), ...]
        跳过了所有空白字符
        """
        tokens = self.lexer.get_tokens(code)
        tokens = list(filter(lambda item: not item[1].isspace(), tokens))  # 惰性计算 iterator，只能调用一次
        return tokens

    def __call__(self, code: str):
        """
        返回格式 ["Token.Keyword", ...], ["if", ...]
        注意 tokens 都是字符串
        """
        if code == '' or code.isspace():
            return [], []
        tokens = self.get_tokens(code)
        tokens, values = [list(t) for t in zip(*tokens)]
        tokens = [str(token) for token in tokens]
        return tokens, values


def collect_function_names(dir_pattern, lan, thres: int = 500):
    """
    统计某一数据集所有代码中的函数名出现的次数，返回前 thres 个名称
    Args:
        dir_pattern: dataset 的目录
        lan: dataset 的语言
        thres: 返回几个名称

    Returns:
        List[str] with length thres
    """
    cs = CodeSpliter(lan)
    names_counter = Counter()
    tbar = tqdm.tqdm(glob.iglob(dir_pattern, recursive=True), desc="Counting function names")
    for filename in tbar:
        with open(filename, 'r') as fp:
            code = fp.read()
        tokens, values = cs(code)
        name_index = set(map(lambda item: item[0], filter(lambda item: "Token.Name" in item[1], enumerate(tokens))))
        lbracket_index = map(lambda item: item[0],
                             filter(lambda item: item[1] == "Token.Punctuation" and values[item[0]] == '(',
                                    enumerate(tokens)))
        lbracket_index_ = set(map(lambda i: i - 1, lbracket_index))
        interested_index = name_index.intersection(lbracket_index_)
        names = list(map(lambda i: values[i], interested_index))
        names_counter.update(names)
    most_common_names = list(map(lambda item: item[0], names_counter.most_common(thres)))
    return most_common_names


class CodeNormalizer(CodeSpliter):
    """
    __call__ 调用，进行了变量名正则化和字符串正则化（顺便去除所有的comment，在获取词汇表时有用）
    call 不需要重载，自动调用原来的
    """
    # %[flags][width][.precision][length]specifier
    __C_CPP_Placeholder_PCRE = r"%(?<flags>-|\+| |#|0)?"\
                               r"(?<width>\d+|\*)?"\
                               r"(.(?<precision>\d+|\*))?"\
                               r"(?<length>hh|h|l|ll|j|z|t|L)?"\
                               r"(?<specifier>d|i|u|o|x|X|f|F|e|E|g|G|a|A|c|s|p|n)"  # `%` is not needed here

    def __init__(self, lan: str, whole_code: str, most_commons=None, rename=True):
        assert lan.lower() in ['c', 'cpp']  # 目前只支持 C, CPP
        super().__init__(lan)
        self.most_commons = most_commons if most_commons is not None else []
        self.whole_tokens = self.lexer.get_tokens_unprocessed(whole_code)
        self.whole_tokens = list(filter(lambda item: not item[2].isspace(), self.whole_tokens))
        self.whole_tokens = self._remove_comment(self.whole_tokens)
        self.renamer = {}
        if rename:
            self.__get_substitute_dict()
        self.__convert_string()
        if rename:
            self.__rename()

    def __get_substitute_dict(self):
        """
        从 tokens 中提取所有的变量名，生成重命名字典
        """
        names = list(map(lambda i: self.whole_tokens[i][2], filter(lambda i: self.whole_tokens[i][1] in Token.Name and self.whole_tokens[i+1][2] != '(' if i+1 < len(self.whole_tokens) else True, range(len(self.whole_tokens)))))
        ordered_names = OrderedSet(names) - OrderedSet(self.most_commons)
        self.renamer = dict(zip(ordered_names.items, map(lambda i: f"var{ordered_names.index(i)}", ordered_names.items)))

    def __rename(self):
        """
        根据重命名字典，完成重命名
        注：不重命名函数名，避免重命名常见函数。统计所有dataset，选排名前多少的函数名
        """
        self.whole_tokens = list(map(
            lambda tk: (tk[0], tk[1], self.renamer[tk[2]]) if tk[1] in Token.Name and tk[2] in self.renamer.keys() else tk,
            self.whole_tokens))

    def __convert_string(self):
        flag = False
        res = []
        for tk in self.whole_tokens:
            if tk[1] == Token.Literal.String and tk[2] == '"':
                res.append(tk)
                flag = not flag
                continue
            if flag:
                res.extend(map(lambda r: (tk[0], Token.Literal.String, r.group()),
                               regex.finditer(CodeNormalizer.__C_CPP_Placeholder_PCRE, tk[2])))
            else:
                res.append(tk)
        self.whole_tokens = res

    def localize_tokens(self, sstart, send):
        assert sstart < send
        filtered_tokens = list(filter(lambda _t: sstart <= _t[0] < send, self.whole_tokens))
        tokens = list(map(lambda item: str(item[1]), filtered_tokens))
        values = list(map(lambda item: item[2], filtered_tokens))
        return tokens, values

    def get_all_tokens(self):
        return self.localize_tokens(0, self.whole_tokens[-1][0]+len(self.whole_tokens[-1][2]))


class Corpus:
    def __init__(self, raw_path, dir_pattern, lan, target="TokenValue"):
        assert os.path.exists(raw_path), "[Corpus] raw_path doesn't exist."
        self.__dir_pattern = dir_pattern
        self.__lan = lan
        with open(os.path.join(raw_path, "most_common_function_names.txt"), 'r') as fp:
            self.most_commons = fp.read().split('\n')
        self.lexical_map = pickle_read(os.path.join(raw_path, "word_lexical_dict.pkl"))
        self.value_map = pickle_read(os.path.join(raw_path, "word_value_dict.pkl"))
        self.__target = target
        self.set_target(target)

    def set_target(self, target):
        assert target in ("TokenType", "TokenValue")
        self.__target = target

    def __iter__(self):
        """
        词汇表是字符串类型的
        """
        for filename in glob.iglob(self.__dir_pattern, recursive=True):
            with open(filename, 'r') as fp:
                code = fp.read()
            cn = CodeNormalizer(self.__lan, code, self.most_commons)
            _lexical, _value = cn.get_all_tokens()
            _converted_lexical = list(map(
                lambda l: str(self.lexical_map.index(l))
                if l in self.lexical_map else str(self.lexical_map.index('<UNK>')), _lexical))
            _converted_value = list(map(
                lambda v: str(self.value_map.index(v))
                if v in self.value_map else str(self.value_map.index('<UNK>')), _value))
            if self.__target == "TokenType":
                yield _converted_lexical
            else:
                yield _converted_value


class Word2VecWrapper:
    """
    Word2vec 预训练模型
    注意词汇表都是字符串格式的数字
    """
    def __init__(self, dataset: DatasetName, model_name):
        save_dir = os.path.join(DATASET_DIR_PATH, f"{dataset.name}_{PreprocessMethod.WORD2VEC.name}")
        print(f"Loading Word2vec model from {os.path.join(save_dir, model_name)} ...", end='')
        self.model = word2vec.Word2Vec.load(os.path.join(save_dir, model_name))
        print("Finished.")

    def __call__(self, tokens):
        """
        _type 指定了使用哪一个模型
        Args:
            tokens: List[str]

        Returns:

        """
        assert len(tokens) != 0

        use_pytorch = False
        if isinstance(tokens, torch.Tensor):
            use_pytorch = True
            assert tokens.dim() == 1
            tokens = tokens.tolist()
        if isinstance(tokens[0], int):
            tokens = list(map(str, tokens))
        if use_pytorch:
            return torch.tensor(self.model.wv[tokens])
        else:
            return self.model.wv[tokens]


def preprocess_raw(preprocess_config: dict):
    # create save dir
    save_dir = os.path.join(DATASET_DIR_PATH, f"{preprocess_config['dataset_name'].name}_{PreprocessMethod.RAW.name}")
    if os.path.exists(save_dir) and len(glob.glob(f"{save_dir}/*")) == 7 and not preprocess_config["force_refresh"]:
        return
    Path(save_dir).mkdir(exist_ok=True)  # 不需要删除旧文件，处理完后会覆盖

    dataset_path = os.path.join(DATASET_DIR_PATH, preprocess_config['dataset_name'].name)
    assert os.path.exists(dataset_path), f"[ERROR] {dataset_path} is invalid."

    print(f"Generating raw dataset of {preprocess_config['dataset_name'].name} ...")

    '''
    # 每个 class 单独分出 train/val/test
    class_classifier = {}
    for filename in glob.iglob(rf"{dataset_path}/*.json"):
        data = json_read(filename)
        cls = data["properties"]["class"]
        for cl in cls:
            if cl not in class_classifier.keys():
                class_classifier[cl] = [filename]
            else:
                class_classifier[cl].append(filename)
    train_filenames = []
    valid_filenames = []
    test_filenames = []
    p1 = preprocess_config['preprocess']['proportion'][0]/sum(preprocess_config['preprocess']['proportion'])
    p2 = sum(preprocess_config['preprocess']['proportion'][0:2])/sum(preprocess_config['preprocess']['proportion'])
    for filenames in class_classifier.values():
        random.shuffle(filenames)
        n1 = round(len(filenames)*p1)
        n2 = round(len(filenames)*p2)
        train_filenames.extend(filenames[:n1])
        valid_filenames.extend(filenames[n1:n2])
        test_filenames.extend(filenames[n2:])
    del class_classifier
    '''
    binary_classifier = {0: [], 1: []}
    filenames = glob.glob(rf"{dataset_path}/*.json")
    for filename in filenames:
        try:
            data = json_read(filename)
        except json.decoder.JSONDecodeError:
            continue
        len_nodes = len(data['nodes'])
        # 数据清洗：删除具有 5000 及以上个节点的图样本
        if len_nodes > 5000 or len_nodes < 2:
            continue
        label = data["properties"]["label"]

        # TODO
        if label is None:
            continue
        binary_classifier[label].append(filename)
    train_filenames = []
    valid_filenames = []
    test_filenames = []
    p1 = preprocess_config['preprocess']['proportion'][0] / sum(preprocess_config['preprocess']['proportion'])
    p2 = sum(preprocess_config['preprocess']['proportion'][0:2]) / sum(preprocess_config['preprocess']['proportion'])
    for filenames2 in binary_classifier.values():
        random.shuffle(filenames2)
        random.shuffle(filenames2)
        random.shuffle(filenames2)
        n1 = round(len(filenames2) * p1)
        n2 = round(len(filenames2) * p2)
        train_filenames.extend(filenames2[:n1])
        valid_filenames.extend(filenames2[n1:n2])
        test_filenames.extend(filenames2[n2:])
    for _ in range(7):
        random.shuffle(train_filenames)
        random.shuffle(valid_filenames)
        random.shuffle(test_filenames)
    del binary_classifier

    # 匹配处理完的文件和源文件，即 preprocess 模块中的 result 和 enhanced
    # result -> uid -> enhanced
    # result2uid = dict(map(lambda path: (path, '-'.join(path.split('/')[-1].split('.')[0].split('-')[:-1])), filenames))
    # uid2enhanced = dict(map(lambda path: (path.split('/')[-1].split('.')[0], path),
    #                         glob.iglob(preprocess_config['preprocess']['origin_data_path_pattern'], recursive=True)))
    # processed2orig = dict(map(lambda item: (item[0], uid2enhanced[item[1]]), result2uid.items()))
    # del result2uid, uid2enhanced, filenames

    # JOERN1
    result2uid = dict(map(lambda path: (path, '-'.join(path.split('/')[-1].split('.')[0].split('-')[:-1])), filenames))
    # JOERN2
    # result2uid = dict(map(lambda path: (path, path.split('/')[-1].split('.')[0]), filenames))
    uid2enhanced = dict(map(lambda path: (path.split('/')[-1].split('.')[0], path),
                            glob.iglob(preprocess_config['preprocess']['origin_data_path_pattern'], recursive=True)))
    processed2orig = dict(map(lambda item: (item[0], uid2enhanced[item[1]]), result2uid.items()))
    del result2uid, uid2enhanced, filenames

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
    # 将字符串转化为数字，减少存储量
    lexical_map = OrderedSet(map(lambda item: item[0], lexical_counter.items()))
    lexical_map.add('<UNK>')
    value_map = OrderedSet(map(lambda item: item[0],
                               filter(lambda item: item[1] > min_count, value_counter.items())))
    value_map.add('<UNK>')
    # joern 自己加入的一个代码，特殊处理
    value_counter.update(["RET"])

    train_graph = {
        "n_graphs": len(train_filenames),
        "graph_id_index": [],
        "graphs": {}
    }
    valid_graph = {
        "n_graphs": len(valid_filenames),
        "graph_id_index": [],
        "graphs": {}
    }
    test_graph = {
        "n_graphs": len(test_filenames),
        "graph_id_index": [],
        "graphs": {}
    }

    print("Generating dataset...")
    cnt = 0
    for split, graph, files in zip(['train', 'valid', 'test'],
                                   [train_graph, valid_graph, test_graph],
                                   [train_filenames, valid_filenames, test_filenames]):
        files = tqdm.tqdm(files, desc=f'{split}')
        for filename in files:
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
            node_type_map.update(sorted_type)
            _type = list(map(node_type_map.index, sorted_type))

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
            # rpst_nodes
            _lines = list(map(lambda item: int(item[1]["line"]) - 1, sorted_data))  # joern 从 1 开始
            _lines_code_length = list(map(lambda item: len(item[1]["code"]), sorted_data))
            max_values = {}
            result_indexes = []
            for i, (v1, v2) in enumerate(zip(_lines, _lines_code_length)):
                # TODO: 第二个脚本中是 0
                if v1 < 0:  # RET 不管了，应该不会有小于三个字符的函数名吧
                    continue
                if v1 not in max_values:
                    max_values[v1] = (v2, i)
                else:
                    if v2 > max_values[v1][0]:
                        max_values[v1] = (v2, i)
            for v1 in max_values:
                result_indexes.append(max_values[v1][1])

            interested_lines = data["properties"]["lines"]
            interested_nodes_indexes = list(map(lambda item: item[0], filter(lambda _l: _l[1] in interested_lines, enumerate(_lines))))

            # links
            _edges = data["edges"]

            graph["graph_id_index"].append(graph_index_id)
            graph["graphs"][graph_index_id] = {
                "node_types": _type,
                "node_lexical_features": converted_lexical,
                "node_value_features": converted_value,
                "edges": _edges,
                "rpst_nodes": {
                    0: list(set(result_indexes) - set(interested_nodes_indexes)),
                    1: list(set(result_indexes) & set(interested_nodes_indexes))
                },
                "class": data["properties"]["class"],
                "label": data["properties"]["label"],
            }

    # store the dataset
    for split, graph in zip(["train", "valid", "test"], [train_graph, valid_graph, test_graph]):
        json_dump(obj=graph, path=f"{save_dir}/{split}.json")
    print(f"Skip {cnt} samples due to failure.")

    # save most commons
    with open(f"{save_dir}/most_common_function_names.txt", 'w') as fp:
        fp.write('\n'.join(most_commons))
    # save the word dict
    pickle_dump(obj=node_type_map, path=f"{save_dir}/node_type_dict.pkl")
    pickle_dump(obj=lexical_map, path=f"{save_dir}/word_lexical_dict.pkl")
    pickle_dump(obj=value_map, path=f"{save_dir}/word_value_dict.pkl")

    print(f"Raw dataset of {preprocess_config['dataset_name'].name} has been generated!")
    sys.stdout.flush()


def preprocess_raw_wo_rename(preprocess_config: dict):
    # create save dir
    save_dir = os.path.join(DATASET_DIR_PATH, f"{preprocess_config['dataset_name'].name}_{PreprocessMethod.RAW_WO_RENAME.name}")
    if os.path.exists(save_dir) and len(glob.glob(f"{save_dir}/*")) == 7 and not preprocess_config["force_refresh"]:
        return
    Path(save_dir).mkdir(exist_ok=True)  # 不需要删除旧文件，处理完后会覆盖

    dataset_path = os.path.join(DATASET_DIR_PATH, preprocess_config['dataset_name'].name)
    assert os.path.exists(dataset_path), f"[ERROR] {dataset_path} is invalid."

    print(f"Generating raw_wo_rename dataset of {preprocess_config['dataset_name'].name} ...")

    binary_classifier = {0: [], 1: []}
    filenames = glob.glob(rf"{dataset_path}/*.json")
    for filename in filenames:
        data = json_read(filename)
        len_nodes = len(data['nodes'])
        # 数据清洗：删除具有 5000 及以上个节点的图样本
        if len_nodes > 5000:
            continue
        label = data["properties"]["label"]
        binary_classifier[label].append(filename)
    train_filenames = []
    valid_filenames = []
    test_filenames = []
    p1 = preprocess_config['preprocess']['proportion'][0] / sum(preprocess_config['preprocess']['proportion'])
    p2 = sum(preprocess_config['preprocess']['proportion'][0:2]) / sum(preprocess_config['preprocess']['proportion'])
    for filenames2 in binary_classifier.values():
        random.shuffle(filenames2)
        random.shuffle(filenames2)
        random.shuffle(filenames2)
        n1 = round(len(filenames2) * p1)
        n2 = round(len(filenames2) * p2)
        train_filenames.extend(filenames2[:n1])
        valid_filenames.extend(filenames2[n1:n2])
        test_filenames.extend(filenames2[n2:])
    for _ in range(7):
        random.shuffle(train_filenames)
        random.shuffle(valid_filenames)
        random.shuffle(test_filenames)
    del binary_classifier

    # 匹配处理完的文件和源文件，即 preprocess 模块中的 result 和 enhanced
    # result -> uid -> enhanced
    result2uid = dict(map(lambda path: (path, '-'.join(path.split('/')[-1].split('.')[0].split('-')[:-1])), filenames))
    uid2enhanced = dict(map(lambda path: (path.split('/')[-1].split('.')[0], path),
                            glob.iglob(preprocess_config['preprocess']['origin_data_path_pattern'], recursive=True)))
    processed2orig = dict(map(lambda item: (item[0], uid2enhanced[item[1]]), result2uid.items()))
    del result2uid, uid2enhanced, filenames

    most_commons = collect_function_names(preprocess_config['preprocess']['origin_data_path_pattern'], 'cpp')

    # 统一用 cpp 处理
    node_type_map = OrderedSet()  # 将字符串转化为数字，减少存储量
    # 先生成完整的字典
    min_count = 10
    lexical_counter = Counter()
    value_counter = Counter()
    for filename in tqdm.tqdm(glob.iglob(preprocess_config['preprocess']['origin_data_path_pattern'], recursive=True),
                              desc="Generating word dict"):
        with open(filename, 'r') as fp:
            code = fp.read()
        cn = CodeNormalizer('cpp', code, most_commons, rename=False)
        _lexical, _value = cn.get_all_tokens()
        lexical_counter.update(_lexical)
        value_counter.update(_value)
    # 将字符串转化为数字，减少存储量
    lexical_map = OrderedSet(map(lambda item: item[0], lexical_counter.items()))
    lexical_map.add('<UNK>')
    value_map = OrderedSet(map(lambda item: item[0],
                               filter(lambda item: item[1] > min_count, value_counter.items())))
    value_map.add('<UNK>')
    # joern 自己加入的一个代码，特殊处理
    value_counter.update(["RET"])

    train_graph = {
        "n_graphs": len(train_filenames),
        "graph_id_index": [],
        "graphs": {}
    }
    valid_graph = {
        "n_graphs": len(valid_filenames),
        "graph_id_index": [],
        "graphs": {}
    }
    test_graph = {
        "n_graphs": len(test_filenames),
        "graph_id_index": [],
        "graphs": {}
    }

    print("Generating dataset...")
    cnt = 0
    for split, graph, files in zip(['train', 'valid', 'test'],
                                   [train_graph, valid_graph, test_graph],
                                   [train_filenames, valid_filenames, test_filenames]):
        files = tqdm.tqdm(files, desc=f'{split}')
        for filename in files:
            with open(processed2orig[filename], 'r') as fp:
                orig_code = fp.read()
            cn = CodeNormalizer('cpp', orig_code, most_commons, rename=False)
            split_lines = orig_code.split('\n')

            data = json_read(filename)
            graph_index_id = filename.split('/')[-1]

            # 先生成str版本的数据集
            sorted_data = sorted(data["nodes"].items(), key=lambda item: int(item[0]))
            # node type
            sorted_type = list(map(lambda item: item[1]["type"], sorted_data))
            node_type_map.update(sorted_type)
            _type = list(map(node_type_map.index, sorted_type))

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

            graph["graph_id_index"].append(graph_index_id)
            graph["graphs"][graph_index_id] = {
                "node_types": _type,
                "node_lexical_features": converted_lexical,
                "node_value_features": converted_value,
                "edges": _edges,
            }
            graph["graphs"][graph_index_id].update(data["properties"])

    # store the dataset
    for split, graph in zip(["train", "valid", "test"], [train_graph, valid_graph, test_graph]):
        json_dump(obj=graph, path=f"{save_dir}/{split}.json")
    print(f"Skip {cnt} samples due to failure.")

    # save most commons
    with open(f"{save_dir}/most_common_function_names.txt", 'w') as fp:
        fp.write('\n'.join(most_commons))
    # save the word dict
    pickle_dump(obj=node_type_map, path=f"{save_dir}/node_type_dict.pkl")
    pickle_dump(obj=lexical_map, path=f"{save_dir}/word_lexical_dict.pkl")
    pickle_dump(obj=value_map, path=f"{save_dir}/word_value_dict.pkl")

    print(f"Raw_wo_rename dataset of {preprocess_config['dataset_name'].name} has been generated!")
    sys.stdout.flush()


def preprocess_wv(preprocess_config: dict):
    preprocess_raw(preprocess_config)

    save_dir = os.path.join(DATASET_DIR_PATH, f"{preprocess_config['dataset_name'].name}_{PreprocessMethod.WORD2VEC.name}")
    num_out_features = preprocess_config['model']['num_features_per_gat_layer'][0]
    lexical_model_path = f"""{os.path.join(save_dir, f"lexical_{num_out_features}.pkl")}"""
    value_model_path = f"""{os.path.join(save_dir, f"value_{num_out_features}.pkl")}"""

    if os.path.exists(lexical_model_path) and os.path.exists(value_model_path) and not preprocess_config["force_refresh"]:
        return
    Path(save_dir).mkdir(exist_ok=True)  # 不需要删除旧文件，处理完后会覆盖

    print(f"Generating word2vec models of {preprocess_config['dataset_name'].name} ...")

    if os.path.exists(lexical_model_path) and os.path.exists(value_model_path) and not preprocess_config['force_refresh']:  # 备注：TIFS 中为每个 cwe 生成一个model
        print("Word2Vec models already exist. Skip.")
        return
    print("Word2Vec model don't exist. Start training...")
    sys.stdout.flush()

    raw_dir = os.path.join(DATASET_DIR_PATH, f"{preprocess_config['dataset_name'].name}_{PreprocessMethod.RAW.name}")
    # 使用 enhanced 数据集进行训练
    dir_pattern = preprocess_config['preprocess']['origin_data_path_pattern']

    sentences = Corpus(raw_dir, dir_pattern, 'cpp')
    # CBOW 适合小数据集
    if os.path.exists(lexical_model_path) and not preprocess_config['force_refresh']:
        print("Lexical model exists!")
    else:
        sentences.set_target("TokenType")
        lexical_model = Word2Vec(sentences, min_count=1, vector_size=num_out_features,
                                 sg=0, window=5, negative=3, sample=0.001, hs=1, workers=8)
        lexical_model.save(lexical_model_path)
        print(f"Word2Vec model has been saved at {lexical_model_path}.")
    sys.stdout.flush()
    if os.path.exists(value_model_path) and not preprocess_config['force_refresh']:
        print("Value model exists!")
    else:
        sentences.set_target("TokenValue")
        value_model = Word2Vec(sentences, min_count=1, vector_size=num_out_features,
                               sg=0, window=5, negative=3, sample=0.001, hs=1, workers=8)
        value_model.save(value_model_path)
        print(f"Word2Vec model has been saved at {value_model_path}.")
    print(f"Word2Vec models are ready.")
    sys.stdout.flush()


def preprocess_cv(preprocess_config: dict):
    pass


def preprocess(preprocess_config: dict):
    if preprocess_config['preprocess']['method'] == PreprocessMethod.RAW:
        preprocess_raw(preprocess_config)
    elif preprocess_config['preprocess']['method'] == PreprocessMethod.WORD2VEC:
        preprocess_wv(preprocess_config)
    elif preprocess_config['preprocess']['method'] == PreprocessMethod.CODE2VEC:
        preprocess_cv(preprocess_config)
    elif preprocess_config['preprocess']['method'] == PreprocessMethod.RAW_WO_RENAME:
        preprocess_raw_wo_rename(preprocess_config)
    else:
        raise Exception()


# split [train/valid/test] and convert .txt to .npy&.json as they are much easier to be loaded
# generate the following files:
# [train/valid/test]_graph.json, [train/valid/test]_graph_id.npy,
# [train/valid/test]_feats.npy, [train/valid/test]_labels.npy
if __name__ == '__main__':
    config = {
        "dataset_name": DatasetName.CroVulNew,
        "preprocess": {
            "method": PreprocessMethod.RAW,
            "proportion": [8, 1, 1],
            # /Users/huanghongjun/FiCoVuL/preprocess/data/enhanced/**/*.c
            # /home/huanghongjun/FiCoVuL/preprocess/data/enhanced/**/*.c
            "origin_data_path_pattern": "../data/datasets/CroVulNew_ORIG/*.c",  # path in server
        },
        "model": {
            "num_features_per_gat_layer": [128],
        },
        "force_refresh": False,
    }
    preprocess(config)
