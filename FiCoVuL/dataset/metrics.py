import glob
import tqdm
import json


if __name__ == '__main__':
    total_graph_num = 0
    total_rpst_0_num = 0
    total_rpst_1_num = 0
    for filename in tqdm.tqdm(glob.glob(r"../data/datasets/CroVul/*.json")):
        with open(filename, 'r') as f:
            data = json.load(f)
        # 先生成str版本的数据集
        sorted_data = sorted(data["nodes"].items(), key=lambda item: int(item[0]))
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
        interested_nodes_indexes = list(
            map(lambda item: item[0], filter(lambda _l: _l[1] in interested_lines, enumerate(_lines))))

        len_0 = len(set(result_indexes) - set(interested_nodes_indexes))
        len_1 = len(set(result_indexes) & set(interested_nodes_indexes))

        total_graph_num += 1
        total_rpst_0_num += len_0
        total_rpst_1_num += len_1

    print(f"total_graph_num: {total_graph_num}")
    print(f"total_rpst_0_num: {total_rpst_0_num}")
    print(f"total_rpst_1_num: {total_rpst_1_num}")
    print(f"#nodes/#graphs: {(total_rpst_0_num + total_rpst_1_num) / total_graph_num}")
