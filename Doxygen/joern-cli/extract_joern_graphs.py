import os
import json

import regex
from ordered_set import OrderedSet
from typing import Tuple, List, Dict
import argparse
from pathlib import Path

JOERN_LINE_FORMAT = r"\"((joern_id_\((?P<joern_id1>\d+)\))_joern_code_(\((?P<joern_code1>.*?)\)_)?joern_type_(\((?P<joern_type1>.*?)\)_)?joern_name_(\((?P<joern_name1>.*?)\)_)?joern_line_(\((?P<joern_line1>\d*)\)_)?joern_column_(\((?P<joern_column1>\d*)\))?)\" -->> \"((joern_id_\((?P<joern_id2>\d+)\))_joern_code_(\((?P<joern_code2>.*?)\)_)?joern_type_(\((?P<joern_type2>.*?)\)_)?joern_name_(\((?P<joern_name2>.*?)\)_)?joern_line_(\((?P<joern_line2>\d*)\)_)?joern_column_(\((?P<joern_column2>\d*)\))?)\""


class Node:
    def __init__(self, id: str, type: str, code: str, line: int, column: int) -> None:
        self.id = id
        self.type = type
        self.code = code
        self.line = line
        self.column = column

    def __repr__(self) -> str:
        return f"({self.id}, {self.type}, {self.code}, {self.line}, {self.column})"

    def __call__(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "code": self.code,
            "line": self.line,
            "column": self.column
        }


class Edge:
    typeMap = {
        "AST": 0,
        "CFG": 1,
        "PDG": 2,
        "REF": 3,
        "CALL": 4,
        "VTABLE": 5,
        "INHERITS_FROM": 6,
        "BINDS_TO": 7,
        "REACHING_DEF": 8,
        "EVAL_TYPE": 9,
        "CONTAINS": 10,
        "PROPAGATE": 11,
    }

    def __init__(self, id: str, inNode: int, outNode: int, type: str) -> None:
        assert type in Edge.typeMap.keys()

        self.id = id
        self.inNode = inNode
        self.outNode = outNode
        self.type = Edge.typeMap[type]

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"{tuple(self())}"

    def __call__(self) -> Tuple[int, int, int]:
        return self.inNode, self.outNode, self.type


class JointGraph(object):
    def __init__(self) -> None:
        self.nodes = {}  # Map: id -> Node
        self.nodesIndex = OrderedSet()  # set(id)
        self.edges = OrderedSet()  # set(Edge)

    def addNode(self, n: Node) -> int:
        if n.id not in self.nodes:
            self.nodes[n.id] = n
            return self.nodesIndex.add(n.id)
        else:
            return self.indexNode(n.id)

    def addEdge(self, e: Edge) -> int:
        return self.edges.add(e)

    def indexNode(self, id: str or List[str]) -> int:
        return self.nodesIndex.index(id)

    def indexEdge(self, e: Edge or List[Edge]) -> int:
        return self.edges.index(e)

    def getNode(self, id: str) -> Node:
        return self.nodes[id]

    def getAllEdges(self):
        return list(set(map(lambda e: e(), self.edges)))

    def __call__(self, annotation) -> Dict:
        result = {
            "properties": {
                "origin_path": annotation["origin_path"],
                "function_name": annotation["function_name"],
                "class": annotation["class"],
                "depth": annotation["depth"],
                "label": annotation["label"],
                "lines": annotation["roi"],
            },
            "nodes": {},
            "edges": self.getAllEdges()
        }
        for k, v in self.nodes.items():
            result["nodes"][self.indexNode(k)] = v()

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str,
                        help="result processed by the scala script")
    parser.add_argument("--annotated_file", "-a", type=str,
                        help="where the annotated json file is stored")
    parser.add_argument("--output_file", "-o", type=str,
                        help="where the result is to be stored")
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        lines = f.readlines()

    graph = JointGraph()

    cur_relation = None
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        if line == 'digraph g {' or line == '{' or line == '}' or line.startswith('edge'):
            continue
        elif line.startswith('#'):
            cur_relation = line[2:]
        else:
            res = regex.match(JOERN_LINE_FORMAT, line)
            if res is None:
                print(f'[ERROR] {line}')
                continue

            joern_id1 = res.group('joern_id1')
            joern_code1 = res.group('joern_code1')
            joern_type1 = res.group('joern_type1')
            # joern_name1 = res.group('joern_name1')
            joern_line1 = res.group('joern_line1')
            joern_column1 = res.group('joern_column1')

            joern_id2 = res.group('joern_id2')
            joern_code2 = res.group('joern_code2')
            joern_type2 = res.group('joern_type2')
            # joern_name2 = res.group('joern_name2')
            joern_line2 = res.group('joern_line2')
            joern_column2 = res.group('joern_column2')

            joern_code1 = joern_code1 if joern_code1 is not None else ''
            joern_code2 = joern_code2 if joern_code2 is not None else ''
            joern_type1 = joern_type1 if joern_type1 is not None else 'UNK'
            joern_type2 = joern_type2 if joern_type2 is not None else 'UNK'
            joern_line1 = joern_line1 if joern_line1 is not None else 0
            joern_line2 = joern_line2 if joern_line2 is not None else 0
            joern_column1 = joern_column1 if joern_column1 is not None else -1
            joern_column2 = joern_column2 if joern_column2 is not None else -1

            node1 = Node(joern_id1, joern_type1, joern_code1, joern_line1, joern_column1)
            node2 = Node(joern_id2, joern_type2, joern_code2, joern_line2, joern_column2)
            graph.addNode(node1)
            graph.addNode(node2)
            inNode, outNode = graph.indexNode([joern_id1, joern_id2])
            graph.addEdge(Edge(f"{joern_id1}-{cur_relation}->{joern_id2}", inNode, outNode, cur_relation))

    # Save
    with open(args.annotated_file, 'r') as f:
        annotation = json.load(f)
    with open(f"{args.output_file}", 'w') as f:
        json.dump(graph(annotation), f)
