import os
import json
from ordered_set import OrderedSet
from typing import Tuple, List, Dict
import argparse
from pathlib import Path


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
        "REACHING_DEF": 2,
        "CDG": 3,
        "REF": 4,
        "CALL": 5,
        "VTABLE": 6,
        "INHERITS_FROM": 7,
        "BINDS_TO": 8,
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
    parser.add_argument("--output_file", "-o", type=str,
                        help="where the results are stored")
    args = parser.parse_args()
    
    savePath = Path(os.getcwd()) / "Data" / "output_doxygen_result"
    savePath.mkdir(exist_ok=True)
    
    with open(args.input_file, 'r') as f:
        content = json.load(f)
    origin_path = content["originPath"]
    functions = content["functions"]
    for func in functions:
        # filter out undefined functions
        if func["filename"] == "N/A":
            continue
        
        method_name = func["function"]
        graphs = JointGraph()
        # First Loop: collect all nodes
        # function Node
        id = func["id"]
        type = func["label"]
        code = func["function"]
        line = -1
        column = -1
        graphs.addNode(Node(id, type, code, line, column))
        for node in func["AST"]+func["CFG"]+func["PDG"]:
            id = node["id"]
            type = node["label"]
            code = node["properties"]["CODE"] if "CODE" in node["properties"] else ""
            line = node["properties"].get("LINE_NUMBER", -1)
            column = node["properties"].get("COLUMN_NUMBER", -1)
            graphs.addNode(Node(id, type, code, line, column))
        # Second Loop: collect all edges
        for node in func["AST"]+func["CFG"]+func["PDG"]:
            for e in node["edges"]:
                inNode, outNode = graphs.indexNode([e["in"], e["out"]])
                graphs.addEdge(Edge(e["id"], inNode, outNode, e["label"]))
        # Log out
        # print(f"A total of {len(graphs.nodes)} vertices and {len(graphs.edges)} edges are collected")

        # Save
        with open(origin_path.replace('.c', '.json'), 'r') as f:
            annotation = json.load(f)
        if method_name != annotation["function_name"]:
            # 线程执行时此警报不会被输出
            print(f'[ERROR]In {func["filename"]} {method_name} != {annotation["function_name"]}')
            continue
        with open(f"{args.output_file}-{method_name}.json", 'w') as f:
            json.dump(graphs(annotation), f)
