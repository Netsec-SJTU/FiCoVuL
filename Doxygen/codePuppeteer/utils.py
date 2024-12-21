from pathlib import Path
import json
from uuid import uuid4
import difflib
from typing import List, Tuple
import sqlite3 as lite
from sqlite3 import Error

from .fextracter import FunctionDecl, FunctionDeclOrig


def save_single_instance(save_path: Path, code: str, annotation: dict, foler_classify: bool = False):
    """
    annotation = {
        "origin_path": str,
        "function_name": str,
        "class": List[str],
        "label": 0 or 1,
        "roi": List[int],
    }
    """
    result_filename = uuid4()
    if foler_classify:
        for cls in annotation["class"]:
            save_path = save_path / cls
            save_path.mkdir(exist_ok=True)
            with open(save_path / f"{result_filename}.c", 'w') as fp:
                fp.write(code)
            with open(save_path / f"{result_filename}.json", 'w') as fp:
                json.dump(annotation, fp=fp)
    else:
        with open(save_path / f"{result_filename}.c", 'w') as fp:
            fp.write(code)
        with open(save_path / f"{result_filename}.json", 'w') as fp:
            json.dump(annotation, fp=fp)


def save_multiple_instances(save_path: Path, results: List, foler_classify: bool = False):
    """

    :param save_path:
    :param results: List[Tuple[fname, code, annotation]]
    :return:
    """
    if len(results) == 0:
        return
    save_path.mkdir(exist_ok=True)
    for result in results:
        save_single_instance(save_path, result[1], result[2], foler_classify=foler_classify)


def longestCommonSubsequence(t1: str, t2: str) -> int:
    m, n = len(t1), len(t2)

    # 构建DP TABLE + base case
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if t1[i - 1] == t2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def get_code_changeline(s1: str, s2: str):
    changeline1 = []
    changeline2 = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=s1, b=s2).get_opcodes():
        if tag != 'equal':
            temp = s1[:i1].count('\n')
            changeline1 += range(temp, temp+s1[i1:i2].count('\n')+1)
            temp = s2[:j1].count('\n')
            changeline2 += range(temp, temp+s2[j1:j2].count('\n')+1)
    return list(set(changeline1)), list(set(changeline2))


def merge(records: List[Tuple[bool, FunctionDeclOrig, FunctionDecl, dict]]):
    """
    区分能够和不能够合并的函数，对能够合并的函数进行函数融合，返回全部结果
    :param records: List[Tuple[is_enhanced, fer, fee, annotation]]
    :return: List[Tuple[fname, code, annotation]]
    """
    functions = {}  # fname -> (fer, fee, annotation)
    results = []  # List[fname, code, annotation]
    for is_enhanced, fer, fee, annotation in records:
        if is_enhanced:
            if fer.fname not in functions:
                functions[fer.fname] = (fer, fee, annotation)
            else:
                # 将重名的函数移出待融合的队列
                temp = functions[fer.fname]
                temp[2]["roi"] = fer.roi
                results.append((temp[0].fname, temp[0].get_orig(), temp[2]))
                del functions[fer.fname]
                # 记录这次处理的函数
                annotation["roi"] = fer.roi
                results.append((fer.fname, fer.get_orig(), annotation))
        else:
            annotation["roi"] = fer.roi
            results.append((fer.fname, fer.get_orig(), annotation))

    callers_with_annotation = list(map(lambda x: (x[0], x[2]), functions.values()))
    callees = list(map(lambda x: x[1], functions.values()))
    for caller, annotation in callers_with_annotation:
        flag, result, roi = caller.expands(callees)
        if not flag:
            result = caller.get_orig()
            roi = caller.roi
        annotation["roi"] = roi
        results.append((caller.fname, result, annotation))

    return results
