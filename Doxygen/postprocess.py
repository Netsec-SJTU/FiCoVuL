import glob
import json
import os
import random

import chardet
import doxmlparser
import networkx as nx
from collections import namedtuple
from itertools import combinations, permutations, chain
from typing import List, Set, Tuple

from configure import DOXYGEN_OUTPUT_JSON_PATH
from doxmlparser.compound import DoxLanguage, DoxCompoundKind, DoxMemberKind, DoxSectionKind
from utils import clean_file, convert_runtime, convert_to_utf8, run_cmd

Pointer = namedtuple("Pointer", ["file", "line", "cwe_ids", "label"])


class Function:
    def __init__(self, id: str, project_path: str, relative_file_path: str, name: str, start: int, end: int, lan: str,
                 return_type: str, definition: str, argsstring: str, params: List):
        self.id = id
        self.project_path = project_path
        self.decl_file = relative_file_path
        self.name = name
        self.start = start
        self.end = end
        self.lois = []  # 0-indexed
        self.body = None
        self.extract_body()
        self.lan = lan
        self.cwe_ids = set()
        self.label = None

        self.puzzles = {
            "return_type": return_type,
            "definition": definition,
            "argsstring": argsstring,
            "params": params
        }

    def extract_body(self):
        try:
            with open(os.path.join(self.project_path, self.decl_file), 'r', errors='ignore') as f:
                lines = f.readlines()
        except UnicodeDecodeError as e:
            with open(os.path.join(self.project_path, self.decl_file), 'rb') as f:
                data = f.read()
                result = chardet.detect(data)
                encoding = result['encoding']
            with open(os.path.join(self.project_path, self.decl_file), 'r', encoding=encoding, errors='ignore') as f:
                lines = f.readlines()
        self.body = "".join(lines[self.start - 1: self.end])

    def set_by_pointer(self, pointer: Pointer):
        self.add_loi(pointer.line)
        if isinstance(pointer.cwe_ids, str):
            pointer.cwe_ids = [pointer.cwe_ids]
        self.add_cwe_ids(pointer.cwe_ids)
        self.set_label(pointer.label)

    def add_loi(self, loi: int):
        loi = loi - self.start
        if loi != 0 and loi not in self.lois:
            self.lois.append(loi)

    def add_cwe_ids(self, cwe_ids: List[str]|Set[str]):
        self.cwe_ids.update(cwe_ids)

    def set_lois(self, lois: List[int]):
        self.lois = list(map(lambda l: l-self.start, lois))

    def set_label(self, label: int):
        assert label in [0, 1]
        if self.label is None:
            self.label = label
        else:
            assert self.label == label

    def transfer2json(self):
        return {
            "member_id": self.id,
            "project_path": self.project_path,
            "decl_file": self.decl_file,
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "code": self.body,
            "lan": self.lan,
            "lois": self.lois,
            "cwe_ids": list(self.cwe_ids),
            "label": self.label,
            "puzzles": self.puzzles
        }

    def __str__(self):
        return f"Function(id={self.id}, name={self.name}, projet_path={self.project_path}, file={self.decl_file}, start={self.start}, end={self.end}, lan={self.lan}, cwe_ids={self.cwe_ids}, lois={self.lois})" \
               f"{'-' * 70}" \
               f"{self.body}" \
               f"{'-' * 70}"


class DoxygenParser:
    functionLikeKind = [DoxMemberKind.FUNCTION,
                        DoxMemberKind.PROTOTYPE,
                        DoxMemberKind.SIGNAL,
                        DoxMemberKind.SLOT,
                        DoxMemberKind.DCOP]

    def __init__(self, project_path, config_path, xml_path):
        self.project_path = str(project_path)
        self.xml_path = xml_path
        # compound -> sesction -> member
        # file
        self.MemberId2Compound = {}  # member.id -> {'kind': compound_kind, 'language': compound_language}, 为了区分 file 中的 function，现在只能处理这种 callee
        # Function Like
        self.Id2FuncLikeMember = {}  # member.id -> {'location': '', 'bodyfile': '', 'bodystart': '', 'bodyend': ''}
        self.Id2Function = {}  # member.id -> Function
        # only file function - filter ✅
        # 多条链重合的情况 ✅
        self.References = {}  # member.id -> [member.id]

        if not self.run_doxyparse(config_path):
            file_extensions = [".c", ".cc", ".cxx", ".cpp", ".ii", ".ixx", ".ipp", ".i++", ".inl", ".c++", ".h", ".H",
                               ".hh", ".HH", ".hxx", ".hpp", ".h++", ".mm"]
            interested_files = list(chain(
                *map(lambda postfix: list(glob.glob(rf"{self.project_path}/**/*{postfix}", recursive=True)),
                     file_extensions)))

            try:
                list(map(convert_to_utf8, interested_files))
            except:
                pass
            if not self.run_doxyparse(config_path):
                raise Exception(f"doxygen parse error.")

        try:
            self.parse_index()
        except:
            clean_file(self.xml_path)
            self.parse_index()
        self.filter_func_like_references()
        self.filter_out_non_exist_references()
        self.filter_empty_references()

        self.FOIs = []

    def run_doxyparse(self, config_path):
        # TODO：存在dump文件
        # if os.path.exists(self.xml_path) and len(os.listdir(self.xml_path)) > 0:
        #     return True
        code, msg = run_cmd(f"doxygen {config_path}")
        if code != 0:
            print(f"[ERROR] doxygen parse error. Msg: {msg}")
            return False
        return True

    def parse_index(self):
        """
        根据 index 中指示的 compound id ，解析所有 compound
        """
        rootObj = doxmlparser.index.parse(os.path.join(self.xml_path, "index.xml"), True)
        list(map(lambda compound: self.parse_compound(compound.get_refid()), rootObj.get_compound()))
        del rootObj

    def parse_compound(self, baseName):
        rootObj = doxmlparser.compound.parse(os.path.join(self.xml_path, f"{baseName}.xml"), True)
        # 只考虑 C/C++ 的组件
        list(map(lambda compounddef: self.parse_sections(compounddef),
                 filter(lambda x: x.get_language() in [DoxLanguage.OBJECTIVEC, DoxLanguage.C_1],
                        rootObj.get_compounddef())))
        del rootObj

    def parse_sections(self, compounddef):
        # 不要这里做限制了，之后会过滤出 FUNCLIKE
        list(map(lambda sectiondef: self.parse_members(compounddef, sectiondef), compounddef.get_sectiondef()))

    def parse_members(self, compounddef, sectiondef):
        # variableLikeKind = [DoxMemberKind.VARIABLE, DoxMemberKind.PROPERTY]
        # 忽略变量定义，此时bodyend为-1
        list(map(lambda memberdef: self.Id2FuncLikeMember.update({memberdef.get_id(): dict(zip(['location', 'bodyfile', 'bodystart', 'bodyend'], [memberdef.get_location(), *self.parse_location(memberdef.get_location())]))}),
                 filter(lambda x: x.get_kind() in DoxygenParser.functionLikeKind, sectiondef.get_memberdef())))
        list(map(lambda memberdef: self.MemberId2Compound.update({memberdef.get_id(): {'kind': compounddef.get_kind(), 'language': compounddef.get_language()}}),
                 filter(lambda x: x.get_kind() in DoxygenParser.functionLikeKind, sectiondef.get_memberdef())))
        # 只关心 FUCNTION
        list(map(lambda memberdef: self.extract_function(memberdef),
                 filter(lambda x: x.get_kind() in [DoxMemberKind.FUNCTION, DoxMemberKind.SLOT], sectiondef.get_memberdef())))
        # 只关心 functionLikeKind reference
        # TODO: references 可能不在 Id2FuncLikeMember 中，要用 subgraph
        list(map(lambda memberdef: self.References.update(
            {memberdef.get_id(): list(map(lambda ref: self.parse_reference(ref), memberdef.get_references()))}),
                 filter(lambda x: x.get_kind() in DoxygenParser.functionLikeKind, sectiondef.get_memberdef())))

    def extract_function(self, memberdef):
        # 只关心 FUCNTION
        assert memberdef.get_kind() in [DoxMemberKind.FUNCTION, DoxMemberKind.SLOT]

        loc = memberdef.get_location()
        if loc is None:
            # print(f"[ERROR] {memberdef.get_id()} has no location.")
            return
        # 1-indexed
        bodyfile, bodystart, bodyend = self.parse_location(loc)
        if bodyfile is None:  # 没有函数体定义
            return
        self.Id2Function[memberdef.get_id()] = Function(memberdef.get_id(), self.project_path, bodyfile,
                                                        memberdef.get_name(), bodystart, bodyend,
                                                        self.MemberId2Compound[memberdef.get_id()]['language'],
                                                        self.parse_linkedText(memberdef.get_type()),
                                                        memberdef.get_definition(), memberdef.get_argsstring(),
                                                        list(map(self.parse_param, memberdef.get_param())))

    def parse_linkedText(self, linkedText):
        if linkedText is None:
            return None
        else:
            t1 = linkedText.get_valueOf_()
            if len(linkedText.get_ref()) != 0:
                t2 = linkedText.get_ref()[0].get_valueOf_()
                t1 = t2 + t1
            return t1

    def parse_param(self, paramdef):
        # print("get_type: {}".format(self.parse_linkedText(paramdef.get_type())))
        # print("get_declname: {}".format(paramdef.get_declname()))
        # print("get_defname: {}".format(paramdef.get_defname()))
        # print("get_array: {}".format(paramdef.get_array()))
        # print("get_defval: {}".format(self.parse_linkedText(paramdef.get_defval())))
        # print("typeconstraint: {}".format(paramdef.get_typeconstraint()))
        """
        get_type: uint      const
        get_declname: p
        get_defname: None
        get_array: []
        get_defval: None
        """
        # self.parse_linkedText(paramdef.get_type()), paramdef.get_declname(), paramdef.get_defname(), paramdef.get_array(), self.parse_linkedText(paramdef.get_defval())
        return {
            "type": self.parse_linkedText(paramdef.get_type()),
            "declname": paramdef.get_declname(),
            "defname": paramdef.get_defname(),
            "array": paramdef.get_array(),
            "defval": self.parse_linkedText(paramdef.get_defval())
        }

    def parse_location(self, locationdef):
        if locationdef is None:
            return None, None, None
        # print("File: {}".format(locationdef.get_file()))
        # 只关心函数体定义在哪里
        return locationdef.get_bodyfile(), locationdef.get_bodystart(), locationdef.get_bodyend()

    def parse_reference(self, referencedef):
        # print("Value: {}".format(referencedef.get_valueOf_()))
        # print("Refid: {}".format(referencedef.get_refid()))
        # print("Compoundref: {}".format(referencedef.get_compoundref()))
        return referencedef.get_refid()

    def filter_out_non_exist_references(self):
        def is_existing_reference(caller_id, callee_id):
            if callee_id not in self.Id2Function or caller_id not in self.Id2Function:
                return False
            callee_name = self.Id2Function[callee_id].name
            return callee_name in self.Id2Function[caller_id].body

        list(map(lambda k: self.References.update({k: list(filter(lambda x: is_existing_reference(k, x), self.References[k]))}), self.References))

    def filter_func_like_references(self):
        # 只关心 functionLikeKind reference
        # 构建完以后调用
        list(map(lambda mid: self.References.update(
            {mid: list(filter(lambda ref: ref in self.Id2FuncLikeMember.keys(), self.References[mid]))}), self.References))

    def filter_empty_references(self):
        self.References = dict(filter(lambda x: len(x[1]) != 0, self.References.items()))

    def get_references(self):
        return self.References

    def get_reference_edges(self):
        return list(chain(*map(lambda x: list(map(lambda y: (x, y), self.References[x])), self.References)))

    def get_functions(self):
        return self.Id2Function

    def get_member_by_loc(self, filename, line):
        res = list(map(lambda x: x[0],
                       filter(lambda x: x[1]['location'] is not None and
                                        x[1]['bodyfile'] == filename and
                                        x[1]['bodystart'] <= line <= x[1]['bodyend'],
                              self.Id2FuncLikeMember.items())))
        if len(res) != 0:
            return res[0]
        return None

    def set_FOIs(self, pointers: List[Pointer]):
        for pointer in pointers:
            self.add_FOI(pointer)

    def get_FOIs(self):
        return self.FOIs

    def add_FOI(self, pointer: Pointer):
        # FOI 一定是有定义的
        m = self.get_member_by_loc(pointer.file, pointer.line)
        if m is not None:
            if m not in self.FOIs:
                self.FOIs.append(m)
            self.Id2Function[m].set_by_pointer(pointer)

    def add_FOI_by_id(self, id: str):
        if id not in self.FOIs:
            self.FOIs.append(id)

    def clear_FOIs(self):
        self.FOIs.clear()


def find_line_correspondence(deleted: List[int], added: List[int], n_old_lines, _n_new_lines):
    """找到文件修改前后的行号对应关系

    Args:
        deleted (List[int]): 1-indexed
        added (List[int]): 1-indexed
    """

    deleted = list(map(lambda x: x - 1, deleted))
    added = list(map(lambda x: x - 1, added))

    n_new_lines = n_old_lines - len(deleted) + len(added)
    assert n_new_lines == _n_new_lines, f"Calculated value is {n_new_lines}, but given value is {_n_new_lines}\n"  # given by count

    old2new = [None] * n_old_lines
    new2old = [None] * n_new_lines

    for i in deleted:
        old2new[i] = -1
    for i in added:
        new2old[i] = -1
    for i in set(range(n_old_lines)) - set(deleted):
        j = new2old.index(None)
        old2new[i] = j
        new2old[j] = i

    return old2new, new2old  # 0-indexed


def FOI_outer_merge(doxygen_parser_old, doxygen_parser_new, diff_parsed):
    """
    diff_parsed = [old_path, new_path, n_old_lines, n_new_lines, diff_parsed']
    """
    old_FOIs = doxygen_parser_old.get_FOIs()  # ids
    new_FOIs = doxygen_parser_new.get_FOIs()  # ids
    old_function_defs_filtered = list(map(lambda i: doxygen_parser_old.get_functions()[i], old_FOIs))
    # print("old_function_defs_filtered:", list(map(lambda f: f.name, old_function_defs_filtered)))
    new_function_defs_filtered = list(map(lambda i: doxygen_parser_new.get_functions()[i], new_FOIs))
    # print("new_function_defs_filtered:", list(map(lambda f: f.name, new_function_defs_filtered)))
    # print("diff_parsed:", diff_parsed)
    for x in diff_parsed:
        # print("***************x:", x)
        old_path, new_path, n_old_lines, n_new_lines, dp = x
        # print(old_function_defs_filtered)
        old_FOIs_filtered = list(filter(lambda f: f.decl_file == old_path, old_function_defs_filtered))
        new_FOIs_filtered = list(filter(lambda f: f.decl_file == new_path, new_function_defs_filtered))
        # print("old_FOIs_filtered:", list(map(lambda f: f.name, old_FOIs_filtered)))
        # print("new_FOIs_filtered:", list(map(lambda f: f.name, new_FOIs_filtered)))
        if len(old_FOIs_filtered) == 0 and len(new_FOIs_filtered) == 0:
            continue
        deleted = list(map(lambda x: x[0], dp['deleted']))
        added = list(map(lambda x: x[0], dp['added']))
        # 0-indexed
        try:
            old2new, new2old = find_line_correspondence(deleted, added, n_old_lines, n_new_lines)
        except AssertionError as e:
            print(e)
            print(f"[DEBUG] old_path: {old_path}, new_path: {new_path}")
            return False

        for foi in old_FOIs_filtered:
            if old2new[foi.start - 1] != -1:
                doxygen_parser_new.add_FOI(Pointer(new_path, old2new[foi.start-1]+1, foi.cwe_ids, int(not bool(foi.label))))
                # print("o-n:", (new_path, old2new[foi.start-1]+1))
        for foi in new_FOIs_filtered:
            if new2old[foi.start - 1] != -1:
                doxygen_parser_old.add_FOI(Pointer(old_path, new2old[foi.start-1]+1, foi.cwe_ids, int(not bool(foi.label))))
                # print("n-o:", (old_path, new2old[foi.start-1]+1))
    return True


def FOI_inner_merge(dox, mixed_tags):
    pass


class CallGraph:
    special_token = ' '

    def __init__(self, doxygen_parser):
        self.doxygen_parser = doxygen_parser
        self.call_graph = nx.DiGraph(self.doxygen_parser.get_reference_edges())
        self.call_graph.add_nodes_from(self.doxygen_parser.get_functions().keys())
        self.FOIs = self.doxygen_parser.get_FOIs()

        self.call_path, self.call_path_annotated = self.find_call_relations()

    def __best_shortest_path(self, caller, callee):
        assert nx.has_path(self.call_graph, caller, callee)
        # 前提 has_path
        candidates = nx.all_shortest_paths(self.call_graph, caller, callee)
        candidates = sorted(candidates, key=lambda x: CallGraph.special_token.join(x))
        candidates = sorted(candidates, key=lambda x: len(
            list(filter(lambda s: self.doxygen_parser.MemberId2Compound[s]['kind'] == DoxCompoundKind.FILE, x))),
                            reverse=True)
        return candidates[0]

    @staticmethod
    def __is_slice(a, b):
        str_a = CallGraph.special_token.join(map(str, a))
        str_b = CallGraph.special_token.join(map(str, b))
        return str_a in str_b

    def find_call_relations(self):
        if len(self.FOIs) < 2:
            return list(map(lambda x: [x], self.FOIs)), list(
                map(lambda x: [str(self.doxygen_parser.MemberId2Compound[x]['kind'])], self.FOIs))

        temp_fois = list(filter(lambda x: self.doxygen_parser.get_functions()[x].label == 1,
                                filter(self.call_graph.has_node, self.FOIs)))

        call_path = map(lambda x: self.__best_shortest_path(x[0], x[1]),
                        filter(lambda x: nx.has_path(self.call_graph, x[0], x[1]), permutations(temp_fois, 2)))
        call_path = sorted(call_path, key=lambda x: len(x))

        # 删除重合的路径，即包含关系
        remove_call_path = list(map(lambda x: x[0], filter(lambda x: self.__is_slice(*x), combinations(call_path, 2))))
        for x in remove_call_path:
            try:
                call_path.remove(x)
            except:
                pass
        used_nodes = set(chain(*call_path))
        unused_nodes = set(self.FOIs) - used_nodes
        call_path.extend(map(lambda x: [x], unused_nodes))

        FOIs = list(chain(*call_path))
        list(map(lambda f: self.doxygen_parser.add_FOI_by_id(f), FOIs))
        self.FOIs = self.doxygen_parser.get_FOIs()

        annotations = list(
            map(lambda x: list(map(lambda y: str(self.doxygen_parser.MemberId2Compound[y]['kind']), x)), call_path))

        return call_path, annotations

    def get_interested_functions_definition(self):
        # print(self.FOIs)
        return list(map(lambda i: self.doxygen_parser.get_functions()[i], self.FOIs))


def extract_call_path_with_function_definition(dox, _repo, _hash=None):
    """
    {
        "repo": "xx/xx",
        "hash": "xxx",
        "call_path": [['xx','yy'], ['zz'], ...],
        "call_path_annotation": [["class", "file"], ["file"], ...],
        "functions": {
            id: {
                "member_id": "",
                "project_path": "",
                "decl_file": "",
                "name": "",
                "start": -1,
                "end": -1,
                "code": "",
                "lan": "",
                "cwe_ids": ["", ...],
                "lois": List[int],
                "label": 0/1,
            }, ...
        }
    }
    """
    cg = CallGraph(dox)
    call_path, call_path_annotation = cg.find_call_relations()
    # print("call_path:", call_path)
    functions = cg.get_interested_functions_definition()
    output_file = DOXYGEN_OUTPUT_JSON_PATH / (f"{_repo.replace('/', '+')}+{_hash}.json" if _hash is not None else f"{_repo.replace('/', '+')}.json")
    if len(functions) == 0:
        print(f"[WARNING] {_repo} {_hash} No interested functions found!")
        with open(output_file, 'w') as f:
            json.dump({}, f)
        return False
    # print(output_file)
    with open(output_file, 'w') as f:
        json.dump({
            "repo": _repo,
            "hash": _hash,
            "call_path": call_path,
            "call_path_annotation": call_path_annotation,
            "functions": dict(map(lambda f: (f.id, f.transfer2json()), functions))
        }, f)
    return True


class CallGraphCustomized:
    special_token = ' '

    def __init__(self, doxygen_parser):
        self.doxygen_parser = doxygen_parser
        self.call_graph = nx.DiGraph(self.doxygen_parser.get_reference_edges())
        self.call_graph.add_nodes_from(self.doxygen_parser.get_functions().keys())
        self.FOIs = self.doxygen_parser.get_FOIs()

    def __find_predecessors(self, node_id):
        """前向节点"""
        predecessors = self.call_graph.predecessors(node_id)
        predecessors = list(filter(lambda predecessor: self.doxygen_parser.MemberId2Compound[predecessor]['kind'] == DoxCompoundKind.FILE, predecessors))
        return predecessors

    def __find_successors(self, node_id):
        """后向节点"""
        successors = self.call_graph.successors(node_id)
        successors = list(filter(lambda successor: self.doxygen_parser.MemberId2Compound[successor]['kind'] == DoxCompoundKind.FILE, successors))
        return successors

    def get_call_relations(self, depth=0, strategy="m"):
        assert strategy in ["s", "p", "m"]
        call_paths = []
        for foi in self.FOIs:
            call_path = [foi]

            i = 0
            while True:
                if len(call_path)-1 == depth:
                    break
                flag = False
                if (strategy == "m" and i % 2 == 0) or strategy == "p":
                    p = self.__find_predecessors(call_path[0])
                    if len(p) != 0:
                        call_path.insert(0, p[0])
                        flag = True
                elif (strategy == "m" and i % 2 == 1) or strategy == "s":
                    s = self.__find_successors(call_path[-1])
                    if len(s) != 0:
                        call_path.append(s[0])
                        flag = True
                if not flag:
                    break
                i += 1
            call_paths.append(call_path)

        annotations = list(map(lambda x: list(map(lambda y: str(self.doxygen_parser.MemberId2Compound[y]['kind']), x)), call_paths))

        return call_paths, annotations

    def get_functions_definition(self, function_ids: List[str]):
        return list(map(lambda i: self.doxygen_parser.get_functions()[i], function_ids))


def get_call_path_with_function_definition(dox, _repo, _tid=None, _depth=0, _strategy="m"):
    """
    {
        "repo": "xx/xx",
        "hash": "xxx",
        "call_path": [['xx','yy'], ['zz'], ...],
        "call_path_annotation": [["class", "file"], ["file"], ...],
        "functions": {
            id: {
                "member_id": "",
                "project_path": "",
                "decl_file": "",
                "name": "",
                "start": -1,
                "end": -1,
                "code": "",
                "lan": "",
                "cwe_ids": ["", ...],
                "lois": List[int],
                "label": 0/1,
            }, ...
        }
    }
    """
    cg = CallGraphCustomized(dox)
    call_path, call_path_annotation = cg.get_call_relations(depth=_depth, strategy=_strategy)
    # print("call_path:", call_path)
    functions = list(chain(*map(cg.get_functions_definition, call_path)))
    output_file = DOXYGEN_OUTPUT_JSON_PATH / (f"WILD+{_repo.replace('/', '+')}+{_tid}.json" if _tid is not None else f"WILD+{_repo.replace('/', '+')}.json")
    if len(functions) == 0:
        print(f"[WARNING] WILD {_repo} No interested functions assigned!")
        with open(output_file, 'w') as f:
            json.dump({}, f)
        return False
    # print(output_file)
    with open(output_file, 'w') as f:
        json.dump({
            "repo": _repo,
            "hash": _tid,
            "call_path": call_path,
            "call_path_annotation": call_path_annotation,
            "functions": dict(map(lambda f: (f.id, f.transfer2json()), functions))
        }, f)
    return True


if __name__ == '__main__':
    dox = DoxygenParser(project_path='Data/database/Wild/curl-curl-7_61_1',
                        config_path='',
                        xml_path='Data/output_doxygen_xml/curl-curl-7_61_1/xml')
    dox.set_FOIs([Pointer("lib/tftp.c", 987, ['CWE-191'], 1), Pointer("lib/tftp.c", 1109, ['CWE-191'], 1),
                  Pointer("lib/tftp.c", 994, ['CWE-191'], 1), Pointer("lib/tftp.c", 499, ['CWE-191'], 1)])
    print(dox.FOIs)
    exit(0)


    import time

    start_time = time.perf_counter()
    dox = DoxygenParser(project_path='',
                        config_path='Data/custom_configs/04906bd5de2f220bf100b605dad37b4a1d9a91a6.cf',
                        xml_path='Data/output_doxygen_xml/KDE/kde1-kdebase/04906bd5de2f220bf100b605dad37b4a1d9a91a6')
    dox.set_FOIs([Pointer("kscreensaver/saver.cpp", 92, ['CWE-191'], 1), Pointer("kscreensaver/saver.cpp", 48, ['CWE-191'], 1)])
    cg = CallGraph(dox)
    res = cg.find_call_relations()

    extract_call_path_with_function_definition(dox, 'KDE/kde1-kdebase', '04906bd5de2f220bf100b605dad37b4a1d9a91a6')

    metrics = {
        "cross": 0,
        "single": 0,
        "merge": 0,
        "origin": 0
    }
    for call_relation, annotation in zip(*res):
        if len(call_relation) > 1:
            metrics["cross"] += 1
        else:
            metrics["single"] += 1
        if len(annotation) > 1 and len(list(filter(lambda x: x != DoxCompoundKind.FILE, annotation[1:]))) == 0:
            metrics["merge"] += 1
        else:
            metrics["origin"] += len(annotation)
            # gen uuid - record function - record call relation
    end_time = time.perf_counter()
    hours, minutes, seconds = convert_runtime(start_time, end_time)
    print(f'Time elapsed to pull the data {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f} (hh:mm:ss).')

    print("call relation: {res[0]}, annotation: {res[1]}")
    print(metrics)
