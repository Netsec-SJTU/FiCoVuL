import json
import multiprocessing
import os.path
import pickle
import re
import shutil
import sys
import time
import uuid
from itertools import chain

import regex
import tqdm

from codePuppeteer import FExtracter
from configure import DOXYGEN_OUTPUT_RESULT_WILD_PATH, DOXYGEN_LOG_DIR_PATH, DOXYGEN_OUTPUT_JSON_WILD_PATH

# all_change = []  # (orig_depth, generated_depth)
lock = multiprocessing.Lock()


def remove_comments(code):
    # 匹配 /* ... */ 风格的注释
    pattern1 = r"/\*[\s\S]*?\*/"
    code = re.sub(pattern1, lambda x: ''.join(['\n'] * x.group().count('\n')), code)

    # 匹配 // 单行注释
    pattern2 = r"//.*?$"
    code = re.sub(pattern2, "", code, flags=re.MULTILINE)

    return code


def process_single_instance(data):
    global lock
    json_file, store_result, _all_changes = data

    res_files = []
    all_change = []
    json_file = str(json_file)

    # print(f"Processing {json_file} ...")
    with open(json_file, 'r') as f:
        content = json.load(f)
    if not content:
        return json_file

    if '+good' in json_file:
        for function in content['functions'].values():
            code = function['code']
            annotation = {
                "origin_path": str(json_file),
                "function_name": function['name'],  # result 添加，replace
                "class": function['cwe_ids'],  # result 添加 extend
                "call_path": [function['member_id']],  # result 添加，extend
                "depth": 0,  # result 添加，int
                "label": 0,  # result 添加，0/1/null（不用特殊考虑）
                "roi": function['lois'],  # result 添加，replace
            }
            all_change.append((0, 0))

            random_id = uuid.uuid4()
            with open(DOXYGEN_OUTPUT_RESULT_WILD_PATH / f"{random_id}.c", 'w') as f:
                f.write(code)
            with open(DOXYGEN_OUTPUT_RESULT_WILD_PATH / f"{random_id}.json", 'w') as f:
                json.dump(annotation, f)
            res_files.append(random_id)
        return json_file


    call_paths = content['call_path']
    call_path_annotations = content['call_path_annotation']
    funcs = content['functions']
    function2annotation = dict(zip(chain(*call_paths), chain(*call_path_annotations)))

    functions = {}
    functions_record = dict(map(lambda k: (k, False), funcs.keys()))
    banned_id = set()
    for k, v in funcs.items():
        project_path = v['project_path']
        decl_file = v['decl_file']
        name = v['name']
        code = v['code']
        lan = v['lan']
        lois = v['lois']
        cwe_ids = v['cwe_ids']
        label = v['label'] if v['label'] is not None else 0

        annotation = {
            "origin_path": str(json_file),
            "function_name": name,  # result 添加，replace
            "class": cwe_ids,  # result 添加 extend
            "call_path": [k],  # result 添加，extend
            "depth": 0,  # result 添加，int
            "label": label,  # result 添加，0/1/null（不用特殊考虑）
            "roi": lois,  # result 添加，replace
        }

        code = remove_comments(code)

        if regex.match(r'(\w+)::~\1|(\w+)::\1', code) or '~' in name or 'operator' in name:
            banned_id.add(k)
            continue

        orig_code = code
        # print("---", function2annotation[k])
        fe = FExtracter(code, lan)
        fus = fe.list_all_functions()
        if len(fus) != 1 or len(list(filter(lambda n: n[2] in ['if', 'for', 'while'], fus))) > 0:
            fe.set_function_name_by_pcre()
            fus = fe.list_all_functions()

        if len(fus) != 1 or len(list(filter(lambda n: n[2] in ['if', 'for', 'while'], fus))) > 0:
            return_type = v['puzzles']['return_type']
            definition = v['puzzles']['definition']
            if definition.replace(" ", "").replace("\t", "").replace("\n", "") not in code.replace(" ", "").replace(
                    "\t", "").replace("\n", ""):
                prefix = ""
                if return_type is not None:
                    if name not in code:
                        prefix = definition
                    else:
                        prefix = definition.replace(name, '')
                code = prefix + ' ' + code
            elif 'return' in code:
                code = 'int  ' + code
            else:
                code = 'void ' + code

            fe = FExtracter(code, lan)
            fus = fe.list_all_functions()

        if len(fus) != 1 or len(list(filter(lambda n: n[2] in ['if', 'for', 'while'], fus))) > 0:
            fe.set_function_name_by_pcre()
            fus = fe.list_all_functions()

        # if len(fus) != 1:
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! len(fus) != 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     banned_id.add(k)
        #     continue

        if decl_file == "test/integration/filters/buffer_continue_filter.cc":
            banned_id.add(k)
            continue

        if len(fus) > 1:
            fus = [sorted(fus, key=lambda x: x[0])[0]]

        if len(fus) != 1:
            if function2annotation[k] != 'file':
                functions[k] = (False, orig_code, lan, annotation)
                continue
            else:
                functions[k] = (False, orig_code, lan, annotation)
                continue
            # assert len(fus) == 1, f"\n---------fus:\n{fus}\nfile: {project_path}/{decl_file}\nannotation:{function2annotation[k]}\n----------orig_code-{lan}:\n{orig_code}----------code-{lan}:\n{code}"

        # print(name, v['puzzles']['params'])
        # flag, is_enhanced, fer, fee = fe.get_function(fus[0][0], rois=lois, _params=v['puzzles']['params'])
        flag, is_enhanced, fer, fee = fe.get_function(fus[0][0], rois=lois)

        if flag and is_enhanced:
            # print('*', name, fee.params)
            # functions[k] = (True, fer, fee, v['puzzles']['params'], fe.get_lan(), annotation)  # annotation 是对 fee 的
            functions[k] = (True, fer, fee, fe.get_lan(), annotation)  # annotation 是对 fee 的
        else:
            print(f"Not processed: {name} due to {flag} {is_enhanced}")
            # print(code)
            functions[k] = (False, code, lan, annotation)

    for i in enumerate(call_paths):
        call_path = i[1]
        # print("call path:", call_path)
        if len(set(call_path) & banned_id) > 0:
            continue
        call_path_annotation = call_path_annotations[i[0]]
        if len(list(filter(lambda a: a != 'file', call_path_annotation[1:]))) > 0:
            print("******** Skip call_path_annotation:", call_path_annotation)
            continue

        fucee = functions[call_path[-1]]
        result = functions[call_path[-1]][1].get_orig() if functions[call_path[-1]][0] else functions[call_path[-1]][1]
        annotation = functions[call_path[-1]][-1].copy()
        functions_record[call_path[-1]] = True

        i = len(call_path) - 1
        while i >= 1:
            caller_id = call_path[i-1]
            fucer = functions[caller_id]

            if fucer[0] and fucee[0]:
                try:
                    flag, _result, roi = fucer[1].expand(fucee[2])
                except IndexError as e:
                    print(f"caller:\n{fucer[1].get_orig()}\ncallee:\n{fucee[1].get_orig()}")
                    flag = False
                except TypeError as e:
                    print(f"caller:\n{fucer[1].get_orig()}\ncallee:\n{fucee[1].get_orig()}")
                    print(fucee[2].params)
                    flag = False
                if flag:
                    result = _result
                    functions_record[caller_id] = True
                    annotation['function_name'] = fucer[1].fname
                    annotation['class'].extend(fucer[-1]['class'])
                    annotation['call_path'].insert(0, caller_id)
                    annotation['depth'] += 1
                    annotation['label'] = annotation['label'] or fucer[-1]['label']
                    annotation['roi'] = roi

                    fe = FExtracter(result, fucer[-2])
                    fus = fe.list_all_functions()

                    if len(fus) != 1:
                        fe.set_function_name_by_pcre()
                        fus = fe.list_all_functions()

                    if len(fus) != 1:
                        break

                    # assert len(fus) == 1, f"\n---------fus:\n{fus}\nname: {fucer[1].fname}\n----------code-{fucer[-2]}:\n{result}"
                    flag, is_enhanced, fer, fee = fe.get_function(fus[0][0], rois=roi)
                    # flag, is_enhanced, fer, fee = fe.get_function(fus[0][0], rois=roi, _params=fucer[-3])
                    if not (flag and is_enhanced):
                        print(f"3 Not processed: {fucer[1].fname} -> {annotation['function_name']} due to {flag} {is_enhanced}")
                        raise
                    else:
                        fucee = (True, fer, fee, fe.get_lan(), annotation)
                else:
                    print(f"2 Not processed: {fucer[1].fname} -> {annotation['function_name']} due to NOT MERGED")
                    # print("--------------caller--------------")
                    # print(fucer[1].get_orig())
                    # print("--------------callee--------------")
                    # print(fucee[1].get_orig())
                    # print("----------------------------------")
                    # print(call_path)
                    # print("2 Not processed----------------------------------")
                    break
            else:
                print("1 Not processed: {} -> {} due to {} {}".format(fucer[-1]['function_name'], annotation['function_name'], fucer[0], fucee[0]))
                break
            i -= 1

        annotation['class'] = list(set(annotation['class']))

        # print('-'*20)
        # print(result)
        # print(annotation)
        # print('-'*20)

        # metrics
        # print(f"O: {len(call_path)-1}, A: {annotation['depth']}")
        all_change.append((len(call_path)-1, annotation['depth']))

        random_id = uuid.uuid4()
        with open(DOXYGEN_OUTPUT_RESULT_WILD_PATH / f"{random_id}.c", 'w') as f:
            f.write(result)
        with open(DOXYGEN_OUTPUT_RESULT_WILD_PATH / f"{random_id}.json", 'w') as f:
            json.dump(annotation, f)
        res_files.append(random_id)

    for k, v in functions_record.items():
        if not v and k not in banned_id:
            annotation = functions[k][-1]
            if annotation['label'] is None:
                continue

            all_change.append((0, 0))

            random_id = uuid.uuid4()
            with open(DOXYGEN_OUTPUT_RESULT_WILD_PATH / f"{random_id}.c", 'w') as f:
                code = functions[k][1].get_orig() if functions[k][0] else functions[k][1]
                f.write(code)
            annotation['class'] = list(set(annotation['class']))
            with open(DOXYGEN_OUTPUT_RESULT_WILD_PATH / f"{random_id}.json", 'w') as f:
                json.dump(annotation, f)
            res_files.append(random_id)

    lock.acquire()
    store_result[json_file] = list(map(str, res_files))
    _all_changes.extend(all_change)
    lock.release()

    return json_file


def error_callback(error):
    print(f"Error info: {error}")


if __name__ == '__main__':
    files = list(DOXYGEN_OUTPUT_JSON_WILD_PATH.glob('*.json'))
    content = []
    all_change = []
    if os.path.exists(DOXYGEN_LOG_DIR_PATH / 'FF_WILD.json'):
        with open(DOXYGEN_LOG_DIR_PATH / 'FF_WILD.json', 'r') as f:
            content = json.load(f)
        finished_files = content.keys()
        files = list(filter(lambda x: x not in finished_files, files))
    if os.path.exists(DOXYGEN_LOG_DIR_PATH / 'WILD_all_change.pkl'):
        with open(DOXYGEN_LOG_DIR_PATH / 'WILD_all_change.pkl', 'rb') as f:
            all_change = pickle.load(f)
    # record = {}
    # tbar = tqdm.tqdm(files)
    # for f in tbar:
    #     tbar.set_description("Processing %s" % f)
    #     if f in ['Data/output_doxygen_json/CVE/vim+vim+9c23f9bb5fe435b28245ba8ac65aa0ca6b902c04.json']:
    #         continue
    #     _, res = process_single_instance(f)
    #     record[str(f)] = res
        # with open(f'{DOXYGEN_LOG_DIR_PATH}/FF_SARD.json', 'w') as fp:
        #     json.dump(record, fp)
    manager = multiprocessing.Manager()
    record = manager.dict()
    record.update(content)
    all_changes = manager.list()
    all_changes.extend(all_change)
    args = [(f, record, all_changes) for f in files]

    # pbar = tqdm.tqdm(args, desc="Process")
    # for arg in pbar:
    #     pbar.set_description("Processing %s" % arg[0])
    #     process_single_instance(arg)

    with multiprocessing.Pool(40) as pool:
        pbar = tqdm.tqdm(pool.imap_unordered(process_single_instance, args), total=len(files), desc="Process")
        for f in pbar:
            pbar.refresh()

    with open(f'{DOXYGEN_LOG_DIR_PATH}/FF_WILD.json', 'w') as f:
        json.dump(dict(record), f)
    with open(f'{DOXYGEN_LOG_DIR_PATH}/WILD_all_change.pkl', 'wb') as f:
        pickle.dump(list(all_changes), f)
    print('done')
