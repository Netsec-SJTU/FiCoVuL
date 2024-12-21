import copy
import json
import os.path
import pickle
import shutil
import uuid

import regex
import tqdm

from codePuppeteer import FExtracter
from configure import DOXYGEN_OUTPUT_JSON_SARD_PATH, DOXYGEN_OUTPUT_RESULT_PATH, DOXYGEN_LOG_DIR_PATH

all_change = []  # (orig_depth, generated_depth)


def process_single_instance(json_file):
    global all_change

    # print(f"Processing {json_file} ...")
    with open(json_file, 'r') as f:
        content = json.load(f)

    call_paths = content['call_path']
    call_path_annotations = content['call_path_annotation']
    funcs = content['functions']

    """
    旧
    annotation = {
                "origin_path": str(file_path),
                "function_name": fer.fname,
                "class": [cls, ...],
                "label": 1,
                "roi": None,
            }
    """

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
            "label": label,  # result 添加，bool or （有 None ，不用特殊考虑）
            "roi": lois,  # result 添加，replace
        }

        fe = FExtracter(code, lan)
        fus = fe.list_all_functions()
        orig_code = code
        if len(fus) == 0:
            if regex.match(r'(\w+)::~\1|(\w+)::\1', code) or '~' in name or 'operator' in name:
                banned_id.add(k)
                continue
            return_type = v['puzzles']['return_type']
            definition = v['puzzles']['definition']
            if definition.replace(" ", "").replace("\t", "").replace("\n", "") not in code.replace(" ", "").replace("\t", "").replace("\n", ""):
                prefix = ""
                if return_type is not None:
                    prefix = definition.replace(name, '')
                code = prefix + ' ' + code
            elif name == "main":
                code = 'int  ' + code

            fe = FExtracter(code, lan)
            fus = fe.list_all_functions()

            if len(fus) == 0:
                fe.set_function_name_by_pcre()
                fus = fe.list_all_functions()

        if len(fus) > 1:
            print(f"More than one function in {project_path}/{decl_file}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return

        assert len(fus) == 1, f"\n---------fus:\n{fus}\nfile: {project_path}/{decl_file}\n----------orig_code-{lan}:\n{orig_code}----------code-{lan}:\n{code}"

        flag, is_enhanced, fer, fee = fe.get_function(fus[0][0], rois=lois)

        if flag and is_enhanced:
            functions[k] = (True, fer, fee, fe.get_lan(), annotation)  # annotation 是对 fee 的
        else:
            print(f"Not processed: {k} due to {flag} {is_enhanced}")
            functions[k] = (False, code, lan, annotation)

    # print("call path:", call_paths)
    for cp in enumerate(call_paths):
        call_path = cp[1]
        if len(set(call_path) & banned_id) > 0:
            continue
        call_path_annotation = call_path_annotations[cp[0]]
        # if len(list(filter(lambda a: a != 'file', call_path_annotation))) > 0:
        #     print("********call_path_annotation:", call_path_annotation)

        fucee = functions[call_path[-1]]
        result = functions[call_path[-1]][1].get_orig() if functions[call_path[-1]][0] else functions[call_path[-1]][1]
        annotation = copy.deepcopy(functions[call_path[-1]][-1])
        functions_record[call_path[-1]] = True

        for i in reversed(range(1, len(call_path))):
            caller_id = call_path[i - 1]
            fucer = functions[caller_id]

            if fucer[0] and fucee[0]:
                flag, result, roi = fucer[1].expand(fucee[2])
                if flag:
                    functions_record[caller_id] = True
                    annotation['function_name'] = fucer[1].fname
                    annotation['class'].extend(fucer[-1]['class'])
                    annotation['call_path'].insert(0, caller_id)
                    annotation['depth'] += 1
                    annotation['label'] = annotation['label'] or fucer[-1]['label']
                    annotation['roi'] = roi

                    fe = FExtracter(result, fucer[-2])
                    fus = fe.list_all_functions()

                    if len(fus) == 0:
                        fe.set_function_name_by_pcre()
                        fus = fe.list_all_functions()

                    assert len(fus) == 1
                    flag, is_enhanced, fer, fee = fe.get_function(fus[0][0], rois=roi)
                    if not (flag and is_enhanced):
                        print(f"3 Not processed: {fucer[1].fname} -> {annotation['function_name']} due to {flag} {is_enhanced}")
                        raise
                    else:
                        fucee = (True, fer, fee, fe.get_lan(), annotation)
                else:
                    print(f"2 Not processed: {fucer[1].fname} -> {annotation['function_name']} due to NOT MERGED")
                    print("--------------caller--------------")
                    print(fucer[1].get_orig())
                    print("--------------callee--------------")
                    print(fucee[1].get_orig())
                    print("----------------------------------")
                    raise
            else:
                print("1 Not processed: {} -> {} due to {} {}".format(fucer[1].fname, annotation['function_name'], fucer[0], fucee[0]))
                raise

        annotation['class'] = list(set(annotation['class']))

        # print('-'*20)
        # print(result)
        # print(annotation)
        # print('-'*20)

        # metrics
        # print(f"O: {len(call_path)-1}, A: {annotation['depth']}")
        all_change.append((len(call_path)-1, annotation['depth']))

        random_id = uuid.uuid4()
        with open(DOXYGEN_OUTPUT_RESULT_PATH / f"{random_id}.c", 'w') as f:
            f.write(result)
        with open(DOXYGEN_OUTPUT_RESULT_PATH / f"{random_id}.json", 'w') as f:
            json.dump(annotation, f)

    for k, v in functions_record.items():
        if not v and k not in banned_id:
            all_change.append((0, 0))

            random_id = uuid.uuid4()
            with open(DOXYGEN_OUTPUT_RESULT_PATH / f"{random_id}.c", 'w') as f:
                code = functions[k][1].get_orig() if functions[k][0] else functions[k][1]
                f.write(code)
            annotation = functions[k][-1]
            annotation['class'] = list(set(annotation['class']))
            with open(DOXYGEN_OUTPUT_RESULT_PATH / f"{random_id}.json", 'w') as f:
                json.dump(annotation, f)


if __name__ == '__main__':
    # if os.path.exists(DOXYGEN_OUTPUT_RESULT_PATH):
    #     print('remove old result')
    #     shutil.rmtree(DOXYGEN_OUTPUT_RESULT_PATH)
    files = list(DOXYGEN_OUTPUT_JSON_SARD_PATH.glob('*.json'))
    tbar = tqdm.tqdm(files)
    for file_name in tbar:
        tbar.set_description("Processing %s" % file_name)
        process_single_instance(file_name)
    pickle.dump(all_change, open(DOXYGEN_LOG_DIR_PATH / 'all_change.pkl', 'wb'))
    print('done')
