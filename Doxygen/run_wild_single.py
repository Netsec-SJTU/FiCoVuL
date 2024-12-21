import glob
import multiprocessing
import os.path
import random
import shutil

import tqdm
import json

from configure import WILD_PATH, modify_doxygen_config_wild, DOXYGEN_OUTPUT_JSON_PATH
from postprocess import DoxygenParser, Pointer, extract_call_path_with_function_definition, \
    get_call_path_with_function_definition

Annotation2Cwe_Map = {
    # "stack_overflow": "CWE-121",
    # "heap_overflow": "CWE-122",
    "overflowed_call": "CWE-190",
    "underflowed_call": "CWE-191",
    "second_free": "CWE-415",
    "use_after_free": "CWE-416"
}


def worker(project_path):
    project_id = project_path.split('/')[-1]
    if not os.path.exists(f"{project_path}/labels.json"):
        print(f"[ERROR] {project_id} label.json not found!")
        return project_id
    if os.path.exists(DOXYGEN_OUTPUT_JSON_PATH / (f"WILD+{project_id.replace('/', '+')}+bad.json")) and \
        os.path.exists(DOXYGEN_OUTPUT_JSON_PATH / (f"WILD+{project_id.replace('/', '+')}+good.json")):
        return project_id

    config_path, output_path = modify_doxygen_config_wild(project_id)
    dox = DoxygenParser(project_path=WILD_PATH / project_id,
                        config_path=config_path,
                        xml_path=output_path)

    with open(f"{project_path}/labels.json", 'r') as f:
        labels = json.load(f)
    cwe_id = None
    for label in labels:
        if label['label'] in Annotation2Cwe_Map:
            cwe_id = Annotation2Cwe_Map[label['label']]
            break
    if cwe_id is None:
        return project_id
    pointers = []
    for label in labels:
        pointers.append(Pointer(label['filename'], int(label['line_number']), [cwe_id], 1))
    dox.set_FOIs(pointers)
    print(f"*********************{len(pointers)}")
    extract_call_path_with_function_definition(dox, project_id, rf"bad")
    output_file = DOXYGEN_OUTPUT_JSON_PATH / (f"{project_id.replace('/', '+')}+bad.json")
    output_file_rename = DOXYGEN_OUTPUT_JSON_PATH / (f"WILD+{project_id.replace('/', '+')}+bad.json")
    shutil.move(output_file, output_file_rename)

    totalset = set(filter(lambda k: dox.MemberId2Compound[k]['kind'] == 'file', dox.get_functions().keys()))
    blackset = set(dox.FOIs)
    candidiates = totalset - blackset
    if len(candidiates) == 0:
        return project_id
    dox.FOIs = candidiates
    for foi in dox.FOIs:
        dox.Id2Function[foi].label = 0

    output_file = DOXYGEN_OUTPUT_JSON_PATH / (f"WILD+{project_id.replace('/', '+')}+good.json")
    with open(output_file, 'w') as f:
        json.dump({
            "repo": project_id,
            "hash": "bad",
            "call_path": list(map(lambda x: [x], dox.get_FOIs())),
            "call_path_annotation": list(map(lambda x: [str(dox.MemberId2Compound[x]['kind'])], dox.get_FOIs())),
            "functions": dict(map(lambda f: (dox.Id2Function[f].id, dox.Id2Function[f].transfer2json()), dox.get_FOIs()))
        }, f)

    return project_id


if __name__ == '__main__':
    args = list(glob.glob(f"{WILD_PATH}/*"))
    with multiprocessing.Pool(20) as pool:
        pbar = tqdm.tqdm(pool.imap_unordered(worker, args), total=len(args), desc="Process")
        for f in pbar:
            pbar.refresh()
