import glob
import multiprocessing
import os.path
import random
import shutil
from itertools import chain

import tqdm
import json

from configure import WILD_PATH, modify_doxygen_config_wild, DOXYGEN_OUTPUT_JSON_PATH
from postprocess import DoxygenParser, Pointer, extract_call_path_with_function_definition, \
    get_call_path_with_function_definition

Annotation2Cwe_Map = {
    "stack_overflow": "CWE-121",
    "heap_overflow": "CWE-122",
    "overflowed_call": "CWE-190",
    "underflowed_call": "CWE-191",
    "second_free": "CWE-415",
    "use_after_free": "CWE-416"
}

interested_projects_classification = {
    'CWE-190': [
        'curl-curl-7_56_1',
        'curl-curl-7_64_1',
        'libcurl-curl-7_61_0',
        'libcurl-curl-7_63_0',
        'libgit2-v0.26.1',
        'libgit2-v0.27.2',
        'libjpeg-turbo-2.0.1',
        'libtiff-v4.0.10',
        'libvncserver-LibVNCServer-0.9.11',
        'sound_exchange-sox-14.4.2'
    ],
    'CWE-121': [
        'poppler-poppler-0.55',
        'sound_exchange-sox-14.4.2'
    ],
    'CWE-122': [
        'curl-curl-7_61_1',
        'graphite2-1.3.5',
        'jasper-version-2.0.22',
        'sound_exchange-sox-14.4.2'
    ],
    'CWE-415': [
        'jasper-version-2.0.11',
        'openjpeg-v2.3.1'
    ],
    'CWE-416': [
        'jasper-version-2.0.11',
        'libical-v1.0.0',
        'libical-v2.0.0',
        'openexr-v2.5.1',
        'openjpeg-v2.3.1',
        'sound_exchange-sox-14.4.2'
    ]
}

root_cause = ['declared_buffer','overflowed_variable','underflowed_variable','first_free','freed_variable']
manifestation = ['stack_overflow','heap_overflow','overflowed_call','underflowed_call','second_free','use_after_free']

# interested_projects = list(set(chain(*interested_projects_classification.values())))
interested_projects = list(map(lambda x: x.split('/')[-2], glob.glob(f"{WILD_PATH}/*/labels.json")))


def worker(project_id):
    project_path = WILD_PATH / project_id
    assert os.path.exists(f"{project_path}/labels.json"), f"[ERROR] {project_id} labels.json not found!"

    config_path, output_path = modify_doxygen_config_wild(project_id)
    dox = DoxygenParser(project_path=WILD_PATH / project_id,
                        config_path=config_path,
                        xml_path=output_path)

    with open(f"{project_path}/labels.json", 'r') as f:
        labels = json.load(f)

    accumulated_fois = []
    cnt = 0
    depth = 0
    cwe_ids = []
    pointers = []
    for label in reversed(labels):
        if label['label'] in root_cause:
            if len(cwe_ids) == 0:
                continue
            pointers.append(Pointer(label['filename'], int(label['line_number']), cwe_ids, 1))
            dox.set_FOIs(pointers)
            extract_call_path_with_function_definition(dox, project_id, rf"bad")
            output_file = DOXYGEN_OUTPUT_JSON_PATH / (f"{project_id.replace('/', '+')}+bad.json")
            output_file_rename = DOXYGEN_OUTPUT_JSON_PATH / (f"WILD+{project_id.replace('/', '+')}+bad+{cnt}.json")
            shutil.move(output_file, output_file_rename)
            cnt += 1
            with open(output_file_rename, 'r') as fp:
                data = json.load(fp)
            _depth = max(map(lambda x: len(x) - 1, data['call_path']))
            if _depth > depth:
                depth = _depth
            accumulated_fois.extend(dox.get_FOIs())
            dox.clear_FOIs()
            cwe_ids.clear()
            pointers.clear()
        elif label['label'] in manifestation:
            cwe_id = Annotation2Cwe_Map[label['label']]
            if cwe_id not in interested_projects_classification:
                continue
            cwe_ids.append(cwe_id)
            pointers.append(Pointer(label['filename'], int(label['line_number']), [cwe_id], 1))
        else:
            raise Exception(f"Unknown label: {label['label']}")

    all_functions = dox.get_functions()
    accumulated_negative_samples = set()
    for label in labels:
        ifp = '/'.join(label['filename'].split('/')[:-1])
        # print("ifp: ", ifp)
        # print(list(map(lambda k: all_functions[k].decl_file, all_functions.keys())))
        negative_samples = list(filter(
            lambda k: all_functions[k].decl_file.startswith(ifp) and all_functions[k].decl_file.count(
                '/') - 1 == ifp.count('/'), all_functions.keys()))
        accumulated_negative_samples.update(negative_samples)
    accumulated_negative_samples = accumulated_negative_samples - set(accumulated_fois)
    for foi in accumulated_negative_samples:
        dox.add_FOI_by_id(foi)
        all_functions[foi].label = 0
    get_call_path_with_function_definition(dox, project_id, rf"good", _depth=depth)

    return project_id


if __name__ == '__main__':
    with multiprocessing.Pool(30) as pool:
        pbar = tqdm.tqdm(pool.imap_unordered(worker, interested_projects), total=len(interested_projects), desc="Process")
        for f in pbar:
            pbar.refresh()
