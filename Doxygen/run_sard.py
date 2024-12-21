"""
===========================================================================
||       SARD_unsorted from https://samate.nist.gov/SARD/search.php       ||
===========================================================================
"""
import glob
import os
from itertools import chain
from multiprocessing import Pool
from typing import List
import tqdm
import xml.dom.minidom
import re
import chardet

from utils import run_cmd, clean_file
from configure import SARD_PATH, SARD_TESTCASES_PATH, modify_doxygen_config_sard, DOXYGEN_CONFIG_DIR_PATH, DOXYGEN_OUTPUT_XML_PATH, DOXYGEN_OUTPUT_JSON_PATH
from postprocess import DoxygenParser, Pointer, extract_call_path_with_function_definition


def extract_full_manifest():
    print(f"Reading xml file from {SARD_PATH}...", end=" ")
    DOMTree = xml.dom.minidom.parse(str(SARD_PATH / "full_manifest.xml"))
    Data = DOMTree.documentElement
    testcases = Data.getElementsByTagName("testcase")
    filtered_testcases = list(filter(lambda testcase: testcase.getAttribute("id") not in ['199276', '199286', '199290', '199296'] and
                                                      testcase.getAttribute("type") == "Source Code" and
                                                      testcase.getAttribute("status") != "Deprecated" and
                                                      testcase.getAttribute("language").lower() in ("c", "c++"),
                                     testcases))
    print("Done.")
    return filtered_testcases


def locate_good_bad_location(testcase):
    # ['file(relative_path)': [[start, end], [start, end], ...], ...], ['file(relative_path)': [[start, end], [start, end], ...], ...]
    def find_lines_with_directive(relative_filename, directive):
        filename = str(SARD_TESTCASES_PATH / relative_filename)
        try:
            with open(filename, 'r') as fp:
                lines = fp.readlines()
        except UnicodeDecodeError as e:
            # 一些编码格式不是 utf-8 ，因此先判断编码格式
            with open(filename, 'rb') as fp:
                cur_encoding = chardet.detect(fp.read())['encoding']
            with open(filename, encoding=cur_encoding) as fp:
                lines = fp.readlines()
        # 1-indexed
        matching_lines = list(map(lambda line: line[0], filter(lambda line: directive in line[1], enumerate(lines, start=1))))
        return matching_lines

    def locate_good_bad_location_per_file(file_path):
        GOOD_START = find_lines_with_directive(file_path, r'#ifndef OMITGOOD')
        GOOD_END = find_lines_with_directive(file_path, r'#endif /* OMITGOOD */')
        BAD_START = find_lines_with_directive(file_path, r'#ifndef OMITBAD')
        BAD_END = find_lines_with_directive(file_path, r'#endif /* OMITBAD */')
        assert len(GOOD_START) == len(GOOD_END)
        assert len(BAD_START) == len(BAD_END)
        return {
            "GOOD": list(zip(GOOD_START, GOOD_END)),
            "BAD": list(zip(BAD_START, BAD_END))
        }

    # 闭区间
    # {'file(relative_path)': {'GOOD': [[start, end], [start, end], ...], 'BAD': [[start, end], [start, end], ...]}, ...}
    temp = dict(map(lambda f: ('/'.join(f.getAttribute("path").split('/')[3:]), locate_good_bad_location_per_file(f.getAttribute("path"))),
                    filter(lambda f: f.getAttribute("path")[:7] != "shared/", testcase.getElementsByTagName("file"))))
    return dict(map(lambda item: (item[0], item[1]["GOOD"]), temp.items())), dict(map(lambda item: (item[0], item[1]["BAD"]), temp.items()))


def predoxygen_repo(repo):
    repo = '/'.join(repo.split('/')[-2:])
    config_path, output_path = modify_doxygen_config_sard(repo)
    if os.path.exists(output_path):
        return repo
    code, msg = run_cmd(f"doxygen {config_path}")
    if code != 0:
        raise Exception(f"doxygen parse error: {msg}")
    clean_file(rf"{output_path}/**/*.xml")
    return repo


def predoxygen_repos():
    print("Pre-doxygening...", end=" ")

    repos = list(filter(os.path.isdir, glob.glob(rf"{SARD_TESTCASES_PATH}/SATE/*"))) + list(filter(os.path.isdir, glob.glob(rf"{SARD_TESTCASES_PATH}/shared/*")))
    with Pool(20) as pool:
        pbar = tqdm.tqdm(total=len(repos))

        def update_progress(result):
            # Callback function to update the progress bar
            pbar.update(1)
            pbar.set_description(f"Finish repo_url {result}")

        # Apply the function to each repository in parallel
        for repo_url in repos:
            pool.apply_async(predoxygen_repo, args=(repo_url,), callback=update_progress)

        # Wait for all tasks to complete
        pool.close()
        pool.join()

        # Close the progress bar
        pbar.close()
    print("Done.")


def classify_functions(dox: DoxygenParser):
    fois = list(map(lambda item: item[0],
                    filter(lambda item: "good" in item[1].name or "bad" in item[1].name, dox.get_functions().items())))
    dox.FOIs.extend(fois)


def process_single_testcase(testcase, CWEs_of_interest: List[str]):
    interested_files = list(filter(lambda f: f.getAttribute("path")[:7] != "shared/" and
                                             len(f.getElementsByTagName("*")) != 0,
                                   testcase.getElementsByTagName("file")))
    tags = list(chain(*map(lambda f: f.getElementsByTagName("*"), interested_files)))
    tags = list(map(lambda tag: tag.tagName, tags))

    if len(tags) == 0:
        return False

    temp = list(filter(lambda p: p[:7] != "shared/",
                       map(lambda f: f.getAttribute("path"), testcase.getElementsByTagName("file"))))
    temp2 = list(map(lambda f: f.getAttribute("path"), testcase.getElementsByTagName("file")))
    if len(temp) > 0 and temp[0].split('/')[0] == "000":
        repo_prefix_n = 3
    else:
        repo_prefix_n = 2
    if len(temp) > 0:
        repo = '/'.join(temp[0].split('/')[:repo_prefix_n])
    else:
        repo = '/'.join(temp2[0].split('/')[:repo_prefix_n])

    is_interested = False
    all_tags = list(chain(*map(lambda f: list(map(lambda t: t.tagName, f.getElementsByTagName("*"))),
                               testcase.getElementsByTagName("file"))))
    LOIs = {}  # {'file(relative_path)': [['mixed', 'cwe_id', line:int], ...], ...}
    for f in testcase.getElementsByTagName("file"):
        file_path = f.getAttribute("path")
        tags = f.getElementsByTagName("*")
        for t in tags:
            searchObj = re.search(r"(?P<cwe_id>^CWE-[0-9]{1,3}):", t.getAttribute("name"))
            if searchObj is None:
                continue
            cwe_id = searchObj.group("cwe_id")
            # print("cwe_id: ", cwe_id)
            if cwe_id not in CWEs_of_interest:
                continue
                # print("cwe_id not in CWEs_of_interest")
            is_interested = True
            line = int(t.getAttribute("line"))
            # print("line: ", line)
            k = '/'.join(file_path.split('/')[repo_prefix_n:])
            if file_path not in LOIs:
                LOIs[k] = []
            LOIs[k].append([t.tagName, cwe_id, line])
    # print(LOIs)

    if not is_interested:
        return False

    if 'mixed' in all_tags:
        return process_mixed(repo, LOIs, repo_prefix_n == 3)
    else:
        return process_flaw_fix(repo, LOIs, repo_prefix_n == 3)


def process_flaw_fix(repo, LOIs, is_000=False):
    # print("Flaw-fix testcase")

    config_path, output_path = modify_doxygen_config_sard(repo)
    dox = DoxygenParser(project_path=SARD_TESTCASES_PATH / repo,
                        config_path=config_path,
                        xml_path=output_path)
    # print("References:", dox.get_references())

    if is_000:
        classify_functions(dox)

    dox.set_FOIs(list(chain(
        *map(lambda item: list(map(lambda an: Pointer(item[0], an[2], [an[1]], 1 if an[0] != 'fix' else 0), item[1])),
             LOIs.items()))))

    # print('+' * 10)
    # print("Functions:", list(map(lambda x: dox.get_functions()[x].name, dox.get_FOIs())))

    if len(dox.get_FOIs()) == 0:
        return False

    extract_call_path_with_function_definition(dox, repo)

    return True


def process_mixed(repo, LOIs, is_000=False):
    # print("Mixed testcase")

    config_path, output_path = modify_doxygen_config_sard(repo, macros="OMITBAD", tag="good")
    dox_good = DoxygenParser(project_path=SARD_TESTCASES_PATH / repo,
                             config_path=config_path,
                             xml_path=output_path)
    config_path, output_path = modify_doxygen_config_sard(repo, macros="OMITGOOD", tag="bad")
    dox_bad = DoxygenParser(project_path=SARD_TESTCASES_PATH / repo,
                            config_path=config_path,
                            xml_path=output_path)

    if is_000:
        classify_functions(dox_good)
        classify_functions(dox_bad)

    dox_good.set_FOIs(list(chain(
        *map(lambda item: list(map(lambda an: Pointer(item[0], an[2], [an[1]], 1 if an[0] != 'fix' else 0), item[1])),
             LOIs.items()))))
    dox_bad.set_FOIs(list(chain(
        *map(lambda item: list(map(lambda an: Pointer(item[0], an[2], [an[1]], 1 if an[0] != 'fix' else 0), item[1])),
             LOIs.items()))))

    # print('+' * 10)
    # print("Functions:", list(map(lambda x: dox_good.get_functions()[x].name, dox_good.get_FOIs())))
    # print("Functions:", list(map(lambda x: dox_bad.get_functions()[x].name, dox_bad.get_FOIs())))

    if len(dox_good.get_FOIs()) == 0 and len(dox_bad.get_FOIs()) == 0:
        return False

    extract_call_path_with_function_definition(dox_good, repo, "good")
    extract_call_path_with_function_definition(dox_bad, repo, "bad")

    return True


if __name__ == "__main__":
    CWE_of_interest = ['CWE-787', 'CWE-125', 'CWE-020', 'CWE-078', 'CWE-416', 'CWE-022', 'CWE-190', 'CWE-476', 'CWE-119',
                       'CWE-200', 'CWE-732', 'CWE-362', 'CWE-400', 'CWE-120', 'CWE-415', 'CWE-059', 'CWE-617', 'CWE-772',
                       'CWE-295', 'CWE-401', 'CWE-284', 'CWE-369', 'CWE-191', 'CWE-763', 'CWE-835', 'CWE-674']

    # for testcase in filter(lambda x: x.getAttribute("id") == "2278", extract_full_manifest()):
    #     process_single_testcase((testcase, CWE_of_interest))

    predoxygen_repos()

    testcases = extract_full_manifest()

    # # 创建进度条
    # pbar = tqdm.tqdm(total=len(testcases))
    #
    # # 定义任务完成时的回调函数
    # def update_progress(future):
    #     pbar.update(1)
    #     pbar.set_description(f"Finish repo_url {future.result()}")
    #
    # # 创建进程池
    # with ProcessPoolExecutor(max_workers=32) as executor:
    #     futures = []
    #
    #     # 提交任务到进程池
    #     for testcase in testcases:
    #         future = executor.submit(process_single_testcase, testcase, CWE_of_interest)
    #         future.add_done_callback(update_progress)
    #         futures.append(future)
    #
    #     # 等待所有任务完成
    #     for future in as_completed(futures):
    #         pass
    #
    # # 关闭进度条
    # pbar.close()

    store_cnt = 0

    tbar = tqdm.tqdm(testcases)
    for testcase in tbar:
        tbar.set_description(f"Processing testcase {testcase.getAttribute('id')}")
        res = process_single_testcase(testcase, CWE_of_interest)
        if res:
            store_cnt += 1
        tbar.set_postfix({"store": store_cnt})
