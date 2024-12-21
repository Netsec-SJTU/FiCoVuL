import glob
import os
import random
import sys
import tqdm
import time
import chardet
import pandas as pd
import multiprocessing
from itertools import chain

from cve_database import group_by_repo_url_generator, group_by_hash_generator, CWE_top26_comprehensive
from configure import modify_doxygen_config_cve, CVE_PATH, CVE_REPO_PATH, DOXYGEN_OUTPUT_JSON_PATH, DOXYGEN_LOG_DIR_PATH
from postprocess import Pointer, DoxygenParser, FOI_outer_merge, extract_call_path_with_function_definition
from utils import git_reset, delete_large_files, convert_to_utf8, run_cmd, clean_file, convert_runtime, LogHandler

lock = multiprocessing.Lock()


def get_diff_parsed(df_data):
    # [old_path, [line1, line2, ...]]
    diff_parsed_before = list(
        map(lambda row: [row.old_path, row.cwe_id, list(map(lambda x: x[0], eval(row.diff_parsed)["deleted"]))],
            df_data.itertuples(index=False)))
    diff_parsed_after = list(
        map(lambda row: [row.new_path, row.cwe_id, list(map(lambda x: x[0], eval(row.diff_parsed)["added"]))],
            df_data.itertuples(index=False)))
    return diff_parsed_before, diff_parsed_after


def doxygen_pairwise(repo, _hash, df_data, input_encoding):
    before_hash = rf"{_hash}^"
    after_hash = _hash

    # print("1", time.time())
    git_reset(repo, before_hash)
    config_path, output_path = modify_doxygen_config_cve(repo, before_hash, input_encoding)
    # delete_large_files(CVE_REPO_PATH / repo)
    doxgen_before = DoxygenParser(project_path=CVE_REPO_PATH / repo, config_path=config_path, xml_path=output_path)

    # print("2", time.time())
    git_reset(repo, after_hash)
    config_path, output_path = modify_doxygen_config_cve(repo, after_hash, input_encoding)
    # delete_large_files(CVE_REPO_PATH / repo)
    doxgen_after = DoxygenParser(project_path=CVE_REPO_PATH / repo, config_path=config_path, xml_path=output_path)

    # print("3", time.time())

    # diff_parsed
    diff_parsed_before, diff_parsed_after = get_diff_parsed(df_data)  # [[path, cwe_id, [line1, line2, ...]], ...]
    doxgen_before.set_FOIs(
        list(chain(*map(lambda x: list(map(lambda y: Pointer(x[0], y, [x[1]], 1), x[2])), diff_parsed_before))))
    doxgen_after.set_FOIs(
        list(chain(*map(lambda x: list(map(lambda y: Pointer(x[0], y, [x[1]], 0), x[2])), diff_parsed_after))))

    # print('-'*10)
    # print("before:", list(map(lambda x: doxgen_before.get_functions()[x].name, doxgen_before.get_FOIs())))
    # print("after:", list(map(lambda x: doxgen_after.get_functions()[x].name, doxgen_after.get_FOIs())))

    # pairwise foi
    diff_parsed = list(map(lambda x: (x.old_path, x.new_path, len(x.code_before.splitlines()) if x.code_before != "None" else 0, len(x.code_after.splitlines()) if x.code_after != "None" else 0, eval(x.diff_parsed)),
                           df_data.itertuples(index=False)))
    res = FOI_outer_merge(doxgen_before, doxgen_after, diff_parsed)
    if not res:
        print(f"[ERROR] {repo} {_hash} FOI_outer_merge error")
        return None, None

    # print('+'*10)
    # print("before:", doxgen_before.get_functions().keys())
    # print("after:", doxgen_after.get_functions().keys())
    # print("before FOIs:", list(map(lambda x: doxgen_before.get_functions()[x].name, doxgen_before.get_FOIs())))
    # print("after FOIs:", list(map(lambda x: doxgen_after.get_functions()[x].name, doxgen_after.get_FOIs())))

    return doxgen_before, doxgen_after


def dxoygen_worker(data):
    repo, group_data = data
    total_length = len(group_data.hash.unique())
    finished = 0
    for _hash, df_data in group_by_hash_generator(group_data):
        project_path = CVE_REPO_PATH / repo
        before_hash = rf"{_hash}^"
        after_hash = _hash

        for _h in (before_hash, after_hash):
            config_path, output_path = modify_doxygen_config_cve(repo, before_hash, 'UTF-8')
            if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
                continue

            git_reset(repo, before_hash)
            # delete_large_files(project_path)
            for filename in filter(os.path.isfile, glob.glob(rf"{project_path}/**/*", recursive=True)):
                try:
                    convert_to_utf8(filename)
                except:
                    pass

            code, msg = run_cmd(f"doxygen {config_path}")
            if code != 0:
                raise Exception(f"[ERROR] doxygen parse error. Msg: {msg}\n{repo} {_hash}")
            clean_file(output_path)

        finished += 1
        print(f"Finished [{finished}/{total_length}] in {repo}")
        sys.stdout.flush()
    return repo


def cve_worker(repo_url, group_data, total_finished, total_length):
    # repo_url, group_data, total_finished, total_length = data

    # detect encoding
    input_encoding = 'utf-8'
    # 只能用 for
    repo_finished = 0
    repo_length = len(group_data.hash.unique())
    log_file = f"{DOXYGEN_LOG_DIR_PATH / repo_url.replace('/', '+')}.txt"
    logger = LogHandler(log_file)
    fp_flag = False
    for _hash, df_data in group_by_hash_generator(group_data):
        output_file_before = DOXYGEN_OUTPUT_JSON_PATH / f"{repo_url.replace('/', '+')}+{_hash}^.json"
        output_file_after = DOXYGEN_OUTPUT_JSON_PATH / f"{repo_url.replace('/', '+')}+{_hash}.json"
        if os.path.exists(output_file_before) and os.path.exists(output_file_after):
            repo_finished += 1
            lock.acquire()
            total_finished.value += 1
            lock.release()
            continue
        try:
            time1 = time.perf_counter()
            lock.acquire()
            print(f"[INFO] [{repo_url} {_hash}] | [Begin] | [{repo_length - repo_finished}/{repo_length}]")
            sys.stdout.flush()
            lock.release()
            logger.log(f"[INFO] [{repo_url} {_hash}] | [Begin] | [{repo_length - repo_finished}/{repo_length}]\n")

            doxgen_before, doxgen_after = doxygen_pairwise(repo_url, _hash, df_data, input_encoding)
            if doxgen_before is None or doxgen_after is None:
                repo_finished += 1
                lock.acquire()
                total_finished.value += 1
                print(f"[WARNING] Skipped: [{repo_url} {_hash}], remaining {total_length.value - total_finished.value}/{total_length.value}\n       Repo remaining: [{repo_length-repo_finished}/{repo_length}]")
                sys.stdout.flush()
                lock.release()
                logger.log(f"[WARNING] Skipped: [{repo_url} {_hash}], remaining {total_length.value - total_finished.value}/{total_length.value}\n       Repo remaining: [{repo_length-repo_finished}/{repo_length}]\n")
                fp_flag = True
                continue

            time2 = time.perf_counter()
            hours, minutes, seconds = convert_runtime(time1, time2)
            lock.acquire()
            print(f"[LOG] [{repo_url} {_hash}] | [Doxygen] | Time for previous stage {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f} | [{repo_length - repo_finished}/{repo_length}]")
            sys.stdout.flush()
            lock.release()
            logger.log(f"[LOG] [{repo_url} {_hash}] | [Doxygen] | Time for previous stage {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f} | [{repo_length - repo_finished}/{repo_length}]\n")

            extract_call_path_with_function_definition(doxgen_before, repo_url, rf"{_hash}^")
            del doxgen_before

            time3 = time.perf_counter()
            hours, minutes, seconds = convert_runtime(time2, time3)
            lock.acquire()
            print(f"[LOG] [{repo_url} {_hash}] | [CG1] | Time for previous stage {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f} | [{repo_length - repo_finished}/{repo_length}]")
            sys.stdout.flush()
            lock.release()
            logger.log(f"[LOG] [{repo_url} {_hash}] | [CG1] | Time for previous stage {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f} | [{repo_length - repo_finished}/{repo_length}]\n")

            extract_call_path_with_function_definition(doxgen_after, repo_url, _hash)
            del doxgen_after

            time4 = time.perf_counter()
            lock.acquire()
            hours, minutes, seconds = convert_runtime(time3, time4)
            print(f"[LOG] [{repo_url} {_hash}] | [CG2] | Time for previous stage {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f} | [{repo_length - repo_finished}/{repo_length}]")
            repo_finished += 1
            total_finished.value += 1
            hours, minutes, seconds = convert_runtime(time1, time4)
            print(f"[INFO] [{repo_url} {_hash}] | [Finish] | Total time elapsed {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}, remaining {total_length.value - total_finished.value}/{total_length.value}\n       Repo remaining: [{repo_length-repo_finished}/{repo_length}]")
            sys.stdout.flush()
            lock.release()
            logger.log(f"[LOG] [{repo_url} {_hash}] | [CG2 -> Finish] | Time for previous stage {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f} | [{repo_length - repo_finished}/{repo_length}]\n")
            logger.log(f"[INFO] [{repo_url} {_hash}] | [Finish] | Total time elapsed {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}, remaining {total_length.value - total_finished.value}/{total_length.value}\n       Repo remaining: [{repo_length-repo_finished}/{repo_length}]\n")
        except Exception as e:
            print(e)
            print("Exception:", repo_url, _hash)
            logger.close()
            os.rename(log_file, "?"+log_file)
            try:
                lock.release()
            except:
                pass
            raise Exception()
    logger.close()
    if not fp_flag:
        os.remove(log_file)
    return repo_url


def error_callback(error):
    print(f"Error info: {error}")


# TODO: 统计数据
if __name__ == '__main__':
    # df_all = get_joint_tables()
    # repos = group_by_repo_url_generator(df_all)

    # # 定义一个进程池，最大进程数 32
    # with Pool(32) as pool:
    #     pbar = tqdm.tqdm(pool.imap_unordered(cve_worker, repos), total=len(repos))
    #     for id in pbar:
    #         pbar.set_description(f"Finish repo_url {id}")

    # use merge=False
    # repo = 'KDE/kde1-kdebase'
    # hashes = ['04906bd5de2f220bf100b605dad37b4a1d9a91a6']

    # from pydriller import Repository
    # repo = 'test/test'
    # hashs = ["0264779e41910afbc49647e572c2c9e10e4ef338", "745db9c42d326c3b100466ffdf1f2a40bd1f2617"]
    # commit_files = []
    # for hash in hashs:
    #     # 先找出新旧记录中的 (filename, funcname)，合并成一个 FOIs
    #     commit = next(Repository(path_to_repo="Data/database/CVE/test/test", single=hash).traverse_commits())
    #
    #     for file in commit.modified_files:
    #         commit_files.append({"hash": hash, "cwe_id": "CWE-119", "old_path": file.old_path, "new_path": file.new_path, "code_before": file.source_code_before, 'code_after': file.source_code, "diff_parsed": file.diff_parsed, "programming_language": "C++"})
    # df_file = pd.DataFrame(commit_files)
    # df_file = df_file.applymap(str)
    # cve_worker((repo, df_file))

    import sqlite3

    print(f"Connecting to database {CVE_PATH / 'CVEfixes-20230813.db'} ...", end="")
    # conn = sqlite3.connect("/Users/huanghongjun/FiCoVuL/preprocess/data/raw/CVEfixes/CVEfixes-2021-06-09.db", timeout=10)
    conn = sqlite3.connect(CVE_PATH / "CVEfixes-20230813.db", timeout=10)
    print("Done")
    df_cwe_classification = pd.read_sql_query(f"""SELECT cve_id, cwe_id FROM cwe_classification WHERE cwe_id IN {tuple(CWE_top26_comprehensive)};""", conn)
    df_fixes = pd.read_sql_query("""SELECT cve_id, hash, repo_url FROM fixes;""", conn)
    df_commits = pd.read_sql_query("""SELECT hash, repo_url FROM commits;""", conn)
    df_file_change = pd.read_sql_query(
        """SELECT file_change_id, hash, filename, old_path, new_path, diff_parsed, code_before, code_after, programming_language FROM file_change WHERE programming_language="C" OR programming_language="C++" OR programming_language="Objective-C";""",
        conn)
    conn.close()
    del conn

    # Unmatched git
    df_commits = df_commits.drop(df_commits[(df_commits["hash"] == "5186ddcf9e35a7aa0ff0539489a930434a1325f4") &
                                            (df_commits["repo_url"] == "https://github.com/Fstark-prog/jhead")].index)

    # Due to Exception: input buffer overflow, can't enlarge buffer because scanner uses REJECT
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/ONLYOFFICE/core"].index)
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/Perl/perl5"].index)
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/perl/perl5"].index)
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/RIOT-OS/RIOT"].index)
    # Due to Exception: invalid character in file
    df_commits = df_commits.drop(df_commits[(df_commits["hash"] == "430403b0029b37decf216d57f810899cab2317dd") &
                                            (df_commits["repo_url"] == "https://github.com/ImageMagick/ImageMagick")].index)

    # Invalid char in code file
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/TeX-Live/texlive-source"].index)
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/eclipse/mosquitto"].index)
    ## old_path: MagickCore/memory.c, new_path: MagickCore/memory.c, Calculated value is 1322, but given value is 1323
    df_commits = df_commits.drop(df_commits[(df_commits["hash"] == "e45e48b881038487d0bc94d92a16c1537616cc0a") &
                                            (df_commits["repo_url"] == "https://github.com/ImageMagick/ImageMagick")].index)

    # Segmentation fault (core dumped)
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/phusion/passenger"].index)
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/mongodb/mongo"].index)

    # Too many files, For acceleration
    """
    torvalds/linux 57519
    mongodb/mongo 33764
    facebook/hhvm 26575
    mysql/mysql-server 12020 -
    openbsd/src 47281
    illumos/illumos-gate 26100
    cesanta/mongoose 100409
    LibreOffice/core 23898
    stoth68000/media-tree (59830)
    apache/openoffice 20114
    libvirt/libvirt (12426)
    oracle/linux-uek 48454
    ONLYOFFICE/core 17769 -
    u-boot/u-boot 13328 -
    gregkh/linux 62564
    """
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/torvalds/linux"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/facebook/hhvm"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/openbsd/src"].index)
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/illumos/illumos-gate"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/cesanta/mongoose"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/LibreOffice/core"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/stoth68000/media-tree"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/apache/openoffice"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/libvirt/libvirt"].index)
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/oracle/linux-uek"].index)
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/gregkh/linux"].index)
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/openbsd/src"].index)
    #
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/libsdl-org/SDL"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/MariaDB/server"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/bminor/binutils-gdb"].index)
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/godotengine/godot"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/wireshark/wireshark"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/fluent/fluent-bit"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/nodejs/node-v0.x-archive"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/qt/qtbase"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/opencv/opencv"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/qemu/qemu"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/u-boot/u-boot"].index)
    df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/bonzini/qemu"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/kamailio/kamailio"].index)
    # df_commits = df_commits.drop(df_commits[df_commits["repo_url"] == "https://github.com/skvadrik/re2c"].index)

    # print(df_cwe_classification.shape)
    # print(df_fixes.shape)
    # print(df_commits.shape)
    # print(df_file_change.shape)

    df_all = pd.merge(df_cwe_classification, df_fixes, how="inner", on="cve_id")  # cve_id 可能重复
    # print(df_all.shape)
    df_all = pd.merge(df_all, df_commits, how="inner", on=["hash", "repo_url"])  # hash 可能重复
    # print(df_all.shape)
    df_all = pd.merge(df_all, df_file_change, how="inner", on="hash")  # hash 可能重复
    # print(df_all.shape)
    df_all['repo_url'] = df_all['repo_url'].apply(lambda x: '/'.join(x.split('/')[-2:]))
    # print(df_all.shape)

    print("Begin to pre-process ...")
    # repo_with_data = group_by_repo_url_generator(df_all)
    #
    # with Pool(10) as pool:
    #     pbar = tqdm.tqdm(pool.imap_unordered(dxoygen_worker, repo_with_data), total=len(df_all.repo_url.unique()), desc="Pre-process")
    #     for id in pbar:
    #         pbar.set_description(f"Finish repo_url {id}")

    repo_with_data = group_by_repo_url_generator(df_all)

    # for rd in repo_with_data:
    #     cve_worker(rd)

    manager = multiprocessing.Manager()
    total_finished = manager.Value('i', 0)
    total_length = manager.Value('i', df_all.groupby(['repo_url', 'hash']).ngroup().nunique())
    pool = multiprocessing.Pool(processes=2)
    repo_with_data = list(repo_with_data)
    random.shuffle(repo_with_data)
    for repo, data in repo_with_data:
        pool.apply_async(cve_worker, args=(repo, data, total_finished, total_length), error_callback=error_callback)
    pool.close()
    pool.join()

    # with multiprocessing.Manager() as manager:
    #     total_finished = manager.Value('i', 0)
    #     total_length = manager.Value('i', df_all.groupby(['repo_url', 'hash']).ngroup().nunique())
    #     args_list = [(*data, total_finished, total_length) for data in repo_with_data]
    #     with multiprocessing.Pool(8) as pool:
    #         pbar = tqdm.tqdm(pool.imap_unordered(cve_worker, args_list), total=len(df_all.repo_url.unique()), desc="Process")
    #         for id in pbar:
    #             pbar.set_description(f"Finish repo_url {id}")
    #             pbar.refresh()
