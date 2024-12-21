import glob

import pandas as pd
import tqdm
from itertools import chain

from configure import CVE_REPO_PATH, CVE_PATH

# import sqlite3
#
# from cve_database import CWE_top30_comprehensive
#
# print(f"Connecting to database {CVE_PATH / 'CVEfixes-20230813.db'} ...", end="")
# # conn = sqlite3.connect("/Users/huanghongjun/FiCoVuL/preprocess/data/raw/CVEfixes/CVEfixes-2021-06-09.db", timeout=10)
# conn = sqlite3.connect(CVE_PATH / "CVEfixes-20230813.db", timeout=10)
# print("Done")
# df_cwe_classification = pd.read_sql_query(f"""SELECT cve_id, cwe_id FROM cwe_classification WHERE cwe_id IN {tuple(CWE_top30_comprehensive)};""", conn)
# df_fixes = pd.read_sql_query("""SELECT cve_id, hash, repo_url FROM fixes;""", conn)
# df_commits = pd.read_sql_query("""SELECT hash, repo_url FROM commits;""", conn)
# df_file_change = pd.read_sql_query(
#     """SELECT file_change_id, hash, filename, old_path, new_path, diff_parsed, code_before, code_after, programming_language FROM file_change WHERE programming_language="C" OR programming_language="C++" OR programming_language="Objective-C";""",
#     conn)
# conn.close()
# del conn
#
# df_all = pd.merge(df_cwe_classification, df_fixes, how="inner", on="cve_id")  # cve_id 可能重复
# # print(df_all.shape)
# df_all = pd.merge(df_all, df_commits, how="inner", on=["hash", "repo_url"])  # hash 可能重复
# # print(df_all.shape)
# df_all = pd.merge(df_all, df_file_change, how="inner", on="hash")  # hash 可能重复
# # print(df_all.shape)
# df_all['repo_url'] = df_all['repo_url'].apply(lambda x: '/'.join(x.split('/')[-2:]))
# # print(df_all.shape)
#
# tbar = tqdm.tqdm(df_all.repo_url.unique())
# for repo in tbar:
#     tbar.set_description(repo)
#     file_extensions = [".c", ".cc", ".cxx", ".cpp", ".ii", ".ixx", ".ipp", ".i++", ".inl", ".c++", ".h", ".H",
#                        ".hh", ".HH", ".hxx", ".hpp", ".h++", ".mm"]
#     n_files = len(list(chain(
#                 *map(lambda postfix: list(glob.glob(rf"{CVE_REPO_PATH / repo}/**/*{postfix}", recursive=True)),
#                      file_extensions))))
#     if n_files > 10000:
#         print(repo, n_files)

file_extensions = [".c", ".cc", ".cxx", ".cpp", ".ii", ".ixx", ".ipp", ".i++", ".inl", ".c++", ".h", ".H",
                   ".hh", ".HH", ".hxx", ".hpp", ".h++", ".mm"]
n_files = len(list(chain(
            *map(lambda postfix: list(glob.glob(rf"{CVE_REPO_PATH}/stoth68000/media-tree/**/*{postfix}", recursive=True)),
                 file_extensions))))
print("stoth68000/media-tree", n_files)
n_files = len(list(chain(
            *map(lambda postfix: list(glob.glob(rf"{CVE_REPO_PATH}/libvirt/libvirt/**/*{postfix}", recursive=True)),
                 file_extensions))))
print("libvirt/libvirt", n_files)
