import sys
import sqlite3
import pandas as pd
from configure import DATABASE_PATH, CVE_PATH
from utils import format_cwe_id


DATABASE = DATABASE_PATH / "CVEFixes.db"

CWE_top26_comprehensive = ['CWE-787', 'CWE-125', 'CWE-20', 'CWE-78', 'CWE-416', 'CWE-22', 'CWE-190', 'CWE-476', 'CWE-119', 'CWE-200', 'CWE-732', 'CWE-362', 'CWE-400', 'CWE-120', 'CWE-415', 'CWE-59', 'CWE-617', 'CWE-772', 'CWE-295', 'CWE-401', 'CWE-284', 'CWE-369', 'CWE-191', 'CWE-763', 'CWE-835', 'CWE-674']


def create_connection(db_file):
    """
    create a connection to sqlite3 database
    """
    try:
        return sqlite3.connect(db_file, timeout=10)  # connection via sqlite3
    except sqlite3.Error as e:
        print(e)
        sys.exit(1)


def get_joint_tables():
    print(f"Connecting to database {CVE_PATH / 'CVEfixes-20230813.db'} ...", end="")
    conn = sqlite3.connect(CVE_PATH / "CVEfixes-20230813.db", timeout=10)
    print("Done")
    df_cwe_classification = pd.read_sql_query(
        f"""SELECT cve_id, cwe_id FROM cwe_classification WHERE cwe_id IN {tuple(CWE_top26_comprehensive)};""", conn)
    df_fixes = pd.read_sql_query("""SELECT cve_id, hash, repo_url FROM fixes;""", conn)
    df_commits = pd.read_sql_query("""SELECT hash, repo_url FROM commits;""", conn)
    df_file_change = pd.read_sql_query(
        """SELECT file_change_id, hash, filename, old_path, new_path, diff_parsed, code_before, code_after, programming_language FROM file_change WHERE programming_language="C" OR programming_language="C++" OR programming_language="Objective-C";""",
        conn)
    conn.close()
    del conn
    df_all = pd.merge(df_cwe_classification, df_fixes, how="inner", on="cve_id")  # cve_id 可能重复
    df_all = pd.merge(df_all, df_commits, how="inner", on=["hash", "repo_url"])  # hash 可能重复
    df_all = pd.merge(df_all, df_file_change, how="inner", on="hash")  # hash 可能重复
    df_all['repo_url'] = df_all['repo_url'].apply(lambda x: '/'.join(x.split('/')[-2:]))
    return df_all


def group_by_repo_url_generator(df_data):
    # groupby
    # 按照 repo_url 进行分组
    grouped_df = df_data.groupby(['repo_url'])

    for group_name, group_data in grouped_df:
        repo_url = group_name[0]
        yield repo_url, group_data


def group_by_hash_generator(df_data):
    # groupby
    # 按照 hash 进行分组
    grouped_df = df_data.groupby(['hash'])

    for group_name, group_data in grouped_df:
        _hash = group_name[0]
        yield _hash, group_data


if __name__ == '__main__':
    pass
