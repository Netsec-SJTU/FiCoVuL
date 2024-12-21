import codecs
import os
import subprocess
import signal
import glob
import math
import ftfy
import datetime
import chardet
import time

from configure import CVE_REPO_PATH


def run_cmd(cmd_string, timeout=None):
    """
    自己实现的超时控制
    优点：能够进行超时的控制，能够杀掉所有的子进程
    缺点：超时之后无法输出命令执行结果
    """
    p = subprocess.Popen(cmd_string, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True, close_fds=True,
                         start_new_session=True)

    format = 'utf-8'

    try:
        (msg, errs) = p.communicate(timeout=timeout)
        ret_code = p.poll()
        if ret_code:
            code = 1
            msg = "[Error]Called Error : " + str(msg.decode(format))
        else:
            code = 0
            msg = str(msg.decode(format))
    except subprocess.TimeoutExpired:
        # 注意：不能只使用p.kill和p.terminate，无法杀干净所有的子进程，需要使用os.killpg
        p.kill()
        p.terminate()
        os.killpg(p.pid, signal.SIGTERM)

        # 注意：如果开启下面这两行的话，会等到执行完成才报超时错误，但是可以输出执行结果
        # (outs, errs) = p.communicate()
        # print(outs.decode('utf-8'))

        code = 1
        msg = "[ERROR]Timeout Error : Command '" + cmd_string + "' timed out after " + str(timeout) + " seconds"
    except Exception as e:
        code = 1
        msg = "[ERROR]Unknown Error : " + str(e)

    return code, msg


def git_reset(repo, hash):
    repo_path = CVE_REPO_PATH / repo
    # 只改变文件，不改变指针
    code, msg = run_cmd(f"cd {repo_path} && git checkout {hash} -- . && cd -")
    if code != 0:
        raise Exception(msg)


def gen_repo_hash_path(repo, hash):
    """abandoned"""
    return f"{repo.split('/')[0]}/{hash}/{repo.split('/')[1]}"


def remove_non_utf8(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    # 使用 ftfy 过滤非UTF-8字符
    content = ftfy.fix_text(content)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def clean_file(path):
    # 过滤非法字符
    list(map(remove_non_utf8, glob.glob(rf"{path}/**/*.xml", recursive=True)))


def format_cwe_id(cwe_id: str):
    if len(cwe_id) == 7:
        return cwe_id
    temp = cwe_id.split('-')
    temp[1] = '0'*(7-len(cwe_id))+temp[1]
    return '-'.join(temp)


def convert_runtime(start_time, end_time):
    """
    converts runtime of the slice of code more readable format
    """
    runtime = end_time - start_time
    hours = runtime / 3600
    minutes = (runtime % 3600) / 60
    seconds = (runtime % 3600) % 60
    return math.floor(hours), math.floor(minutes), round(seconds)


def convert_to_utf8(file_path):
    # 检测文件编码
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    # 转换为UTF-8编码
    with codecs.open(file_path, 'r', encoding=encoding) as f:
        content = f.read()

    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    remove_non_utf8(file_path)


def delete_large_files(directory, max_kb=250):
    for file_path in filter(os.path.isfile, glob.glob(f"{directory}/**/*", recursive=True)):
        if os.path.getsize(file_path) > max_kb * 1024:
            os.remove(file_path)


class LogHandler:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        self.log_file = open(log_file_path, 'w')

    def log(self, msg):
        now = datetime.datetime.now()
        self.log_file.write(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}]\n{msg}")
        self.log_file.flush()

    def close(self):
        self.log_file.close()

    def __del__(self):
        self.close()
