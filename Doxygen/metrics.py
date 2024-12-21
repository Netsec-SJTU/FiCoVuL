import glob
import json
import tqdm

from configure import DOXYGEN_OUTPUT_JSON_PATH
from utils import git_reset

import subprocess

def get_file_count(directory):
    command = f"find {directory} -type f | wc -l"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output = process.communicate()[0]
    file_count = int(output.strip())
    return file_count

if __name__ == '__main__':
    # cnt_repo = {0: 0, 1: 0}
    # cnt_depth = {}
    # for filename in tqdm.tqdm(glob.glob(f"{DOXYGEN_OUTPUT_JSON_PATH / '*.json'}")):
    #     with open(filename, 'r') as f:
    #         data = json.load(f)
    #     if data:
    #         flag = False
    #         for call_path in data['call_path']:
    #             l = len(call_path) - 1
    #             if l in cnt_depth:
    #                 cnt_depth[l] += 1
    #             else:
    #                 cnt_depth[l] = 1
    #             if l > 1:
    #                 flag = True
    #         if flag:
    #             cnt_repo[1] += 1
    #         else:
    #             cnt_repo[0] += 1
    # print(f"Metrics [repo distribution]:")
    # print(f"{cnt_repo[0]} repos do not have cross-function vuls, {cnt_repo[1]} repos have cross-function vuls. Proportion: {cnt_repo[1] / (cnt_repo[0] + cnt_repo[1])}")
    # print(f"Metrics [depth distribution]:")
    # # 对字典的键进行排序
    # sorted_keys = sorted(cnt_depth.keys())
    #
    # # 按照排序后的键顺序输出字典的内容
    # for key in sorted_keys:
    #     value = cnt_depth[key]
    #     print(f'{key}: {value}', end=', ')
    # print()
    #
    # print(sum(cnt_depth.values()))

    import glob
    import json
    import os

    import tqdm

    cnt_repo = {0: set(), 1: set()}
    sizes = {0: [], 1: []}
    temp = ['torvalds/linux', 'mongodb/mongo', 'facebook/hhvm', 'openbsd/src', 'illumos/illumos-gate',
            'cesanta/mongoose', 'LibreOffice/core', 'stoth68000/media-tree', 'apache/openoffice', 'libvirt/libvirt',
            'oracle/linux-uek', 'gregkh/linux', 'libsdl-org/SDL', 'MariaDB/server', 'bminor/binutils-gdb',
            'godotengine/godot', 'wireshark/wireshark', 'fluent/fluent-bit', 'nodejs/node-v0.x-archive', 'qt/qtbase',
            'opencv/opencv', 'qemu/qemu', 'u-boot/u-boot', 'bonzini/qemu', 'kamailio/kamailio', 'skvadrik/re2c']

    files = list(filter(lambda x: '^' in x, glob.glob(f"{DOXYGEN_OUTPUT_JSON_PATH / '*.json'}")))
    for filename in tqdm.tqdm(files):
        with open(filename, 'r') as f:
            data = json.load(f)
        if data:
            flag = False
            for call_path in data['call_path']:
                l = len(call_path) - 1
                if l > 0:
                    flag = True

            repo = data['repo']
            if repo in temp:
                temp.remove(repo)
            if repo not in cnt_repo[int(flag)]:
                git_reset(repo, "HEAD")
                size = get_file_count(f"Data/database/CVE/repos/{repo}")
                sizes[int(flag)].append(size)
            cnt_repo[int(flag)].add(repo)
    for repo in temp:
        cnt_repo[1].add(repo)
        git_reset(repo, "HEAD")
        size = get_file_count(f"Data/database/CVE/repos/{repo}")
        sizes[1].append(size)

    sizes[0] = sum(sizes[0]) / len(sizes[0])
    sizes[1] = sum(sizes[1]) / len(sizes[1])
    print(f"Average size of repos without cross-function vuls: {sizes[0]:.2f}")
    print(f"Average size of repos with cross-function vuls: {sizes[1]:.2f}")
