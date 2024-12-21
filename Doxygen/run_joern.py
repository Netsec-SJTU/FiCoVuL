import argparse
import os
import sys
import glob
import shutil
import tqdm
from pathlib import Path
from multiprocessing import Pool

from utils import run_cmd
from configure import JOERN_BINARY_PATH, JOERN_TEMP_PATH, JOERN_RESULT_PATH, \
    DOXYGEN_OUTPUT_RESULT_CVE_PATH, DOXYGEN_OUTPUT_RESULT_SARD_PATH, DOXYGEN_OUTPUT_RESULT_WILD_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['CVE', 'SARD', 'WILD'], default="CVE")
    parser.add_argument("--conti", type=bool, default=True,
                        help="continue from the last interruption?")
    args = parser.parse_args()

    if args.dataset == 'CVE':
        PATH = DOXYGEN_OUTPUT_RESULT_CVE_PATH
    elif args.dataset == 'SARD':
        PATH = DOXYGEN_OUTPUT_RESULT_SARD_PATH
    else:
        PATH = DOXYGEN_OUTPUT_RESULT_WILD_PATH

    current_path = Path(os.path.abspath(os.path.join(__file__, "..")))

    if not args.conti:
        if os.path.exists(JOERN_BINARY_PATH):
            shutil.rmtree(JOERN_BINARY_PATH)
        if os.path.exists(JOERN_TEMP_PATH):
            shutil.rmtree(JOERN_TEMP_PATH)
        if os.path.exists(JOERN_RESULT_PATH):
            shutil.rmtree(JOERN_RESULT_PATH)
    JOERN_BINARY_PATH.mkdir(exist_ok=True)
    JOERN_TEMP_PATH.mkdir(exist_ok=True)
    JOERN_RESULT_PATH.mkdir(exist_ok=True)

    filenames = glob.glob(str(PATH) + r"/*.c")
    processed_ids = []
    if args.conti:
        processed_files = glob.glob(str(JOERN_RESULT_PATH) + r"/*")
        processed_ids = list(map(lambda x: '-'.join(x.split('/')[-1].split('-')[:-1]), processed_files))
        print(f"{len(processed_ids)} files have been processed.")
        sys.stdout.flush()


    def conscious_worker(filename):
        global processed_ids

        id = filename.split('/')[-1].split('.')[0]

        if args.conti and id in processed_ids:
            return id

        binary_file_path = f"{JOERN_BINARY_PATH / str(id)}.bin"
        code, msg = run_cmd(f"sh {current_path}/joern-cli/joern-parse {filename} --out {binary_file_path}",
                            timeout=500)
        if code != 0:
            print(msg)
            sys.stdout.flush()
            return id

        temp_file_path = f"{JOERN_TEMP_PATH / str(id)}"
        code, msg = run_cmd(f"sh {current_path}/joern-cli/joern "
                            f"--script {current_path}/joern-cli/scripts/graph-for-funcs.sc "
                            f"--params originFile={filename},cpgFile={binary_file_path},outDir={temp_file_path}",
                            timeout=600)
        if code != 0:
            print(msg)
            sys.stdout.flush()
            return id

        code, msg = run_cmd(f"python3 {current_path}/joern-cli/extract_joern.py "
                            f"-i {temp_file_path}/all.json -o {JOERN_RESULT_PATH}/{str(id)}",
                            timeout=250)
        if code != 0:
            print(msg)
            sys.stdout.flush()
            return id

        return id


    # 定义一个进程池，最大进程数 10
    with Pool(16) as pool:
        pbar = tqdm.tqdm(pool.imap_unordered(conscious_worker, filenames), total=len(filenames))
        for id in pbar:
            pbar.set_description(f"Finish process {id}")
