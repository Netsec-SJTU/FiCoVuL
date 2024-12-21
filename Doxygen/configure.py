import os
from pathlib import Path

DATA_PATH = Path('Data')
DATABASE_PATH = DATA_PATH / 'database'  # store original repos
CVE_PATH = DATABASE_PATH / 'CVE'
CVE_REPO_PATH = CVE_PATH / 'repos'
SARD_PATH = DATABASE_PATH / 'SARD'
WILD_PATH = DATABASE_PATH / 'WILD'
# SARD_PATH = Path("/Users/huanghongjun/FiCoVuL/preprocess/data/raw/SARD")
SARD_TESTCASES_PATH = SARD_PATH / "testcases"
DOXYGEN_CONFIG_DIR_PATH = DATA_PATH / 'custom_configs'
DOXYGEN_LOG_DIR_PATH = DATA_PATH / 'logs'
TEMPLATE_DOXYGEN_CONFIG_DIR_PATH = DOXYGEN_CONFIG_DIR_PATH / 'template.cf'
DOXYGEN_OUTPUT_XML_PATH = DATA_PATH / 'output_doxygen_xml'
DOXYGEN_OUTPUT_JSON_PATH = DATA_PATH / 'output_doxygen_json'
DOXYGEN_OUTPUT_JSON_CVE_PATH = DOXYGEN_OUTPUT_JSON_PATH / 'CVE'
DOXYGEN_OUTPUT_JSON_SARD_PATH = DOXYGEN_OUTPUT_JSON_PATH / 'SARD'
DOXYGEN_OUTPUT_JSON_WILD_PATH = DOXYGEN_OUTPUT_JSON_PATH / 'WILD'
DOXYGEN_OUTPUT_RESULT_PATH = DATA_PATH / 'output_doxygen_result'
DOXYGEN_OUTPUT_RESULT_CVE_PATH = DOXYGEN_OUTPUT_RESULT_PATH / 'CVE'
DOXYGEN_OUTPUT_RESULT_SARD_PATH = DOXYGEN_OUTPUT_RESULT_PATH / 'SARD'
DOXYGEN_OUTPUT_RESULT_WILD_PATH = DOXYGEN_OUTPUT_RESULT_PATH / 'WILD'
ORIG_OUTPUT_RESULT_PATH = DATA_PATH / 'output_orig_result'
ORIG_OUTPUT_RESULT_CVE_PATH = ORIG_OUTPUT_RESULT_PATH / 'CVE'
ORIG_OUTPUT_RESULT_CVE_PATH_SAMPLES = ORIG_OUTPUT_RESULT_CVE_PATH / 'samples'
ORIG_OUTPUT_RESULT_SARD_PATH = ORIG_OUTPUT_RESULT_PATH / 'SARD'
ORIG_OUTPUT_RESULT_SARD_PATH_SAMPLES = ORIG_OUTPUT_RESULT_SARD_PATH / 'samples'
ORIG_OUTPUT_RESULT_WILD_PATH = ORIG_OUTPUT_RESULT_PATH / 'WILD'
ORIG_OUTPUT_RESULT_WILD_PATH_SAMPLES = ORIG_OUTPUT_RESULT_WILD_PATH / 'samples'
JOERN_BINARY_PATH = DATA_PATH / 'temp_joern_binary'
JOERN_TEMP_PATH = DATA_PATH / 'temp_joern_scala'
JOERN_RESULT_PATH = DATA_PATH / 'output_joern_result'

DOXYGEN_LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
DOXYGEN_OUTPUT_XML_PATH.mkdir(parents=True, exist_ok=True)

DOXYGEN_OUTPUT_JSON_PATH.mkdir(parents=True, exist_ok=True)
DOXYGEN_OUTPUT_JSON_CVE_PATH.mkdir(parents=True, exist_ok=True)
DOXYGEN_OUTPUT_JSON_SARD_PATH.mkdir(parents=True, exist_ok=True)
DOXYGEN_OUTPUT_JSON_WILD_PATH.mkdir(parents=True, exist_ok=True)

ORIG_OUTPUT_RESULT_PATH.mkdir(parents=True, exist_ok=True)
ORIG_OUTPUT_RESULT_CVE_PATH.mkdir(parents=True, exist_ok=True)
ORIG_OUTPUT_RESULT_CVE_PATH_SAMPLES.mkdir(parents=True, exist_ok=True)
ORIG_OUTPUT_RESULT_SARD_PATH.mkdir(parents=True, exist_ok=True)
ORIG_OUTPUT_RESULT_SARD_PATH_SAMPLES.mkdir(parents=True, exist_ok=True)
ORIG_OUTPUT_RESULT_WILD_PATH.mkdir(parents=True, exist_ok=True)
ORIG_OUTPUT_RESULT_WILD_PATH_SAMPLES.mkdir(parents=True, exist_ok=True)

DOXYGEN_OUTPUT_RESULT_PATH.mkdir(parents=True, exist_ok=True)
DOXYGEN_OUTPUT_RESULT_CVE_PATH.mkdir(parents=True, exist_ok=True)
DOXYGEN_OUTPUT_RESULT_SARD_PATH.mkdir(parents=True, exist_ok=True)
DOXYGEN_OUTPUT_RESULT_WILD_PATH.mkdir(parents=True, exist_ok=True)

# 输出的行号是1开始的

# MAX_DOT_GRAPH_DEPTH
# CALLER_GRAPH 函数被哪些函数调用


def modify_doxygen_config_cve(repo_path, hash, input_encoding='UTF-8'):
    global CVE_REPO_PATH, DOXYGEN_OUTPUT_XML_PATH, TEMPLATE_DOXYGEN_CONFIG_DIR_PATH
    
    # 读取配置文件
    with open(TEMPLATE_DOXYGEN_CONFIG_DIR_PATH, 'r') as file:
        config_lines = file.readlines()

    input_path = CVE_REPO_PATH / repo_path
    
    # 修改项目名选项
    i = next(filter(lambda i: config_lines[i].startswith('PROJECT_NAME'), range(len(config_lines))))
    config_lines[i] = f'PROJECT_NAME           = "{input_path}"\n'

    # 修改输入目录选项
    i = next(filter(lambda i: config_lines[i].startswith('INPUT'), range(len(config_lines))))
    config_lines[i] = f'INPUT                  = "{input_path}"\n'
    i = next(filter(lambda i: config_lines[i].startswith('STRIP_FROM_PATH'), range(len(config_lines))))
    config_lines[i] = f'STRIP_FROM_PATH        = "{input_path}"\n'
    i = next(filter(lambda i: config_lines[i].startswith('INPUT_ENCODING'), range(len(config_lines))))
    config_lines[i] = f'INPUT_ENCODING         = "{input_encoding}"\n'

    # 修改输出目录选项
    output_dir = DOXYGEN_OUTPUT_XML_PATH / repo_path / hash
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    i = next(filter(lambda i: config_lines[i].startswith('OUTPUT_DIRECTORY'), range(len(config_lines))))
    config_lines[i] = f'OUTPUT_DIRECTORY       = "{output_dir}"\n'
    # i = next(filter(lambda i: config_lines[i].startswith('XML_OUTPUT'), range(len(config_lines))))
    # config_lines[i] = f'XML_OUTPUT             = "{hash}"\n'

    # 保存修改后的配置文件
    store_path = DOXYGEN_CONFIG_DIR_PATH / f'{repo_path.replace("/", "+")}+{hash}.cf'
    with open(store_path, 'w') as file:
        file.writelines(config_lines)
    
    return store_path, output_dir/'xml'


def modify_doxygen_config_sard(repo_path, macros: str = "", tag: str = None):
    global SARD_TESTCASES_PATH, DOXYGEN_OUTPUT_XML_PATH, TEMPLATE_DOXYGEN_CONFIG_DIR_PATH

    if macros != "":
        assert tag is not None, "tag must be specified when macros is not empty"

    # 读取配置文件
    with open(TEMPLATE_DOXYGEN_CONFIG_DIR_PATH, 'r') as file:
        config_lines = file.readlines()

    input_path = SARD_TESTCASES_PATH / repo_path
    tid = repo_path.replace('/', '-')

    # 修改项目名选项
    i = next(filter(lambda i: config_lines[i].startswith('PROJECT_NAME'), range(len(config_lines))))
    config_lines[i] = f'PROJECT_NAME           = "{input_path}"\n'

    # 修改输入目录选项
    i = next(filter(lambda i: config_lines[i].startswith('INPUT'), range(len(config_lines))))
    config_lines[i] = f'INPUT                  = "{input_path}"\n'
    i = next(filter(lambda i: config_lines[i].startswith('STRIP_FROM_PATH'), range(len(config_lines))))
    config_lines[i] = f'STRIP_FROM_PATH        = "{input_path}"\n'

    # 修改输出目录选项
    output_dir = DOXYGEN_OUTPUT_XML_PATH / tid
    if tag is not None:
        output_dir = output_dir / tag
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    i = next(filter(lambda i: config_lines[i].startswith('OUTPUT_DIRECTORY'), range(len(config_lines))))
    config_lines[i] = f'OUTPUT_DIRECTORY       = "{output_dir}"\n'

    # 修改宏定义选项
    i = next(filter(lambda i: config_lines[i].startswith('PREDEFINED'), range(len(config_lines))))
    config_lines[i] = f'PREDEFINED             = {macros}\n'

    # 保存修改后的配置文件
    if tag is None:
        store_path = DOXYGEN_CONFIG_DIR_PATH / f'{tid}.cf'
    else:
        store_path = DOXYGEN_CONFIG_DIR_PATH / f'{tid}-{tag}.cf'
    with open(store_path, 'w') as file:
        file.writelines(config_lines)

    return store_path, output_dir/'xml'


def modify_doxygen_config_wild(repo_path):
    global WILD_PATH, DOXYGEN_OUTPUT_XML_PATH, TEMPLATE_DOXYGEN_CONFIG_DIR_PATH

    if os.path.exists(f'{DOXYGEN_CONFIG_DIR_PATH / repo_path}.cf'):
        store_path = DOXYGEN_CONFIG_DIR_PATH / f'{repo_path}.cf'
        output_dir = DOXYGEN_OUTPUT_XML_PATH / repo_path
        return store_path, output_dir/'xml'

    # 读取配置文件
    with open(TEMPLATE_DOXYGEN_CONFIG_DIR_PATH, 'r') as file:
        config_lines = file.readlines()

    input_path = WILD_PATH / repo_path
    tid = repo_path

    # 修改项目名选项
    i = next(filter(lambda i: config_lines[i].startswith('PROJECT_NAME'), range(len(config_lines))))
    config_lines[i] = f'PROJECT_NAME           = "{input_path}"\n'

    # 修改输入目录选项
    i = next(filter(lambda i: config_lines[i].startswith('INPUT'), range(len(config_lines))))
    config_lines[i] = f'INPUT                  = "{input_path}"\n'
    i = next(filter(lambda i: config_lines[i].startswith('STRIP_FROM_PATH'), range(len(config_lines))))
    config_lines[i] = f'STRIP_FROM_PATH        = "{input_path}"\n'

    # 修改输出目录选项
    output_dir = DOXYGEN_OUTPUT_XML_PATH / tid
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    i = next(filter(lambda i: config_lines[i].startswith('OUTPUT_DIRECTORY'), range(len(config_lines))))
    config_lines[i] = f'OUTPUT_DIRECTORY       = "{output_dir}"\n'

    # 保存修改后的配置文件
    store_path = DOXYGEN_CONFIG_DIR_PATH / f'{tid}.cf'
    with open(store_path, 'w') as file:
        file.writelines(config_lines)

    return store_path, output_dir/'xml'


if __name__ == '__main__':
    # repo = 'KDE/kde1-kdebase'
    # hash = '04906bd5de2f220bf100b605dad37b4a1d9a91a6'
    # modify_doxygen_config_cve(repo, hash)

    repo = '000/002/278'
    modify_doxygen_config_sard(repo)
