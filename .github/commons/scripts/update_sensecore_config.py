#!/usr/bin/env python3
"""
update_yaml.py - 将代理地址，commit_id（通过参数传入）注入 YAML 的 Envs，并将 cmd.sh 内容设为 Entrypoint

用法:
    python update_yaml.py
"""

import sys
import os
import subprocess

# 尝试导入 PyYAML，若失败则自动安装
try:
    import yaml
except ImportError:
    print("PyYAML not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml


def update_env(env: str, new_envs: str) -> str:
    """
    更新环境变量字符串。
    
    参数:
        env: 原始环境变量字符串，格式如 "key1:value1,key2:value2"
        new_envs: 新的环境变量字符串，格式相同
    
    返回:
        合并更新后的字符串，保持原有顺序，新键追加在末尾。
    """
    env_dict = {}
    if env.strip():
        for part in env.split(','):
            if ':' in part:
                key, value = part.split(':', 1)
                env_dict[key.strip()] = value.strip()
    
    # 解析新字符串并合并（更新已有键，插入新键）
    if new_envs.strip():
        for part in new_envs.split(','):
            if ':' in part:
                key, value = part.split(':', 1)
                env_dict[key.strip()] = value.strip()
    
    # 将字典转换回字符串
    return ','.join(f"{k}:{v}" for k, v in env_dict.items())

def update_yaml():
    """主逻辑：读取 YAML，修改 Envs 和 Command，写入文件"""
    
    # 读取环境变量
    template_file = os.environ.get('TEMPLATE_FILE')
    cmd_file = os.environ.get('CMD_FILE')
    commit_id = os.environ.get('COMMIT_ID')
    output_file = os.environ.get('OUTPUT_FILE')
    case_cmd = os.environ.get('CASE_CMD')
    remote_dir = os.environ.get('REMOTE_DIR')
    aoss_ak = os.environ.get('AOSS_AK')
    aoss_sk = os.environ.get('AOSS_SK')
    http_proxy = os.environ.get('HTTP_PROXY')
    https_proxy = os.environ.get('HTTPS_PROXY')

    # 1. 读取 YAML 模板
    with open(template_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 确保 Envs 字段存在且为string
    if "Env" not in data or data["Env"] is None:
        data["Env"] = ""

    # 2. 构建环境变量
    new_envs = []
    if http_proxy:
        new_envs.append(f"http_proxy:'{http_proxy}'")
    if https_proxy:
        new_envs.append(f"https_proxy:'{https_proxy}'")
    if commit_id:
        new_envs.append(f"commit_id:'{commit_id}'")
    if aoss_ak:
        new_envs.append(f"ak:'{aoss_ak}'")
    if aoss_sk:
        new_envs.append(f"sk:'{aoss_sk}'")
    if remote_dir:
        if not remote_dir.endswith("/"):
            remote_dir += "/"
        new_envs.append(f"remote_dir:'{remote_dir}'")
    new_envs_string = ",".join(new_envs)
    data["Env"] = f'{update_env(data["Env"], new_envs_string)}'
    # 3. 读取 cmd.sh 内容，作为 Entrypoint
    with open(cmd_file, "r", encoding="utf-8") as f:
        cmd_content = f.read()
    
    # 因为大装置acp环境变量不支持带空格的value,所以直接替换到command中去
    if "$case_cmd" in cmd_content:
        cmd_content = cmd_content.replace("$case_cmd", case_cmd)
    data["Command"] = cmd_content

    if os.environ.get("WORKSPACE_NAME", ""):
        data["WorkspaceName"] = os.environ.get("WORKSPACE_NAME")

    # 修改资源配置
    gpu_num = get_gpu_num(case_cmd)
    data["WorkerSpec"] = f"n12lp.nn.i10.{gpu_num}"


    # 写入新的 YAML
    output_file = output_file or template_file  # 默认覆盖原文件
    with open(output_file, "w", encoding="utf-8") as f:
        # 使用 safe_dump，保持字段顺序（Python 3.7+ 字典有序）
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"✅ YAML 文件已更新: {output_file}")

import re

def get_gpu_num(case_cmd: str) -> int:
    """
    处理case_cmd环境变量，按规则返回数字：
    1. 检查命令中是否包含"multi_gpu"（对应multi_gpu_inference.sh）
    2. 若包含，提取最后一个参数，判断是否为纯数字
    3. 是数字返回该数字，否则返回1；不包含也返回1
    """
    if "multi_gpu" not in case_cmd:
        return 1
    
    cmd_parts = case_cmd.strip().split()
    if not cmd_parts:  # 极端情况：命令为空
        return 1
    
    last_param = cmd_parts[-1]

    if re.match(r"^[0-9]+$", last_param):
        return int(last_param)
    else:
        return 1

def main():
    update_yaml()


if __name__ == "__main__":
    main()