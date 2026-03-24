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

resource_map={
        1: {
            "CPU": 13.000,
            "Memory": 234.000,
            "GPUNum": 1
        },
        4:{
            "CPU": 52.000,
            "Memory": 937.000,
            "GPUNum": 4
        },
        8:{
            "CPU": 105.000,
            "Memory": 1875.000,
            "GPUNum": 8
        }
    }


def update_yaml():
    """主逻辑：读取 YAML，修改 Envs 和 Entrypoint，写入文件"""
    
    # 读取环境变量
    template_file = os.environ.get('TEMPLATE_FILE')
    cmd_file = os.environ.get('CMD_FILE')
    http_proxy = os.environ.get('HTTP_PROXY')
    https_proxy = os.environ.get('HTTPS_PROXY')
    commit_id = os.environ.get('COMMIT_ID')
    output_file = os.environ.get('OUTPUT_FILE')
    case_cmd = os.environ.get('CASE_CMD')
    remote_dir = os.environ.get('REMOTE_DIR')
    aoss_ak = os.environ.get('AOSS_AK')
    aoss_sk = os.environ.get('AOSS_SK')

    # 1. 读取 YAML 模板
    with open(template_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 确保 Envs 字段存在且为列表
    if "Envs" not in data or data["Envs"] is None:
        data["Envs"] = []

    # 2. 构建环境变量
    envs = []
    if http_proxy:
        envs.append({"Name": "http_proxy", "Value": http_proxy})
    if https_proxy:
        envs.append({"Name": "https_proxy", "Value": https_proxy})
    if commit_id:
        envs.append({"Name": "commit_id", "Value": commit_id})
    if case_cmd:
        envs.append({"Name": "case_cmd", "Value": case_cmd})
    if aoss_ak:
        envs.append({"Name": "ak", "Value": aoss_ak})
    if aoss_sk:
        envs.append({"Name": "sk", "Value": aoss_sk})

    if remote_dir:
        if not remote_dir.endswith("/"):
            remote_dir += "/"
        envs.append({"Name": "remote_dir", "Value": remote_dir})

    for env in envs:
        data["Envs"] = [e for e in data["Envs"] if e.get("Name") != env["Name"]]
    data["Envs"].extend(envs)

    # 3. 读取 cmd.sh 内容，作为 Entrypoint
    with open(cmd_file, "r", encoding="utf-8") as f:
        cmd_content = f.read()

    data["Entrypoint"] = cmd_content

    # 4. 修改资源配置
    gpu_num = get_gpu_num(case_cmd)
    if gpu_num in resource_map.keys():
        for role in data.get("TaskRoleSpecs", []):
            if role.get("RoleName") == "worker":
                # 修改ResourceSpec中的配置
                resource_spec = role["ResourceSpec"]
                resource_spec["CPU"] = resource_map[gpu_num]["CPU"]
                resource_spec["Memory"] = resource_map[gpu_num]["Memory"]
                resource_spec["GPUNum"] = resource_map[gpu_num]["GPUNum"]
                break 
    else:
        print(f"❌Unsupport gpu_num:{gpu_num}!!!")


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