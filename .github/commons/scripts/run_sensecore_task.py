#!/usr/bin/env python3
"""
Sensecore ACP Task 管理脚本
功能：
- 提交任务前，检查是否存在同名任务：
  - 存在最新的同名任务且状态为(SUCCEEDED, RUNNING, PENDING, QUEUEING) → 复用
  - 存在同名任务但状态为其他 → 取消该任务并提交新任务
  - 无同名任务 → 提交新任务
- 将TASK_ID写入文件task_id.txt中
- 等待任务进入 RUNNING 状态（超时控制）。
- 任务结束后检查最终状态，成功则退出 0，否则退出 1。
- 根据状态获取日志
环境变量：
  TASK_NAME          : 任务名称
  TASK_CONFIG_FILE   : 任务配置文件路径
  PENDDING_TIME_OUT  : 等待 Running 的超时时间（秒，默认 1800）
  INTERVAL           : 状态检查间隔（秒，默认 30）
  TASKID_FILE        : 保存Task ID的文件路径（必填）
"""

import os
import sys
import time
import json
import subprocess
import yaml
import re

# Python 3.7+ 支持 reconfigure
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
else:
    # 兼容老版本 Python：每次 print 强制 flush
    import functools
    print = functools.partial(print, flush=True)

def run_cmd(cmd, check=True, capture_output=True):
    """执行命令，返回输出或抛出异常"""
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}")
        sys.exit(result.returncode)
    return result

def run_cmd_live(cmd):
    print(f"CMD: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
    process.wait()
    return process.returncode

def get_task_status(task_id):
    """获取任务状态，返回状态字符串，失败时返回 None"""
    config = read_config()
    try:
        result = run_cmd(f'sco acp jobs describe --workspace-name={config["WorkspaceName"]} --format json  {task_id}', check=True)
        data = json.loads(result.stdout)
        return data["state"]
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Warning: Failed to get task status: {e}")
        return None

def cancel_task(task_id):
    config = read_config()
    """取消任务"""
    print(f"Cancelling task {task_id}...")
    run_cmd(f"sco acp jobs stop --workspace-name {config['WorkspaceName']} {task_id}", check=False)

def submit_task(config_file, task_name):
    config = read_config(config_file)
    """提交任务，返回 task_id"""
    print(f"Submitting task with name: {task_name}")
    cmd = f'sco acp jobs create \
            --job-name \'{task_name}\' \
            --workspace-name \'{config["WorkspaceName"]}\' \
            --aec2-name \'{config["Aec2Name"]}\' \
            --priority \'{config["priority"]}\' \
            --worker-spec  \'{config["WorkerSpec"]}\' \
            --training-framework \'{config["TrainingFramework"]}\' \
            --container-image-url \'{config["Image"]}\' \
            --storage-mount  \'{config["MountInfo"]}\' \
            --env \'{config["Env"]}\' \
            --command=\'{config["Command"]}\''
    result = run_cmd(cmd)
    task_id = extract_job_id(result.stdout)
    if not task_id:
        print(f"Failed to get task ID from output: {result.stdout}")
        sys.exit(1)
    print(f"Task submitted, ID: {task_id}")
    return task_id

def wait_for_running(task_id, timeout, interval):
    """等待任务进入 Running 状态，超时或失败则退出"""
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"⏰ Timeout reached after {timeout}s. Task did not become Running.")
            print(f"❌ Task failed to start, current status: {get_task_status(task_id)}")
            cancel_task(task_id)
            sys.exit(1)

        status = get_task_status(task_id)
        if status is None:
            # 获取失败，继续等待
            time.sleep(interval)
            continue

        print(f"Current status: {status}")

        if status == "RUNNING":
            print(f"✅ Task is now {status}.")
            break
        elif status in ("FAILED", "CANCELED"):
            print(f"❌ Task failed with status: {status}")
            sys.exit(1)
        elif status == "SUCCEEDED":
            print(f"✅ Task is {status}")
            break
        else:
            time.sleep(interval)
    return status

def wait_for_completion(task_id, interval):
    """等待任务进入终态，返回最终状态"""
    while True:
        status = get_task_status(task_id)
        if status is None:
            # 获取失败，继续等待
            time.sleep(interval)
            continue

        print(f"Current status: {status}")

        # 终态判断
        if status in ("SUCCEEDED", "FAILED", "CANCELED"):
            return status
        # 其他状态继续等待
        time.sleep(interval)

def fetch_logs_on_failure():
    """如果任务失败，获取并打印日志"""
    print("Fetching logs due to task failure...")
    print("="*50)
    remote_dir = os.environ.get('REMOTE_DIR')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = f'python {script_dir}/download_remote.py && tail -n 50 {remote_dir}/run.log'
    try:
        run_cmd_live(cmd)
    except Exception as e:
        print(f"Failed to fetch logs: {e}")


def fetch_logs_on_success():
    """任务成功后获取性能日志"""
    remote_dir = os.environ.get('REMOTE_DIR')

    print("Performance results...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = f'python {script_dir}/download_remote.py && cat {remote_dir}/run.log | python {script_dir}/filter_logs.py'
    try:
        run_cmd_live(cmd)
    except Exception as e:
        print(f"Failed to fetch logs: {e}")

def write_taskid_to_file(task_id, file_path):
    with open(file_path, "w") as f:
        f.write(task_id)
    print(f"已将task_id:{task_id}成功写入文件{file_path}")

def read_config(config_file=None):
    if config_file is None:
        config_file = os.environ.get('TASK_CONFIG_FILE')
    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def extract_job_id(output: str) -> str:
    """
    从 acp jobs create 命令的输出中提取 job id。
    
    支持两种格式：
    1. 第二行包含 "job id : <job_id>"
    2. 第一行包含 "job <job_id> submitted"
    
    参数:
        output: 命令输出的字符串
        
    返回:
        提取到的 job id 字符串，如果未找到则返回 None
    """
    # 尝试匹配 "job id : <job_id>" 模式
    match = re.search(r'job\s+id\s*:\s*(\S+)', output, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # 尝试匹配 "job <job_id> submitted" 模式
    match = re.search(r'job\s+(\S+)\s+submitted', output, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # 如果都不匹配，返回 None
    return None

def main():
    
    # 读取环境变量
    task_name = os.environ.get('TASK_NAME', "test_5090_ci")
    config_file = os.environ.get('TASK_CONFIG_FILE')
    timeout = int(os.environ.get('PENDDING_TIME_OUT', 1800))
    interval = int(os.environ.get('INTERVAL', 30))
    taskid_file = os.environ.get('TASKID_FILE', "task_id.txt")

    if not task_name or not config_file or not taskid_file:
        print("Error: TASK_NAME, TASK_CONFIG_FILE and TASKID_FILE must be set.")
        sys.exit(1)

    config = read_config(config_file)
    # 定义可复用的任务状态列表
    REUSABLE_STATUSES = ('SUCCEEDED', 'RUNNING', 'PENDING', 'QUEUEING', 'INIT', 'STARTING')
    
    # 1. 检查同名任务（所有状态）
    print(f"Checking for existing tasks with name: {task_name}")
    cmd = f"sco acp jobs list --workspace-name={config['WorkspaceName']} --page-size 20 -o json"
    result = run_cmd(cmd, check=True)
    tasks = json.loads(result.stdout)
    
    # 筛选同名任务并按创建时间排序（最新的在前）
    same_name_tasks = sorted(
        [t for t in tasks if t.get('display_name') == task_name],
        key=lambda x: x.get('create_time', ''),
        reverse=True
    )

    task_id = None

    if same_name_tasks:
        # 获取最新的同名任务
        latest_task = same_name_tasks[0]
        task_id = latest_task['name']
        task_status = latest_task.get('state', 'Unknown')
        
        print(f"Found existing task: ID={task_id}, Status={task_status}")
        
        if task_status in REUSABLE_STATUSES:
            # 状态符合要求，复用该任务
            print(f"Task status '{task_status}' is reusable, reusing task {task_id}")
        else:
            # 状态不符合，取消任务并重新提交
            print(f"Task status '{task_status}' is not reusable, cancelling and submitting new task")
            cancel_task(task_id)
            task_id = submit_task(config_file, task_name)
    else:
        # 无同名任务，提交新任务
        print("No existing task found, submitting new task")
        task_id = submit_task(config_file, task_name)

    # 写入task id到文件
    write_taskid_to_file(task_id, taskid_file)
    
    current_status = get_task_status(task_id)
    
    # 等待 Running
    if current_status not in ("SUCCEEDED", "FAILED", "SUSPENDED"):
        current_status = wait_for_running(task_id, timeout, interval)
    else:
        print(f"Task is already in final status: {current_status}, skip waiting for Running")

    # 等待任务完成（如果任务还没到终态）
    if current_status not in ("SUCCEEDED", "FAILED", "CANCELED"):
        final_status = wait_for_completion(task_id, interval)
    else:
        final_status = current_status

    # 根据最终状态处理
    if final_status == "SUCCEEDED":
        print("✅ Task succeeded!")
        fetch_logs_on_success()
        sys.exit(0)
    else:
        # # 失败，获取日志
        fetch_logs_on_failure()
        print(f"❌ Task failed! status: {final_status}")
        sys.exit(1)

if __name__ == "__main__":
    main()