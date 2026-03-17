import os
import sys
import time
import json
import subprocess
import signal
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

def main():
    
    # 读取环境变量
    remote_dir = os.environ.get('REMOTE_DIR')
    ak = os.environ.get('AOSS_AK')
    sk = os.environ.get('AOSS_SK')
    remote_endpoint = "white-bucket.aoss.cn-sh-01b.sensecoreapi-oss.cn"
    if not os.path.exists("./ads-cli"):
        run_cmd("wget https://quark.aoss.cn-sh-01.sensecoreapi-oss.cn/ads-cli/release/v1.10.0/ads-cli")
    run_cmd("chmod +x ads-cli")
    remote_path = os.path.join(f"s3://{ak}:{sk}@{remote_endpoint}", remote_dir)
    local_path = os.path.join(remote_dir)
    local_dir_path = os.path.dirname(local_path)
    run_cmd(f"mkdir -p {local_dir_path} || true && ./ads-cli -q cp {remote_path} {local_path}")
    run_cmd_live(f'echo "download artifacts to {local_path}"')
    

if __name__ == "__main__":
    main()