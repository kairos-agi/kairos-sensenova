#!/usr/bin/env python3
import sys
import re

def trim_before_gpu(s):
    idx = s.find("[GPU")
    if idx != -1:
        return s[idx:]
    else:
        # 未找到时返回原字符串
        return s

def extract_gpu_stat_blocks(lines):
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # 匹配 "GPU X 显存统计：" 行
        m = re.match(r'\[GPU\s+(\d+)\]\s+显存统计：', line)
        if m:
            device_id = int(m.group(1))
            block = [line.rstrip()]
            i += 1
            # 收集后续缩进行
            while i < n and (lines[i].startswith('  ') or lines[i].strip() == ''):
                if lines[i].strip() != '':  # 忽略空行？但统计块内可能有空行分隔？根据之前，统计块是连续缩进行直到空行。我们保留非空行。
                    block.append(lines[i].rstrip())
                else:
                    # 遇到空行，结束块
                    break
                i += 1
            blocks.append((device_id, block))
        else:
            i += 1
    # 按device_id排序
    blocks.sort(key=lambda x: x[0])
    return blocks

def extract_tables(lines):
    in_table = False
    end_table = False
    table_lines = []
    tables = []
    for line in lines:
        if not in_table and "PyTorch CUDA memory summary, device ID" in line:
            in_table = True
            table_lines.append("|===========================================================================|")
            table_lines.append(line.rstrip())
            continue
        if in_table:
            table_lines.append(line.rstrip())
        if in_table and "Oversize GPU segments" in line:
            end_table = True
            continue

        if in_table and end_table:
            # 解析 device ID
            device_id = None
            for line in table_lines:
                m = re.search(r'device ID (\d+)', line)
                if m:
                    device_id = int(m.group(1))
                    break
            if device_id is not None:
                tables.append((device_id, table_lines))
                # 初始化
                in_table = False
                end_table = False
                table_lines = []
        
    # 按 device_id 排序
    tables.sort(key=lambda x: x[0])
    return tables

def extract_max_memory_allocated(lines):
    max_memorys = []
    for line in lines:
        if "max_memory_allocated (GB)" in line:
            match = re.search(r'\[GPU (\d+)\]', line)
            if match:
                gpu_id = int(match.group(1))
                max_memorys.append((gpu_id, trim_before_gpu(line.rstrip())))
    max_memorys.sort(key=lambda x: x[0])
    return max_memorys

def extract_cost_times(lines):
    cost_times = []
    for line in lines:
        if " infer time (s): " in line:
            match = re.search(r'\[GPU (\d+)\]', line)
            if match:
                gpu_id = int(match.group(1))
                cost_times.append((gpu_id, trim_before_gpu(line.rstrip())))
    cost_times.sort(key=lambda x: x[0])
    return cost_times

def main():
    lines = sys.stdin.readlines()

    # 提取各个部分
    gpu_stat_blocks = extract_gpu_stat_blocks(lines)
    tables = extract_tables(lines)
    memorys = extract_max_memory_allocated(lines)
    times = extract_cost_times(lines)
    # 输出
    # 1. GPU 显存统计块
    for _, block in gpu_stat_blocks:
        print("\n".join(block))
        print()  # 空行分隔

    for device_id, table_lines in tables:
        if len(times) == len(tables):
            print(times[device_id][1])
        if len(memorys) == len(tables):
            print(memorys[device_id][1])
        print("\n".join(table_lines))
        print()  # 表格之间空一行分隔

if __name__ == "__main__":
    main()