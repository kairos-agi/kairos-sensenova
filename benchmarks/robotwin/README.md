# RoboTwin Benchmark (KairosWAM)

在 RoboTwin 仿真环境中评测 KairosWAM 策略。评测采用 **双进程架构**：

1. **WAM 推理服务** - 多 GPU HTTP 服务，加载 DiT checkpoint，返回 action
2. **RoboTwin 评测客户端** - RoboTwin/SAPIEN 仿真，通过 HTTP policy adapter 调用 WAM 服务

两个组件使用 **不同的 conda 环境**，不可混装。

## 目录结构

```
benchmarks/robotwin/
├── README.md
├── run.sh                          # 一键：启动 WAM + 等待就绪 + 跑评测
├── RoboTwin_client.sh              # 仅启动 RoboTwin 评测客户端
├── configs/
│   └── robotwin_wam_infer_config.py
├── robotwin_dataset_stats.json     # action/state 归一化统计
├── requirements-eval.txt           # RoboTwin 评测环境依赖
├── robotwin_eval/                  # RoboTwin 评测入口和 policy adapter
│   ├── run_chunked_eval.py
│   ├── eval_single.py
│   ├── resummarize_chunked_results.py
│   ├── config/
│   ├── policy/
│   └── robotwin_scripts/
└── third_party/
    └── RoboTwin/                   # 需用户下载/解压，GitHub 仓库不直接包含
```

WAM FastAPI 服务、等待脚本和 server 依赖文件复用 `benchmarks/common/` 下的公共实现。

当前开源配置在 RoboTwin 上的成功率为 **clean: 96.9%**、**random: 95.2%**。

## 前置条件

| 资源 | 说明 | 默认路径 |
|------|------|----------|
| RoboTwin 仿真器 | 官方 [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 代码，下载或解压后放置 | `third_party/RoboTwin` |
| WAM DiT checkpoint | `.safetensors` 权重 | 通过 `WAM_PRETRAINED_DIT` 指定 |
| Dataset stats | 归一化 JSON | `robotwin_dataset_stats.json` |
| Eval step limit | RoboTwin 每个任务的步数配置 | `robotwin_eval/config/_eval_step_limit.yml` |
| 共享模型 | Qwen3.5、Wan VAE 等 | 见 `configs/robotwin_wam_infer_config.py` |
| GPU | WAM 默认 4 卡；评测默认使用同一组卡 | `WAM_GPU_IDS` / `EVAL_GPU_IDS` |

RoboTwin 体积较大，不随 GitHub 代码仓库发布。请按照官方仓库 [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 下载或解压 RoboTwin；建议将官方 RoboTwin 目录放到：

```bash
benchmarks/robotwin/third_party/RoboTwin
```

如需放到其他位置，可通过 `ROBOTWIN_ROOT` 覆盖。

## 安装

需要 **两个** conda 环境。WAM 推理环境用于启动 HTTP 服务；RoboTwin 评测环境用于运行 SAPIEN 仿真。

### 1. WAM 推理环境

LIBERO-Plus 和 RoboTwin 共用同一个 WAM server 环境。请从 `benchmarks` 根目录安装公共依赖。

```bash
conda create -n kairos-wam python=3.10.12 -y
conda activate kairos-wam

cd benchmarks

python -m pip install -U pip setuptools wheel ninja packaging
python -m pip install -r common/requirment_server.txt
python -m pip install flash-attn==2.6.3 --no-build-isolation

# causal-conv1d 需要与 torch/CUDA ABI 匹配，普通预编译 wheel 可能 import 失败。
CUDA_HOME=/usr/local/cuda \
CAUSAL_CONV1D_FORCE_BUILD=TRUE \
CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE \
python -m pip install causal-conv1d==1.5.3.post1 --no-binary causal-conv1d --no-build-isolation
```

`kairos.modules.dits.*` 依赖 NVIDIA Apex 的 `FusedRMSNorm`, 请使用 NVIDIA Apex 源码或与 `torch==2.6.0+cu126` 匹配的内部 wheel：

```bash
cd /tmp
git clone --depth 1 https://github.com/NVIDIA/apex.git
cd apex

python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" .
```

安装后建议先验证关键 CUDA 扩展：

```bash
python - <<'PY'
import flash_attn
import causal_conv1d_cuda
from apex.normalization.fused_layer_norm import FusedRMSNorm
print("WAM server environment OK")
PY
```

也可直接将 `SERVER_PYTHON` 指向已按上述流程安装好的 KairosWAM server 环境 Python。

### 2. RoboTwin 评测环境

请先按照官方仓库 [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 准备 RoboTwin 代码：

```bash
cd benchmarks/robotwin
mkdir -p third_party
# 下载/解压官方 RoboTwin 到 third_party/RoboTwin
```

然后用本目录的 `requirements-eval.txt` 创建 RoboTwin 评测环境，并按 RoboTwin 官方说明补齐仿真器需要的系统依赖：

```bash
conda create -n robotwin-eval python=3.10 -y
conda activate robotwin-eval

cd benchmarks/robotwin
pip install -r requirements-eval.txt
```

完成后将 `CLIENT_PYTHON` 指向该环境的 Python，例如：

```bash
export CLIENT_PYTHON=/path/to/robotwin-eval/bin/python
```

`RoboTwin_client.sh` 会检查 `sapien` 和 `yaml` 是否可导入。

## 运行

### 方式一：一键运行（推荐）

在 shell 中执行：

```bash
cd benchmarks/robotwin

export WAM_PRETRAINED_DIT=/path/to/checkpoint.safetensors
export CLIENT_PYTHON=/path/to/robotwin-eval/bin/python

bash run.sh
```

默认行为：

- WAM 端口：`8006`（`WAM_PORT`）
- WAM GPU：`0,1,2,3`（`WAM_GPU_IDS`）
- 评测 GPU：默认与 `WAM_GPU_IDS` 一致（`EVAL_GPU_IDS`）
- RoboTwin 路径：`third_party/RoboTwin`（`ROBOTWIN_ROOT`）
- 输出根目录：`../../outputs/robotwin`（`OUTPUT_ROOT`）
- 评测输出目录：`${OUTPUT_ROOT}/eval`（`EVAL_OUTPUT_DIR`）
- WAM 日志：`${OUTPUT_ROOT}/wam_server.log`

### 方式二：连接已有 WAM 服务跑评测

如果 WAM 服务已经通过其他流程启动，并且 `/health` 返回 `workers_loaded=true`，可以只运行 RoboTwin 评测客户端：

```bash
cd benchmarks/robotwin

export CLIENT_PYTHON=/path/to/robotwin-eval/bin/python
export MODEL_ENDPOINT=http://127.0.0.1:8006
export DATASET_STATS_PATH=robotwin_dataset_stats.json
export EVAL_OUTPUT_DIR=../../outputs/robotwin/eval

bash RoboTwin_client.sh
```

### 等待 WAM 就绪

WAM 启动后多个 worker 加载模型通常需 **数分钟**。可手动检查：

```bash
curl http://127.0.0.1:8006/health
# workers_loaded=true 表示可以开始评测
```

## 常用环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WAM_PRETRAINED_DIT` | `run.sh` 内默认路径 | DiT checkpoint |
| `WAM_PORT` | `8006` | WAM HTTP 端口 |
| `WAM_GPU_IDS` | `0,1,2,3` | WAM worker 绑定的 GPU |
| `MODEL_ENDPOINT` | `http://127.0.0.1:${WAM_PORT}` | 评测客户端连接的 WAM 地址 |
| `DATASET_STATS_PATH` | `${PROJECT_ROOT}/robotwin_dataset_stats.json` | dataset stats JSON |
| `OUTPUT_ROOT` | `../../outputs/robotwin` | 运行日志、缓存和默认输出根目录 |
| `EVAL_OUTPUT_DIR` | `${OUTPUT_ROOT}/eval` | 评测结果输出目录 |
| `EVAL_GPU_IDS` | `${WAM_GPU_IDS}` | RoboTwin 评测 worker 使用的 GPU |
| `ROBOTWIN_ROOT` | `${PROJECT_ROOT}/third_party/RoboTwin` | RoboTwin 官方代码目录 |
| `EVAL_STEP_LIMIT_FILE` | `robotwin_eval/config/_eval_step_limit.yml` | 每个任务的评测步数配置 |
| `SERVER_PYTHON` | `run.sh` 内默认路径 | WAM server Python |
| `CLIENT_PYTHON` | 需用户指定或使用 `run.sh` 内默认路径 | RoboTwin eval Python |
| `EVAL_NUM_EPISODES` | `100` | 每个任务/阶段评测 episode 数 |
| `CHUNK_SIZE` | `5` | 每个 chunk 的 episode 数 |
| `MAX_TASKS_PER_GPU` | `run.sh`: `2`；单独运行 `RoboTwin_client.sh`: `5` | 每张评测 GPU 上并发任务数 |
| `RESUME` | `false` | 是否从已有输出继续 |
| `NUM_INFERENCE_STEPS` | `30` | WAM 采样步数 |

## 结果汇总

RoboTwin chunked eval 会将每个任务的输出写到 `EVAL_OUTPUT_DIR`。如需重新汇总已有结果，可运行：

```bash
cd benchmarks/robotwin
python -m robotwin_eval.resummarize_chunked_results <eval_output_dir>
```

如汇总脚本参数有调整，请以 `python -m robotwin_eval.resummarize_chunked_results --help` 输出为准。

## 故障排查

| 现象 | 处理 |
|------|------|
| `CLIENT_PYTHON missing sapien or yaml` | 确认 `CLIENT_PYTHON` 指向 RoboTwin 评测环境 |
| `ModuleNotFoundError: robotwin_eval` | 确认在 `benchmarks/robotwin` 下运行，或检查 `PYTHONPATH` |
| 找不到 RoboTwin 代码 | 将官方 RoboTwin 放到 `third_party/RoboTwin`，或设置 `ROBOTWIN_ROOT` |
| WAM `/health` 超时 | 增大 `WAM_READY_TIMEOUT_SEC`；查看 `${OUTPUT_ROOT}/wam_server.log` |
| 端口占用 | 修改 `WAM_PORT` 或停止已有服务 |
| GPU 任务过多 | 减小 `MAX_TASKS_PER_GPU` 或调整 `EVAL_GPU_IDS` |
