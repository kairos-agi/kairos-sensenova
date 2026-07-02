# LIBERO-Plus Benchmark (KairosWAM)

在 LIBERO-Plus 扩展基准上评测 KairosWAM。架构与 LIBERO 相同：**WAM 推理服务** + **仿真评测客户端**，使用两个独立 conda 环境。

LIBERO-Plus 支持按 **category** 分批跑任务；开源入口统一为 `run.sh`。

## 目录结构

```
benchmarks/libero_plus/
├── README.md
├── run.sh                      # 一键：启动 WAM + 等待就绪 + 按 category 评测
├── configs/
│   └── libero_wam_infer_config.py
├── kairos_wam/
├── libero_plus_dataset_stats.json
├── requirements-eval.txt
└── third_party/
    └── LIBERO-plus/              # 需用户克隆/解压，GitHub 仓库不直接包含
```

WAM FastAPI 服务、等待脚本和 server 依赖文件复用 `benchmarks/common/` 下的公共实现。


## 安装

需要 **两个** conda 环境。

### 1. 评测环境（LIBERO-Plus 仿真 + Hydra）

```bash
conda create -n libero-plus-eval python=3.10.20 -y
conda activate libero-plus-eval

cd benchmarks/libero_plus
pip install -r requirements-eval.txt
```

### 2. WAM 推理环境

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

### 3. LIBERO-Plus 组件安装

`run.sh` 默认从 `benchmarks/libero_plus/third_party/LIBERO-plus` 读取 LIBERO-Plus。推荐放置方式：

```bash
cd benchmarks/libero_plus
mkdir -p third_party
git clone https://github.com/sylvestf/LIBERO-plus.git third_party/LIBERO-plus
```

如果放在其他位置，运行前设置：

```bash
export LIBERO_PKG_ROOT=/path/to/LIBERO-plus
```

### 4. Headless EGL / MuJoCo 渲染要求

LIBERO-Plus 使用 robosuite / MuJoCo 离屏渲染。无显示器环境下推荐使用 NVIDIA EGL，而不是 X11。运行前请确认 eval 环境能加载系统 EGL 与 NVIDIA EGL vendor：

```bash
python - <<'PY'
import ctypes

ctypes.CDLL("libEGL.so.1")
ctypes.CDLL("libEGL_nvidia.so.0")

from mujoco.egl import egl_ext as EGL
print("eglQueryDevicesEXT count =", len(EGL.eglQueryDevicesEXT()))
PY
```

`eglQueryDevicesEXT count` 应大于 0。`run.sh` 会自动生成临时 NVIDIA EGL vendor JSON，并设置 `__EGL_VENDOR_LIBRARY_FILENAMES` 指向该 JSON，以避免 GLVND 默认选到 Mesa vendor 导致队列环境中 EGL device 枚举为 0。

如果上述检查失败，优先修复容器 / 队列环境：

```bash
ldconfig -p | grep -E 'libEGL.so.1|libEGL_nvidia.so.0|libGLdispatch.so.0'
ls -l /usr/share/glvnd/egl_vendor.d /etc/glvnd/egl_vendor.d 2>/dev/null || true
ls -l /dev/nvidia* /dev/dri 2>/dev/null || true
```

常见处理方式是安装或挂载与宿主 NVIDIA driver 匹配的 GLVND / EGL 运行库，例如:

```bash
ln -s /path/to/libEGL.so.1.1.0 /usr/lib/x86_64-linux-gnu/libEGL.so.1
```

## 运行

### 一键按 category 跑

`run.sh` 会按 category 启动 WAM、等待 `/health` 就绪，然后运行评测。

```bash
cd benchmarks/libero_plus

export WAM_PRETRAINED_DIT=/path/to/step-XXXXX-ema.safetensors

bash run.sh
```

可指定 category 或 run id：

```bash
bash run.sh "Camera Viewpoints" "Sensor Noise"
bash run.sh run2 run7
```

## 常用环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WAM_PRETRAINED_DIT` | category 配置中的占位路径 | DiT checkpoint，运行前应设置为实际存在的文件 |
| `WAM_PORT` | `8005` | 与 LIBERO 的 8002 不同 |
| `MODEL_ENDPOINT` | `http://127.0.0.1:${WAM_PORT}` | 评测连接地址 |
| `WAM_GPU_IDS` | 按 category 配置 | WAM worker 绑定的 GPU；用 `bash run.sh --list` 查看 |
| `EVAL_GPU_IDS` | 按 category 配置 | 仿真评测 GPU；用 `bash run.sh --list` 查看 |
| `NUM_GPUS` / `MAX_TASKS_PER_GPU` | 按 category 配置 | 并行评测槽位数；用 `bash run.sh --list` 查看 |
| `DATASET_STATS_PATH` | `libero_plus_dataset_stats.json` | 归一化统计 |
| `OUTPUT_ROOT` | `../../outputs/libero_plus` | 运行日志和缓存根目录 |
| `EVAL_OUTPUT_ROOT` | `${OUTPUT_ROOT}/eval` | category 批次输出根目录 |
| `EVAL_OUTPUT_DIR` | 自动生成带时间戳路径 | 单个 category 批次输出目录 |
| `LIBERO_PKG_ROOT` | `${PROJECT_ROOT}/third_party/LIBERO-plus` | LIBERO-Plus 包路径 |
| `SERVER_PYTHON` | `run.sh` 内默认路径 | WAM server Python |
| `EVAL_PYTHON` | `run.sh` 内默认路径 | LIBERO-Plus eval Python |
| `LIBERO_EGL_VENDOR_MODE` | `nvidia` | EGL vendor 选择；`nvidia` 会强制使用 NVIDIA EGL vendor，`system` 使用系统 GLVND 默认选择 |
| `LIBERO_NVIDIA_EGL_VENDOR_JSON` | `${OUTPUT_ROOT}/nvidia_egl_vendor.json` | 自动生成的 NVIDIA EGL vendor JSON 路径 |
| `LIBERO_EGL_DEVICE_ID_MODE` | `physical` | EGL device id 模式；默认与物理 GPU id 对齐 |

## 结果汇总

```bash
cd kairos_wam
python experiments/libero/summarize_results.py --output_dir <eval_output_dir>
```

category 批次输出示例：

```
.../outputs/libero_plus/eval/libero_uncond_2cam224_1e-4/by_category_<run_id>_YYYYmmdd_HHMMSS/
```

## 故障排查

| 现象 | 处理 |
|------|------|
| `checkpoint not found` | 设置 `WAM_PRETRAINED_DIT` 并确认文件存在 |
| `ModuleNotFoundError: libero` | 检查 `LIBERO_PKG_ROOT` 与 `PYTHONPATH` |
| category 名称报错 | 名称须与 `task_classification.json` 完全一致 |
| 端口占用 | 修改 `WAM_PORT` 或分机器跑 run |
| WAM 加载慢 | 首次 torch inductor 编译较慢，可增大 `WAM_READY_TIMEOUT_SEC` |
| `eglQueryDevicesEXT returned 0` | 没有正确暴露 NVIDIA EGL；检查 `libEGL_nvidia.so.0`、GLVND vendor JSON、`/dev/nvidia*` 和 `/dev/dri` |
| `MUJOCO_EGL_DEVICE_ID ... between 0 and -1/0` | EGL device 枚举为空或数量不匹配；先按 Headless EGL 检查修复环境 |
