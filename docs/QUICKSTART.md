
# Fetch Models

Download the required checkpoints and place them under a unified directory (e.g. `models/`) for easier management.
## Directory Layout
```text
models/
├── Kairos-model/
├── Qwen/
│   ├── Qwen2.5-VL-7B-Instruct/
│   └── Qwen3-VL-8B-Instruct/
└── Wan2.1-T2V-14B/
```

## Step1. Install download tools
```bash
pip install -U huggingface_hub modelscope
```
## Step2. Create target directories
```bash
mkdir -p models/Kairos-model models/Qwen models/Wan2.1-T2V-14B
```

## Step3. Download checkpoints
### Kairos checkpoint (Hugging Face)

- Model ID: **[kairos-agi/kairos-sensenova-common](https://huggingface.co/kairos-agi/kairos-sensenova-common)** from Hugging Face  
- Target directory: `models/Kairos-model/`
#### Option A (Recommended): Hugging Face CLI

Best for one-shot download with resume / incremental updates.
```bash
# If the repo is gated/private, authenticate first (optional)
hf auth login

# Download to the target directory
hf download kairos-agi/kairos-sensenova-common \
  --local-dir models/Kairos-model \
  --local-dir-use-symlinks False
```
#### Option B: Python
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="kairos-agi/kairos-sensenova-common",
    local_dir="models/Kairos-model",
)
```
> Note: Hugging Face may create a small metadata/cache folder under local_dir for incremental updates. This is expected.

### Text encoder model (ModelScope)

- Model ID: **[Qwen/Qwen2.5-VL-7B-Instruct-AWQ](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct-AWQ)** from ModelScope
- Target directory: `models/Qwen/Qwen2.5-VL-7B-Instruct-AWQ/`
#### Python:
```python
from modelscope import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    local_dir="models/Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
)
```
### Prompt Rewriter model (ModelScope)

- Model ID: **[Qwen/Qwen3-VL-8B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)** from ModelScope
- Target directory: `models/Qwen/Qwen3-VL-8B-Instruct/`
#### Python:
```python
from modelscope import snapshot_download

snapshot_download(
    "Qwen/Qwen3-VL-8B-Instruct",
    local_dir="models/Qwen/Qwen3-VL-8B-Instruct",
)
```
### Video VAE model (ModelScope)

- Model ID: **[Wan-AI/Wan2.1-T2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B)**  from ModelScope
- Target directory: `models/Wan2.1-T2V-14B/`
#### Python:
```python
from modelscope import snapshot_download

snapshot_download(
    "Wan-AI/Wan2.1-T2V-14B",
    local_dir="models/Wan2.1-T2V-14B",
)
```

# Model Inference

> 💡 Platform Notes
>
> The setup and inference procedure are largely the same for **A800/A100**, **RTX 5090**, and **Metax C500**.
> The following instructions describe the common workflow.
>
> Platform-specific differences mainly involve environment selection, such as choosing the correct Docker image for your hardware.

## Prepare inference configs 
> 💡 If you are not using the default save paths above, update the model paths in your config.  
> See **[`kairos/configs/kairos_4b_config.py`](kairos/configs/kairos_4b_config.py)** and **[`kairos/configs/kairos_4b_config_DMD.py`](kairos/configs/kairos_4b_config_DMD.py)** for examples.  
> In general, use `kairos_4b_config_DMD.py` for **`kairos-sensenova-robot-4B-480P-distilled`**, and `kairos_4b_config.py` for other Kairos models.

Kairos-sensenova-4B supports `T2V`/`I2V`/`TI2V` modes, select mode by setting the value of `prompt` and `input_image`. The main differences among these modes are shown below.
```python
# parameters for generating a sample is a dictionary.

# mode: t2v
{
    "prompt":"your prompt here",
    "input_image":"",
    "negative_prompt" : "",
    # other args ...
}

# mode: ti2v
{
    "prompt":"your prompt here",
    "input_image":"assets/demo_image.jpg",
    "negative_prompt" : "",
    # other args ...
}

# mode: i2v
{
    "prompt":"",
    "input_image":"assets/demo_image.jpg",
    "negative_prompt" : "",
    # other args ...
}
```
> 💡 Tips: For more information of generating parameters, refer to `examples/example_*.json` and `__call__` function in the `kairos/pipelines/pipelines/kairos_video_pipeline_DMD.py`.

## Run Generation using single-GPU without Prompt Rewriter

> 💡 Model / Config Matching
>
> Different models should be used with the corresponding `--config` file:
>
> - **`kairos-sensenova-robot-4B-480P-distilled`** → use **`kairos/configs/kairos_4b_config_DMD.py`**
> - **other Kairos models** → use **`kairos/configs/kairos_4b_config.py`**
>
> In addition, the example JSON should match the selected model and target resolution for the best results.
>
> - For **`kairos-sensenova-robot-4B-480P-distilled`**, please use the **480P JSON configs** such as:
>   - `examples/example_t2v_480P.json`
>   - `examples/example_ti2v_480P.json`
>   - `examples/example_i2v_480P.json`
>
> This distilled model is optimized for **480P**, and its performance at **720P** is not ideal.
>
- example of t2v inference
```bash
bash examples/inference.sh examples/_example_t2v_480P.json kairos/configs/kairos_4b_config_DMD.py
```

- example of ti2v inference
```bash
bash examples/inference.sh examples/example_ti2v_480P.json kairos/configs/kairos_4b_config_DMD.py
```

- example of i2v inference
```bash
bash examples/inference.sh examples/example_i2v_480P.json kairos/configs/kairos_4b_config_DMD.py
```

> 💡 Tips: The default video resolution is 480x640. You can modify the resolution in the example.json. For example, if you want to generate a 720P video, use `examples/example_i2v.json`, `examples/example_t2v.json` and `examples/example_ti2v.json`.

## Run Generation using Multi-GPU (torchrun)
> ⚠️ Multi-GPU supports two modes:
> - **Pure-TP** (no CFG-parallel): `tp_size = WORLD_SIZE`, requires `num_heads % WORLD_SIZE == 0`.
> - **CFG-parallel**: splits positive/negative branches across GPUs, currently supported **only for WORLD_SIZE = 4 or 8**.
> - **`kairos-sensenova-robot-4B-480P-distilled` does not support CFG-parallel**. Please use it without CFG-parallel enabled.
> ⚙️ Additional Config for Multi-GPU Inference

> When running with multi_gpu_inference.sh, you must enable tensor / sequence parallel related flags in the model config.

**How to enable**
> Edit kairos/configs/kairos_4b_config.py and set:
```python
pipeline = dict (
    ...

    # Enable for multi-GPU inference
    "use_seq_parallel": True,
    "use_tp_in_getaeddeltanet": True,
    "use_tp_in_self_attn": True,
)
```
> ⚠️ If these flags remain False, the pipeline may still run,
but the model will not actually shard computation across GPUs,
leading to only one GPU being heavily utilized.

### CFG-parallel (Optional, 4/8 GPUs)

CFG guidance normally evaluates both **positive (cond)** and **negative (uncond)** branches.  
With **CFG-parallel**, we run these two branches **on different GPUs in parallel**:

- `cfg_size = 2` (positive / negative)
- `tp_size = WORLD_SIZE // cfg_size`
- Each `cfg_group` contains exactly two ranks for the same `tp_rank`: `[positive, negative]`

**How to enable**
1) Launch with 4 or 8 GPUs (torchrun).
2) Enable CFG-parallel via input example_*.json:
```python
pipeline = dict(
    ...

    # Enable for cfg parallel
    use_cfg_parallel=True
)
```
>💡 Notes: `use_cfg_parallel` here enables **CFG-parallel** (distributed cond/uncond execution).
>💡 Example commands below use **4 GPUs** with `kairos/configs/kairos_4b_config_DMD.py`, and are intended for **`kairos-sensenova-robot-4B-480P-distilled`** with the corresponding **480P** example JSONs.
- example of t2v inference
```bash
bash examples/multi_gpu_inference.sh examples/example_t2v_480P.json kairos/configs/kairos_4b_config_DMD.py 4
```

- example of ti2v inference
```bash
bash examples/multi_gpu_inference.sh examples/example_ti2v_480P.json kairos/configs/kairos_4b_config_DMD.py 4
```

- example of i2v inference
```bash
bash examples/multi_gpu_inference.sh examples/example_i2v_480P.json kairos/configs/kairos_4b_config_DMD.py 4
```
## Enable TeaCache
> ⚙️ TeaCache (Optional, works on both single-GPU and multi-GPU)

> TeaCache can be enabled for both single-GPU and multi-GPU inference to reduce redundant DiT computation and improve throughput.

**How to enable**
> Edit kairos/configs/kairos_4b_config.py and uncomment:
```python
pipeline = dict(
    ...

    # Enable for TeaCache
    tea_cache_l1_thresh = 0.1,
    tea_cache_model_id = "Wan2.1-T2V-1.3B",
)
```
>💡 Notes: If TeaCache is disabled, inference still works, but may be slower depending on the workload.

>💡 Recommendation: We suggest starting from tea_cache_l1_thresh = 0.1 and then tuning it based on observed recomputation rate and output quality.

> 💡 Tips: other inference instructions are similar. See the `use_prompt_rewriter`  in  `examples/inference.py` for details.

## Enable vram_management
> ⚙️ vram_management (Optional, works on both single-GPU and multi-GPU)

> vram_management can be enabled for both single-GPU and multi-GPU inference to offload model between GPU and CPU under limited hardware resources.

**How to enable**
> Edit kairos/configs/kairos_4b_config.py and set:
```python
pipeline = dict(
    ...

    # Enable for vram_management
    vram_management_enabled = True
)
```