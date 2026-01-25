
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

- Model ID: **[Qwen/Qwen2.5-VL-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)** from ModelScope
- Target directory: `models/Qwen/Qwen2.5-VL-7B-Instruct/`
#### Python:
```python
from modelscope import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir="models/Qwen/Qwen2.5-VL-7B-Instruct",
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

## Prepare inference configs 
> 💡 If you are not using the default save paths above, update the model paths in your config.  
> See **[`kairos/configs/kairos_4b_config.py`](kairos/configs/kairos_4b_config.py)** for examples.

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
> 💡 Tips: For more information of generating parameters, refer to `examples/example_*.json` and `__call__` function in the `kairos/pipelines/pipelines/kairos_video_pipeline.py`.


## Run Generation using single-GPU without Prompt Rewriter
> 💡 These commands require ≥80GB GPU VRAM. Recommended GPUs: NVIDIA A100 / A800 .

- example of t2v inference
```bash
bash examples/inference.sh examples/example_t2v.json
```

- example of ti2v inference
```bash
bash examples/inference.sh examples/example_ti2v.json
```

- example of i2v inference
```bash
bash examples/inference.sh examples/example_i2v.json
```

> 💡 Tips: other inference instructions are similar. See the `use_prompt_rewriter`  in  `examples/inference.py` for details.
