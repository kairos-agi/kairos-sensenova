
# Fetch Models

Download the required checkpoints and place them under a unified directory (e.g. `models/`) for easier management.

### Kairos checkpoint (Hugging Face)

- Download **[Kairos-sensenova-4B-pretrained-480p](https://huggingface.co/kairos-agi/kairos-sensenova-common)** from Hugging Face  
- Save to: `models/kairos-model/`

### Text/VLM encoder checkpoints (ModelScope)

- Download **[Qwen2.5-VL-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)** from ModelScope
- Save to: `models/Qwen/Qwen2.5-VL-7B-Instruct/`

- Download **[Qwen3-VL-8B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)** from ModelScope
- Save to: `models/Qwen/Qwen3-VL-8B-Instruct/`

### Video VAE checkpoint (ModelScope)

- Download **[Wan2.1-T2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B)**  from ModelScope
- Save to: `models/Wan2.1-T2V-14B/`


# Model Inference

## Prepare inference configs 
> 💡 If you are not using the default save paths above, update the model paths in your config.  
> See **[`kairos/configs/kairos_4b_config.py`](kairos/configs/kairos_4b_config.py)** for examples.

Kairos-sensenova-4B supports `T2V`/`I2V`/`TI2V` mode, select mode by setting the value of `prompt` and `input_image`. The main differences among these modes are shown below.
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
> 💡 These commands require ≥80GB GPU VRAM. Recommended GPUs: NVIDIA A100 80GB / A800 .

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


## Run Generation with Prompt Rewriter
> 💡 These commands require ≥80GB GPU VRAM. Recommended GPUs: NVIDIA A100 80GB / A800 .

Adding the parameter `true` at the end of all the above commands will enable prompt rewriter.

- example of t2v inference
```bash
bash examples/inference.sh examples/example_t2v.json true
```

> 💡 Tips: other inference instructions are similar. See the `use_prompt_rewriter`  in  `examples/inference.py` for details.
