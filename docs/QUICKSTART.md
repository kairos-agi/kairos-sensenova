
#  Fetch Models

> Download models
- Download kairos model from <a href="https://huggingface.co/kairos-agi/kairos-sensenova-common">Kairos-sensenova-4B-pretrained-480p</a>

- Download Qwen2.5-VL-7B-Instruct from ModelScope

- Download Qwen/Qwen3-VL-8B-Instruct from ModelScope

> Configure the model path in the configuration file you are using.
Refer to the model path settings in [`kairos/configs/kairos_4b_config.py`](kairos/configs/kairos_4b_config.py).

# Model Inference

## prepare inference configs 
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
> 💡 Tips: For more information of generating parameters, refer to `examples/kairos/example_*.json` and `__call__` function in the `kairos/pipelines/pipelines/kairos_video_pipeline.py`.


## Run Generation using single-GPU without Prompt Rewriter
> 💡 These command can run on a GPU with at least 80GB VRAM.

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
bash examples/inference.sh xamples/example_i2v.json
```


## Run Generation with Prompt Rewriter
> 💡 These command can run on a GPU with at least 80GB VRAM.

Adding the parameter `true` at the end of all the above commands will enable prompt rewriter.

- example of t2v inference
```
examples/inference.sh kairos/configs/kairos_4b_config.py none examples/kairos/example_t2v.json output/t2v true
```

> 💡 Tips: other inference instructions are similar. See the `use_prompt_rewriter`  in  `examples/inference.py` for details.
