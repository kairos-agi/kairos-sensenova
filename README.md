# Kairos 3.0

<p align="center">
    <img src="assets/logo_kairos.png" width="500"/>
<p>

<p align="center">
    💜 <a href="https://kairos.acerobotics.com">Kairos Platform</a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/kairos-agi">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/kairos-agi">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="">Paper</a> &nbsp&nbsp 

-----
Hi, it’s **Kairos** here. 

**Kairos 3.0** is an efficient **world foundation model** designed to learn the **dynamics, causality, and physical laws of the real world** by rendering them into long-horizon videos. This release marks the **first open-sourced version of Kairos**, with a strong emphasis on **scalable temporal modeling** and **physically consistent video synthesis**.

Key highlights of Kairos 3.0 include:

- 😀 **Temporally-Linear DiT Architecture**: Kairos 3.0 introduces a LinearDiT backbone that scales *linearly* with video length. By replacing quadratic temporal attention with a carefully designed hybrid of local and linear attentions, the model efficiently supports long video sequences while preserving temporal coherence.


- 😊 **Physically Grounded and Causally Consistent Generation**: The DiT architecture is explicitly structured to respect temporal causality. Through gated linear attention, Kairos 3.0 propagates global state information across time, enabling stable object permanence, coherent interactions, and physically plausible event evolution.


- 😋 **Complex and Long-Range Motion Modeling**: By interleaving sliding-window, dilated, and global linear attentions, Kairos 3.0 captures motion across multiple temporal scales—from fine-grained local dynamics to long-range dependencies spanning several seconds. This design enables the generation of complex, multi-stage motions with strong long-term consistency.

- 😍 **Massive World Data**: Kairos 3.0 is trained on hundreds of millions of video clips spanning diverse domains and data sources, such as *human-centric manipulation* and *physical phenomenon*. We further design a high-quality data curation pipeline tailored for world modeling, emphasizing informative, physically meaningful, and temporally rich training samples.


## 🔥 Latest News
* Dec 18, 2025: 👋 We have released the inference code of Kairos-sensenova-common-4B model. The checkpoints are coming soon. Stay tuned.

## 📑 Open-source Plan
- [x] Inference code
- [ ] Checkpoints of the pretrained and post-trained models
- [ ] Checkpoints of the distilled models
- [ ] Technical report

## 🪩 Model Architecture

### Overall Architecture

<p align="center">
    <img src="assets/framework_kairos.png" width="1000"/>
<p>

Kairos 3.0 is built upon a diffusion-based world modeling framework that integrates a high-compression video VAE, a multimodal (VLM-based) conditioning encoder, and a temporally scalable DiT backbone. The overall design emphasizes **long-horizon temporal modeling**, **physical consistency**, and **computational efficiency**.

- **Video VAE**: Kairos 3.0 adopts the WAN2.1 VAE, which provides strong reconstruction fidelity under aggressive compression. Specifically, a video of shape `3 × T × H × W` is encoded into a latent representation of size `16 × T/4 × H/8 × W/8`, corresponding to a compression ratio of 48×.

- **VLM-based Conditioning Encoder**: Text prompts are embedded using a vision-language model (VLM), enabling semantically rich conditioning. 

- **LinearDiT Backbone**:  
  At the core of Kairos 3.0 is a temporally-linear DiT architecture that replaces standard quadratic temporal attention with a hybrid design composed of linear- and local-attention mechanisms. This enables efficient modeling of long video sequences while maintaining strong temporal coherence and causal reasoning ability.

---

### Hybrid Linear Attention

<p align="center">
    <img src="assets/architecture_kairos.png" width="1000"/>
<p>

To achieve linear temporal complexity without sacrificing global consistency, Kairos 3.0 interleaves multiple complementary attention mechanisms. The LinearDiT backbone is organized into `M = 8` groups of hybrid blocks, where each group contains:
- `2 ×` Sliding Window Attention (SWA) blocks  
- `1 ×` Dilated Sliding Window Attention (DSWA) block  
- `1 ×` Gated Linear Attention (GLA) block  

This hierarchical composition balances **local motion modeling**, **mid-range temporal interaction**, and **global causal dependency capture**.

- **Sliding Window Attention (SWA)**: SWA focuses on fine-grained local temporal dynamics. Each SWA block attends to a window of `6 × L` tokens, where `L` denotes the number of spatial tokens per frame. This design is effective for modeling short-term motion continuity and local physical interactions.

- **Dilated Sliding Window Attention (DSWA)**: DSWA extends the temporal receptive field by introducing dilation. It uses the same window size (`6 × L`) but with a dilation factor of `6`, corresponding to one second at 24 FPS. This allows the model to capture longer-range temporal dependencies while maintaining linear complexity.

- **Gated Linear Attention (GLA)**: To model global temporal causality, Kairos 3.0 employs GatedDeltaNet, a modern gated linear attention variant. GLA enables information propagation across the entire video sequence in linear time, supporting long-horizon reasoning, object permanence, and physically consistent event evolution.


## Model Zoo

| Models              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| Kairos-sensenova-4B-pretrained    | release soon   | pretrained model |
| Kairos-sensenova-4B-posttrained    | release soon  | posttrained model |

## Run Kairos 3.0

#### Installation
Clone the repo:
```bash
git clone https://github.com/kairos-agi/kairos-sensenova.git
cd kairos-sensenova
```

Install dependencies:
```bash
# Ensure torch >= 2.4.0
# recommend python==3.10.12&&torch==2.6.0&&cuda==12.6

# 1. install torch fisrst
# ref to https://pytorch.org/get-started/locally/

# 2. install flash-attn & einops
pip install einops==0.8.1 psutil
pip install flash-attn==2.6.3 --no-build-isolatio

# 3. install apex
# ref to https://github.com/NVIDIA/apex

# 4. install other requirements 
pip install -r requirements/requirements.txt

```





#### Model Inference

##### prepare inference configs 
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
    "input_image":"examples/kairos/demo_image.jpg",
    "negative_prompt" : "",
    # other args ...
}

# mode: i2v
{
    "prompt":"",
    "input_image":"examples/kairos/demo_image.jpg",
    "negative_prompt" : "",
    # other args ...
}
```
> 💡 Tips: For more information of generating parameters, refer to `examples/kairos/example_*.json` and `__call__` function in the `kairos/pipelines/pipelines/kairos_video_pipeline.py`.


##### Run Generation using single-GPU
> 💡 These command can run on a GPU with at least 80GB VRAM.

- example of t2v inference
```bash
tools/inference.sh configs/kairos_4b/kairos_4b_config.py none examples/kairos/example_t2v.json output/t2v
```

- example of ti2v inference
```bash
tools/inference.sh configs/kairos_4b/kairos_4b_config.py none examples/kairos/example_ti2v.json output/ti2v
```

- example of i2v inference
```bash
tools/inference.sh configs/kairos_4b/kairos_4b_config.py none examples/kairos/example_i2v.json output/i2v
```

##### Run Generation using multi-GPUs
> 💡 These command can run on a GPU with at least 80GB VRAM.

- example of multi-gpu inference
```bash
tools/inference_multi_gpu.sh configs/kairos_4b/kairos_4b_config.py none examples/kairos/example_list.json output/multi_gpu
```


## Citation
If you find our work helpful, please cite us.

```
@article{kairos,
}
```

## License Agreement
This project is licensed under the Apache License, Version 2.0.  You may use, modify, and distribute this software in compliance with the License.  See the [LICENSE](LICENSE) file for details. Besides, this project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.


## Acknowledgements

We would like to thank the contributors to the [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and [HuggingFace](https://huggingface.co) repositories, for their open research.



