# Docker Image Usage Guide

The project supports two primary usage modes:

- 🐳 **Docker (recommended)** — zero environment setup. This project provides **pre-built Docker images** to simplify environment setup and enable quick usage without local compilation.
- 📦 **flexible installation** — use pip to build your flexible local development.



The official images are hosted on **GitHub Container Registry (GHCR)**.

---

## 🐳 Docker (recommended)

- Registry: **GitHub Container Registry**
- Image name: `ghcr.io/kairos-agi/kairos-sensenova:v0.0.1`
- Platform: Linux (CUDA-enabled)


## 1. Pull the Docker Image
- you need to login to ghcr.io first through Personal Access Token (PAT)
```bash
echo ghp_xxxxxxxxxxxxxxxxx | docker login ghcr.io -u username --password-stdin
```
- then pull image 
```bash
docker pull gghcr.io/kairos-agi/kairos-sensenova:v0.0.1
```

## 2. Create a container using Docker
```bash
docker run --rm -it \
  --gpus all \
  -v $(pwd):/workspace \
  ghcr.io/kairos-agi/kairos-sensenova:v0.0.1 \
  bash
```

## 📦 flexible installation
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
pip install -r requirements.txt

```
