# RoboTwin Evaluation Client

This folder contains the qworld WAM client for RoboTwin evaluation.

## Layout

- `run_chunked_eval.py`: multi-task, multi-GPU chunk manager with resume support.
- `eval_single.py`: launches one RoboTwin task/config/chunk.
- `policy/qworld_wam_policy/`: RoboTwin policy adapter for the qworld WAM HTTP service.
- `robotwin_scripts/eval_policy_chunked.py`: patched RoboTwin evaluation script used by the client. It is kept here so `third_party/RoboTwin` can stay as an upstream checkout/symlink.
- `config/`: default RoboTwin task split files.

`third_party/RoboTwin` is expected to point to a RoboTwin checkout. In this workspace it is the relative symlink `../../RoboTwin-main`, resolving to:



## Client Only

If a qworld WAM server is already running:

```bash
python -m robotwin_eval.run_chunked_eval \
  --robotwin-root third_party/RoboTwin \
  --endpoint http://127.0.0.1:8006 \
  --dataset-stats-path /path/to/dataset_stats.json \
  --output-dir outputs/robotwin/test_run \
  --eval-step-limit-file robotwin_eval/config/evalstep_8parts/_eval_step_limit_1.yml \
  --eval-num-episodes 100 \
  --chunk-size 10 \
  --gpu-ids 0,1,2,3 \
  --max-tasks-per-gpu 2 \
  --resume
```

## Server + Client Entry Script

```bash
DATASET_STATS_PATH=/path/to/dataset_stats.json \
WAM_PRETRAINED_DIT=/path/to/model.safetensors \
WAM_GPU_IDS=0,1,2,3 \
EVAL_GPU_IDS=0,1,2,3 \
bash run.sh
```

The entry script starts the shared WAM service, waits for `/health`, then launches the chunked RoboTwin client.
