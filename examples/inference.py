import sys
import os
import argparse

from mmengine import Config
import mmengine
from mmengine.dist import init_dist, get_dist_info

import torch.distributed as dist
from kairos.apis.builder import build_model_pipeline
from kairos.modules.utils.prompt_rewriter import PromptRewriter
import torch
from PIL import Image
from kairos.modules.utils import save_video, save_image, parallel_state

# default 0（general devices）
IS_METAX = os.environ.get("IS_METAX", "0") == "1"

if IS_METAX:
    print("metax devices")
    if not hasattr(torch, "maca"):
        torch.maca = torch.cuda

    if hasattr(torch, "get_autocast_dtype"):
        _orig_get_autocast_dtype = torch.get_autocast_dtype

        def _patched_get_autocast_dtype(device_type: str):
            # fla / triton ~G~L~B~^~\| ~F 'maca'~L~_~@~S~H~P 'cuda'
            if device_type == "maca":
                device_type = "cuda"
            return _orig_get_autocast_dtype(device_type)

        torch.get_autocast_dtype = _patched_get_autocast_dtype

    if hasattr(torch, "is_autocast_enabled"):
        _orig_is_autocast_enabled = torch.is_autocast_enabled

        def _patched_is_autocast_enabled(device_type: str = "cuda"):
            if device_type == "maca":
                device_type = "cuda"
            return _orig_is_autocast_enabled(device_type)

        torch.is_autocast_enabled = _patched_is_autocast_enabled
else:
    print("general devices")



def parse_args():
    parser = argparse.ArgumentParser(description='TRAIN_MODEL_LOOP')
    parser.add_argument('--input_file', default='', help='input_file')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    init_dist(launcher='pytorch')
    rank, world_size = get_dist_info()
 
    print('RANK: {} || WORLD_SIZE: {}'.format(rank, world_size))

    args = parse_args()

    cfg_path = "kairos/configs/kairos_4b_config.py"

    if cfg_path == '':
        ValueError('config path is empty')
        exit()

    input_file = args.input_file
    input_args_d = mmengine.load(input_file)
    
    output_dir = input_args_d['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    save_path = f'{output_dir}/output.mp4'

    cfg = Config.fromfile(cfg_path)

    use_prompt_rewriter = input_args_d['use_prompt_rewriter']
    if rank == 0 and use_prompt_rewriter:
        prompt_rewriter_path = cfg.prompt_rewriter_path
        prompt_rewriter = PromptRewriter(prompt_rewriter_path)

    # ***********************************************************
    # Initialize parallel state
    # ---- defaults: force no parallel unless multi-GPU + dist initialized ----
    use_cfg_parallel = cfg.pipeline.get("use_cfg_parallel")
    parallel_state.reset_cfg()
    is_multi_gpu_dist = (world_size > 1) and dist.is_initialized()

    # Enforce: no parallelism in non-multi-GPU environments
    if not is_multi_gpu_dist:
        use_cfg_parallel = False  # treat cfg-parallel as OFF
        if rank == 0:
            print(f"[init] dist=OFF or WORLD_SIZE=1 -> force parallel OFF ", flush=True)
    else:
        parallel_state.set_vae_group(dist.group.WORLD)

        if use_cfg_parallel:
            cfg_size = 2  # fixed 2-way CFG (pos/neg)
            assert world_size in (4, 8), (
                f"CFG-parallel only supports WORLD_SIZE in {{4, 8}}, got {world_size}. "
                f"Please re-launch with 4 or 8 GPUs, or disable CFG-parallel."
            )
            tp_size = world_size // cfg_size  # -> 2 or 4
            assert tp_size in (2, 4), (
                f"CFG-parallel derived tp_size={tp_size} is not supported. "
                f"Supported tp_size: {{2, 4}}"
            )
        else:
            cfg_size = 1
            tp_size = world_size
            # attention head number
            NUM_HEADS = 20
            assert NUM_HEADS % tp_size == 0, (
                f"Pure-TP mode requires num_heads({NUM_HEADS}) % tp_size({tp_size}) == 0. "
                f"Please change GPU count or enable CFG-parallel."
            )

        # ---- TP group ----
        if tp_size == world_size:
            tp_group = dist.group.WORLD
            tp_rank = rank
            tp_gid = 0
        else:
            tp_group, tp_rank, tp_gid = parallel_state.init_tp_groups(tp_size=tp_size)

        parallel_state.set_tp_group(tp_group)

        # ---- CFG group (only when cfg-parallel enabled) ----
        if use_cfg_parallel:
            # build all cfg groups in same order on all ranks
            cfg_groups = []
            for tp_r in range(tp_size):
                ranks_g = [tp_r + i * tp_size for i in range(cfg_size)]
                cfg_groups.append(dist.new_group(ranks=ranks_g))

            cfg_group = cfg_groups[tp_rank]
            cfg_rank = rank // tp_size
            parallel_state.set_cfg_group(cfg_group, cfg_rank=cfg_rank, cfg_size=cfg_size)
            print(
                f"[init] rank={rank} tp_rank={tp_rank} cfg_rank={cfg_rank} "
                f"tp_group={dist.get_world_size(tp_group)} "
                f"cfg_group={dist.get_world_size(cfg_group)}",
                flush=True
            )
        else:
            if rank == 0:
                print(f"[init] cfg_parallel=OFF tp_size={tp_size}", flush=True)

    print('build pipeline ...')
    pipeline = build_model_pipeline(cfg.pipeline)
    print('build pipeline done')

    print('start infer ...')
    raw_prompt = input_args_d.get('prompt','')
    if raw_prompt.strip() != '':
        if use_prompt_rewriter:
            rewritten_prompt = prompt_rewriter.rewrite_prompt(raw_prompt, image_path=input_args_d.get('input_image',''))
        else:
            rewritten_prompt = raw_prompt
        
        input_args_d['raw_prompt'] = raw_prompt
        input_args_d['prompt'] = rewritten_prompt

        print('rewritten prompt from [{}] to [{}]'.format(raw_prompt, rewritten_prompt))

    input_args_d.pop('output_dir')
    input_args_d.pop('use_prompt_rewriter')

    # open image
    if 'input_image' in input_args_d and isinstance(input_args_d['input_image'], str):
        if not input_args_d['input_image']:
            input_args_d.pop('input_image')
        else:
            raw_imgage = input_args_d['input_image']
            image = Image.open(input_args_d['input_image'])
            input_args_d['input_image'] = [image]

    input_args_d.pop('raw_prompt', '') 

    # add prompt prefix in ti2v mode
    if raw_prompt.strip() != '' and 'input_image' in input_args_d:
        prompt_prefix = 'high-quality video, realistic motion, single continuous shot, no jump cuts, smooth motion. '
        input_args_d['prompt'] = prompt_prefix + input_args_d['prompt']

    video = pipeline(**input_args_d)
    print('infer done')

    # save video
    if rank == 0:
        save_fps = 16
        if len(video) == 1:
            save_path = save_path.replace('.mp4', '.jpg')
            save_image(video[0], save_path)
        else:
            save_video(video, save_path, fps=save_fps, quality=5)

    if dist.is_initialized():
        dist.destroy_process_group()
