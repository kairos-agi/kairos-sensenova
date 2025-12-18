import sys
import os
import argparse
from tqdm import tqdm
from mmengine import Config
import mmengine
from mmengine.dist import init_dist, get_dist_info
import torch.distributed as dist

from kairos.pipelines.builder import build_model_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description='TRAIN_MODEL_LOOP')
    parser.add_argument('--config', default='', help='train config file path')
    parser.add_argument('--checkpoint', default='', help='model checkpoint')
    parser.add_argument('--input_file', default='', help='input_file')
    parser.add_argument('--output_dir', default='', help='output_dir')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    init_dist(launcher='pytorch')
    rank, world_size = get_dist_info()

    print('RANK: {} || WORLD_SIZE: {}'.format(rank, world_size))

    args = parse_args()

    cfg_path = args.config
    checkpoint = args.checkpoint

    if cfg_path == '':
        ValueError('config path is empty')
        exit()

    input_file = args.input_file
    if not input_file:
        ValueError('input_file path is empty')
        exit()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    input_infos = mmengine.load(input_file)

    input_args_lists = []
    for idx, input_info in enumerate(input_infos):
        save_path = f'{output_dir}/output_{idx}.mp4'
        input_info['save_path'] = save_path
        input_args_lists.append(input_info)

    cfg = Config.fromfile(cfg_path)

    if checkpoint and checkpoint.lower() != 'none':
        cfg.pipeline.pretrained_dit = checkpoint

    print('build pipeline ...')
    pipeline = build_model_pipeline(cfg.pipeline)
    print('build pipeline done')


    curr_rank_infos = input_args_lists[rank::world_size]

    print('start infer ...')
    if rank == 0:
        pbar = tqdm(desc='infer', total=len(input_args_lists))
    for curr_rank_info in curr_rank_infos:
        pipeline(**curr_rank_info)
        if rank == 0:
            pbar.update(world_size)
    
    if rank == 0:
        pbar.close()
    print('infer done')

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
