import sys
import os
import argparse

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
    input_args_d = mmengine.load(input_file)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = f'{output_dir}/output.mp4'

    input_args_d['save_path'] = save_path

    cfg = Config.fromfile(cfg_path)

    if checkpoint and checkpoint.lower() != 'none':
        cfg.pipeline.pretrained_dit = checkpoint

    print('build pipeline ...')
    pipeline = build_model_pipeline(cfg.pipeline)
    print('build pipeline done')

    print('start infer ...')
    pipeline(**input_args_d)
    print('infer done')

    if dist.is_initialized():
        dist.destroy_process_group()
