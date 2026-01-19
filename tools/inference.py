import sys
import os
import argparse

from mmengine import Config
import mmengine
from mmengine.dist import init_dist, get_dist_info

import torch.distributed as dist
from kairos.pipelines.builder import build_model_pipeline
from kairos.utils.prompt_rewriter import PromptRewriter


def parse_args():
    parser = argparse.ArgumentParser(description='TRAIN_MODEL_LOOP')
    parser.add_argument('--config', default='', help='train config file path')
    parser.add_argument('--checkpoint', default='', help='model checkpoint')
    parser.add_argument('--input_file', default='', help='input_file')
    parser.add_argument('--output_dir', default='', help='output_dir')
    parser.add_argument('--use_prompt_rewriter', default='false', help='use_prompt_rewriter')

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

    use_prompt_rewriter = args.use_prompt_rewriter.lower().strip() in ['1', 'true', 'yes']
    if use_prompt_rewriter:
        prompt_rewriter_path = cfg.prompt_rewriter_path
        prompt_rewriter = PromptRewriter(prompt_rewriter_path)

    print('build pipeline ...')
    pipeline = build_model_pipeline(cfg.pipeline)
    print('build pipeline done')

    print('start infer ...')

    raw_prompt = input_args_d.get('prompt','')
    if raw_prompt.strip() != '':
        rewritten_prompt = prompt_rewriter.rewrite_prompt(raw_prompt, image_path=input_args_d.get('input_image',''))
        input_args_d['raw_prompt'] = raw_prompt
        input_args_d['prompt'] = rewritten_prompt
        print('rewritten prompt from [{}] to [{}]'.format(raw_prompt, rewritten_prompt))

    pipeline(**input_args_d)
    print('infer done')

    if dist.is_initialized():
        dist.destroy_process_group()
