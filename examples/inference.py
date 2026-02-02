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
from kairos.modules.utils import save_video, save_image

def parse_args():
    parser = argparse.ArgumentParser(description='TRAIN_MODEL_LOOP')
    parser.add_argument('--input_file', default='', help='input_file')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
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
    if use_prompt_rewriter:
        prompt_rewriter_path = cfg.prompt_rewriter_path
        prompt_rewriter = PromptRewriter(prompt_rewriter_path)

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
    save_fps = 16
    if len(video) == 1:
        save_path = save_path.replace('.mp4', '.jpg')
        save_image(video[0], save_path)
    else:
        save_video(video, save_path, fps=save_fps, quality=5)
    
    if dist.is_initialized():
        dist.destroy_process_group()
