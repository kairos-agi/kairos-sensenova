from typing import Any, Callable, Dict, List, Optional, Union

import os
import torch
import torch.nn as nn

from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
from transformers import AutoProcessor

class Qwen3_5_TextEncoder(nn.Module):
    def __init__(self, 
        dtype=torch.bfloat16, 
        device='cuda',
        from_pretrained='',
        tokenizer_max_length = 1024,
        prompt_template_encode_start_idx=97,
        prompt_template_encode_post_len=5,
        max_pixels=448 * 448,
        system_prompt='',
        ):
        super().__init__()
        self.tokenizer_max_length = tokenizer_max_length
        self.prompt_template_encode_start_idx = prompt_template_encode_start_idx
        self.prompt_template_encode_post_len = prompt_template_encode_post_len
        self.max_pixels = max_pixels

        print('Loading text encoder (Qwen3_5_ForConditionalGeneration)')

        if from_pretrained:
            pretrain_path = from_pretrained
        else:
            QWORLD_HF_CHECKPOINTS_ROOT = os.environ.get('QWORLD_HF_CHECKPOINTS_ROOT','')
            pretrain_path = f'{QWORLD_HF_CHECKPOINTS_ROOT}/Qwen/Qwen3.5-2B'

        print(f'Loading text encoder from {pretrain_path} with dtype {dtype} and device {device}')
        self.qw_rm_sys_prompt_in_vlm = int(os.environ.get('QW_RM_SYS_PROMPT_IN_VLM','0')) == 1
        self.qw_drop_img_in_vlm = int(os.environ.get('QW_DROP_IMG_IN_VLM','0')) == 1

        print('RNV: qw_rm_sys_prompt_in_vlm: {}'.format(self.qw_rm_sys_prompt_in_vlm))
        print('RNV: qw_drop_img_in_vlm: {}'.format(self.qw_drop_img_in_vlm))

        self.text_encoder = Qwen3_5ForConditionalGeneration.from_pretrained(pretrain_path, dtype=dtype)
        print('Loading text tokenizer (Qwen3_5Tokenizer)')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)

        self.processor = AutoProcessor.from_pretrained(pretrain_path, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "right"

        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = (
                "Describe the video by detailing the following aspects: "
                "1. The main content and theme of the video. "
                "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
                "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
                "4. background environment, light, style and atmosphere. "
                "5. camera angles, movements, and transitions used in the video:"
            )
        
        print('TEXT_ENCODER-setting|| pretrain_path:  {}'.format(pretrain_path))


    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        images: Optional[Union[Any, List[Any]]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Returns:
            prompt_embeds: [B, L, D]  (only prompt-text token span, padded)
            encoder_attention_mask: [B, L] bool
        """
        device = device or torch.device("cuda")
        dtype = dtype or self.text_encoder.dtype
        self.text_encoder.to(device=device, dtype=dtype)

        # Normalize batch
        if isinstance(prompt, str):
            prompt_list = [prompt]
        else:
            prompt_list = prompt
        bsz = len(prompt_list)

        # Normalize images
        if images is None:
            images_list = [None] * bsz
        else:
            if isinstance(images, list):
                images_list = images
            else:
                images_list = [images]
            if len(images_list) != bsz:
                raise ValueError(f"len(images)={len(images_list)} must match len(prompt)={bsz}")

        chat_texts = []
        pre_texts = []
        post_texts = []
        for p, img in zip(prompt_list, images_list):
            if img is not None:
                user_content = [
                    {"type": "image"},
                    {"type": "text", "text": p},
                ]
            else:
                user_content = [
                    {"type": "text", "text": p},
                ]

            if self.qw_rm_sys_prompt_in_vlm:
                messages = [
                    {"role": "user", "content": user_content},
                ]
            else:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                ]

            s = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            chat_texts.append(s)

        # Encode multimodal inputs (processor will insert image tokens properly)
        model_inputs = self.processor(
            text=chat_texts,
            images=[img for img in images_list if img is not None] if any(img is not None for img in images_list) else None,
            padding=True,
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
            images_kwargs=dict(max_pixels=448 * 448, do_resize=True),
        )

        model_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in model_inputs.items()}

        out = self.text_encoder(
            **model_inputs,
            output_hidden_states=True,
        )
        hidden_states = out.hidden_states[-1]
        input_ids = model_inputs["input_ids"]
        attn_mask = model_inputs.get("attention_mask", torch.ones_like(input_ids, dtype=torch.long))

        prompt_h_list = []
        prompt_m_list = []
        for i in range(bsz):
            valid_len = int(attn_mask[i].sum().item())
            ids_i = input_ids[i, :valid_len].detach().cpu()

            start = self.prompt_template_encode_start_idx
            end = valid_len

            h = hidden_states[i, start:end, :]  # [Li, D]
            m = torch.ones((h.shape[0],), device=device, dtype=torch.bool)

            prompt_h_list.append(h)
            prompt_m_list.append(m)

        max_L = max(h.shape[0] for h in prompt_h_list) if bsz > 0 else 0
        if max_L == 0:
            D = hidden_states.shape[-1]
            prompt_embeds = hidden_states.new_zeros((bsz, 1, D))
            encoder_attention_mask = torch.zeros((bsz, 1), device=device, dtype=torch.bool)
            return prompt_embeds.to(dtype=dtype), encoder_attention_mask

        D = hidden_states.shape[-1]
        prompt_embeds = hidden_states.new_zeros((bsz, max_L, D))
        encoder_attention_mask = torch.zeros((bsz, max_L), device=device, dtype=torch.bool)

        for i, (h, m) in enumerate(zip(prompt_h_list, prompt_m_list)):
            L = h.shape[0]
            prompt_embeds[i, :L] = h
            encoder_attention_mask[i, :L] = m

        return prompt_embeds.to(dtype=dtype), encoder_attention_mask

    def encode_prompt(self, prompt, images=None, positive=True, device='cuda'):
        if self.qw_drop_img_in_vlm:
            _images = None
        else:
            _images = images

        embeds, attention_mask = self._get_qwen_prompt_embeds(
            prompt=prompt, 
            images=_images, 
            device=device
        )
        if embeds.shape[0] == 1:
            attention_mask = None
        return embeds, attention_mask

