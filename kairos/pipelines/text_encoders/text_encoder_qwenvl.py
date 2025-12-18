from typing import Any, Callable, Dict, List, Optional, Union

import os
import torch
import torch.nn as nn

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer


class QwenVLTextEncoder(nn.Module):
    def __init__(self, dtype=torch.bfloat16, device='cuda', from_pretrained=''):
        super().__init__()
        self.tokenizer_max_length = 384
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        
        if from_pretrained:
            print()
            text_encoder_path = f'{from_pretrained}/text_encoder'
            text_tokenizer_path = f'{from_pretrained}/tokenizer'
        else:
            KAIROS_HF_CHECKPOINTS_ROOT = os.environ.get('KAIROS_HF_CHECKPOINTS_ROOT','.')
            text_encoder_path = f'{KAIROS_HF_CHECKPOINTS_ROOT}/Qwen/Qwen-Image/text_encoder'
            text_tokenizer_path = f'{KAIROS_HF_CHECKPOINTS_ROOT}/Qwen/Qwen-Image/tokenizer'

        print('Loading text encoder (Qwen2_5_VLForConditionalGeneration)')
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(text_encoder_path)
        print('Loading text tokenizer (Qwen2Tokenizer)')
        self.tokenizer = Qwen2Tokenizer.from_pretrained(text_tokenizer_path)
        self.text_encoder.to(device=device, dtype=dtype)

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.text_encoder.to(device=device, dtype=dtype)
        device = device or self.text_encoder.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        ).bool()

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask
    
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def encode_prompt(self, prompt, positive=True, device='cuda'):
        embeds, attention_mask = self._get_qwen_prompt_embeds(prompt, device=device)
        if embeds.shape[0] == 1:
            attention_mask = None
        return embeds, attention_mask
    
