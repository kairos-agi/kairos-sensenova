from collections import defaultdict
import sys
import os
import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo as dynamo

dynamo.config.cache_size_limit = 4096
dynamo.config.accumulated_cache_size_limit = 4096

from safetensors import safe_open
from einops import rearrange
from torch.nn.attention.flex_attention import flex_attention

from .kairos_dit import KairosDiT
from .kairos_dit_v2 import KairosDiTV2
from kairos.third_party.fla.models.utils import Cache as fla_Cahce

from kairos.apis.builder import DITS

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

CUSTOM_DIT_D = dict(
    KairosDiT=KairosDiT,
    KairosDiTV2=KairosDiTV2,
)


@torch.compile(dynamic=False, mode='max-autotune')
def flex_attn_compiled(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_mod = None,
        block_mask = None,
        scale = None,
        enable_gqa= False,
        return_lse = False,
        kernel_options = None,):

    return flex_attention(query=query, key=key, value=value, block_mask=block_mask, score_mod=score_mod, scale=scale, enable_gqa=enable_gqa, return_lse=return_lse, kernel_options=kernel_options)

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor, 
    num_heads: int,
    ctx_mask: Optional[torch.Tensor] = None, 
    window_size=(-1,-1),
    compatibility_mode=False, 
    attn_method='flex',
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
):

    return_clone = True if attn_method == 'flex' and (use_gradient_checkpointing or use_gradient_checkpointing_offload) else False

    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            return torch.utils.checkpoint.checkpoint(
                real_flash_attention,
                q=q, k=k, v=v, num_heads=num_heads, ctx_mask=ctx_mask, window_size=window_size,
                compatibility_mode=compatibility_mode, attn_method=attn_method,return_clone=return_clone,
                use_reentrant=False,
            )
    elif use_gradient_checkpointing:
        return torch.utils.checkpoint.checkpoint(
            real_flash_attention,
            q=q, k=k, v=v, num_heads=num_heads, ctx_mask=ctx_mask, window_size=window_size,
            compatibility_mode=compatibility_mode, attn_method=attn_method,return_clone=return_clone,
            use_reentrant=False,
        )
    else:
        return real_flash_attention(
            q=q, k=k, v=v, num_heads=num_heads, ctx_mask=ctx_mask, window_size=window_size,
            compatibility_mode=compatibility_mode, attn_method=attn_method,return_clone=return_clone)

def real_flash_attention(
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        ctx_mask: Optional[torch.Tensor] = None,
        window_size=(-1,-1),
        compatibility_mode=False,
        attn_method='flex',
        return_clone=False):

    if compatibility_mode or attn_method =='torch_compat':
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        return x
    
    elif attn_method == 'flex':
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = flex_attn_compiled(query=q, key=k, value=v, block_mask=ctx_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        if not return_clone:
            return x
        else:
            return x.clone()
    elif attn_method == 'flash':
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(
            q, k, v, window_size=window_size,
        )
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
        return x
    else:
        raise NotImplementedError("only compatibility_mode & flex supported !!! curr attn_method: {}".format(attn_method))

@DITS.register_module()
class Simple_Multi_DIT_Wrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for dit_name, dit_config in kwargs.items():
            dit_type = dit_config.pop('dit_type')

            pretrained_path = dit_config.pop('pretrained_path', '')

            dit_cls = CUSTOM_DIT_D[dit_type]
            dit = dit_cls(**dit_config)
            total_params = sum(p.numel() for p in dit.parameters()) / 1e9
            print(f"Total parameters of {dit_name} DiT: {total_params:.3f} B")


            if pretrained_path != '':
                print('loading pretrained weights from {} for {}'.format(pretrained_path, dit_name))
                if pretrained_path.endswith('.pth') or pretrained_path.endswith('.pt'):
                    ckpt = torch.load(pretrained_path, map_location='cpu')
                elif pretrained_path.endswith('.safetensors'):
                    ckpt = {}
                    with safe_open(pretrained_path, framework="pt", device='cpu') as f:
                        for k in f.keys():
                            ckpt[k] = f.get_tensor(k)
                else:
                    raise ValueError('Unsupported pretrained file format: {}'.format(pretrained_path))
                
                dit_state = {}
                for k, v in ckpt.items():
                    if k.startswith('pipe.dit.{}.'.format(dit_name)):
                        dit_state[k[len('pipe.dit.{}.'.format(dit_name)):]] = v
                    elif k.startswith('dit.{}.'.format(dit_name)):
                        dit_state[k[len('dit.{}.'.format(dit_name)):]] = v
                    elif k.startswith('{}.'.format(dit_name)):
                        dit_state[k[len('{}.'.format(dit_name)):]] = v
                    else:
                        dit_state[k] = v
                missing, unexpected = dit.load_state_dict(dit_state, strict=True)


            setattr(self, dit_name, dit)

        total_params = sum(p.numel() for p in self.parameters()) / 1e9
        print(f"Total parameters of Simple_Multi_DIT_Wrapper: {total_params:.3f} B")

    def forward(self, x):
        raise NotImplementedError()


# *************************************************************************************
# mot code

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)

def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


# ***************************
# dit-block part1: beforce self-attn

def dit_block_forward_get_qkv(
        dit_block,
        tokens: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
    
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            return torch.utils.checkpoint.checkpoint(
                real_dit_block_forward_get_qkv,
                dit_block,
                tokens,
                freqs,
                t_mod,
                use_reentrant=False,
            )
    elif use_gradient_checkpointing:
        return torch.utils.checkpoint.checkpoint(
            real_dit_block_forward_get_qkv,
            dit_block,
            tokens,
            freqs,
            t_mod,
            use_reentrant=False,
        )
    else:
        return real_dit_block_forward_get_qkv(dit_block, tokens, freqs, t_mod)


def real_dit_block_forward_get_qkv(
    dit_block,
    tokens: torch.Tensor,
    freqs: torch.Tensor,
    t_mod: torch.Tensor,
):
    has_seq = len(t_mod.shape) == 4
    chunk_dim = 2 if has_seq else 1
    len_t_mod_shape = len(t_mod.shape)

    block = dit_block
    x = tokens

    scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = (
        block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
    if has_seq:
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = (
            scale_msa.squeeze(2), shift_msa.squeeze(2), gate_msa.squeeze(2),
            scale_mlp.squeeze(2), shift_mlp.squeeze(2), gate_mlp.squeeze(2),
        )

    use_seq_parallel = block.use_seq_parallel

    full_x = x

    input_x = modulate(block.self_attn_norm(full_x), shift_msa, scale_msa)

    if block.use_linear_attn:
        if input_x.shape[1] > block.gated_delta_chunk_size:
            cache = fla_Cahce.from_legacy_cache()
            outputs = []
            for start in range(0, input_x.shape[1], block.gated_delta_chunk_size):
                x_chunk = input_x[:, start:start+block.gated_delta_chunk_size, :]
                out_chunk, _, cache = block.gated_delta(
                    x_chunk,
                    past_key_values=cache,
                    use_cache=True
                )
                outputs.append(out_chunk)
            attn_out = torch.cat(outputs, dim=1)
        else:
            attn_out, _, _ = block.gated_delta(input_x)
        
        q = None
        k = None
        v = None
        gated_delta_out = attn_out
    else:
        q, k, v= block.self_attn.get_atten_qkv(input_x, f=None, freqs=freqs, L=None)
        gated_delta_out = None
    return q, k, v, full_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gated_delta_out, len_t_mod_shape


# ***************************
# dit-block part3: after self-attn [cross-atten & ffn]


def dit_block_forward_cross_ffn(
    dit_block,
    residual_x: torch.Tensor,
    gate_msa: torch.Tensor,
    shift_mlp: torch.Tensor,
    scale_mlp: torch.Tensor,
    gate_mlp: torch.Tensor,
    atten_out: list[int],
    context : torch.Tensor,
    context_mask : torch.Tensor,
    len_t_mod_shape: int,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
):
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            return torch.utils.checkpoint.checkpoint(
                real_dit_block_forward_cross_ffn,
                dit_block,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                atten_out,
                context,
                context_mask,
                len_t_mod_shape,
                use_reentrant=False,
            )
    elif use_gradient_checkpointing:
        return torch.utils.checkpoint.checkpoint(
            real_dit_block_forward_cross_ffn,
            dit_block,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            atten_out,
            context,
            context_mask,
            len_t_mod_shape,
            use_reentrant=False,
        )
    else:
        return real_dit_block_forward_cross_ffn(
            dit_block,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            atten_out,
            context,
            context_mask,
            len_t_mod_shape,
        )

def real_dit_block_forward_cross_ffn(
    dit_block,
    residual_x: torch.Tensor,
    gate_msa: torch.Tensor,
    shift_mlp: torch.Tensor,
    scale_mlp: torch.Tensor,
    gate_mlp: torch.Tensor,
    atten_out: list[int],
    context : torch.Tensor,
    context_mask : torch.Tensor,
    len_t_mod_shape: int,
):

    block = dit_block

    if not block.use_linear_attn:
        atten_out = block.self_attn.o(atten_out)
    else:
        pass

    full_x = block.gate(residual_x, gate_msa, atten_out)
    distributed_x = full_x
    # cross-attention
    distributed_input_x = block.cross_attn_norm(distributed_x)

    attn_out = block.cross_attn(distributed_input_x, context, attn_mask=context_mask)
    distributed_x = distributed_x + attn_out

    distributed_input_x = modulate(block.ffn_norm(distributed_x), shift_mlp, scale_mlp)
    
    distributed_ffn_out = block.ffn(distributed_input_x)

    distributed_x = block.gate(distributed_x, gate_mlp, distributed_ffn_out)

    x = distributed_x

    return x


def self_attention_swa_wrapper(
    *,
    q,
    k,
    v,
    num_heads,
    ctx_mask,
    extra_attn_info,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    attn_method):
    if attn_method == 'flex':
        attn_out = flash_attention(
            q=q,
            k=k,
            v=v,
            num_heads=num_heads,
            ctx_mask=ctx_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            attn_method='flex',
        )
    elif attn_method == 'flash':
        video_tokens_per_frame = extra_attn_info['video_tokens_per_frame']
        window_size = extra_attn_info['window_size']
        assert window_size > 0
        real_window_size = (video_tokens_per_frame*window_size, video_tokens_per_frame*window_size)
        attn_out = flash_attention(
            q=q,
            k=k,
            v=v,
            num_heads=num_heads,
            ctx_mask=None, 
            window_size=real_window_size,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            attn_method='flash',
        )    
    else:
        raise NotImplementedError('not supported attn_method: {} !!!'.format(attn_method))
    return attn_out


def self_attention_dswa_wrapper(
    *,
    q,
    k,
    v,
    num_heads,
    ctx_mask,
    extra_attn_info,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    attn_method):

    if attn_method == 'flex':
        attn_out = flash_attention(
            q=q,
            k=k,
            v=v,
            num_heads=num_heads,
            ctx_mask=ctx_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            attn_method='flex',
        )
    elif attn_method == 'flash':
        tokens_per_frame = extra_attn_info['video_tokens_per_frame']
        dilated_length = extra_attn_info['dilated_length']
        window_size = extra_attn_info['window_size']
        seq_len = extra_attn_info['dswa_seq_len']

        assert seq_len % tokens_per_frame == 0, "seq_len should be divisible by tokens_per_frame"
        assert seq_len == q.shape[1], "seq_len should be equal to the second dimension of q"

        use_dilated = dilated_length > 1 and seq_len // (dilated_length * tokens_per_frame) > 1

        assert window_size > 0
        real_window_size = (tokens_per_frame*window_size, tokens_per_frame*window_size)

        L = tokens_per_frame
        if use_dilated:
            assert seq_len % L == 0, "L should equal to the num of tokens per frame"
            pad_len = dilated_length * L - seq_len % (dilated_length * L)
            if pad_len != 0:
                q = F.pad(q, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))
            q = rearrange(q, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)
            k = rearrange(k, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)
            v = rearrange(v, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)

        attn_out = flash_attention(
            q=q,
            k=k,
            v=v,
            num_heads=num_heads,
            ctx_mask=None,
            window_size=real_window_size,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            attn_method='flash',
        )
        if use_dilated:
            attn_out = rearrange(attn_out, "(b d) (n l) c -> b (n d l) c", l=L, d=dilated_length)
            if pad_len != 0:
                attn_out = attn_out[:, :-pad_len]

    else:
        raise NotImplementedError('not supported attn_method: {} !!!'.format(attn_method))
    return attn_out


def self_attention_fa_wrapper(
    q,
    k,
    v,
    num_heads,
    ctx_mask,
    extra_attn_info,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    attn_method):
    if attn_method == 'flex':
        attn_out = flash_attention(
            q=q,
            k=k,
            v=v,
            num_heads=num_heads,
            ctx_mask=ctx_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            attn_method='flex',
        )
    elif attn_method == 'flash':
        window_size = extra_attn_info['window_size']
        assert window_size <= 0
        real_window_size = (-1, -1)
        attn_out = flash_attention(
            q=q,
            k=k,
            v=v,
            num_heads=num_heads,
            ctx_mask=None, 
            window_size=real_window_size,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            attn_method='flash',
        )    
    else:
        raise NotImplementedError('not supported attn_method: {} !!!'.format(attn_method))
    return attn_out


def self_attention_video_action_mixed_wrapper(
    *,
    q_chunks,
    k_chunks,
    v_chunks,
    num_heads,
    ctx_mask,
    extra_attn_info,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    block_type,
    attn_method):
    if attn_method == 'flex':
        q_cat = torch.cat(q_chunks, dim=1)
        k_cat = torch.cat(k_chunks, dim=1)
        v_cat = torch.cat(v_chunks, dim=1)
        attn_out = flash_attention(
            q=q_cat,
            k=k_cat,
            v=v_cat,
            num_heads=num_heads,
            ctx_mask=ctx_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            attn_method='flex',
        )
    elif attn_method == 'flash':
        raise NotImplementedError()
    return attn_out



# ***************************
# video-dit pre

def mot_pre_video_dit(
    dit,
    first_frame_latents,
    latents,
    context,
    context_mask,
    timestep=None,
    fuse_vae_embedding_in_latents=False,
    ):

    with torch.amp.autocast('cuda', dtype=torch.float32):
        if dit.seperated_timestep and fuse_vae_embedding_in_latents:

            if timestep is None:
                assert first_frame_latents is not None, 'predict future video, should set timestep'
                assert (latents.shape[2] - first_frame_latents.shape[2]) == 0, 'predict future video, should set timestep'
                timestep = 0

            timestep = torch.concat([
                torch.zeros((first_frame_latents.shape[2], latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
                torch.ones((latents.shape[2] - first_frame_latents.shape[2], latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
            ]).flatten()
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
            t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
        else:
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
            t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    t = t.to(dtype=latents.dtype)
    t_mod = t_mod.to(dtype=latents.dtype)
    timestep = timestep.to(dtype=latents.dtype)

    context = dit.text_embedding(context)

    x = latents
    x, (f, h, w) = dit.patchify(x, None)
    grid_size = (f, h, w)

    tokens_per_frame = h * w

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    output = {
        "x": x,
        "t": t,
        "t_mod": t_mod,
        "context": context,
        "context_mask": context_mask,
        "freqs": freqs,
        "grid_size": grid_size,
        "tokens_per_frame": tokens_per_frame,
        "batch_size": latents.shape[0],
    }
    return output


# ***************************
# action-dit pre

def mot_pre_action_dit(
    dit,
    action_tokens,
    timestep,
    context,
    context_mask,
    ):

    seq_len = action_tokens.shape[1]
    if seq_len > dit.freqs.shape[0]:
        raise ValueError(
            f"Action token length {seq_len} exceeds RoPE cache {dit.freqs.shape[0]}."
        )

    with torch.amp.autocast('cuda', dtype=torch.float32):
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    t = t.to(dtype=action_tokens.dtype)
    t_mod = t_mod.to(dtype=action_tokens.dtype)
    timestep = timestep.to(dtype=action_tokens.dtype)

    tokens = dit.action_encoder(action_tokens)

    context = dit.text_embedding(context)
    freqs = dit.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

    output = {
        "tokens": tokens,
        "t": t,
        "t_mod": t_mod,
        "context": context,
        "context_mask": context_mask,
        "freqs": freqs,
        "seq_len": seq_len,
        "batch_size": tokens.shape[0],
    }
    return output


# ***************************
# video-dit post

def mot_post_video_dit(dit, video_tokens, video_pre):
    x = video_tokens
    x = dit.head(x, video_pre['t'])
    x = dit.unpatchify(x, video_pre['grid_size'])
    return x


# ***************************
# action-dit post
def mot_post_action_dit(dit, action_tokens, action_pre):
    return dit.head(action_tokens)


# ***************************
# mot_blocks forward


def forward_one_layer(
    *,
    layer_idx,
    dit_d,
    available_dits,
    tokens_all,
    freqs_d,
    t_mod_d,
    attention_mask_d,
    context_d,
    embed_d,
    video_seq_len,
    input_kv_cache,
    return_kv_cache,
    kv_cache,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    gradient_checkpointing_level,
    attn_method,
): 

    assert gradient_checkpointing_level in ['block', 'op']

    block_level_grad_ckpt = gradient_checkpointing_level == 'block'

    if use_gradient_checkpointing_offload and block_level_grad_ckpt:
        with torch.autograd.graph.save_on_cpu():
            return torch.utils.checkpoint.checkpoint(
                _forward_one_layer,
                layer_idx=layer_idx,
                dit_d=dit_d,
                available_dits=available_dits,
                tokens_all=tokens_all,
                freqs_d=freqs_d,
                t_mod_d=t_mod_d,
                attention_mask_d=attention_mask_d,
                context_d=context_d,
                embed_d=embed_d,
                video_seq_len=video_seq_len,
                input_kv_cache=input_kv_cache,
                return_kv_cache=return_kv_cache,
                kv_cache=kv_cache,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
                attn_method=attn_method,
                use_reentrant=False,
            )
    elif use_gradient_checkpointing and block_level_grad_ckpt:
        return torch.utils.checkpoint.checkpoint(
            _forward_one_layer,
            layer_idx=layer_idx,
            dit_d=dit_d,
            available_dits=available_dits,
            tokens_all=tokens_all,
            freqs_d=freqs_d,
            t_mod_d=t_mod_d,
            attention_mask_d=attention_mask_d,
            context_d=context_d,
            embed_d=embed_d,
            video_seq_len=video_seq_len,
            input_kv_cache=input_kv_cache,
            return_kv_cache=return_kv_cache,
            kv_cache=kv_cache,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
            attn_method=attn_method,
            use_reentrant=False,
        )

    else:
        return _forward_one_layer(
            layer_idx=layer_idx,
            dit_d=dit_d,
            available_dits=available_dits,
            tokens_all=tokens_all,
            freqs_d=freqs_d,
            t_mod_d=t_mod_d,
            attention_mask_d=attention_mask_d,
            context_d=context_d,
            embed_d=embed_d,
            video_seq_len=video_seq_len,
            input_kv_cache=input_kv_cache,
            return_kv_cache=return_kv_cache,
            kv_cache=kv_cache,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            attn_method=attn_method,
        )

def _forward_one_layer(
    *,
    layer_idx,
    dit_d,
    available_dits,
    tokens_all,
    freqs_d,
    t_mod_d,
    attention_mask_d,
    context_d,
    embed_d,
    video_seq_len,
    input_kv_cache,
    return_kv_cache,
    kv_cache,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    attn_method,
):
    """Forward one DiT layer and return updated states."""
    q_chunks = []
    k_chunks = []
    v_chunks = []
    cached = {}
    gated_delta_outs = []

    block_types = []
    block_names = []

    for name in available_dits:
        block = dit_d[name].blocks[layer_idx]
        block_names.append(name)
        block_type = dit_d[name].layers_settings[layer_idx]
        block_types.append(block_type)
        q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gated_delta_out, len_t_mod_shape = dit_block_forward_get_qkv(
            dit_block=block,
            tokens=tokens_all[name],
            freqs=freqs_d[name],
            t_mod=t_mod_d[name],
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        q_chunks.append(q)
        k_chunks.append(k)
        v_chunks.append(v)
        gated_delta_outs.append(gated_delta_out)

        if return_kv_cache:
            kv_cache[name].append({"k": k, "v": v})

        cached[name] = {
            "residual_x": residual_x,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
            "len_t_mod_shape": len_t_mod_shape,
        }

    atten_out_slices_d = {}

    if "video" in embed_d:
        assert video_seq_len == embed_d["video"].shape[1]

    if "GATED" in block_types or "SKIP" in block_types or len(block_types) == 1:
        # only single dit / split infer
        for block_idx, block_type in enumerate(block_types):
            block_name = block_names[block_idx]
            block = dit_d[block_name].blocks[layer_idx]

            if block_type == "GATED":
                attn_out = gated_delta_outs[block_idx]
            elif block_type == "SKIP":
                atten_out_slices_d[block_name] = cached[block_name]["tokens"]
                continue
            elif block_type in ["FA", "SWA", "DSWA"]:
                if block_name == "video":
                    sub_mask, extra_attn_info = attention_mask_d[f"video_only@{block_type}"]
                    q = q_chunks[block_idx]
                    k = k_chunks[block_idx]
                    v = v_chunks[block_idx]
                    num_heads = block.self_attn.num_heads
                    
                    if block_type == "SWA":
                        attn_out = self_attention_swa_wrapper(
                            q=q,
                            k=k,
                            v=v,
                            num_heads=num_heads,
                            ctx_mask=sub_mask,
                            extra_attn_info=extra_attn_info,
                            use_gradient_checkpointing=use_gradient_checkpointing,
                            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                            attn_method=attn_method,
                        )
                    elif block_type == "DSWA":
                        attn_out = self_attention_dswa_wrapper(
                            q=q,
                            k=k,
                            v=v,
                            num_heads=num_heads,
                            ctx_mask=sub_mask,
                            extra_attn_info=extra_attn_info,
                            use_gradient_checkpointing=use_gradient_checkpointing,
                            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                            attn_method=attn_method,
                        )
                    elif block_type == "FA":
                        attn_out = self_attention_fa_wrapper(
                            q=q,
                            k=k,
                            v=v,
                            num_heads=num_heads,
                            ctx_mask=sub_mask,
                            extra_attn_info=extra_attn_info,
                            use_gradient_checkpointing=use_gradient_checkpointing,
                            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                            attn_method=attn_method,
                        )

                else:
                    # action branch
                    if len(input_kv_cache) != 0:
                        curr_kv_cache = input_kv_cache["video"][layer_idx]
                        k_video = curr_kv_cache["k"]
                        v_video = curr_kv_cache["v"]
                    else:
                        k_video = None
                        v_video = None

                    q = q_chunks[block_idx]
                    k = k_chunks[block_idx]
                    v = v_chunks[block_idx]
                    num_heads = block.self_attn.num_heads

                    if k_video is None:
                        # action split infer
                        sub_mask, extra_attn_info = attention_mask_d["action_only@FA"]
                        attn_out = self_attention_fa_wrapper(  # action is FA
                            q=q,
                            k=k,
                            v=v,
                            num_heads=num_heads,
                            ctx_mask=sub_mask,
                            extra_attn_info=extra_attn_info,
                            use_gradient_checkpointing=use_gradient_checkpointing,
                            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                            attn_method=attn_method,
                        )
                    else:
                        sub_mask, _ = attention_mask_d["action_with_video_cache@SWA"] if 'action_with_video_cache@SWA' in attention_mask_d else attention_mask_d["action_with_video_cache@FA"]
                        k_cat = torch.cat([k_video, k], dim=1)
                        v_cat = torch.cat([v_video, v], dim=1)
                        attn_out = self_attention_fa_wrapper(
                            q=q,
                            k=k_cat,
                            v=v_cat,
                            num_heads=num_heads,
                            ctx_mask=sub_mask,
                            extra_attn_info=dict(window_size=-1),
                            use_gradient_checkpointing=use_gradient_checkpointing,
                            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                            attn_method=attn_method,
                        )
            else:
                raise ValueError(f"Invalid block type: {block_type}")

            atten_out_slices_d[block_name] = attn_out
    else:
        block_type = dit_d["video"].layers_settings[layer_idx]
        atten_mask, extra_attn_info = attention_mask_d[f"video_action_mixed@{block_type}"]
        block = dit_d[block_names[0]].blocks[layer_idx]
        num_heads = block.self_attn.num_heads

        mixed_attn_out = self_attention_video_action_mixed_wrapper(
            q_chunks=q_chunks,
            k_chunks=k_chunks,
            v_chunks=v_chunks,
            num_heads=num_heads,
            ctx_mask=atten_mask,
            extra_attn_info=extra_attn_info,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            block_type=block_type,
            attn_method=attn_method,
        )
        video_attn_out = mixed_attn_out[:, :video_seq_len, :]
        action_attn_out = mixed_attn_out[:, video_seq_len:, :]
        atten_out_slices_d["video"] = video_attn_out
        atten_out_slices_d["action"] = action_attn_out

    out_tokens = dict()
    for name in available_dits:
        if cached[name].get("skip", False):
            out_tokens[name] = cached[name]["tokens"]
            continue

        block = dit_d[name].blocks[layer_idx]
        out = dit_block_forward_cross_ffn(
            dit_block=block,
            residual_x=cached[name]["residual_x"],
            gate_msa=cached[name]["gate_msa"],
            shift_mlp=cached[name]["shift_mlp"],
            scale_mlp=cached[name]["scale_mlp"],
            gate_mlp=cached[name]["gate_mlp"],
            len_t_mod_shape=cached[name]["len_t_mod_shape"],
            atten_out=atten_out_slices_d[name],
            context=context_d[name]["context"],
            context_mask=context_d[name]["context_mask"],
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        out_tokens[name] = out

    return out_tokens, kv_cache



def mot_dit_blocks_forward(
    dit_d,
    attention_mask_d,
    embed_d,
    freqs_d,
    context_d,
    t_mod_d,
    video_seq_len,
    input_kv_cache=dict(),
    return_kv_cache=False,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    gradient_checkpointing_level='block',
):  
    '''
        support mode: 
            1. video_cache_forward
            2. action_forward_with_video_cache
            3. video_forward
            4. video_action_joint_forward
    '''

    assert 'video' in dit_d or 'action' in dit_d
    if 'video' not in dit_d:
        assert 'video' in input_kv_cache
    if len(input_kv_cache) > 0:
        assert 'video' not in dit_d
        assert 'action' in dit_d and len(dit_d) == 1
        assert return_kv_cache == False
    if return_kv_cache:
        assert len(input_kv_cache) == 0

    if 'video' in dit_d:
        available_dits = ['video']
    else:
        available_dits = []
    for key in dit_d:
        if key not in available_dits:
            available_dits.append(key)

    # chech input mode
    # ***********************************
    _attn_methods = [dit_d[key].attn_method for key in dit_d]
    assert len(set(_attn_methods)) == 1, 'all dit should have the same attn_methods'
    attn_method = _attn_methods[0]

    num_layers_list  = [dit.num_layers for dit in dit_d.values()]
    assert len(set(num_layers_list)) == 1, 'all dit should have the same number of layers'
    num_layers = num_layers_list[0]

    tokens_all = {k: v for k, v in embed_d.items()}

    kv_cache = defaultdict(list)

    for layer_idx in range(num_layers):
        tokens_all, kv_cache = forward_one_layer(
            layer_idx=layer_idx,
            dit_d=dit_d,
            available_dits=available_dits,
            tokens_all=tokens_all,
            freqs_d=freqs_d,
            t_mod_d=t_mod_d,
            attention_mask_d=attention_mask_d,
            context_d=context_d,
            embed_d=embed_d,
            video_seq_len=video_seq_len,
            input_kv_cache=input_kv_cache,
            return_kv_cache=return_kv_cache,
            kv_cache=kv_cache,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            gradient_checkpointing_level=gradient_checkpointing_level,
            attn_method=attn_method,
        )

    if return_kv_cache:
        for key in kv_cache:
            assert len(kv_cache[key]) == num_layers
        tokens_all['kv_cache'] = kv_cache
        
    return tokens_all


def mot_infer_pure_video_dit_once(
    video_dit,
    latents,
    first_frame_latents,
    fuse_vae_embedding_in_latents,
    timestep,
    context,
    context_mask,
    attention_mask_d,

    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    gradient_checkpointing_level='block',
):
    # video-forward
    video_pre = mot_pre_video_dit(dit=video_dit,
                    first_frame_latents=first_frame_latents,
                    latents=latents, 
                    context=context, 
                    context_mask=context_mask,
                    timestep=timestep,
                    fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                    )
    video_tokens = video_pre["x"]
    output_d = mot_dit_blocks_forward(
        dit_d = {
            "video" : video_dit,
        },
        attention_mask_d=attention_mask_d,
        embed_d={
            "video": video_tokens,
        },
        freqs_d={
            "video": video_pre["freqs"],
        },
        context_d={
            "video": {
                'context': video_pre["context"],
                'context_mask' : video_pre["context_mask"],
            },
        },
        t_mod_d={
            "video": video_pre["t_mod"],
        },
        video_seq_len=video_tokens.shape[1],
        return_kv_cache=False,

        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        gradient_checkpointing_level=gradient_checkpointing_level,
    )
    video_tokens = output_d['video']
    pred_video = mot_post_video_dit(video_dit,video_tokens, video_pre)
    return pred_video


def mot_infer_pure_action_dit_once_with_video_cache(
    action_dit,
    latents_action,
    timestep_action,
    context,
    context_mask,
    attention_mask_d,
    video_kv_cache,
    video_seq_len,
):
    # typically use this func in for loop, so gen `attention_mask_d` outside this func

    action_pre = mot_pre_action_dit(
        dit=action_dit,
        action_tokens=latents_action,
        timestep=timestep_action,
        context=context,
        context_mask=context_mask,
    )
    action_dit_out = mot_dit_blocks_forward(
        dit_d = {
            "action" : action_dit,
        },
        attention_mask_d=attention_mask_d,
        embed_d={
            "action": action_pre['tokens'],
        },
        freqs_d={
            "action": action_pre["freqs"],
        },
        context_d={
            "action": {
                'context': action_pre["context"],
                'context_mask' : action_pre["context_mask"],
            },
        },
        t_mod_d={
            "action": action_pre["t_mod"],
        },
        video_seq_len=video_seq_len,
        input_kv_cache={
            'video' : video_kv_cache,
        },
        return_kv_cache=False,
    )
    action_tokens = action_dit_out['action']
    pred_action = mot_post_action_dit(action_dit, action_tokens, action_pre)
    return pred_action


def mot_infer_joint_video_action_dit_once(
    video_dit,
    action_dit,

    # *************************************************
    # shared inputs
    context: torch.Tensor = None,
    context_mask: torch.Tensor = None,
    attention_mask_d: dict = None,

    # *************************************************
    # video inputs
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,

    fuse_vae_embedding_in_latents: bool = False,
    first_frame_latents: Optional[torch.Tensor] = None,

    # *************************************************
    # action inputs
    action_latents: Optional[torch.Tensor] = None,
    timestep_action: Optional[torch.Tensor] = None,

    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    gradient_checkpointing_level='block',
):

    # typically use this func in for loop, so gen `attention_mask_d` outside this func

    video_pre = mot_pre_video_dit(dit=video_dit,
                    first_frame_latents=first_frame_latents,
                    latents=latents, 
                    context=context, 
                    context_mask=context_mask,
                    timestep=timestep,
                    fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                    )

    action_pre = mot_pre_action_dit(
        dit=action_dit,
        action_tokens=action_latents,
        timestep=timestep_action,
        context=context,
        context_mask=context_mask,
    )

    video_tokens = video_pre["x"]
    action_tokens = action_pre["tokens"]

    tokens_out = mot_dit_blocks_forward(
        dit_d = {
            "video" : video_dit,
            "action" : action_dit,
        },
        attention_mask_d=attention_mask_d,
        embed_d={
            "video": video_tokens,
            "action": action_tokens,
        },
        freqs_d={
            "video": video_pre["freqs"],
            "action": action_pre["freqs"],
        },
        context_d={
            "video": {
                'context': video_pre["context"],
                'context_mask': video_pre["context_mask"],
            },
            "action": {
                'context': action_pre["context"],
                'context_mask': action_pre["context_mask"],
            },
        },
        t_mod_d={
            "video": video_pre["t_mod"],
            "action": action_pre["t_mod"],
        },
        video_seq_len=video_tokens.shape[1],

        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        gradient_checkpointing_level=gradient_checkpointing_level,
    ) 

    pred_video = mot_post_video_dit(video_dit,tokens_out["video"], video_pre)

    pred_action = mot_post_action_dit(action_dit,tokens_out["action"], action_pre)

    return pred_video, pred_action

