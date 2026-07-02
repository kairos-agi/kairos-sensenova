"""WAM inference config for LIBERO-Plus evaluation (kairos_wam).

Aligned with benchmarks/libero/configs/libero_wam_infer_config.py:
- Qwen2.5-VL-7B text encoder (text_dim=3584)
- LIBERO action dims (7-DOF + gripper state dim 8)
"""

import os

KAIROS_MODEL_DIR = os.environ.get('KAIROS_MODEL_DIR', "")
_DEFAULT_pretrained_dit = ''
pretrained_dit = os.environ.get('WAM_PRETRAINED_DIT', _DEFAULT_pretrained_dit)

vae_path = f'{KAIROS_MODEL_DIR}/Wan-AI/Wan2.2-T2V-A14B/Wan2.1_VAE.pth'

pipeline = dict(
    type='KairosEmbodiedAPI',
    pretrained_dit=pretrained_dit,
    use_cfg_parallel=False,
    pipeline_type='KairosEmbodiedWAMPipeline',
    pipeline_args=dict(
        load_dit_fn='strict_load',
        vae_path=vae_path,
        text_encoder_config=dict(
            type='QwenVLTextEncoder',
            from_pretrained=f'{KAIROS_MODEL_DIR}/Qwen/Qwen2.5-VL-7B-Instruct',
        ),
        dit_config=dict(
            dit_type='Simple_Multi_DIT_Wrapper',
            video_dit={
                'dit_type': 'KairosDiTV2',
                'has_image_input': False,
                'patch_size': [1, 2, 2],
                'in_dim': 16,
                'dim': 2560,
                'ffn_dim': 10240,
                'freq_dim': 256,
                'text_dim': 3584,
                'out_dim': 16,
                'num_heads': 20,
                'num_layers': 32,
                'layers_settings': [
                    'SWA', 'SWA', 'DSWA', 'GATED',
                    'SWA', 'SWA', 'DSWA', 'GATED',
                    'SWA', 'SWA', 'DSWA', 'GATED',
                    'SWA', 'SWA', 'DSWA', 'GATED',
                    'SWA', 'SWA', 'DSWA', 'GATED',
                    'SWA', 'SWA', 'DSWA', 'GATED',
                    'SWA', 'SWA', 'DSWA', 'GATED',
                    'SWA', 'SWA', 'DSWA', 'GATED',
                ],
                'eps': 1e-6,
                'seperated_timestep': True,
                'require_clip_embedding': False,
                'require_vae_embedding': False,
                'fuse_vae_embedding_in_latents': True,
                'attn_method': 'flex',
                'dilated_lengths': [4],
                'use_first_frame_cond': False,
                'use_seq_parallel': False,
                'use_tp_in_getaeddeltanet': False,
                'use_tp_in_self_attn': False,
                'attend_k0': False,
            },
            action_dit={
                'dit_type': 'KairosDiTV2',
                'has_image_input': False,
                'patch_size': [1, 2, 2],
                'in_dim': 7,
                'out_dim': 7,
                'attn_hidden_dim': 2560,
                'dim': 1024,
                'ffn_dim': 4096,
                'num_heads': 20,
                'num_layers': 32,
                'freq_dim': 256,
                'text_dim': 3584,
                'layers_settings': [
                    'FA', 'FA', 'FA', 'FA',
                    'FA', 'FA', 'FA', 'FA',
                    'FA', 'FA', 'FA', 'FA',
                    'FA', 'FA', 'FA', 'FA',
                    'FA', 'FA', 'FA', 'FA',
                    'FA', 'FA', 'FA', 'FA',
                    'FA', 'FA', 'FA', 'FA',
                    'FA', 'FA', 'FA', 'FA',
                ],
                'action_head_cfg': {
                    'action_state_dim': 8,
                    'action_dim': 7,
                },
                'eps': 1e-6,
                'seperated_timestep': False,
                'require_clip_embedding': False,
                'require_vae_embedding': False,
                'fuse_vae_embedding_in_latents': False,
                'attn_method': 'flex',
                'dilated_lengths': [4],
                'use_first_frame_cond': False,
                'use_seq_parallel': False,
                'use_tp_in_getaeddeltanet': False,
                'use_tp_in_self_attn': False,
                'attend_k0': False,
            },
        ),
    ),
)
