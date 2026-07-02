

KAIROS_MODEL_DIR = 'models'

pretrained_dit = f'{KAIROS_MODEL_DIR}/kairos-4B-robot-RoboTwin2.0/kairos-4B-robot-RoboTwin2.0.safetensors'
# text_encoder_path =  f'{KAIROS_MODEL_DIR}/Qwen/Qwen2.5-VL-7B-Instruct/'
vae_path = f'{KAIROS_MODEL_DIR}/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
prompt_rewriter_path = f'{KAIROS_MODEL_DIR}/Qwen/Qwen3-VL-8B-Instruct/'

pipeline = dict(
    type='KairosEmbodiedAPI',
    pretrained_dit = pretrained_dit,
    # tea_cache_l1_thresh= 0.1,
    # tea_cache_model_id= "Wan2.1-T2V-1.3B",
    # parallel_mode="cp",
    use_cfg_parallel=False,
    pipeline_type='KairosEmbodiedWAMPipeline',
    pipeline_args = dict(
        # trainable_keys=None,
        load_dit_fn='strict_load',
        vae_path=vae_path,
        text_encoder_config=dict(
            type='Qwen3_5_TextEncoder',
            from_pretrained=f'{KAIROS_MODEL_DIR}/Qwen/Qwen3.5-2B/'
        ),
        dit_config = dict(
            dit_type = 'Simple_Multi_DIT_Wrapper',
            video_dit = {
                "dit_type" : 'KairosDiTV2',
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 2560,
                "ffn_dim": 10240,
                "freq_dim": 256,
                "text_dim": 2048,  # 4096 for T5
                "out_dim": 16,
                "num_heads": 20,
                "num_layers": 32,
                "layers_settings": [
                    'SWA', 'SWA', 'DSWA', 'GATED',
                    'SWA', 'SWA', 'DSWA', 'GATED',
                    'SWA', 'SWA', 'DSWA', 'GATED',
                    'SWA', 'SWA', 'DSWA', 'GATED',
                    'SWA', 'SWA', 'DSWA', 'GATED',
                    'SWA', 'SWA', 'DSWA', 'GATED',
                    'SWA', 'SWA', 'DSWA', 'GATED',
                    'SWA', 'SWA', 'DSWA', 'GATED',
                ],
                "eps": 1e-6,
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
                "attn_method": 'flex',
                "restrict_history_query_to_history": True,
                "dilated_lengths": [4],
                "use_first_frame_cond": False,
                "use_seq_parallel": False,
                "use_tp_in_getaeddeltanet": False,
                "use_tp_in_self_attn": False,
                "attend_k0": False,
            },
            action_dit = {
                "dit_type" : 'KairosDiTV2',
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 7,       
                "out_dim": 7,       
                "attn_hidden_dim": 2560, # = video-dit dim
                "dim": 1024,
                "ffn_dim": 4096, # ffn_dim = dim * 4 (by default)
                "num_heads": 20,  # dim // num_heads == 2^^N
                "num_layers": 32,
                "freq_dim": 256,
                "text_dim": 2048,  # vlm
                "layers_settings": [
                    'FA', 'FA', 'FA', 'FA',
                    'FA', 'FA', 'FA', 'FA',
                    'FA', 'FA', 'FA', 'FA',
                    'FA', 'FA', 'FA', 'FA',
                    'FA', 'FA', 'FA', 'FA',
                    'FA', 'FA', 'FA', 'FA',
                    'FA', 'FA', 'FA', 'FA',
                    'FA', 'FA', 'FA', 'FA',
                ],
                "action_head_cfg": {
                    "action_state_dim": 14,  # action
                    "action_dim": 14,         # action
                },
                "eps": 1e-6,
                "seperated_timestep": False,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": False,
                "attn_method": 'flex',
                "dilated_lengths": [4],
                "use_first_frame_cond": False,
                "use_seq_parallel": False,
                "use_tp_in_getaeddeltanet": False,
                "use_tp_in_self_attn": False,
                "attend_k0": False,
            },
        ),
    ),
)

