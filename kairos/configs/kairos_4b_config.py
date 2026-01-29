

KAIROS_MODEL_DIR = 'models'

pretrained_dit = f'{KAIROS_MODEL_DIR}/Kairos-model/kairos-model.safetensors'
text_encoder_path =  f'{KAIROS_MODEL_DIR}/Qwen/Qwen2.5-VL-7B-Instruct/'
vae_path = f'{KAIROS_MODEL_DIR}/Wan2.1-T2V-14B/Wan2.1_VAE.pth' 
prompt_rewriter_path = f'{KAIROS_MODEL_DIR}/Qwen/Qwen3-VL-8B-Instruct/'

pipeline = dict(
    type='KairosEmbodiedAPI',
    pretrained_dit = pretrained_dit,
    # tea_cache_l1_thresh= 0.1,
    # tea_cache_model_id= "Wan2.1-T2V-1.3B",
    pipeline_type='KairosEmbodiedPipeline',
    pipeline_args = dict(
        vae_path=vae_path,
        text_encoder_path=text_encoder_path,
        load_dit_fn='strict_load',
        dit_config = {
            "dit_type" : 'KairosDiT',
            "has_image_input": False,
            "patch_size": [1, 2, 2],
            "in_dim": 16,
            "dim": 2560,
            "ffn_dim": 10240,
            "freq_dim": 256,
            "text_dim": 3584,
            "out_dim": 16,
            "num_heads": 20,
            "num_layers": 32,
            "eps": 1e-6,
            "seperated_timestep": True,
            "require_clip_embedding": False,
            "require_vae_embedding": False,
            "fuse_vae_embedding_in_latents": True,
            "dilated_lengths": [1, 1, 6, 1],
            "use_seq_parallel": False,
            "use_tp_in_getaeddeltanet": False,
            "use_tp_in_self_attn": False,
        },
    ),
)

