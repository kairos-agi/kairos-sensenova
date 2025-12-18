

KAIROS_MODEL_DIR = '/your/download/model_path'

pretrained_dit = f'{KAIROS_MODEL_DIR}/model.safetensors'
text_encoder_path =  f'{KAIROS_MODEL_DIR}/Qwen-Image' # Qwen-Image
vae_path = f'{KAIROS_MODEL_DIR}/Wan2.1-T2V-14B/Wan2.1_VAE.pth'  # Wan2.1-VAE


pipeline = dict(
    type='KairosPipeline',
    exec_mode='infer',
    pretrained_dit = pretrained_dit,
    pipeline_type='KairosVideoPipeline',
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
        },
    ),
)

