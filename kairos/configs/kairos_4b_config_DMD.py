
pretrained_dit = '/kairos_vepfs_volc/action/lipu/workshop/kairos_dmd_20260217_step_4_rcm/step-3000.safetensors'
# pretrained_dit = '/kairos-engineer/ModelZoo/Moda//Common/Predict/WM-Predict-LatentFeature-LatentFeature-Common-BF16-v0.0.1/model.safetensors'
text_encoder_path = '/kairos-engineer/ModelZoo/Moda/Common/GEncoder/WM-GEncoder-VL-LatentFeature-Common-BF16-v0.0.1/'
vae_path = '/kairos-engineer/ModelZoo/Moda//Common/GDecoder/WM-GDecoder-LatentFeature-Video-Common-BF16-v2.1.0//Wan2.1_VAE.pth' 

tokenizer_path=None

pipeline = dict(
    type='KairosEmbodiedAPI',
    pretrained_dit = pretrained_dit,
    parallel_mode="cp",
    use_cfg_parallel=False,
    pipeline_type='KairosEmbodiedPipeline_DMD',
    # trainable_models='dit',
    pipeline_args = dict(
        vae_path=vae_path,
        text_encoder_path=text_encoder_path,
        load_dit_fn='strict_load',
        selected_sampling_time = [1000, 800, 500, 100],
        dit_config = {
            "dit_type" : 'KairosDiT',
            "has_image_input": False,
            "patch_size": [1, 2, 2],
            "in_dim": 16,
            "dim": 2560,
            "ffn_dim": 10240,
            "freq_dim": 256,
            "text_dim": 3584,  # 4096 for T5
            "out_dim": 16,
            "num_heads": 20,
            "num_layers": 32,
            "eps": 1e-6,
            "seperated_timestep": True,
            "require_clip_embedding": False,
            "require_vae_embedding": False,
            "fuse_vae_embedding_in_latents": True,
            "dilated_lengths": [1, 1, 4, 1],
            "use_first_frame_cond": False,
            "use_seq_parallel": True,
            "use_tp_in_getaeddeltanet": True,
            "use_tp_in_self_attn": True,
            "attend_k0": False,
        },
    ),
)