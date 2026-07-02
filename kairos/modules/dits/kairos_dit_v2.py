import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
from einops import rearrange
from transformers.activations import ACT2CLS
from apex.normalization.fused_layer_norm import FusedRMSNorm

FLASH_ATTN_2_AVAILABLE = False
FLASH_ATTN_3_AVAILABLE = False
SAGE_ATTN_AVAILABLE = False

try:
    import flash_attn_interface
    # FLASH_ATTN_3_AVAILABLE = True
    # NOTE: forcing disable flash_attn3 for a800
    FLASH_ATTN_3_AVAILABLE = False
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

IS_CUDA = torch.cuda.is_available()
if not IS_CUDA:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

from kairos.modules.utils import FLAGS_KAIROS_CUDA_SM

SUPPORTED_ARCHS = {80, 89, 120, 121}
if FLAGS_KAIROS_CUDA_SM in SUPPORTED_ARCHS:
    try:
        from sageattention import sag_attention_with_window,sageattn
        SAGE_ATTN_AVAILABLE = True
    except ModuleNotFoundError:
        SAGE_ATTN_AVAILABLE = False
try:
    from fla.layers import GatedDeltaNet
except ModuleNotFoundError:
    pass

from kairos.apis.builder import DITS

from kairos.third_party.fla.layers import GatedDeltaNet
from kairos.third_party.fla.layers.gated_deltanet_with_tp import GatedDeltaNet as GatedDeltaNetWithTP
from kairos.third_party.fla.models.utils import Cache as fla_Cahce


import torch.distributed as dist


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, 
                    compatibility_mode=False, attn_mask=None, window_size=(-1, -1), return_attn_probs=False):
    if compatibility_mode or attn_mask is not None:
        q = rearrange(q, "b s n d -> b n s d")
        k = rearrange(k, "b s n d -> b n s d")
        v = rearrange(v, "b s n d -> b n s d")
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = rearrange(x, "b n s d -> b s n d")
    elif FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_func(
            q, k, v, window_size=window_size,
            return_attn_probs=return_attn_probs
        )
        if return_attn_probs:
            x, probs = x[0], x[1]
            return x, probs
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_func(
            q, k, v, window_size=window_size,
            return_attn_probs=return_attn_probs
        )
        if return_attn_probs:
            x, probs = x[0], x[1]
            return x, probs
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s n d -> b n s d")
        k = rearrange(k, "b s n d -> b n s d")
        v = rearrange(v, "b s n d -> b n s d")
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s n d")
    else:
        raise RuntimeError("do not use pytorch attention")
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    x.mul_(1 + scale)
    x.add_(shift)
    return x


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    assert dim % 2 == 0
    half = dim // 2
    h_half = half // 3
    w_half = half // 3
    f_half = half - h_half - w_half
    f_freqs_cis = precompute_freqs_cis(2 * f_half, end, theta)
    h_freqs_cis = precompute_freqs_cis(2 * h_half, end, theta)
    w_freqs_cis = precompute_freqs_cis(2 * w_half, end, theta)
    
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)

def rope_apply_for3d(x, num_frames, freqs, num_heads):
    B, L, D = x.shape
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)

    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)

    x_out = x_out.reshape(B, L, D)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, attn_mask=None, window_size=(-1, -1)):
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, attn_mask=attn_mask, window_size=window_size)
        x = rearrange(x, "b s n d -> b s (n d)", n=self.num_heads)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim: int, attn_hidden_dim: int, num_heads: int, eps: float = 1e-6, dilated_length=1, window_size=3, attend_k0=False,attend_k0_with_module=False):
        super().__init__()
        self.dim = dim
        self.attn_hidden_dim = attn_hidden_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dilated_length = dilated_length
        self.window_size = window_size
        self.attend_k0 = attend_k0
        self.attend_k0_with_module = attend_k0_with_module

        self.q = nn.Linear(dim, attn_hidden_dim)
        self.k = nn.Linear(dim, attn_hidden_dim)
        self.v = nn.Linear(dim, attn_hidden_dim)
        self.o = nn.Linear(attn_hidden_dim, dim)
        self.norm_q = FusedRMSNorm(attn_hidden_dim, eps=eps)
        self.norm_k = FusedRMSNorm(attn_hidden_dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)
    
    def extra_repr(self):
        return f'dilated_length={self.dilated_length}, window_size={self.window_size}'
    
    def forward(self, x, f, freqs, L=1):
        dilated_length = self.dilated_length

        l_q = self.q(x)
        l_k = self.k(x)

        q = self.norm_q(l_q)
        k = self.norm_k(l_k)

        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        use_dilated = dilated_length > 1 and x.shape[1] // (dilated_length * L) > 1
        

        if use_dilated:
            assert x.shape[1] % L == 0, "L should equal to the num of tokens per frame"
            pad_len = dilated_length * L - x.shape[1] % (dilated_length * L)
            if pad_len != 0:
                q = F.pad(q, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))
            q = rearrange(q, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)
            k = rearrange(k, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)
            v = rearrange(v, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)
        x = self.attn(q, k, v, window_size=(L*self.window_size, L*self.window_size))

        out = self.o(x)
        if use_dilated:
            out = rearrange(out, "(b d) (n l) c -> b (n d l) c", l=L, d=dilated_length)
            if pad_len != 0:
                out = out[:, :-pad_len]
        return out

    def get_atten_qkv(self, x, f, freqs, L=1):
        dilated_length = self.dilated_length

        l_q = self.q(x)
        l_k = self.k(x)

        q = self.norm_q(l_q)
        k = self.norm_k(l_k)

        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        return q, k, v


class CrossAttention(nn.Module):
    def __init__(self, dim: int, attn_hidden_dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False, sp_group=None, block_idx=0):
        super().__init__()
        self.dim = dim
        self.attn_hidden_dim = attn_hidden_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sp_group = sp_group
        self.q = nn.Linear(dim, attn_hidden_dim)
        self.k = nn.Linear(dim, attn_hidden_dim)
        self.v = nn.Linear(dim, attn_hidden_dim)
        self.o = nn.Linear(attn_hidden_dim, dim)
        self.norm_q = FusedRMSNorm(attn_hidden_dim, eps=eps)
        self.norm_k = FusedRMSNorm(attn_hidden_dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = FusedRMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

        self.block_idx = block_idx
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        self.context_group_size = dist.get_world_size(group=sp_group) if dist.is_initialized() else 1


    def forward(self, x: torch.Tensor, y: torch.Tensor, attn_mask=None):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y

        if attn_mask is not None:
            B, L, S = x.shape[0], x.shape[1], ctx.shape[1]
            attn_mask = attn_mask.view(B, 1, 1, S).expand(B, 1, L, S)

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)

        x = self.attn(q, k, v, attn_mask=attn_mask)

        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y

        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):

    def __init__(
        self,
        # *************************
        # block params
        has_image_input: bool,
        dim: int,
        attn_hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        use_linear_attn=True,
        dilated_length=1,
        window_size=3,
        gated_delta_chunk_size=10240,
        # *************************
        # seq parallel params
        block_idx=-1,
        gateddeltanet_layer_idx = -1,
        is_first_block=False,
        is_last_block=False,
        use_seq_parallel=False,
        use_tp_in_getaeddeltanet=False,
        use_tp_in_self_attn=False,
        attend_k0=False,
        attend_k0_with_module=False,
        tp_chunk_list=None,
        use_cross_attn=True,
    ):
        super().__init__()
        self.dim = dim
        self.attn_hidden_dim = attn_hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.use_linear_attn = use_linear_attn
        self.gated_delta_chunk_size = gated_delta_chunk_size

        self.block_idx=block_idx
        self.is_first_block=is_first_block
        self.is_last_block=is_last_block
        self.use_seq_parallel=use_seq_parallel
        
        self.use_tp_in_getaeddeltanet = use_tp_in_getaeddeltanet
        self.use_tp_in_self_attn = use_tp_in_self_attn

        self.use_cross_attn = use_cross_attn


        self.context_group_rank = 0
        self.context_group_size = 1
        self.context_group = None

        self.window_size = window_size
        self.dilated_length = dilated_length

        if is_first_block:
            print(f'{self.__class__.__name__} use_seq_parallel: {use_seq_parallel} context_group_size: {self.context_group_size}')

        if self.use_linear_attn:
            assert gateddeltanet_layer_idx >= 0


            self.gated_delta = GatedDeltaNet(hidden_size=dim, 
                                            num_heads=num_heads,
                                            mode='chunk', 
                                            use_gate=True,
                                            norm_eps=eps,
                                            layer_idx=gateddeltanet_layer_idx,
                                            )
        else:
            assert gateddeltanet_layer_idx == -1
   
            self.self_attn = SelfAttention(dim, attn_hidden_dim, num_heads, eps, dilated_length=dilated_length, window_size=window_size, attend_k0=attend_k0,attend_k0_with_module=attend_k0_with_module)

        self.cross_attn = CrossAttention(
            dim, attn_hidden_dim, num_heads, eps, has_image_input=has_image_input, sp_group=self.context_group, block_idx=self.block_idx)

        self.self_attn_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.cross_attn_norm = nn.LayerNorm(dim, eps=eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), 
            ACT2CLS['silu'](),
            nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()


    def forward(self, x, context, t_mod, freqs, grid_size, context_mask=None):
        (f, h, w) = grid_size
        B, _, D = x.shape
        L = h * w

        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        # scale & gate
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = (
                scale_msa.squeeze(2), shift_msa.squeeze(2), gate_msa.squeeze(2),
                scale_mlp.squeeze(2), shift_mlp.squeeze(2), gate_mlp.squeeze(2),
            )

        full_x = x

        # self-attention
        input_x = modulate(self.self_attn_norm(full_x), shift_msa, scale_msa)
        if self.use_linear_attn:
            if input_x.shape[1] > self.gated_delta_chunk_size:
                cache = fla_Cahce.from_legacy_cache()
                outputs = []
                for start in range(0, input_x.shape[1], self.gated_delta_chunk_size):
                    x_chunk = input_x[:, start:start+self.gated_delta_chunk_size, :]


                    out_chunk, _, cache = self.gated_delta(
                        x_chunk,
                        past_key_values=cache,
                        use_cache=True
                    )
                    outputs.append(out_chunk)
                attn_out = torch.cat(outputs, dim=1)
            else:
                attn_out, _, _ = self.gated_delta(input_x)
        else:
            attn_out = self.self_attn(input_x, f, freqs, L=L)
        full_x = self.gate(full_x, gate_msa, attn_out)


        distributed_x = full_x

        # cross-attention
        distributed_input_x = self.cross_attn_norm(distributed_x)


        attn_out = self.cross_attn(distributed_input_x, context, attn_mask=context_mask)
        distributed_x = distributed_x + attn_out

        distributed_input_x = modulate(self.ffn_norm(distributed_x), shift_mlp, scale_mlp)

        distributed_ffn_out = self.ffn(distributed_input_x)
        distributed_x = self.gate(distributed_x, gate_mlp, distributed_ffn_out)

        return distributed_x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            ACT2CLS['silu'](),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


def build_2d_sincos_pos_embed(embed_dim: int, h: int, w: int, device=None, dtype=None):
    """
    (1, h*w, embed_dim)  2D sin-cos positional embedding.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4."
    device = device or "cpu"
    dtype = dtype or torch.float32

    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # (h, w)
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)

    dim_each = embed_dim // 2
    omega = torch.arange(dim_each // 2, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / (dim_each // 2)))

    out_y = yy[:, None] * omega[None, :]
    out_x = xx[:, None] * omega[None, :]

    pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)

    pos = torch.cat([pos_y, pos_x], dim=1).unsqueeze(0)
    return pos


class PosEmbed2D(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, h: int, w: int):
        # x: (B, N, C), N = h*w
        B, N, C = x.shape
        assert C == self.embed_dim and N == h * w
        pos = build_2d_sincos_pos_embed(C, h, w, device=x.device, dtype=torch.float32)
        return x + pos.to(x.dtype)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x

@DITS.register_module()
class KairosDiTV2(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        attn_hidden_dim=None,
        layers_settings: List[str] = None,
        has_image_input: bool = False,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        window_size = 3,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        dilated_lengths = [1, 1, 6, 1],
        gated_delta_chunk_size=10240,
        flex_block_size=128,
        attn_method='flash',
        all_atten_history=False,
        restrict_history_query_to_history: bool = True,
        action_head_cfg = dict(),
        double_patch_embeding_cfg=dict(),
        use_first_frame_cond: bool = False,
        use_seq_parallel=False,
        use_tp_in_getaeddeltanet=False,
        use_tp_in_self_attn=False,
        attend_k0=False,
        attend_k0_with_module=False,
        tp_chunk_list=None,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.use_first_frame_cond = use_first_frame_cond

        if attn_hidden_dim is None:
            attn_hidden_dim = dim
        self.attn_hidden_dim = attn_hidden_dim

        if not action_head_cfg: 
            self.patch_embedding = nn.Conv3d(
                in_dim, dim, kernel_size=patch_size, stride=patch_size)


        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            ACT2CLS['silu'](),
            nn.Linear(dim, dim),
        )
        if self.use_first_frame_cond:
            self.image_downsample = nn.Sequential(
                nn.Conv2d(in_dim, dim, 3, stride=2, padding=1),
                nn.Conv2d(dim, dim, 3, stride=2, padding=1),
            )
            self.image_embedding = MLP(dim, dim, has_pos_emb=False)
            self.image_pos_embed = PosEmbed2D(dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6)
        )

        self.action_head_cfg = action_head_cfg
        if action_head_cfg:
            action_state_dim = action_head_cfg["action_state_dim"]
            action_dim = action_head_cfg["action_dim"]

            action_dim_weight_path = action_head_cfg.get("action_dim_weight_path",'')

            self.proprio_encoder = nn.Sequential(
                nn.Linear(action_state_dim, text_dim),
                ACT2CLS['silu'](),
                nn.Linear(text_dim, text_dim),
            )
            self.action_encoder = nn.Linear(action_dim, dim)
            self.action_dim = action_dim

            if action_dim_weight_path is not None and action_dim_weight_path != '':
                print('loading action_dim_weight from path: ', action_dim_weight_path)
                action_dim_weight = torch.load(action_dim_weight_path, map_location='cpu', weights_only=False).to(torch.float32)
                assert action_dim_weight.numel() == action_dim, "action_dim_weight.numel() must be equal to action_dim"
                assert action_dim_weight.sum() - 1 < 1e-6, "action_dim_weight.sum() must be close to 1"
                action_dim_weight = action_dim_weight.reshape([1,1,action_dim]) # [D] -> [B,T,D]
            else:
                action_dim_weight = torch.ones(size=[1, 1, self.action_dim], dtype=torch.float32) / self.action_dim
            self.register_buffer('action_dim_weight', action_dim_weight, persistent=False)
        else:
            self.proprio_encoder = None
            self.action_encoder = None
            self.action_dim = None
            self.action_dim_weight = None        

        self.num_layers = num_layers
        self.layers_settings = layers_settings

        self.window_size = window_size
        self.dilated_length = max(dilated_lengths)
        self.flex_block_size = flex_block_size
        self.attn_method = attn_method

        self.atten_modes = list(set(layers_settings))
        self.atten_modes = [vi for vi in self.atten_modes if vi not in ['GATED']]
        self.all_atten_history = all_atten_history
        self.restrict_history_query_to_history = restrict_history_query_to_history

        _blocks = []
        if not layers_settings:
            raise NotImplementedError()
        else:
            assert len(layers_settings) == num_layers, "layers_settings must be the same length as num_layers"
            assert len(dilated_lengths) == 1, "dilated_lengths for DSWA"
            gidx = 0
            for layer_idx, layer_desc in enumerate(layers_settings):

                use_cross_attn = True

                if layer_desc == 'SWA':
                    use_linear_attn = False
                    dilated_length = 1
                    real_window_size = window_size

                elif layer_desc == 'DSWA':
                    use_linear_attn = False
                    dilated_length = dilated_lengths[0]
                    real_window_size = window_size
                elif layer_desc == 'GATED':
                    use_linear_attn = True
                    dilated_length = 1
                    real_window_size = -1
                elif layer_desc == 'FA':
                    use_linear_attn = False
                    dilated_length = 1
                    real_window_size = -1
                else: 
                    raise ValueError(f"Invalid layer description: {layer_desc}")
                _block = DiTBlock(has_image_input, dim, attn_hidden_dim, num_heads, ffn_dim, eps, use_linear_attn=use_linear_attn, dilated_length=dilated_length,
                        block_idx=layer_idx,
                        gateddeltanet_layer_idx=gidx if use_linear_attn else -1,
                        gated_delta_chunk_size=gated_delta_chunk_size,
                        is_first_block=(layer_idx == 0),
                        is_last_block= (layer_idx == (num_layers - 1)),
                        use_seq_parallel=use_seq_parallel,
                        use_tp_in_getaeddeltanet=use_tp_in_getaeddeltanet,
                        use_tp_in_self_attn=use_tp_in_self_attn,
                        tp_chunk_list=tp_chunk_list,
                        attend_k0=attend_k0,
                        attend_k0_with_module=attend_k0_with_module,
                        window_size=real_window_size,
                        use_cross_attn=use_cross_attn,
                )

                if layer_desc == 'GATED':
                    gidx += 1
                _blocks.append(_block)
        self.blocks = nn.ModuleList(_blocks)
        print(f'{self.__class__.__name__} use_seq_parallel: {use_seq_parallel}')
        print(f'{self.__class__.__name__} use_tp_in_getaeddeltanet: {use_tp_in_getaeddeltanet}')
        print(f'{self.__class__.__name__} use_tp_in_self_attn: {use_tp_in_self_attn}')
        print(f'{self.__class__.__name__} gated_delta_chunk_size: {gated_delta_chunk_size}')
        print(f'{self.__class__.__name__} attn_method: {attn_method}')
        head_dim = dim // num_heads

        if self.action_encoder is not None:
            assert attn_hidden_dim % num_heads == 0
            attn_head_dim = attn_hidden_dim // num_heads
            self.freqs = precompute_freqs_cis(attn_head_dim)
            self.head = nn.Linear(dim, self.action_dim)
        else:
            self.freqs = precompute_freqs_cis_3d(head_dim)
            self.head = Head(dim, out_dim, patch_size, eps)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                first_frame_latent: Optional[torch.Tensor] = None,
                **kwargs,
                ):
        
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if first_frame_latents is not None and self.use_first_frame_cond:
            # shape: (b, c, t, h, w)
            first_frame_latents = first_frame_latents.to(context.device)
            img_context = self.image_downsample(first_frame_latents.squeeze(2))
            fb, fc, fh, fw = img_context.shape
            img_context = img_context.flatten(2).transpose(-2, -1)
            img_context = self.image_embedding(img_context)
            img_context = self.image_pos_embed(img_context, h=fh, w=fw)
            context = torch.cat([img_context, context], dim=1)
            if context_mask is not None:
                context_mask = torch.cat([
                    torch.ones(context.shape[0], img_context.shape[1], dtype=context_mask.dtype, device=context_mask.device),
                    context_mask
                ], dim=1)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)
        grid_size = (f, h, w)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, grid_size, context_mask=context_mask,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, grid_size, context_mask=context_mask,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs, grid_size, context_mask=context_mask)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
