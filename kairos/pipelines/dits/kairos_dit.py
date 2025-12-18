import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from transformers.activations import ACT2CLS
from apex.normalization.fused_layer_norm import FusedRMSNorm
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

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    from fla.layers import GatedDeltaNet
except ModuleNotFoundError:
    pass

from kairos.pipelines.builder import DITS


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, attn_mask=None, window_size=(-1, -1)):
    if compatibility_mode or attn_mask is not None:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v, window_size=window_size)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v, window_size=window_size)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        raise RuntimeError("do not use pytorch attention")
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
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
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, attn_mask=attn_mask, window_size=window_size)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, dilated_length=1, window_size=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dilated_length = dilated_length
        self.window_size = window_size

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = FusedRMSNorm(dim, eps=eps)
        self.norm_k = FusedRMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def extra_repr(self):
        return f'dilated_length={self.dilated_length}, window_size={self.window_size}'

    def forward(self, x, f, freqs, L=1):
        dilated_length = self.dilated_length
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply_for3d(q, f, freqs, self.num_heads)
        k = rope_apply_for3d(k, f, freqs, self.num_heads)
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


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = FusedRMSNorm(dim, eps=eps)
        self.norm_k = FusedRMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = FusedRMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

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
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6, use_linear_attn = True, dilated_length=1, window_size=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.use_linear_attn = use_linear_attn

        if self.use_linear_attn:
            self.gated_delta = GatedDeltaNet(hidden_size=dim, num_heads=num_heads, mode='chunk', use_gate=True, norm_eps=eps)
        else:
            self.self_attn = SelfAttention(dim, num_heads, eps, dilated_length=dilated_length, window_size=window_size)

        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)

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

        # self-attention
        input_x = modulate(self.self_attn_norm(x), shift_msa, scale_msa)
        if self.use_linear_attn:
            chunk_size = 40768 
            if input_x.shape[1] > chunk_size:
                outputs = []
                for start in range(0, input_x.shape[1], chunk_size):
                    x_chunk = input_x[:, start:start+chunk_size, :]
                    if start == 0:
                        past_key_values = None

                    out_chunk, _, past_key_values = self.gated_delta(
                        x_chunk,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    outputs.append(out_chunk)
                attn_out = torch.cat(outputs, dim=1)
            else:
                attn_out, _, _ = self.gated_delta(input_x)
        else:
            attn_out = self.self_attn(input_x, f, freqs, L=L)
        x = self.gate(x, gate_msa, attn_out)

        # cross-attention
        input_x = self.cross_attn_norm(x)
        attn_out = self.cross_attn(input_x, context, attn_mask=context_mask)
        x = x + attn_out

        input_x = modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
        ffn_out = self.ffn(input_x)
        x = self.gate(x, gate_mlp, ffn_out)
        return x


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
class KairosDiT(torch.nn.Module):
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
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        dilated_lengths = [1, 1, 6, 1]
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

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            ACT2CLS['silu'](),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6)
        )

        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps, use_linear_attn=(i + 1) % 4 == 0, dilated_length=dilated_lengths[i % 4])
            for i in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads

        self.freqs = precompute_freqs_cis_3d(head_dim)

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
                **kwargs,
                ):

        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

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
