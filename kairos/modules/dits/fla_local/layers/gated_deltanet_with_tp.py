# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F
import torch.distributed as dist

from kairos.modules.dits.fla_local.layers.utils import get_unpad_data, index_first_axis, pad_input
from kairos.modules.dits.fla_local.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from kairos.modules.dits.fla_local.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


@torch.compile
def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


@torch.compile
def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

class TPLinearFullWeight(nn.Module):
    def __init__(self, in_dim, out_dim, tp_group):
        super().__init__()

        self.tp_group = tp_group
        self.tp_rank = dist.get_rank(tp_group)
        self.tp_size = dist.get_world_size(tp_group)
        assert out_dim % self.tp_size == 0
        self.shard = out_dim // self.tp_size

        # self.global_rank = dist.get_rank()

        self.current_device = torch.cuda.current_device()


        full = torch.empty(out_dim, in_dim, device=None, dtype=None)
        nn.init.kaiming_uniform_(full, a=math.sqrt(5))

        full_t = full.to(device=self.current_device) # 默认的 dtype=torch.float32

        dist.broadcast(full_t, src=0)
        full = full_t.cpu()

        # only keep shard as parameter
        self.weight = nn.Parameter(full)

    def forward(self, x):
        start = self.tp_rank * self.shard
        end = start + self.shard
        return x.matmul(self.weight[start:end].t())

class TPLinear(nn.Module):
    def __init__(self, in_dim, out_dim, tp_group):
        super().__init__()

        self.tp_group = tp_group
        self.tp_rank = dist.get_rank(tp_group)
        self.tp_size = dist.get_world_size(tp_group)
        assert out_dim % self.tp_size == 0
        self.shard = out_dim // self.tp_size
        # self.global_rank = dist.get_rank()

        self.current_device = torch.cuda.current_device()
        
        full = torch.empty(out_dim, in_dim, device=None, dtype=None)
        nn.init.kaiming_uniform_(full, a=math.sqrt(5))

        full_t = full.to(device=self.current_device) #, dtype=torch.bfloat16

        # no need to broadcast if all rand seeds are aligned
        dist.broadcast(full_t, src=0)

        full = full_t.cpu()

        # slice local shard
        start = self.tp_rank * self.shard
        end = start + self.shard
        local = full[start:end, :]

        # only keep shard as parameter
        self.weight = nn.Parameter(local.contiguous())

    def forward(self, x, gather_output=False):        
        y_local = x @ self.weight.t()
        
        if gather_output:
            # ColumnParallel 的标准操作：gather 输出
            out_list = [torch.empty_like(y_local) for _ in range(self.tp_size)]
            dist.all_gather(out_list, y_local, group=self.tp_group)
            return torch.cat(out_list, dim=-1)
        else:
            return y_local


def _all2all_head_to_seq(x: torch.Tensor, context_group):
    if not dist.is_available() or not dist.is_initialized():
        return x
    world = dist.get_world_size(group=context_group) if context_group is not None else dist.get_world_size()
    if world <= 1:
        return x
    B, T, H_local, D = x.shape

    if T % world != 0:
        x_resh = x.reshape(B * T, H_local, D).contiguous()
        gathered = _gather_input(x_resh, context_group)
        h_full = gathered.shape[1]
        return gathered.reshape(B, T, h_full, D)

    T_local = T // world
    input_tensors = [x[:, r * T_local:(r + 1) * T_local, :, :].contiguous() for r in range(world)]
    output_tensors = [torch.empty_like(input_tensors[0]) for _ in range(world)]

    dist.all_to_all(output_tensors, input_tensors, group=context_group)
    out = torch.cat(output_tensors, dim=2)
    return out

def _all2all_seq_to_head(x: torch.Tensor, context_group):
    if not dist.is_available() or not dist.is_initialized():
        return x
    world = dist.get_world_size(group=context_group) if context_group is not None else dist.get_world_size()
    if world <= 1:
        return x

    B, T_local, H_full, D = x.shape
    H_local = H_full // world
    input_tensors = [x[:, :, r * H_local:(r + 1) * H_local, :].contiguous() for r in range(world)]
    output_tensors = [torch.empty_like(input_tensors[0]) for _ in range(world)]

    dist.all_to_all(output_tensors, input_tensors, group=context_group)
    out = torch.cat(output_tensors, dim=1)
    return out

def _gather_input(x: torch.Tensor, context_group):
    if not dist.is_available() or not dist.is_initialized():
        return x
    context_group_world_size = dist.get_world_size(group=context_group)
    context_group_rank = dist.get_rank(group=context_group)
    if context_group_world_size <= 1:
        return x

    # move gather dim to axis=1 if needed by caller; caller should provide tensor
    # shaped as [B, N_local, ...] when possible. We will assume first two dims
    # are batch and the local dim to gather.
    B, N_local = x.shape[0], x.shape[1]
    rest = x.shape[2:]
    start_idx = context_group_rank * N_local
    end_idx = start_idx + N_local

    full_output = torch.zeros((B, N_local * context_group_world_size, *rest),
                              device=x.device,
                              dtype=x.dtype)
    full_output[:, start_idx:end_idx, ...] = x
    dist.all_reduce(full_output, op=dist.ReduceOp.SUM, group=context_group)

    return full_output


def _distribute_input(x: torch.Tensor, context_group):
    if not dist.is_available() or not dist.is_initialized():
        return x
    ws = dist.get_world_size(group=context_group)
    rank = dist.get_rank(group=context_group)
    if ws <= 1:
        return x
    # assume x shape [B, N_full, ...]
    B, N_full = x.shape[0], x.shape[1]
    per = N_full // ws
    start = rank * per
    return x[:, start:start + per, ...].contiguous()

class GatedDeltaNet(nn.Module):
    """
    The layer implementaion for [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).  # noqa

    Similar to Mamba2, each layer contains around 6*hidden_size*hidden_size parameters.

    Parameter alloation when use_gate=True:
        - 0.75 * hidden_size * hidden_size for the q_proj and k_proj each
        - 1.5 * hidden_size * hidden_size for the v_proj, g_proj and o_proj each
        - Others are ignorably small.
        - In total = 0.75 * 2 + 1.5 * 3 = 6 * hidden_size * hidden_size
    NOTE: num_heads * head_dim = 0.75 * hidden_size, please make sure to set the correct num_heads and head_dim.

    Parameter allocation when use_gate=False:
        - 1 * hidden_size * hidden_size for the q_proj and k_proj each
        - 2 * hidden_size * hidden_size for the v_proj and o_proj each
        - Others are ignorably small.
        - In total = 1 * 2 + 2 * 2 = 6 * hidden_size * hidden_size

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA is applied if `num_v_heads` > `num_heads`. Default: `None`.
        mode (str, Optional):
            Which Gated DeltaNet kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_beta (bool, Optional):
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference: [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int = None,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        tp_num_splits: int = 1,
        tp_group = None,
        **kwargs
    ) -> GatedDeltaNet:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx
        # Number of head-wise splits to perform sequentially to reduce peak memory.
        # When >1, q/k/v (and corresponding initial states) will be sliced along the head
        # dimension and processed sequentially then concatenated. This reduces peak memory
        # while preserving correct autograd flow.
        self.tp_num_splits = int(tp_num_splits)
        self.tp_group = tp_group

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear."
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}."
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated."
            )
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        def _all_reduce_grad(g):
            '''对规模较小的参数，采用各卡保有，算完同步的方式处理'''
            if g is None:
                return None
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(g, op=dist.ReduceOp.SUM, group=self.tp_group)
            return g

        self.chunk_step = 0
        self.save = False

        # Create either full projections (single-device) or local sharded projections
        # when tensor-parallel (TP) is enabled via `tp_num_splits`.
        if self.tp_num_splits <= 1:
            # hidden_size = 2560
            self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False) # self.key_dim=5120
            self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False) # self.value_dim=10240
            self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
            self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False) # self.num_v_heads=20
            self._tp_enabled = False
        else:
            splits = self.tp_num_splits
            if self.key_dim % splits != 0:
                raise ValueError("key_dim must be divisible by tp_num_splits")
            if self.value_dim % splits != 0:
                raise ValueError("value_dim must be divisible by tp_num_splits")
            # Local output sizes for each rank
            self.key_dim_local = self.key_dim // splits
            self.value_dim_local = self.value_dim // splits
            self.num_v_heads_local = self.num_v_heads // splits

            self.q_proj = TPLinearFullWeight(hidden_size, self.key_dim, self.tp_group)
            self.k_proj = TPLinearFullWeight(hidden_size, self.key_dim, self.tp_group)
            self.v_proj = TPLinearFullWeight(hidden_size, self.value_dim, self.tp_group)
            # keep original impl
            self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
            self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False) # self.num_v_heads=20        
            self._tp_enabled = True

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
                tp_num_splits=tp_num_splits,
                tp_group=tp_group,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
                tp_num_splits=tp_num_splits,
                tp_group=tp_group,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
                tp_num_splits=tp_num_splits,
                tp_group=tp_group,
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
            )
        if use_gate:
            if self.tp_num_splits <= 1:
                self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            else:
                self.g_proj = TPLinearFullWeight(hidden_size, self.value_dim, self.tp_group)

            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None

        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.save: torch.save({"hidden_states": hidden_states.cpu()}, f"saved_tensors/gated-inputx_{self.chunk_step}-{rank}.pt")

        def _all_reduce_grad(g):
            '''对规模较小的参数，采用各卡保有，算完同步的方式处理'''
            if g is None:
                return None
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(g, op=dist.ReduceOp.SUM, group=self.tp_group)
            return g

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
 
            if self.save: torch.save({"q": q.cpu(), "k": k.cpu(), "v": v.cpu()}, f"saved_tensors/gated-conv_{self.chunk_step}-{rank}.pt")
            if conv_state_q is not None:
                if self.save: torch.save({"conv_state_q": conv_state_q.cpu(), "conv_state_k": conv_state_k.cpu(), "conv_state_v": conv_state_v.cpu()}, f"saved_tensors/gated-conv_state_{self.chunk_step}-{rank}.pt")

        else:
            if not getattr(self, '_tp_enabled', False):
                q = F.silu(self.q_proj(hidden_states))
                k = F.silu(self.k_proj(hidden_states))
                v = F.silu(self.v_proj(hidden_states))
            else:
                q = F.silu(self.q_proj(hidden_states))
                k = F.silu(self.k_proj(hidden_states))
                v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        if self.save: torch.save({"q": q.cpu(), "k": k.cpu(), "v": v.cpu()}, f"saved_tensors/gated-remap_{self.chunk_step}-{rank}.pt")

        if self.num_v_heads > self.num_heads:
            q, k = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads), (q, k))

        # beta and g should stay full even in TP.
        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.
        if self.save: torch.save({"beta": beta.cpu()}, f"saved_tensors/gated-beta_{self.chunk_step}-{rank}.pt")
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)
        if self.save: torch.save({"g": g.cpu()}, f"saved_tensors/gated-g_{self.chunk_step}-{rank}.pt")

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        # If TP is not enabled, use original dense call. Otherwise each rank
        # processes only its local head slice and we all_gather results.
        if not getattr(self, '_tp_enabled', False):
            if mode == 'chunk':
                o, recurrent_state = chunk_gated_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True
                )
            elif mode == 'fused_recurrent':
                o, recurrent_state = fused_recurrent_gated_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True
                )
            else:
                raise NotImplementedError(f"Not supported mode `{mode}`.")
            
            if self.save: torch.save({"output": o.cpu()}, f"saved_tensors/gated-output_{self.chunk_step}-{rank}.pt")
        else:
            # Distributed case: each process handles one local slice determined by
            # its rank. We require the distributed process group to be initialized.
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Distributed not initialized but tp_num_splits>1")
            world_size = dist.get_world_size(self.tp_group)
            splits = self.tp_num_splits
            if world_size != splits:
                raise RuntimeError(f"world_size ({world_size}) must equal tp_num_splits ({splits})")
            rank = dist.get_rank(self.tp_group)
            # q/k/v here are already local (per-rank) tensors with shape [..., H_local, ...].
            # Use the local head count directly.
            H_local = q.shape[-2]
            h_local = H_local
            h0, h1 = rank * h_local, (rank + 1) * h_local

            init_i = None
            if recurrent_state is not None:
                init_i = recurrent_state[:, h0:h1, :, :]

            # shape: [batch, seq, head]
            g = g[:, :, h0:h1]
            beta = beta[:, :, h0:h1]

            if mode == 'chunk':
                o_local, final_local = chunk_gated_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=init_i,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True
                )
            elif mode == 'fused_recurrent':
                o_local, final_local = fused_recurrent_gated_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=init_i,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True
                )
            else:
                raise NotImplementedError(f"Not supported mode `{mode}`.")

            # reshape to [B*T, H_local, D] for gather implementation
            b, t, h_loc, d = o_local.shape
            # world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
            if world_size > 1:
                # head->seq: [B, T, H_local, D] -> [B, T_local, H_full, D]
                o_seq = _all2all_head_to_seq(o_local, self.tp_group)
            else:
                o_seq = o_local
            if self.save: torch.save({"output": o_seq.cpu()}, f"saved_tensors/gated-output_{self.chunk_step}-{rank}.pt")

            # Gather final recurrent states if requested
            if final_local is not None:
                recurrent_state = _gather_input(final_local, self.tp_group)
                if self.save: torch.save({"recurrent_state": recurrent_state.cpu()}, f"saved_tensors/recurrent_state_{self.chunk_step}-{rank}.pt")
            else:
                recurrent_state = None

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len
            )

        # o_norm may not support tensor-parallel execution or lower precision.
        # Cast to float32 (single precision) for normalization then cast back.
        if self.use_gate:
            # Compute gating for o_norm. If TP enabled, compute local gating and
            # all_gather to form the full gating tensor required by o_norm.
            if not getattr(self, '_tp_enabled', False):
                g_norm = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            else:
                # local gating
                g_local = self.g_proj(hidden_states)
                
                # all_gather to form full gating
                world_size = dist.get_world_size(self.tp_group) if (dist.is_available() and dist.is_initialized()) else 1
                if world_size > 1:
                    # g_local shaped [B, T, H_local * head_v_dim]
                    b, t, ld = g_local.shape
                    h_loc = ld // self.head_v_dim
                    # reshape to [B, T, H_local, D]
                    g_local_4 = rearrange(g_local, 'b t (h d) -> b t h d', h=h_loc, d=self.head_v_dim)
                    # perform head->seq transpose directly
                    g_seq = _all2all_head_to_seq(g_local_4, self.tp_group)
                    # flattened full gating for later use if needed
                    g_full = rearrange(g_seq, 'b t h d -> b t (h d)')
                else:
                    g_full = g_local
                g_norm = rearrange(g_full, '... (h d) -> ... h d', d=self.head_v_dim)

            if self.save: torch.save({"g_norm": g_norm.cpu()}, f"saved_tensors/gated-g_norm_{self.chunk_step}-{rank}.pt")

            # Determine distributed world size
            world_size = dist.get_world_size(self.tp_group) if (dist.is_available() and dist.is_initialized()) else 1
            if world_size > 1:
                # determine gating sequence layout (g_seq). If g_seq was
                # produced earlier, reuse it; otherwise build from g_full.
                if 'g_seq' in locals():
                    g_seq_final = g_seq
                else:
                    g_full_4 = rearrange(g_full, 'b t (h d) -> b t h d', d=self.head_v_dim)
                    g_seq_final = _all2all_head_to_seq(g_full_4, self.tp_group)
                # choose input for norm: prefer o_seq (head->seq result) when available
                o_in = o_seq if 'o_seq' in locals() else o
                o_seq_norm = self.o_norm(o_in, g_seq_final)
                o_flat = rearrange(o_seq_norm, 'b t h d -> b t (h d)')
                o = self.o_proj(o_flat)
                if self.save: torch.save({"o_norm": o.cpu()}, f"saved_tensors/gated-o_norm_{self.chunk_step}-{rank}.pt")
            else:
                # single-rank case: normal path
                o = self.o_norm(o, g_norm)
                if self.save: torch.save({"o_norm": o.cpu()}, f"saved_tensors/gated-o_norm_{self.chunk_step}-{rank}.pt")
        else:
            o = self.o_norm(o)
        if o.dim() == 4:
            o = rearrange(o, 'b t h d -> b t (h d)')
            o = self.o_proj(o)
        if self.save: torch.save({"o_proj": o.cpu()}, f"saved_tensors/gated-o_proj_{self.chunk_step}-{rank}.pt")

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        self.chunk_step += 1

        if world_size > 1:            
            o_gatherd = _gather_input(o, self.tp_group)
            return o_gatherd, None, past_key_values

        return o, None, past_key_values