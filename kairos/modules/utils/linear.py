from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from abc import ABC, abstractmethod
from kairos.modules.utils import parallel_state
from typing import Any
import torch.nn as nn
import torch.distributed as dist

def ensure_divisibility(numerator, denominator) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def _prefix_sum(xs):
    ps = [0]
    s = 0
    for v in xs:
        s += int(v)
        ps.append(s)
    return ps

def _build_plan(full_dim: int, tp_rank: int, tp_size: int, tp_chunk_list=None):
    if tp_chunk_list is None:
        if full_dim % tp_size != 0:
            raise RuntimeError(f"full_dim({full_dim}) must be divisible by tp_size({tp_size}) in balanced mode")
        local = full_dim // tp_size
        start = tp_rank * local
        end = start + local
        return start, end, local

    if len(tp_chunk_list) != tp_size:
        raise RuntimeError(f"len(tp_chunk_list)={len(tp_chunk_list)} != tp_size={tp_size}")
    if any(int(v) <= 0 for v in tp_chunk_list):
        raise RuntimeError(f"tp_chunk_list must be positive ints, got {tp_chunk_list}")

    total = sum(int(v) for v in tp_chunk_list)
    chunk_dim = full_dim // total
    ps = _prefix_sum(tp_chunk_list)
    start = ps[tp_rank] * chunk_dim
    end = ps[tp_rank + 1] * chunk_dim
    return start, end, end - start

class ColumnParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        tp_chunk_list=None):
        # Divide the weight matrix along the last dimension.
        self.tp_rank = parallel_state.get_context_parallel_rank()
        self.tp_size = parallel_state.get_context_parallel_world_size()
        self.tp_group = parallel_state.get_context_parallel_group()
        self.gather_output = gather_output
        self.in_features_full = in_features
        self.out_features_full = out_features

        self.start, self.end, self.out_features_per_partition = _build_plan(
            full_dim=out_features,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            tp_chunk_list=tp_chunk_list,
        )

        super().__init__(
            in_features,
            self.out_features_per_partition,
            bias=bias,
        )

        self._register_load_state_dict_pre_hook(
            self._load_state_dict_pre_hook,
            with_module=True,
        )

    def forward(self, input_: torch.Tensor):
        output = F.linear(input_, self.weight, self.bias)
        if self.gather_output:
            # output = self._all_gather(output)
            output = self._all_gather_varlen(output)
        return output

    def _all_gather_varlen(self, x: torch.Tensor):
        device = x.device
        local_len = torch.tensor([x.shape[-1]], device=device, dtype=torch.int64)
        lens = [torch.empty_like(local_len) for _ in range(self.tp_size)]
        dist.all_gather(lens, local_len, group=self.tp_group)
        lens = [int(t.item()) for t in lens]

        if all(l == lens[0] for l in lens):
            outs = [torch.empty_like(x) for _ in range(self.tp_size)]
            dist.all_gather(outs, x, group=self.tp_group)
            return torch.cat(outs, dim=-1)

        max_len = max(lens)
        if x.shape[-1] < max_len:
            x_pad = F.pad(x, (0, max_len - x.shape[-1]), "constant", 0)
        else:
            x_pad = x

        outs = [torch.empty_like(x_pad) for _ in range(self.tp_size)]
        dist.all_gather(outs, x_pad, group=self.tp_group)
        outs = [o[..., :lens[i]] for i, o in enumerate(outs)]
        return torch.cat(outs, dim=-1)

    def _load_state_dict_pre_hook(self, module, state_dict, prefix, *args):
        weight_key = prefix + "weight"
        if weight_key not in state_dict:
            return

        full_weight = state_dict[weight_key]
        # full ckpt: [out, in] -> shard rows
        if full_weight.ndim == 2 and full_weight.shape[0] == self.out_features_full:
            state_dict[weight_key] = full_weight[self.start:self.end, :]

        if self.bias is not None:
            bias_key = prefix + "bias"
            if bias_key in state_dict:
                full_bias = state_dict[bias_key]
                if full_bias.ndim == 1 and full_bias.shape[0] == self.out_features_full:
                    state_dict[bias_key] = full_bias[self.start:self.end]

class RowParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        tp_chunk_list=None,
    ):
        self.tp_rank = parallel_state.get_context_parallel_rank()
        self.tp_size = parallel_state.get_context_parallel_world_size()
        self.tp_group = parallel_state.get_context_parallel_group()
        self.input_is_parallel = input_is_parallel

        # full dims
        self.in_features_full = in_features
        self.out_features_full = out_features

        self.start, self.end, self.local_in = _build_plan(
            full_dim=in_features,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            tp_chunk_list=tp_chunk_list,
        )

        super().__init__(
            # self.in_features_per_partition,
            self.local_in,
            out_features,
            bias=bias,
        )

        self._register_load_state_dict_pre_hook(
            self._load_state_dict_pre_hook,
            with_module=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            x = x[..., self.start:self.end]
        else:
            if x.shape[-1] != self.local_in:
                raise RuntimeError(
                    f"RowParallelLinear got input lastdim={x.shape[-1]}, expect local_in={self.local_in} "
                    f"(rank={self.tp_rank}, slice={self.start}:{self.end})"
                )

        out = F.linear(x, self.weight, None)
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)

        if self.bias is not None:
            out = out + self.bias

        return out

    def _load_state_dict_pre_hook(self, module, state_dict, prefix, *args):
        weight_key = prefix + "weight"
        if weight_key not in state_dict:
            return

        w = state_dict[weight_key]          # ckpt weight
        expect = self.weight.shape          # (out, local_in)

        if tuple(w.shape) == tuple(expect):
            pass
        else:
            if w.ndim == 2 and w.shape[0] == self.out_features_full and w.shape[1] == self.in_features_full:
                state_dict[weight_key] = w[:, self.start:self.end]
            else:
                raise RuntimeError(
                    f"[{prefix}] unexpected weight shape in ckpt: {tuple(w.shape)}, "
                    f"expect shard {tuple(expect)} or full ({self.out_features_full}, {self.in_features_full})"
                )

        if self.bias is not None:
            bias_key = prefix + "bias"
            if bias_key in state_dict:
                b = state_dict[bias_key]
                if tuple(b.shape) != tuple(self.bias.shape):
                    raise RuntimeError(
                        f"[{prefix}] unexpected bias shape in ckpt: {tuple(b.shape)}, expect {tuple(self.bias.shape)}"
                    )