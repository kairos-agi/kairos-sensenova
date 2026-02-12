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

class ColumnParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False):
        # Divide the weight matrix along the last dimension.
        self.tp_rank = parallel_state.get_context_parallel_rank()
        self.tp_size = parallel_state.get_context_parallel_world_size()
        self.tp_group = parallel_state.get_context_parallel_group()
        self.out_features_per_partition = divide(out_features, self.tp_size)
        self.gather_output = gather_output
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
            output = self._all_gather(output)
        return output

    def _all_gather(self, x):
        outs = [torch.empty_like(x) for _ in range(self.tp_size)]
        dist.all_gather(outs, x, group=self.tp_group)
        return torch.cat(outs, dim=-1)

    def _load_state_dict_pre_hook(
        self,
        module,
        state_dict,
        prefix,
        *args,
    ):
        weight_key = prefix + "weight"
        if weight_key not in state_dict:
            return

        full_weight = state_dict[weight_key]
        shard_size = self.out_features_per_partition
        start = self.tp_rank * shard_size
        end = start + shard_size

        # 替换成 shard
        state_dict[weight_key] = full_weight[start:end, :]

        if self.bias is not None:
            bias_key = prefix + "bias"
            if bias_key in state_dict:
                full_bias = state_dict[bias_key]
                state_dict[bias_key] = full_bias[start:end]

class RowParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
    ):
        self.tp_rank = parallel_state.get_context_parallel_rank()
        self.tp_size = parallel_state.get_context_parallel_world_size()
        self.tp_group = parallel_state.get_context_parallel_group()
        self.input_is_parallel = input_is_parallel

        assert in_features % self.tp_size == 0, "in_features must be divisible by tp_size"
        self.in_features_per_partition = in_features // self.tp_size

        super().__init__(
            self.in_features_per_partition,
            out_features,
            bias=bias,
        )

        # 注册 pre-hook 支持加载完整权重
        self._register_load_state_dict_pre_hook(
            self._load_state_dict_pre_hook,
            with_module=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果输入不是并行的，先切分
        if not self.input_is_parallel:
            x = torch.chunk(x, self.tp_size, dim=-1)[self.tp_rank]

        x = super().forward(x)

        # 对行并行，需要 All-Reduce 聚合输出
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.tp_group)

        return x

    # ===============================
    # state_dict 加载 hook
    # ===============================
    def _load_state_dict_pre_hook(
        self,
        module,
        state_dict,
        prefix,
        *args,
    ):
        weight_key = prefix + "weight"
        if weight_key not in state_dict:
            return

        full_weight = state_dict[weight_key]
        shard_size = self.in_features_per_partition
        start = self.tp_rank * shard_size
        end = start + shard_size

        # 替换成分片
        state_dict[weight_key] = full_weight[:, start:end]

        if self.bias is not None:
            bias_key = prefix + "bias"
            if bias_key in state_dict:
                # row-parallel bias不需要切分，全局bias直接使用
                state_dict[bias_key] = state_dict[bias_key]
