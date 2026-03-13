from typing import List, Optional
import torch
import torch.distributed as dist


def _prefix_sum(xs: List[int]) -> List[int]:
    ps = [0]
    s = 0
    for v in xs:
        s += int(v)
        ps.append(s)
    return ps


def build_tp_chunk_list(num_heads: int, tp_size: int) -> List[int]:
    if tp_size <= 0:
        raise RuntimeError(f"tp_size must be > 0, got {tp_size}")
    if num_heads <= 0:
        raise RuntimeError(f"num_heads must be > 0, got {num_heads}")

    base = num_heads // tp_size
    rem = num_heads % tp_size
    return [base + (1 if r < rem else 0) for r in range(tp_size)]


def build_plan(full_dim: int, tp_rank: int, tp_size: int, tp_chunk_list: Optional[List[int]] = None):
    if tp_chunk_list is None:
        if full_dim % tp_size != 0:
            raise RuntimeError(
                f"full_dim({full_dim}) must be divisible by tp_size({tp_size}) in balanced mode"
            )
        local = full_dim // tp_size
        start = tp_rank * local
        end = start + local
        return start, end, local

    if len(tp_chunk_list) != tp_size:
        raise RuntimeError(f"len(tp_chunk_list)={len(tp_chunk_list)} != tp_size={tp_size}")
    if any(int(v) <= 0 for v in tp_chunk_list):
        raise RuntimeError(f"tp_chunk_list must be positive ints, got {tp_chunk_list}")

    total = sum(int(v) for v in tp_chunk_list)
    if full_dim % total != 0:
        raise RuntimeError(
            f"full_dim({full_dim}) must be divisible by sum(tp_chunk_list)({total})"
        )

    chunk_dim = full_dim // total
    ps = _prefix_sum(tp_chunk_list)
    start = ps[tp_rank] * chunk_dim
    end = ps[tp_rank + 1] * chunk_dim
    return start, end, end - start

def _gather_input_sp(x, context_group):
    if not dist.is_available() or not dist.is_initialized():
        return x

    world = dist.get_world_size(group=context_group)
    if world <= 1:
        return x
    
    gather_list = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(gather_list, x, group=context_group)
    return torch.cat(gather_list, dim=1)

def _distribute_input_sp(x, context_group):
    context_group_world_size = dist.get_world_size(group=context_group)
    context_group_rank = dist.get_rank(group=context_group)
    if context_group_world_size <= 1:
        return x
        
    B, N, D = x.shape
    
    assert N % context_group_world_size == 0, 'cannot split tensor {} vs {}'.format(x.shape, context_group_world_size)

    local_dim = N // context_group_world_size
    start_idx = context_group_rank * local_dim
    end_idx = start_idx + local_dim if context_group_rank != context_group_world_size - 1 else N

    local_x = x[:, start_idx:end_idx, :].contiguous()
    
    return local_x

def all2all_seq_to_head(
    x: torch.Tensor,
    context_group,
    tp_chunk_list=None,
    head_dim=None,
):
    if not dist.is_available() or not dist.is_initialized():
        return x

    group = context_group
    world = dist.get_world_size(group=group) if group is not None else dist.get_world_size()
    if world <= 1:
        return x

    rank = dist.get_rank(group=group) if group is not None else dist.get_rank()

    B, N_local, D_full = x.shape

    if tp_chunk_list is None:
        # balanced head split
        if D_full % world != 0:
            raise RuntimeError(f"D_full={D_full} must be divisible by world={world}")

        D_local = D_full // world
        input_tensors = [
            x[:, :, r * D_local:(r + 1) * D_local].contiguous()
            for r in range(world)
        ]
        output_tensors = [
            x.new_empty(B, N_local, D_local)
            for _ in range(world)
        ]
    else:
        # unbalanced head split
        if head_dim is None:
            raise RuntimeError("head_dim must be provided when tp_chunk_list is not None")

        if len(tp_chunk_list) != world:
            raise RuntimeError(
                f"len(tp_chunk_list)={len(tp_chunk_list)} != world={world}"
            )
        if any(int(v) <= 0 for v in tp_chunk_list):
            raise RuntimeError(f"tp_chunk_list must be positive ints, got {tp_chunk_list}")

        # tp_chunk_list: number of heads on each rank
        dim_chunk_list = [int(v) * int(head_dim) for v in tp_chunk_list]

        if sum(dim_chunk_list) != D_full:
            raise RuntimeError(
                f"sum(dim_chunk_list)={sum(dim_chunk_list)} != D_full={D_full}, "
                f"tp_chunk_list={tp_chunk_list}, head_dim={head_dim}"
            )

        offsets = _prefix_sum(dim_chunk_list)

        # sender: split full hidden dim according to destination rank's head chunk
        input_tensors = [
            x[:, :, offsets[r]:offsets[r + 1]].contiguous()
            for r in range(world)
        ]

        # receiver: from every peer receive this rank's local head chunk
        D_local = dim_chunk_list[rank]
        output_tensors = [
            x.new_empty(B, N_local, D_local)
            for _ in range(world)
        ]

    dist.all_to_all(output_tensors, input_tensors, group=group)

    # [B, N_local * world, D_local]
    out = torch.cat(output_tensors, dim=1)
    return out


def all2all_head_to_seq(
    x: torch.Tensor,
    context_group,
    tp_chunk_list=None,
    head_dim=None,
):
    """
    Input:
        balanced:
            x: [B, N_full, D_local]
               D_local = D_full // world

        unbalanced:
            x: [B, N_full, D_local]
               D_local = tp_chunk_list[rank] * head_dim

    Output:
        out: [B, N_local, D_full]
             N_local = N_full // world
    """
    if not dist.is_available() or not dist.is_initialized():
        return x

    group = context_group
    world = dist.get_world_size(group=group) if group is not None else dist.get_world_size()
    if world <= 1:
        return x

    rank = dist.get_rank(group=group) if group is not None else dist.get_rank()

    B, N_full, D_local = x.shape

    if N_full % world != 0:
        raise RuntimeError(
            f"all2all_head_to_seq currently only supports seq evenly divisible by world size, "
            f"got N_full={N_full}, world={world}"
        )

    N_local = N_full // world

    if tp_chunk_list is None:
        # balanced
        input_tensors = [
            x[:, r * N_local:(r + 1) * N_local, :].contiguous()
            for r in range(world)
        ]
        output_tensors = [
            torch.empty_like(input_tensors[0])
            for _ in range(world)
        ]

        dist.all_to_all(output_tensors, input_tensors, group=group)
        out = torch.cat(output_tensors, dim=2)
        return out

    # unbalanced
    if head_dim is None:
        raise RuntimeError("head_dim must be provided when tp_chunk_list is not None")
    if len(tp_chunk_list) != world:
        raise RuntimeError(f"len(tp_chunk_list)={len(tp_chunk_list)} must equal world={world}")
    if any(int(v) <= 0 for v in tp_chunk_list):
        raise RuntimeError(f"tp_chunk_list must be positive ints, got {tp_chunk_list}")
    if D_local % head_dim != 0:
        raise RuntimeError(f"D_local={D_local} must be divisible by head_dim={head_dim}")

    expected_d_local = int(tp_chunk_list[rank]) * int(head_dim)
    if D_local != expected_d_local:
        raise RuntimeError(f"D_local={D_local}, expected {expected_d_local}")

    max_h_local = max(int(v) for v in tp_chunk_list)
    max_d_local = max_h_local * int(head_dim)

    if D_local < max_d_local:
        pad_shape = list(x.shape)
        pad_shape[2] = max_d_local - D_local
        zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([x, zeros], dim=2)

    input_tensors = [
        x[:, r * N_local:(r + 1) * N_local, :].contiguous()
        for r in range(world)
    ]
    output_tensors = [
        torch.empty_like(input_tensors[0])
        for _ in range(world)
    ]

    dist.all_to_all(output_tensors, input_tensors, group=group)

    cleaned_tensors = []
    for r in range(world):
        real_d = int(tp_chunk_list[r]) * int(head_dim)
        cleaned_tensors.append(output_tensors[r][:, :, :real_d])

    out = torch.cat(cleaned_tensors, dim=2)
    return out