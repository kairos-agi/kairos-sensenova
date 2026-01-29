import torch
import torch.distributed as dist

class SimpleParallelState:
    """
    A lightweight fallback for Megatron's parallel_state.

    Notes / Limitations:
    - This implementation treats "context parallel (CP)" as the *entire* WORLD process group:
        CP group == dist.group.WORLD
        CP size  == dist.get_world_size()
        CP rank  == dist.get_rank()
    - It is suitable when you only need rank/world_size/group for basic coordination
      (e.g., logging, simple WORLD communication, or local tensor chunking).
    - It does NOT create CP sub-groups. If you need cp_size < world_size (multiple CP groups),
      you must build groups via dist.new_group(...) yourself or use Megatron parallel_state.
    """
    @staticmethod
    def get_context_parallel_group():
        if dist.is_initialized():
            return dist.group.WORLD
        return None

    @staticmethod
    def get_context_parallel_world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def get_context_parallel_rank():
        if dist.is_initialized():
            return dist.get_rank()
        return 0

    @staticmethod
    def init_parallel_state(backend: str | None = None):
        if not dist.is_initialized():
            if backend is None:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

parallel_state = SimpleParallelState()