from __future__ import annotations

import os

import torch
import torch.distributed as distributed
from torch import Tensor


def init_distributed_from_env(backend: str = "nccl") -> bool:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False
    selected_backend = backend
    if selected_backend == "nccl" and not torch.cuda.is_available():
        selected_backend = "gloo"
    if not distributed.is_initialized():
        distributed.init_process_group(backend=selected_backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True


def is_distributed() -> bool:
    return distributed.is_available() and distributed.is_initialized()


def get_rank() -> int:
    if not is_distributed():
        return 0
    return distributed.get_rank()


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return distributed.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        distributed.barrier()


def all_gather_1d_tensor(values: Tensor) -> Tensor:
    if not is_distributed():
        return values
    local_values = values.detach().contiguous()
    local_size = torch.tensor([local_values.numel()], device=local_values.device, dtype=torch.long)
    size_list = [torch.zeros_like(local_size) for _ in range(get_world_size())]
    distributed.all_gather(size_list, local_size)
    max_size = int(torch.stack(size_list).max().item())

    padded = torch.empty(max_size, device=local_values.device, dtype=local_values.dtype)
    padded[: local_values.numel()] = local_values
    if local_values.numel() < max_size:
        padded[local_values.numel() :] = 0

    gathered = [torch.empty_like(padded) for _ in range(get_world_size())]
    distributed.all_gather(gathered, padded)
    trimmed = [tensor[: int(size.item())] for tensor, size in zip(gathered, size_list, strict=True)]
    return torch.cat(trimmed, dim=0)