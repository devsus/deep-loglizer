import os
import torch
import torch.distributed as dist

def setup() -> tuple[bool, int, int]:
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))   # gloo ?
        world_size = dist.get_world_size()
        return True, local_rank, world_size
    return False, 0, 1

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0

def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1
