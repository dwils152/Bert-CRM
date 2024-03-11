import psutil
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, DistributedSampler
import datetime
from tqdm import tqdm

def print_memory_usage():
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"Memory Usage: {memory.percent}% | Total: {memory.total / (1024**3):.2f} GB | Used: {memory.used / (1024**3):.2f} GB | Free: {memory.available / (1024**3):.2f} GB")
    print(f"Swap Usage: {swap.percent}% | Total: {swap.total / (1024**3):.2f} GB | Used: {swap.used / (1024**3):.2f} GB | Free: {swap.free / (1024**3):.2f} GB")


def print_gpu_memory_usage():
    if torch.cuda.is_available():
        print(
            f"Allocated GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(
            f"Reserved GPU Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("CUDA is not available. No GPU memory usage info.")


def setup(rank, world_size):
    # Sets up the environment for distributed training
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=10_000))


def cleanup():
    # Cleanup the distributed process group
    dist.destroy_process_group()


def tqdm_wrapper(iterable, rank, *args, **kwargs):
    if rank == 0:
        return tqdm(iterable, *args, **kwargs)
    else:
        return iterable


def supervised_collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    coordinates = [item[3] for item in batch]  # List of coordinate lists

    return input_ids, attention_mask, labels, coordinates

def unsupervised_collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    coordinates = [item[2] for item in batch]  # List of coordinate lists

    return input_ids, attention_mask, coordinates

