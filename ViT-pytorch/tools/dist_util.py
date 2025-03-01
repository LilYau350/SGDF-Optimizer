"""
Helpers for distributed training.
"""

import io
import os
import socket

#import blobfile as bf
#from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
#GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def is_main_process():
    """Helper function to check if the current process is the main one."""
    # If the distributed environment is not initialised, it returns True, because in single card mode, 
    # the current process is the master process.
    if not dist.is_available() or not dist.is_initialized():
        return True
    # In a multi-card distributed environment, only the process with a rank of 0 is the master process
    return dist.get_rank() == 0

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    # Dynamically get the number of GPUs
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
    GPUS_PER_NODE = len(visible_devices.split(",")) if visible_devices else th.cuda.device_count()

    local_rank = int(os.getenv('LOCAL_RANK', 0))
    th.cuda.set_device(local_rank)

    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "127.0.0.1")  
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12345")  

    # Setting RANK and WORLD_SIZE
    os.environ["RANK"] = os.getenv("RANK", str(local_rank))  
    os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", str(GPUS_PER_NODE))  

    # Initialise process group, use nccl for GPU communication
    dist.init_process_group(backend="nccl", init_method="env://")

def cleanup_dist():
    """
    Cleanup a distributed process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        
def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()