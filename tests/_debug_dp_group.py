"""Debug: check dp_mesh.get_group() behavior with 2D mesh."""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

mesh_2d = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
dp_mesh = mesh_2d["dp"]
tp_mesh = mesh_2d["tp"]

dp_group = dp_mesh.get_group()
tp_group = tp_mesh.get_group()

print(
    f"[rank {rank}] "
    f"dp_group: size={dp_group.size()}, my_rank_in_group={dp_group.rank()}, "
    f"tp_group: size={tp_group.size()}, my_rank_in_group={tp_group.rank()}"
)

dist.destroy_process_group()
