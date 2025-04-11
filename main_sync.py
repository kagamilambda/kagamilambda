import argparse
import torch
from mpi4py import MPI
import ezkfg as ez
from loguru import logger
import time
import shutil

from server import ServerSync
from client import ClientSync
from utils import set_seed, parse_args, get_cfg
from model import get_model

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


args = parse_args()
cfg = get_cfg(args)

logger.add(
    f"{cfg.log_dir}/{cfg.log_path}/run_{rank}.log",
    format="{time} {level} {message}",
    level="INFO",
    rotation="1 MB",
    compression="zip",
    enqueue=False,
)

logger.info(cfg)


# Set device to use for training
use_cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")

device_count = torch.cuda.device_count()
device = torch.device("cuda:" + str(rank % device_count) if use_cuda else "cpu")
if rank == 0:
    logger.info(f"total device count: {device_count}")
    device = torch.device("cpu")


cfg.update({"device": device, "rank": rank, "size": size, "use_cuda": use_cuda})

# Set seed for reproducibility
set_seed(cfg.seed)

# Initialize the server and clients
if rank == 0:
    model = get_model(
        cfg.model_name,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        img_size=cfg.img_size,
    )
    logger.info(model)
    server = ServerSync(cfg, model)
    server.run()

else:

    # Divide clients among the MPI ranks, rank 0 is reserved for the server
    client_size = size - 1
    # logger.info(f"client_size: {client_size}")
    clients_per_rank = cfg.num_clients // client_size
    remainder = cfg.num_clients % client_size
    if rank <= remainder:
        start_client_idx = (rank - 1) * (clients_per_rank + 1)
        end_client_idx = start_client_idx + clients_per_rank + 1
    else:
        start_client_idx = (rank - 1) * clients_per_rank + remainder
        end_client_idx = start_client_idx + clients_per_rank
    clients = []
    for i in range(start_client_idx, end_client_idx):
        client = ClientSync(cfg, i)
        clients.append(client)
    logger.info(
        f"Rank {rank} has {len(clients)} clients, from {start_client_idx} to {end_client_idx}"
    )

    client_num_samples = []
    client_ids = []
    for client in clients:
        client_num_samples.append(client.num_samples)
        client_ids.append(client.id)

    # Send the number of samples to the server
    comm.send((client_num_samples, client_ids), dest=0)

    while True:
        # Receive the global model from the server
        model = comm.recv(source=0)

        logger.info(f"Rank {rank} received the global model")
        if model == "done":
            break

        local_weights = []
        num_samples = []
        local_ids = []

        # Update the local model
        for client in clients:
            logger.info(f"Rank {rank} is updating client {client.id}")
            local_weight, num_sample = client.update_model(model)

            local_weights.append(local_weight)
            num_samples.append(num_sample)
            local_ids.append(client.id)

        # Send the local model to the server
        comm.send((local_weights, local_ids), dest=0)

# Finalize MPI
logger.info(f"Rank {rank} is done")
comm.Barrier()
MPI.Finalize()
