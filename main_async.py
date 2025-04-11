import argparse
import datetime
import torch
import random
from mpi4py import MPI
import ezkfg as ez
import time
from loguru import logger
from model import get_model
import shutil
import time

from server import ServerAsync
from client import ClientAsync
from algor import require_num_samples
from utils import parse_args, set_seed, get_cfg

# 记录开始时间
start_time = time.time()
start_time_formatted = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')

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
    enqueue=True,
)

logger.info(cfg)

# Set device to use for training
use_cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")

device_count = torch.cuda.device_count()
device = torch.device("cuda:" + str(rank % device_count) if use_cuda else "cpu")
if rank == 0:
    logger.info(f"total device count: {device_count}")
    logger.info(f"开始时间: {start_time_formatted}")  # 记录开始时间到日志

    device = torch.device("cpu")

cfg.update({"device": device, "rank": rank, "size": size, "use_cuda": use_cuda})

# Set seed for reproducibility
set_seed(cfg.seed)

# Asset the number of clients is correct
assert cfg.num_clients == size - 1, "The number of clients is not correct"

# Initialize the server and clients
if rank == 0:
    model = get_model(
        cfg.model_name,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        img_size=cfg.img_size
    )
    logger.info(model)
    server = ServerAsync(cfg, model)
    server.run()


else:

    client = ClientAsync(cfg, rank - 1)
    logger.info(f"Rank {rank} has one client with id {client.id}")

    if require_num_samples(cfg):
        comm.send((client.num_samples, client.id), dest=0)

    while True:
        # Receive the global model from the server
        model = comm.recv(source=0)

        logger.info(f"Rank {rank} received the global model")
        if model == "done":
            break

        # Update the local model
        local_weight, num_sample = client.update_model(model)

        # Send the local model to the server
        comm.send((local_weight, num_sample, client.id), dest=0)
# ===== 新增代码开始 =====
# 记录结束时间
end_time = time.time()
end_time_formatted = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
training_time = end_time - start_time

# 只在主进程(rank=0)计算和打印总时间
if rank == 0:
    # 格式化训练耗时（时:分:秒）
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{seconds:.2f}"

    # 打印结束时间和总训练时间
    print(f"程序结束时间: {end_time_formatted}")
    print(f"程序总运行时间: {formatted_time}")
# ===== 新增代码结束 =====

# Finalize MPI
logger.info(f"Rank {rank} is done")
comm.Barrier()
MPI.Finalize()
