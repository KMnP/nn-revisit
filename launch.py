#!/usr/bin/env python3
"""
for launch locally or in flow
"""
import argparse
import os
import sys
import pprint
import PIL
from collections import defaultdict
from tabulate import tabulate
from typing import Tuple

import torch
from knn.utils.file_io import PathManager
from knn.utils import logging
from knn.utils.distributed import get_rank, get_world_size


def collect_torch_env() -> str:
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def get_env_module() -> Tuple[str]:
    var_name = "ENV_MODULE"
    return var_name, os.environ.get(var_name, "<not set>")


def collect_env_info() -> str:
    data = []
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(get_env_module())
    data.append(("PyTorch", torch.__version__))
    data.append(("PyTorch Debug Build", torch.version.debug))

    has_cuda = torch.cuda.is_available()
    data.append(("CUDA available", has_cuda))
    if has_cuda:
        data.append(("CUDA ID", os.environ["CUDA_VISIBLE_DEVICES"]))
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))
    data.append(("Pillow", PIL.__version__))

    try:
        import cv2

        data.append(("cv2", cv2.__version__))
    except ImportError:
        pass
    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


def default_argument_parser():
    """
    create a parser
    """
    parser = argparse.ArgumentParser(description="kNN-revisited")
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_false",
        help="whether to attempt to resume from the checkpoint directory",
        default=True  # always true for jobs in cluster
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only")
    # parser.add_argument(
    #     "--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument(
        "--pretrain", action="store_true", help="perform pretraining instead of downstream finetuning tasks")
    parser.add_argument(
        "--train-type", default="", help="training types")
    # parser.add_argument(
    #     "--init_method",
    #     help="Initialization method, includes TCP or shared file-system",
    #     default="tcp://localhost:9999",
    #     type=str,
    # )
    parser.add_argument(
        "--shard_id",
        default="0",
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        default="1",
        type=int,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


def train_setup(args, cfg) -> None:
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        PathManager.mkdirs(output_dir)

    logger = logging.setup_logging(
        cfg.NUM_GPUS, get_world_size(), output_dir, name="nearest_neighbors")

    # Log basic information about environment, cmdline arguments, and config
    rank = get_rank()
    logger.info(
        f"Rank of current process: {rank}. World size: {get_world_size()}")
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                PathManager.open(args.config_file, "r").read()
            )
        )
    # Show the config
    logger.info("Training with config:")
    logger.info(pprint.pformat(cfg))
    # cudnn benchmark has large overhead.
    # It shouldn't be used considering the small size of typical val set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
