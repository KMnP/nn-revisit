"""
major actions here: fine-tune the features and evaluate different settings
tune lr, wd, knn_coff here
"""
import os
import warnings
import torch

from time import sleep
from random import randint

from knn.configs.config import get_cfg
from knn.utils.file_io import PathManager

from train import train as train_main
from launch import default_argument_parser
warnings.filterwarnings("ignore")


def setup(args, lr, wd, check_runtime=True):
    """
    Create configs and perform basic setups.
    overwrite the 2 parameters in cfg and args
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup dist
    cfg.DIST_INIT_PATH = "tcp://{}:4000".format(os.environ["SLURMD_NODENAME"])

    # overwrite below four parameters
    lr = lr / 256 * cfg.DATA.BATCH_SIZE  # update lr based on the batchsize
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.WEIGHT_DECAY = wd

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    knn_coeff = cfg.MODEL.KNN_LAMBDA
    knn_feat_pool = cfg.DSTORE.FEATUREMAP_POOL
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE,
        f"lr{lr}_wd{wd}_lambda{knn_coeff}_{knn_feat_pool}"
    )
    # output_folder = os.path.splitext(os.path.basename(args.config_file))[0]

    # train cfg.RUN_N_TIMES times
    if check_runtime:
        count = 1
        while count <= cfg.RUN_N_TIMES:
            output_path = os.path.join(output_dir, output_folder, f"run{count}")
            # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
            sleep(randint(1, 5))
            if not PathManager.exists(output_path):
                PathManager.mkdirs(output_path)
                cfg.OUTPUT_DIR = output_path
                break
            else:
                count += 1
        if count > cfg.RUN_N_TIMES:
            raise ValueError(
                f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")
    else:
        # only used for dummy config file
        output_path = os.path.join(output_dir, output_folder, f"run1")
        cfg.OUTPUT_DIR = output_path

    cfg.freeze()
    return cfg


def main(args):
    """main function to call from workflow"""
    if args.pretrain:
        lr_range = [0.1]
    else:
        lr_range = [0.025, 0.0025, 0.00025]
        # lr_range = [0.025, 0.25, 0.0025]

    if args.train_type == "finetune":
        lr_range = [0.05, 0.025, 0.005, 0.0025, 0.0005, 0.00025]
        wd_range = [0.01, 0.001, 0.0001, 0.00001]
        for wd in wd_range:
            for lr in lr_range:
                # set up cfg and args
                try:
                    cfg = setup(args, lr, wd)
                except ValueError:
                    continue
                train_main(cfg, args)

    elif args.train_type == "linear":
        if "inat2021_mini_moco_v2" in args.opts or "inat2021_mini_swav" in args.opts or "imagenet_swav" in args.opts or "imagenet_moco_v2" in args.opts or "imagenet_barlowtwins" in args.opts or "dino_resnet50" in args.opts or "mocov3_vits" in args.opts or "mocov3_vitb" in args.opts or "mocov3_rn50_ep300" in args.opts or "mocov3_rn50_ep1000" in args.opts or "mocov3_rn50_ep100" in args.opts: # noqa
            lr_range = [40, 20, 10, 5.0, 2.5, 1.0, 0.5, 0.05, 0.1]
        else:
            lr_range = [
                0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.0025, 0.001
            ]
        wd_range = [0.01, 0.001, 0.0001, 0.0]

        for lr in sorted(lr_range, reverse=True):
            for wd in wd_range:
                # set up cfg and args
                try:
                    cfg = setup(args, lr, wd)
                except ValueError:
                    continue
                train_main(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)

