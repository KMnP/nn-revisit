"""
major actions here: fine-tune the features and evaluate different settings
"""
import os
import torch
import warnings

from time import sleep
from random import randint

import knn.utils.logging as logging
from knn.configs.config import get_cfg
from knn.data import loader as data_loader
from knn.engine.evaluator import Evaluator
from knn.engine.knn_evaluator import KNNEvaluator
from knn.engine.linear_trainer import LinearTrainer
from knn.engine.trainer import Trainer
from knn.models.build_model import build_model
from knn.utils.file_io import PathManager

from launch import default_argument_parser, train_setup
warnings.filterwarnings("ignore")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup dist
    cfg.DIST_INIT_PATH = "tcp://{}:12399".format(os.environ["SLURMD_NODENAME"])

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    knn_coeff = cfg.MODEL.KNN_LAMBDA
    knn_feat_pool = cfg.DSTORE.FEATUREMAP_POOL
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE,
        f"lr{lr}_wd{wd}_lambda{knn_coeff}_{knn_feat_pool}"
    )
    # output_folder = os.path.splitext(os.path.basename(args.config_file))[0]

    # train cfg.RUN_N_TIMES times
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        sleep(randint(3, 30))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
            break
        else:
            count += 1
    if count > cfg.RUN_N_TIMES:
        raise ValueError(
            f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")

    cfg.freeze()
    return cfg


def get_loaders(cfg, logger):
    # support two training paradims:
    # 1) train / val / test, using val to tune
    # 2) train / val: for imagenet

    logger.info("Loading training data...")
    train_loader = data_loader.construct_train_loader(cfg)

    logger.info("Loading validation data...")
    val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        logger.info("...no test data is constructed")
        test_loader = None
    else:
        test_loader = data_loader.construct_test_loader(cfg)
    return train_loader,  val_loader, test_loader


def train(cfg, args):
    # main training / eval actions here
    # setup training env including loggers
    train_setup(args, cfg)
    logger = logging.get_logger("nearest_neighbors")

    train_loader, val_loader, test_loader = get_loaders(cfg, logger)
    logger.info("Constructing models...")
    model, cur_device = build_model(cfg, train_loader)

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    if cfg.MODEL.FROZEN:
        trainer = LinearTrainer(cfg, model, evaluator, cur_device)
    else:
        trainer = Trainer(cfg, model, evaluator, cur_device)

    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_loader)
    else:
        print("No train loader presented. Exit")

    # add one eval step
    if "base" not in cfg.MODEL.TYPE:
        logger.info("=" * 80)
        logger.info("Getting joint inference results...")
        logger.info("=" * 80)

        knn_evaluator = KNNEvaluator(cfg, trainer.model, evaluator, cur_device)
        knn_evaluator.get_results(
            train_loader, val_loader, test_loader, build=False)
        if not cfg.MODEL.FROZEN:
            knn_evaluator.get_results(
                train_loader, val_loader, test_loader, build=True)


def main(args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
