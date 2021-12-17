#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from .config_node import CfgNode


# Global config object
_C = CfgNode()
# Example usage:
#   from configs.config import cfg

_C.DBG = False
_C.OUTPUT_DIR = "./output"
_C.RUN_N_TIMES = 5
# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN_BENCHMARK = False

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1
_C.NUM_SHARDS = 1
# _C.SHARD_ID = 0

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.RNG_SEED = 1

# ----------------------------------------------------------------------
# Model options
# ----------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.WEIGHT_PATH = ""

_C.MODEL.TYPE = "knn_base"
_C.MODEL.FROZEN = False
_C.MODEL.MLP_NUM = 0

# _C.MODEL.FEATURE_DIM = 2048
_C.MODEL.IMAGENET_PRETRAIN = True
_C.MODEL.POOL = "avg"  # "max"
_C.MODEL.KNN_LAMBDA = 0.1

_C.MODEL.LINEAR = CfgNode()
_C.MODEL.LINEAR.BN_FIRST = False
_C.MODEL.LINEAR.MLP_SIZES = []
_C.MODEL.LINEAR.DROPOUT = 0.1


# ----------------------------------------------------------------------
# Datastore options
# ----------------------------------------------------------------------
_C.DSTORE = CfgNode()

# most important three for model type
_C.DSTORE.RETURN_PROBS = True  # if true, weighted averge the labels, otherwise, weighted average the features  # noqa
_C.DSTORE.SEP_HEAD = False  # if true, there are two seperate heads  # noqa
_C.DSTORE.LOSS = True  # if true,
_C.DSTORE.USE_CACHE = False  # if true, will reduce training time dstore computations by storing the training knn probs

_C.DSTORE.TEMP_RANGE = "full"

_C.DSTORE.NORM_TYPE = "L2NC"  # l2 normalize the features before query
_C.DSTORE.POOL_TYPE = "neg_sum"
_C.DSTORE.DIST_TYPE = "l2"  # or "cosine"
_C.DSTORE.FEATUREMAP_POOL = "max"  # "avg"
_C.DSTORE.TEMP = 1.0
_C.DSTORE.TOPK = 20
_C.DSTORE.TOPK_TYPE = None  # "max", or "avg_cls_count" or None which is an arbitary number
_C.DSTORE.FEAT_MLP = False  # wheather to add a mlp to reduce the features for knn

# some fancy training settings
_C.DSTORE.ALTERNATE_LOSS = False
_C.DSTORE.TUNE_EVERY_EPOCH = True  # Tune temp, topk, coeff every-epoch
_C.DSTORE.TUNE_WITH_KNN_ONLY = False


# params for reading the filename
_C.DSTORE.INDEX_FOLDER = ""

# ----------------------------------------------------------------------
# Solver options
# ----------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.LOSS = "softmax"  # or "knn_reg" for joint training
_C.SOLVER.LOSS_GAMMA = 2.0

_C.SOLVER.OPTIMIZER = "sgd"  # or "adamw"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.PATIENCE = 300

# scheduler choices: "decay", "cosine", "constant",
#     "cosine_hardrestart"
_C.SOLVER.SCHEDULER = "decay"

_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.LR_DECAY_SCHEDULE = [6000, 8000]  # for decay
_C.SOLVER.LR_DECAY_FACTOR = 0.5             # for plateau

_C.SOLVER.WARMUP_EPOCH = 5
_C.SOLVER.TOTAL_EPOCH = 30
_C.SOLVER.LOG_EVERY_N = 1000

# # Number of images per batch for sampler
# _C.SOLVER.SAMPLE_PER_BATCH = 1
# _C.SOLVER.DEBUG_OUTPUT = True
# ----------------------------------------------------------------------
# Dataset options
# ----------------------------------------------------------------------
_C.DATA = CfgNode()

_C.DATA.NAME = ""
_C.DATA.KNN_PERCENTAGE = 1.0
_C.DATA.PERCENTAGE = 1.0
_C.DATA.DSTORE_FEATURE = ""
_C.DATA.NUMBER_CLASSES = -1
_C.DATA.MULTILABEL = False
_C.DATA.CLASS_WEIGHTS_TYPE = "none"   # one of "none", "inv", "inv_sqrt"
_C.DATA.DATAPATH = ""
_C.DATA.FEATURE = ""  # e.g. inat2021_supervised
_C.DATA.CROPSIZE = 224  # or 448

_C.DATA.NO_TEST = False
_C.DATA.BATCH_SIZE = 32
# Number of data loader workers per training process
_C.DATA.NUM_WORKERS = 4
# Load data to pinned host memory
_C.DATA.PIN_MEMORY = True

_C.DIST_BACKEND = "nccl"
_C.DIST_INIT_PATH = "env://"
_C.DIST_INIT_FILE = ""


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
