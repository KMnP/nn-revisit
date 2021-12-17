#!/usr/bin/env python
"""
linear eval with imagenet via distributed training
heavily borrowed from https://github.com/facebookresearch/moco-v3/blob/main/main_lincls.py
"""
import argparse
import glob
import math
import numpy as np
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import knn.utils.logging as logging
from knn.configs.config import get_cfg
from knn.data.loader import _DATASET_CATALOG
from knn.models.build_model import _MODEL_TYPES
OUTPUT_ROOT=""


best_acc1 = 0


def setup_config(args):
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

    output_path = os.path.join(output_dir, output_folder, "run1")
    cfg.OUTPUT_DIR = output_path
    print("output path: {}".format(output_path))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cfg.freeze()
    return cfg


def imagenet_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=4096, type=int,
                        metavar='N',
                        help='mini-batch size (default: 4096), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # additional configs:
    # here config file only used to construct model
    parser.add_argument('--knn-coeff', default=0., type=float,
                        dest='knn_coeff')
    parser.add_argument(
            "--config-file", default="configs/pretrain/imagenet.yaml",
            metavar="FILE", help="path to config file")
    parser.add_argument(
            "opts",
            help="Modify config options using the command-line",
            default=None,
            nargs=argparse.REMAINDER,
        )
    return parser


def main(args):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # setup cfg
    cfg = setup_config(args)
    args.dist_url = cfg.DIST_INIT_PATH
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, cfg, args)
        )
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, cfg, args)


def main_worker(gpu, ngpus_per_node, cfg, args):
    global best_acc1
    args.gpu = gpu

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # setup logger after initializating distributed learning
    logger = logging.setup_logging(
        ngpus_per_node, int(args.world_size / ngpus_per_node),
        cfg.OUTPUT_DIR, name="nearest_neighbors")

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    # create model
    logger.info("=> creating model {} / {}".format(
        cfg.MODEL.TYPE, cfg.DATA.FEATURE))
    train_type = cfg.MODEL.TYPE
    if cfg.DATA.FEATURE == "dino_resnet50":
        train_type = train_type.replace("vit_", "")
    elif"swin" in cfg.DATA.FEATURE or "dino" in cfg.DATA.FEATURE or "sup_vit" in cfg.DATA.FEATURE or "sup_xcit" in cfg.DATA.FEATURE or "mocov3_vit" in cfg.DATA.FEATURE:
        if not train_type.startswith("vit_"):
            train_type = "vit_" + train_type
    model = _MODEL_TYPES[train_type](cfg, None)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(reduction="none").cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # weight, bias

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    ckpt_path = os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth.tar')
    if os.path.exists(ckpt_path):
        logger.info("=> loading checkpoint '{}'".format(ckpt_path))
        if args.gpu is None:
            checkpoint = torch.load(ckpt_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(ckpt_path, map_location=loc)
        if checkpoint['epoch'] == args.epochs:
            logger.info("Training of this model is finished")
            return

        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            ckpt_path, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(ckpt_path))

    cudnn.benchmark = True

    # Data loading
    logger.info("Loading imagenet training/validation data...")
    dataset_name = cfg.DATA.NAME
    train_dataset = _DATASET_CATALOG[dataset_name](cfg, "train")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        _DATASET_CATALOG[dataset_name](cfg, "val"),
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args, logger)

        # train for one epoch
        train(
            train_loader, model, criterion,
            optimizer, epoch, cfg.DATA.FEATURE,
            logger, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, logger, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_str = "best-" if is_best else ""
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0): # only the first GPU saves checkpoint
            ckpt_path = os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth.tar')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=ckpt_path)
            logger.info(f"Saved {best_str}checkpoints to {ckpt_path}")
            # if epoch == args.start_epoch:
            #     sanity_check(
            #         model.state_dict(), args.pretrained, "last_layer", logger)


def compute_loss(
    criterion, output, target,
    coeff=None, knn_reg_array=None, image_ids=None
):
    loss = criterion(output, target)
    # add knn reg, access a image_id2knnloss array
    if image_ids is not None and knn_reg_array is not None:
        # knn_loss
        image_ids = list(image_ids)
        knn_loss = torch.from_numpy(knn_reg_array[image_ids]).float().to(
            target.device)
        loss += torch.mul(loss, knn_loss * coeff)
    loss = torch.sum(loss) / target.shape[0]
    return loss


def train(
    train_loader, model, criterion,
    optimizer, epoch, model_type, logger, args
):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        logger,
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    # TODO: move this part outside train() to save time
    coeff = torch.tensor(
        args.knn_coeff,
        dtype=torch.float,
        device=args.gpu
    )
    if args.knn_coeff > 0:
        logger.info("Loading pre-computed knn regualization array")
        # load the knn_reg_array
        df_paths = glob.glob(f"{OUTPUT_ROOT}/imagenet/{model_type}/lr0.01_wd0.0001_lambda0.1_max/run1/knn_loss.npy")

        assert len(df_paths) == 1
        with open(df_paths[0], "rb") as f:
            knn_reg_array = np.load(f)
    else:
        knn_reg_array = None

    end = time.time()
    for i, input_data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = input_data["image"]
        target = input_data["label"]
        image_ids = input_data["id"]

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images, imagenet=True)
        loss = compute_loss(
            criterion, output, target, coeff, knn_reg_array, image_ids)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        logger,
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, input_data in enumerate(val_loader):

            images = input_data["image"]
            target = input_data["label"]
            # image_ids = input_data["id"]
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images, imagenet=True)
            loss = compute_loss(criterion, output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        best_filename = filename.replace(
            'checkpoint.pth.tar', 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def sanity_check(state_dict, pretrained_weights, linear_keyword, logger):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    logger.info("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore linear layer
        if '%s.weight' % linear_keyword in k or '%s.bias' % linear_keyword in k:
            continue

        # name in pretrained model
        k_pre = 'module.base_encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.base_encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    logger.info("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, logger, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args, logger):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    logger.info(
        "Learning rate adjusted to {:.3f} for epoch {}".format(cur_lr, epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    args = imagenet_parser().parse_args()
    main(args)
