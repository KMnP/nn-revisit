#!/usr/bin/env python3
import math

import torch.optim as optim
from fvcore.common.config import CfgNode
from torch.optim.lr_scheduler import LambdaLR


def make_scheduler(
    optimizer: optim.Optimizer, train_params: CfgNode
) -> LambdaLR:
    warmup = train_params.WARMUP_EPOCH
    total_iters = train_params.TOTAL_EPOCH

    if train_params.SCHEDULER == "constant":
        scheduler = WarmupConstantLRSchedule(
            optimizer, warmup_steps=warmup)
    elif train_params.SCHEDULER == "linear":
        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=warmup,
            t_total=total_iters
        )
    elif train_params.SCHEDULER == "cosine":
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=warmup,
            t_total=total_iters
        )
    elif train_params.SCHEDULER == "cosine_hardrestart":
        scheduler = WarmupCosineWithHardRestartsSchedule(
            optimizer,
            warmup_steps=warmup,
            t_total=total_iters
        )
    elif train_params.SCHEDULER == "decay":
        scheduler = WarmupConstantDecaySchedule(
            optimizer,
            warmup_steps=warmup,
            lr_decay=train_params.LR_DECAY_SCHEDULE
        )
    elif train_params.SCHEDULER == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "max",
            patience=5,
            verbose=True,
            factor=train_params.LR_DECAY_FACTOR,
        )
    else:
        scheduler = None
    return scheduler


class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(
            optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantLRSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        else:
            return 1.0


class WarmupConstantDecaySchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over
        `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, lr_decay, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantDecaySchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)
        self.decay = lr_decay

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        elif step < self.decay[0]:
            return 0.1 ** 0
        elif step < self.decay[1]:
            return 0.1 ** 1
        else:
            return 0.1 ** 2


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps`.
        Linearly decreases learning rate from 1. to 0.
            over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(
            max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps`.
        Decreases learning rate from 1. to 0. over remaining
            `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate
            follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(
            1, self.t_total - self.warmup_steps))
        return max(
            0.0,
            0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )


class WarmupCosineWithHardRestartsSchedule(LambdaLR):
    """ Linear warmup and then cosine cycles with hard restarts.
        Linearly increases learning rate from 0 to 1 over `warmup_steps`.
        If `cycles` (default=1.) is different from default, learning rate
            follows `cycles` times a cosine decaying learning rate
            (with hard restarts).
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=1., last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(
            0.0,
            0.5 * (1. + math.cos(
                math.pi * ((float(self.cycles) * progress) % 1.0)))
        )
