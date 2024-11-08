import torch.nn as nn
from .build import LOSS_FUNC_REGISTRY


@LOSS_FUNC_REGISTRY.register()
def mean_square_error():
    return nn.MSELoss()


@LOSS_FUNC_REGISTRY.register()
def l1_loss():
    return nn.L1Loss()


@LOSS_FUNC_REGISTRY.register()
def smooth_l1_loss():
    return nn.SmoothL1Loss()

@LOSS_FUNC_REGISTRY.register()
def binary_cross_entropy():
    # print('Using BCELoss')
    return nn.BCELoss()

@LOSS_FUNC_REGISTRY.register()
def cross_entropy():
    return nn.CrossEntropyLoss()
