import os
import torch
import random
import numpy as np
import argparse


def setup_config(args, cfg):
    # cfg.DATA_ROOT = args.data_root_dir if args.data_root_dir else cfg.DATA_ROOT
    cfg.LR_INIT = args.lr if args.lr else cfg.LR_INIT
    cfg.TRAIN_BATCH_SIZE = args.bs if args.bs else cfg.TRAIN_BATCH_SIZE
    cfg.MAX_EPOCH = args.max_epoch if args.max_epoch else cfg.MAX_EPOCH
    cfg.RESUME = args.resume if args.resume else cfg.RESUME
    return cfg


def setup_seed(seed=12345):
    np.random.seed(seed) # numpy 
    random.seed(seed) # python
    torch.manual_seed(seed) # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = True


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='deep learning on personality')
    parser.add_argument(
        '-c',
        '--cfg_file',
        help="experiment config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        '--weight',
        dest='weight',
        help='initialize with pretrained model weights',
        type=str,
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="only test model on specified weights",
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='learning rate',
    )
    parser.add_argument(
        '--bs',
        default=None,
        help='training batch size',
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="saved model path to last training epoch",
    )
    parser.add_argument(
        "-m",
        '--max_epoch',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        help='set config keys',
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='extract sample_size frames per video',
    )
    parser.add_argument(
        '--use_wandb',
        type=str,
        default="True",
        help='use wandb to log',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='NUM_WORKERS for dataloader',
    )
    parser.add_argument(
        '--prefetch_factor',
        type=int,
        default=None,
        help='PREFETCH_FACTOR for dataloader',
    )
    parser.add_argument(
        '--num_fold',
        type=int,
        default=None,
        help='num of fold for cross validation',
    )
    parser.add_argument(
        '--max_split_size_mb',
        type=int,
        default=32,
        help='os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:xxx"',
    )
    parser.add_argument(
        '--au_class',
        type=str,
        default="12,8",
        help='AU class',
    )
    parser.add_argument(
        '--backbone_input',
        type=str,
        default="video",
        help='backbone input: video or frame',
    )
    parser.add_argument(
        '--prediction_feat',
        type=str,
        default="cl_edge",
        help='prediction feature: cl_edge or cl',
    )
    parser.add_argument(
        '--fusion_type',
        type=str,
        default="feature_fusion",
        help='fusion type: feature_fusion or decision_fusion',
    )
    parser.add_argument(
        '--use_amp',
        type=str,
        default="False",
        help='use Pytorch AMP',
    )
    parser.add_argument(
        '--use_half',
        type=str,
        default="False",
        help='use half precision',
    )
    args = parser.parse_args()
    return args


def get_device(gpu=0):
    device = torch.device(
        f'cuda:{gpu}'
        if torch.cuda.is_available() and gpu is not None
        else 'cpu')
    return device


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output
