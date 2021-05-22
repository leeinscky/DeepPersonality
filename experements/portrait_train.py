# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os
import numpy as np
import pickle
import argparse
import torch.optim as optim
from datetime import datetime
from dpcv.engine.portrait_model_trainer import ModelTrainer
from dpcv.utiles.common import setup_seed
from dpcv.utiles.draw import show_confMat, plot_line
from dpcv.utiles.logger import make_logger
from dpcv.modeling.build import get_portrait_model
from dpcv.config.portraint_config import cfg
from dpcv.checkpoint.save import save_model
from dpcv.data.datasets.portrait import make_data_loader
from dpcv.modeling.loss.label_smooth import LabelSmoothLoss

# import sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(BASE_DIR, '..'))


def main(analysis_bad_case=False):
    setup_seed(12345)

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--lr', default=None, help='learning rate')
    parser.add_argument('--bs', default=None, help='training batch size')
    parser.add_argument('--max_epoch', default=None)
    parser.add_argument('--data_root_dir',
                        default=r"/home/rongfan/11-personality_traits/DeepPersonality/datasets/portrait",
                        help="path to your dataset")
    args = parser.parse_args()

    cfg.DATA_ROOT = args.data_root_dir if args.data_root_dir else cfg.DATA_ROOT
    cfg.LR_INIT = args.lr if args.lr else cfg.LR_INIT
    cfg.TRAIN_BATCH_SIZE = args.bs if args.bs else cfg.TRAIN_BATCH_SIZE
    cfg.MAX_EPOCH = args.max_epoch if args.max_epoch else cfg.MAX_EPOCH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create logger
    res_dir = os.path.join("..", "results")
    logger, log_dir = make_logger(res_dir)

    # step1： get dataset
    train_loader = make_data_loader(cfg, mode="train")
    valid_loader = make_data_loader(cfg, mode="test")

    # step2: set model
    model = get_portrait_model()

    # step3: loss functions and optimizer
    # if cfg.label_smooth:
    #     loss_f = LabelSmoothLoss(cfg.label_smooth_eps)
    # else:
    loss_f = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR_INIT,  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.FACTOR, milestones=cfg.MILESTONE)

    # step4: 迭代训练
    # 记录训练所采用的模型、损失函数、优化器、配置参数cfg
    logger.info(
        "cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}".format(
            cfg, loss_f, scheduler, optimizer, model
        )
    )

    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0
    for epoch in range(cfg.MAX_EPOCH):
        # train for one epoch
        loss_train, acc_train, mat_train, path_error_train = ModelTrainer.train(
            train_loader, model, loss_f, optimizer, scheduler, epoch, device, cfg, logger
        )
        # eval for after training for a epoch
        loss_valid, acc_valid, mat_valid, path_error_valid = ModelTrainer.valid(
            valid_loader, model, loss_f, device
        )
        # display info for that training epoch
        logger.info(
            "Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}".
            format(
                epoch + 1, cfg.max_epoch, acc_train,
                acc_valid, loss_train, loss_valid,
                optimizer.param_groups[0]["lr"]
            )
        )
        # update training lr every epoch
        scheduler.step()

        # # 记录训练信息
        # loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        # acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        # # 保存混淆矩阵图
        # show_confMat(mat_train, cfg.flower_cls_names, "train", log_dir, epoch=epoch, verbose=(epoch == cfg.max_epoch - 1))
        # show_confMat(mat_valid, cfg.flower_cls_names, "valid", log_dir, epoch=epoch, verbose=(epoch == cfg.max_epoch - 1))
        # # 保存loss曲线， acc曲线
        # plt_x = np.arange(1, epoch + 2)
        # plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        # plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)
        #
        # # 模型保存
        # if acc_valid > best_acc:
        #     save_model(epoch, best_acc, model, optimizer, log_dir, cfg)
        #     best_epoch = epoch
        #     best_acc = acc_valid
        #
        # if analysis_bad_case:
        #     # 保存错误图片的路径
        #     err_ims_name = "error_imgs_{}.pkl".format(epoch) if epoch == (cfg.max_epoch - 1) else "error_imgs_best.pkl"
        #     path_err_imgs = os.path.join(log_dir, err_ims_name)
        #     error_info = {"train": path_error_train, "valid": path_error_valid}
        #     pickle.dump(error_info, open(path_err_imgs, 'wb'))

    logger.info(
        "{} done, best acc: {} in :{}".format(
            datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_acc, best_epoch
        )
    )


if __name__ == "__main__":
    main()