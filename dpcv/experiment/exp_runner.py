import os
import json
import numpy as np
import torch
from datetime import datetime
from dpcv.data.datasets.build import build_dataloader
from dpcv.modeling.networks.build import build_model
from dpcv.modeling.loss.build import build_loss_func
from dpcv.modeling.solver.build import build_solver, build_scheduler
from dpcv.engine.build import build_trainer
from dpcv.evaluation.summary import TrainSummary
from dpcv.checkpoint.save import save_model, resume_training, load_model
from dpcv.evaluation.metrics import compute_pcc, compute_ccc
from dpcv.tools.logger import make_logger
import wandb


class ExpRunner:

    def __init__(self, cfg, feature_extract=None):
        """ run exp from config file

        arg:
            cfg_file: config file of an experiment
        """

        """
        construct certain experiment by the following template
        step 1: prepare dataloader
        step 2: prepare model and loss function
        step 3: select optimizer for gradient descent algorithm
        step 4: prepare trainer for typical training in pytorch manner
        """
        print('[DeepPersonality/dpcv/experiment/exp_runner.py] def __init__ - start')
        self.cfg = cfg
        self.logger, self.log_dir = make_logger(cfg.TRAIN.OUTPUT_DIR)
        self.log_cfg_info()
        if not feature_extract:
            self.data_loader = self.build_dataloader()
            print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - data_loader: ', self.data_loader)

        self.model = self.build_model()
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - self.model:', self.model)
        self.loss_f = self.build_loss_function()
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - self.loss_f:', self.loss_f)

        self.optimizer = self.build_solver()
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - self.optimizer:', self.optimizer)
        self.scheduler = self.build_scheduler()
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - self.scheduler:', self.scheduler)

        self.collector = TrainSummary()
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - self.collector:', self.collector)
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - 准备执行：self.trainer = self.build_trainer()')
        self.trainer = self.build_trainer()
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - 结束执行：self.trainer = self.build_trainer()')
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - self.trainer:', self.trainer)
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - end')

    def build_dataloader(self):
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def build_dataloader')
        return build_dataloader(self.cfg)

    def build_model(self):
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def build_model')
        return build_model(self.cfg)

    def build_loss_function(self):
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def build_loss_function')
        return build_loss_func(self.cfg)

    def build_solver(self):
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def build_solver')
        return build_solver(self.cfg, self.model)

    def build_scheduler(self):
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def build_scheduler')
        return build_scheduler(self.cfg, self.optimizer)

    def build_trainer(self):
        print('[deeppersonality/dpcv/experiment/exp_runner.py] 开始执行 def build_trainer')
        return build_trainer(self.cfg, self.collector, self.logger)

    def before_train(self, cfg):
        # cfg = self.cfg.TRAIN
        if cfg.RESUME:
            self.model, self.optimizer, epoch = resume_training(cfg.RESUME, self.model, self.optimizer)
            cfg.START_EPOCH = epoch
            self.logger.info(f"resume training from {cfg.RESUME}")
        if self.cfg.SOLVER.RESET_LR:
            self.logger.info("change learning rate form [{}] to [{}]".format(
                self.optimizer.param_groups[0]["lr"],
                self.cfg.SOLVER.LR_INIT,
            ))
            self.optimizer.param_groups[0]["lr"] = self.cfg.SOLVER.LR_INIT

    def train_epochs(self, cfg):
        # cfg = self.cfg.TRAIN
        for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
            # 训练用到的文件路径 self.trainer: dpcv/engine/bi_modal_trainer.py;  self.model: dpcv/modeling/networks/audio_visual_residual.py;  self.data_loader["train"]:  备注：结合config/demo/bimodal_resnet18_udiva.yaml里的信息就可以看到class类名
            self.trainer.train(self.data_loader["train"], self.model, self.loss_f, self.optimizer, epoch)
            if epoch % cfg.VALID_INTERVAL == 0: # if epoch % 1 == 0 即每个epoch都进行验证
                self.trainer.valid(self.data_loader["valid"], self.model, self.loss_f, epoch)
            self.scheduler.step() # 每个epoch都进行学习率调整

            if self.collector.model_save and epoch % cfg.VALID_INTERVAL == 0: #  if epoch % 1 == 0 即每个epoch都进行验证
                save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, cfg)
                self.collector.update_best_epoch(epoch)
            if epoch == (cfg.MAX_EPOCH - 1): # 最后一个epoch
                save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, cfg)

    def after_train(self, cfg):
        # cfg = self.cfg.TRAIN
        # self.collector.draw_epo_info(log_dir=self.log_dir)
        self.logger.info(
            "{} done, best acc: {} in :{}".format(
                datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                self.collector.best_valid_acc,
                self.collector.best_epoch,
            )
        )

    def train(self):
        cfg = self.cfg.TRAIN
        self.before_train(cfg) # 实际实验没有执行 只是检查下
        self.train_epochs(cfg)
        self.after_train(cfg)

    def test(self, weight=None):
        self.logger.info("Test only mode")
        cfg = self.cfg.TEST
        cfg.WEIGHT = weight if weight else cfg.WEIGHT

        if cfg.WEIGHT:
            self.model = load_model(self.model, cfg.WEIGHT)
        else:
            try:
                weights = [file for file in os.listdir(self.log_dir) if file.endswith(".pkl") and ("last" not in file)]
                weights = sorted(weights, key=lambda x: int(x[11:-4]))
                weight_file = os.path.join(self.log_dir, weights[-1])
            except IndexError:
                weight_file = os.path.join(self.log_dir, "checkpoint_last.pkl")
            self.logger.info(f"test with model {weight_file}")
            self.model = load_model(self.model, weight_file)

        if not self.cfg.TEST.FULL_TEST: # "FULL_TEST":false
            ocean_acc_avg, ocean_acc, dataset_output, dataset_label, mse = self.trainer.test(
                self.data_loader["test"], self.model
            )
            self.logger.info("mse: {} mean: {}".format(mse[0], mse[1]))
        else:
            ocean_acc_avg, ocean_acc, dataset_output, dataset_label = self.trainer.full_test(
                self.data_loader["full_test"], self.model
            )
        self.logger.info("acc: {} mean: {}".format(ocean_acc, ocean_acc_avg))

        if cfg.COMPUTE_PCC: # "COMPUTE_PCC":true 计算皮尔逊相关系数，即线性相关系数，值域[-1,1]，1表示完全正相关，-1表示完全负相关，0表示不相关，0.8表示强相关
            pcc_dict, pcc_mean = compute_pcc(dataset_output, dataset_label, self.cfg.DATA_LOADER.DATASET_NAME)
            self.logger.info(f"pcc: {pcc_dict} mean: {pcc_mean}")

        if cfg.COMPUTE_CCC: # "COMPUTE_CCC":true 计算皮尔逊相关系数，即线性相关系数，值域[-1,1]，1表示完全正相关，-1表示完全负相关，0表示不相关，0.8表示强相关
            ccc_dict, ccc_mean = compute_ccc(dataset_output, dataset_label, self.cfg.DATA_LOADER.DATASET_NAME)
            self.logger.info(f"ccc: {ccc_dict} mean: {ccc_mean}")

        if cfg.SAVE_DATASET_OUTPUT: # "SAVE_DATASET_OUTPUT":""
            os.makedirs(cfg.SAVE_DATASET_OUTPUT, exist_ok=True)
            torch.save(dataset_output, os.path.join(cfg.SAVE_DATASET_OUTPUT, "pred.pkl"))
            torch.save(dataset_label, os.path.join(cfg.SAVE_DATASET_OUTPUT, "label.pkl"))
        wandb.log({
            "test_mse": mse[1],         # 最终测试得到的mse均值，评价模型效果以这个为准
            "test_acc":ocean_acc_avg,   # 最终测试得到的acc均值，评价模型效果以这个为准
            "test_pcc":pcc_mean,        # 最终测试得到的pcc均值，评价模型效果以这个为准
            "test_ccc":ccc_mean,        # 最终测试得到的ccc均值，评价模型效果以这个为准
            })
        
        return

    def run(self):
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def run - train start ======================')
        self.train()
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def run - train end ======================')
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def run - test start ======================')
        self.test()
        print('[deeppersonality/dpcv/experiment/exp_runner.py] def run - test end ======================')

    def log_cfg_info(self):
        """
        record training info for convenience of results analysis
        """
        string = json.dumps(self.cfg, sort_keys=True, indent=4, separators=(',', ':'))
        self.logger.info(string)

    def data_extract(self, dataloader, output_dir):

        return self.trainer.data_extract(self.model, dataloader, output_dir)




'''
def __init__ - self.model: AudioVisualResNet18(
  (audio_branch): AudioVisualResNet(
    (init_stage): AudInitStage(
      (conv1): Conv2d(1, 32, kernel_size=(1, 49), stride=(1, 4), padding=(0, 24), bias=False)
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=(1, 9), stride=(1, 4), padding=(0, 4), dilation=1, ceil_mode=False)
    )
    (layer1): Sequential(
      (0): BiModalBasicBlock(
        (conv1): Conv2d(32, 32, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(32, 32, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BiModalBasicBlock(
        (conv1): Conv2d(32, 32, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(32, 32, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer2): Sequential(
      (0): BiModalBasicBlock(
        (conv1): Conv2d(32, 64, kernel_size=(1, 9), stride=(1, 4), padding=(0, 4), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 4), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BiModalBasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer3): Sequential(
      (0): BiModalBasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(1, 9), stride=(1, 4), padding=(0, 4), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 4), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BiModalBasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer4): Sequential(
      (0): BiModalBasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(1, 9), stride=(1, 4), padding=(0, 4), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 4), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BiModalBasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  )
  (visual_branch): AudioVisualResNet(
    (init_stage): VisInitStage(
      (conv1): Conv2d(3, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )
    (layer1): Sequential(
      (0): BiModalBasicBlock(
        (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BiModalBasicBlock(
        (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer2): Sequential(
      (0): BiModalBasicBlock(
        (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BiModalBasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer3): Sequential(
      (0): BiModalBasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BiModalBasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer4): Sequential(
      (0): BiModalBasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BiModalBasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  )
  (linear): Linear(in_features=512, out_features=5, bias=True)
)
'''