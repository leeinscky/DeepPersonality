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
        # print('[DeepPersonality/dpcv/experiment/exp_runner.py] def __init__ - start')
        self.cfg = cfg
        # self.logger, self.log_dir = make_logger(cfg.TRAIN.OUTPUT_DIR, cfg.DATA.SAMPLE_SIZE)
        self.logger, self.log_dir = make_logger(cfg.TRAIN.OUTPUT_DIR, cfg.DATA.SAMPLE_SIZE, cfg.DATA_LOADER.TRAIN_BATCH_SIZE)
        if self.cfg.TRAIN.USE_WANDB:
            wandb.config.log_dir = self.log_dir
        self.log_cfg_info()
        if not feature_extract:
            self.data_loader = self.build_dataloader()
            # print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - data_loader: ', self.data_loader)

        self.model = self.build_model()
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - self.model:', self.model)
        self.loss_f = self.build_loss_function()
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - self.loss_f:', self.loss_f)

        self.optimizer = self.build_solver()
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - self.optimizer:', self.optimizer)
        self.scheduler = self.build_scheduler()
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - self.scheduler:', self.scheduler)

        self.collector = TrainSummary()
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - self.collector:', self.collector)
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - 准备执行：self.trainer = self.build_trainer()')
        self.trainer = self.build_trainer()
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - 结束执行：self.trainer = self.build_trainer()')
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - self.trainer:', self.trainer)
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def __init__ - end')

    def build_dataloader(self):
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def build_dataloader')
        return build_dataloader(self.cfg)

    def build_model(self):
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def build_model')
        return build_model(self.cfg)

    def build_loss_function(self):
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def build_loss_function')
        return build_loss_func(self.cfg)

    def build_solver(self):
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def build_solver')
        # 优化器的定义逻辑在 dpcv/modeling/solver/optimize.py
        return build_solver(self.cfg, self.model)

    def build_scheduler(self):
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def build_scheduler')
        return build_scheduler(self.cfg, self.optimizer)

    def build_trainer(self):
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] 开始执行 def build_trainer')
        return build_trainer(self.cfg, self.collector, self.logger)

    def before_train(self, cfg):
        cfg = self.cfg.TRAIN
        
        # # 手动加载预训练模型用于测试 
        # # DeepPersonality代码库提供的预训练模型ResNet：checkpoint_297.pkl  Reference: https://github.com/liaorongfan/DeepPersonality
        # checkpoint_path = 'dpcv/modeling/networks/pretrain_model/deeppersonality_resnet_pretrain_checkpoint_297.pkl'
        # checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage) # lambda storage, loc: storage表示的是将模型加载到内存中
        # # 将 visual_branch.init_stage.conv1.weight 的shape从torch.Size([32, 3, 7, 7]) 改为 torch.Size([32, 6, 7, 7]), 即将3通道的图像改为6通道的图像, 通过复制3通道的图像，得到6通道的图像
        # checkpoint["model_state_dict"]['visual_branch.init_stage.conv1.weight'] = torch.cat((checkpoint["model_state_dict"]['visual_branch.init_stage.conv1.weight'], checkpoint["model_state_dict"]['visual_branch.init_stage.conv1.weight']), 1) # 1表示沿着第1个维度进行拼接，即沿着通道维度进行拼接
        # checkpoint["model_state_dict"].pop('audio_branch.init_stage.conv1.weight') # 如果不pop掉这些key，会报错：size mismatch for ....
        # # checkpoint["model_state_dict"].pop('visual_branch.init_stage.conv1.weight')
        # checkpoint["model_state_dict"].pop('linear.weight')
        # checkpoint["model_state_dict"].pop('linear.bias')
        # self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # wandb.config.update({"resume": checkpoint_path}, allow_val_change=True)
        # wandb.config.note = "use checkpoint:{}".format(checkpoint_path)
        
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
        cfg = self.cfg.TRAIN
        for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
            # 训练用到的文件路径 self.trainer: dpcv/engine/bi_modal_trainer.py;  self.model: dpcv/modeling/networks/audio_visual_residual.py;  self.data_loader["train"]:  备注：结合config/demo/bimodal_resnet18_udiva.yaml里的信息就可以看到class类名
            
            ### 1. train
            print(f'\n================================== Epo:{epoch+1} [train_epochs] start train... {datetime.now()} ==================================') 
            self.trainer.train(self.data_loader["train"], self.model, self.loss_f, self.optimizer, epoch)
            
            ## 2. valid
            print(f'\n================================== Epo:{epoch+1} [train_epochs] start valid...     ==================================')
            if epoch % cfg.VALID_INTERVAL == 0: # if epoch % 1 == 0 即每个epoch都进行验证
                self.trainer.valid(self.data_loader["valid"], self.model, self.loss_f, self.scheduler, epoch)
            if cfg.TRAINER != "SSASTTrainer": # SSASTTrainer 的学习率调整已经在valid函数里面进行，此处不需要再进行学习率调整
                self.scheduler.step() # 每个epoch都进行学习率调整
            
            ### 3. test
            print(f'\n================================== Epo:{epoch+1} [train_epochs] start test...     ==================================')
            if epoch % cfg.TEST_INTERVAL == 0: # if epoch % 1 == 0 即每个epoch都在数据集上进行测试
                self.trainer.test(self.data_loader["test"], self.model, epoch)
            
            # print('current epoch:', epoch+1, ', self.collector.model_save =', self.collector.model_save, ', cfg.VALID_INTERVAL=', cfg.VALID_INTERVAL, ', epoch % cfg.VALID_INTERVAL =', epoch % cfg.VALID_INTERVAL)
            
            # # 定期保存模型：当epoch=4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100时，保存模型
            # if (epoch+1) in [40]:
            #     print('current epoch:', epoch+1, ', save model with specific epoch')
            #     save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, cfg)
            
            # if self.collector.model_save and epoch % cfg.VALID_INTERVAL == 0: #  cfg.VALID_INTERVAL=1, if epoch % 1 == 0 即每个epoch都进行模型保存
            #     print('current epoch:', epoch+1, ', save model with best valid acc')
            #     save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, cfg)
            #     self.collector.update_best_epoch(epoch)
            
            # if epoch == (cfg.MAX_EPOCH - 1) and cfg.MAX_EPOCH >= 10: #  最后一个epoch时且epoch数大于等于10时，保存模型
            #     save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, cfg)

    def after_train(self, cfg):
        cfg = self.cfg.TRAIN
        # self.collector.draw_epo_info(log_dir=self.log_dir)
        self.logger.info("{} Train done, best valid acc: {:.5f} in epoch:{}".format(
                datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                self.collector.best_valid_acc,
                self.collector.best_epoch + 1,
            )
        )

    def train(self):
        # cfg = self.cfg.TRAIN
        cfg = self.cfg
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
            self.logger.info(f"Test with model {weight_file}")
            self.model = load_model(self.model, weight_file)

        if not self.cfg.TEST.FULL_TEST: # "FULL_TEST":false
            ocean_acc_avg, ocean_acc, dataset_output, dataset_label, mse, test_acc = self.trainer.test(self.data_loader["test"], self.model)
            self.logger.info("mse: {} mean: {}".format(mse[0], mse[1]))
        else:
            ocean_acc_avg, ocean_acc, dataset_output, dataset_label = self.trainer.full_test(self.data_loader["full_test"], self.model)
        self.logger.info("acc: {} mean: {}".format(ocean_acc, ocean_acc_avg))
        # 记录测试的准确率
        self.logger.info("test_acc: {}".format(test_acc))

        if cfg.COMPUTE_PCC: # "COMPUTE_PCC":true 计算皮尔逊相关系数，即线性相关系数，值域[-1,1]，1表示完全正相关，-1表示完全负相关，0表示不相关，0.8表示强相关
            pcc_dict, pcc_mean = compute_pcc(dataset_output, dataset_label, self.cfg.DATA_LOADER.DATASET_NAME)
            self.logger.info(f"pcc: {pcc_dict} mean: {pcc_mean}")
            if self.cfg.TRAIN.USE_WANDB:
                wandb.log({"test_pcc":pcc_mean})

        if cfg.COMPUTE_CCC: # "COMPUTE_CCC":true 计算皮尔逊相关系数，即线性相关系数，值域[-1,1]，1表示完全正相关，-1表示完全负相关，0表示不相关，0.8表示强相关
            ccc_dict, ccc_mean = compute_ccc(dataset_output, dataset_label, self.cfg.DATA_LOADER.DATASET_NAME)
            self.logger.info(f"ccc: {ccc_dict} mean: {ccc_mean}")
            if self.cfg.TRAIN.USE_WANDB:
                wandb.log({"test_ccc":ccc_mean})

        if cfg.SAVE_DATASET_OUTPUT: # "SAVE_DATASET_OUTPUT":""
            os.makedirs(cfg.SAVE_DATASET_OUTPUT, exist_ok=True)
            torch.save(dataset_output, os.path.join(cfg.SAVE_DATASET_OUTPUT, "pred.pkl"))
            torch.save(dataset_label, os.path.join(cfg.SAVE_DATASET_OUTPUT, "label.pkl"))
        if self.cfg.TRAIN.USE_WANDB:
            wandb.log({
                "test_mse": mse[1],             # 最终测试得到的mse均值，评价模型效果以这个为准
                "test_acc":test_acc,            # 最终测试得到的acc均值，评价模型效果以这个为准
                # "test_pcc":pcc_mean,          # 最终测试得到的pcc均值，评价模型效果以这个为准
                # "test_ccc":ccc_mean,          # 最终测试得到的ccc均值，评价模型效果以这个为准
                })
        
        return

    def test_classification(self, weight=None):
        self.logger.info("Test only mode")
        cfg = self.cfg.TEST
        cfg.WEIGHT = weight if weight else cfg.WEIGHT

        if cfg.WEIGHT:
            self.model = load_model(self.model, cfg.WEIGHT)
        else:
            try: # 从log_dir中找到最新的模型, 具体逻辑: 从log_dir中找到所有以.pkl结尾的文件，然后找到最新的文件, 如果没有找到最新的模型, 则使用checkpoint_last.pkl
                weights = [file for file in os.listdir(self.log_dir) if file.endswith(".pkl") and ("last" not in file)] # 找到所有以.pkl结尾的文件, 并且不包含last
                weights = sorted(weights, key=lambda x: int(x[11:-4])) # 从文件名中找到最新的模型, 具体逻辑: 从文件名中找到数字, 然后按照数字排序, 最后取最后一个, 也就是最新的模型, 例如: checkpoint_100.pkl, checkpoint_200.pkl, checkpoint_300.pkl, 则取checkpoint_300.pkl
                weight_file = os.path.join(self.log_dir, weights[-1]) # 最新的模型的路径
            except IndexError:
                weight_file = os.path.join(self.log_dir, "checkpoint_last.pkl") # 如果没有找到最新的模型, 则使用checkpoint_last.pkl
            self.logger.info(f"[TEST] Test with model {weight_file}")
            
            # 如果load_model报错, 则打印警告，不退出程序
            try:
                self.model = load_model(self.model, weight_file)
            except Exception as e:
                self.logger.warning(f"[TEST] Error when loading model {weight_file}: {e}")
                return

        if not self.cfg.TEST.FULL_TEST: # "FULL_TEST":false
            test_acc = self.trainer.test(self.data_loader["test"], self.model)
        else:
            ocean_acc_avg, ocean_acc, dataset_output, dataset_label = self.trainer.full_test(self.data_loader["full_test"], self.model)

        if cfg.SAVE_DATASET_OUTPUT: # "SAVE_DATASET_OUTPUT":"" 即默认不保存
            os.makedirs(cfg.SAVE_DATASET_OUTPUT, exist_ok=True)
            torch.save(dataset_output, os.path.join(cfg.SAVE_DATASET_OUTPUT, "pred.pkl"))
            torch.save(dataset_label, os.path.join(cfg.SAVE_DATASET_OUTPUT, "label.pkl"))
        return

    def run(self):
        print('\n================================== [run] start training... ==================================')
        self.train()
        print('\n================================== [run] start test...     ==================================')
        if self.cfg.TRAIN.TRAINER in ['BiModalTrainerUdiva', 'SSASTTrainer']:
            self.test_classification() # test for classification task
        else:
            self.test() # test for regression task
        print('================================== [run] done test ...     ==================================')
        # print('[deeppersonality/dpcv/experiment/exp_runner.py] def run - test end ======================')

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