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
from sklearn.model_selection import StratifiedKFold
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
        self.cfg = cfg
        # self.logger, self.log_dir = make_logger(cfg.TRAIN.OUTPUT_DIR, cfg.DATA.SAMPLE_SIZE)
        # self.logger, self.log_dir = make_logger(cfg.TRAIN.OUTPUT_DIR, cfg.DATA.SAMPLE_SIZE, cfg.DATA_LOADER.TRAIN_BATCH_SIZE, cfg.TRAIN.MAX_EPOCH)
        self.logger, self.log_dir = make_logger(cfg.TRAIN.SAVED_MODEL_DIR, cfg.DATA.SAMPLE_SIZE, cfg.DATA_LOADER.TRAIN_BATCH_SIZE, cfg.TRAIN.MAX_EPOCH)
        if self.cfg.TRAIN.USE_WANDB:
            wandb.config.log_dir = self.log_dir
        self.log_cfg_info()
        if not feature_extract and cfg.DATA_LOADER.DATASET_NAME != 'NOXI': # NoXi 数据集需要交叉验证，因此使用的是def cross_validation里定义的数据加载器，不需要使用 build_dataloader 函数
            self.data_loader = self.build_dataloader()

        self.model = self.build_model()
        self.loss_f = self.build_loss_function()

        self.optimizer = self.build_solver()
        self.scheduler = self.build_scheduler()

        self.collector = TrainSummary()
        self.trainer = self.build_trainer()
        
        self.scaler = self.build_scaler()

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

    def build_scaler(self):
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        scaler = None
        return scaler

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
            pass

    def cross_validation(self, cfg):
        fold_valid_loss = []
        fold_valid_acc = []
        
        # if cfg.CROSS_RESUME:
        #     _, _, _, fold_id = resume_cross_validation(cfg.CROSS_RESUME, self.model, self.optimizer)
        #     cfg.TRAIN.START_FOLD = fold_id
        #     self.logger.info(f"[cross_validation] resume cross validation from fold:{cfg.TRAIN.START_FOLD}")
        
        # for fold_id in range(cfg.TRAIN.START_FOLD, cfg.DATA_LOADER.NUM_FOLD):
        for fold_id in range(cfg.DATA_LOADER.NUM_FOLD):
            print('\n================================== start K-fold cross validation, fold_id:', fold_id, '==================================')
            self.dataloader = build_dataloader(self.cfg, fold_id)
            
            cfg = self.cfg.TRAIN
            if cfg.CROSS_RESUME:
                self.model, self.optimizer, cfg.START_EPOCH = resume_training(cfg.CROSS_RESUME, self.model, self.optimizer)
                # self.model, self.optimizer, cfg.START_EPOCH, _ = resume_cross_validation(cfg.CROSS_RESUME, self.model, self.optimizer)
                self.logger.info(f"[cross_validation] resume cross validation training from {cfg.CROSS_RESUME}, start epoch:{cfg.START_EPOCH}")
            else:
                self.model = build_model(self.cfg)
            
            best_valid_loss = 1e10
            best_valid_acc = 0
            for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
                # 训练用到的文件路径 self.trainer: dpcv/engine/bi_modal_trainer.py;  self.model: dpcv/modeling/networks/audio_visual_residual.py;  self.data_loader["train"]:  备注：结合config/demo/bimodal_resnet18_udiva.yaml里的信息就可以看到class类名
                
                ### 1. train
                print(f'\n==================================Fold:{fold_id} Epo:{epoch+1} [train_epochs] start train... {datetime.now()} ==================================') 
                self.trainer.train(self.dataloader["train"], self.model, self.loss_f, self.optimizer, epoch, self.scaler)
                
                ## 2. valid
                print(f'\n==================================Fold:{fold_id} Epo:{epoch+1} [train_epochs] start valid...     ==================================')
                if epoch % cfg.VALID_INTERVAL == 0: # if epoch % 1 == 0 即每个epoch都进行验证
                    best_valid_loss, best_valid_acc= self.trainer.valid(self.dataloader["valid"], self.model, self.loss_f, self.scheduler, epoch)
                if cfg.TRAINER != "SSASTTrainer": # SSASTTrainer 的学习率调整已经在valid函数里面进行，此处不需要再进行学习率调整
                    self.scheduler.step() # 每个epoch都进行学习率调整
                
                print('[cross_validation] Fold:', fold_id, ', Epo:', epoch+1, ', best_valid_loss:', best_valid_loss, ', best_valid_acc:', best_valid_acc, ', self.collector.model_save:', self.collector.model_save) # type(best_valid_acc): <class 'torch.Tensor'>
                # ### 3. test
                # print(f'\n================================== Epo:{epoch+1} [train_epochs] start test...     ==================================')
                # if epoch % cfg.TEST_INTERVAL == 0: # if epoch % 1 == 0 即每个epoch都在数据集上进行测试
                #     self.trainer.test(self.data_loader["test"], self.model, epoch)
                
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
                
                if self.collector.model_save and self.collector.best_valid_acc >= cfg.ACC_THRESHOLD: # 当best_valid_acc大于等于Acc阈值时，保存模型
                    print('[Cross_Validation] Current epoch:', epoch+1, ', save model with best valid acc:', self.collector.best_valid_acc, '>= cfg.ACC_THRESHOLD:', cfg.ACC_THRESHOLD)
                    save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, cfg, fold_id)
                    self.collector.update_best_epoch(epoch)
                    self.collector.update_best_fold(fold_id)
            
            fold_valid_loss.append(best_valid_loss)
            fold_valid_acc.append(best_valid_acc)
        print('[Cross_Validation] fold_valid_loss =', fold_valid_loss, ', fold_valid_acc =', fold_valid_acc)
        print('[Cross_Validation] average fold_valid_loss =', np.mean(fold_valid_loss), ', average fold_valid_acc =', round(np.mean(fold_valid_acc), 6))
        if self.cfg.TRAIN.USE_WANDB:
            wandb.log({
                "avg_fold_valid_loss": float(np.mean(fold_valid_loss)),
                "avg_fold_valid_acc": float(np.mean(fold_valid_acc)),
            })

    def after_train(self, cfg):
        cfg = self.cfg.TRAIN
        # self.collector.draw_epo_info(log_dir=self.log_dir)
        if self.cfg.DATA_LOADER.DATASET_NAME == "NOXI":
            self.logger.info("{} Train done, best valid acc: {:.5f} in fold:{} epoch:{}".format(
                    datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                    self.collector.best_valid_acc,
                    self.collector.best_fold,
                    self.collector.best_epoch,
                ))
        elif self.cfg.DATA_LOADER.DATASET_NAME == "UDIVA":
            self.logger.info("{} Train done, best valid acc: {:.5f} in epoch:{}".format(
                    datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                    self.collector.best_valid_acc,
                    self.collector.best_epoch,
                ))

    def train(self):
        # cfg = self.cfg.TRAIN
        cfg = self.cfg
        self.before_train(cfg) # 实际实验没有执行 只是检查下
        if cfg.DATA_LOADER.DATASET_NAME == "NOXI":
            self.cross_validation(cfg) # K折交叉验证
        elif cfg.DATA_LOADER.DATASET_NAME == "UDIVA":
            self.train_epochs(cfg) # 正常按照划分的训练集验证集测试集来训练
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
        weight_file = None
        max_acc = -1
        max_fold = -1
        max_acc_file = None

        if cfg.WEIGHT:
            self.model = load_model(self.model, cfg.WEIGHT)
        else:
            #### original logic
            # try: # 从log_dir中找到最新的模型, 具体逻辑: 从log_dir中找到所有以.pkl结尾的文件，然后找到最新的文件, 如果没有找到最新的模型, 则使用checkpoint_last.pkl
            #     weights = [file for file in os.listdir(self.log_dir) if file.endswith(".pkl") and ("last" not in file)] # 找到所有以.pkl结尾的文件, 并且不包含last
            #     weights = sorted(weights, key=lambda x: int(x[11:-4])) # 从文件名中找到最新的模型, 具体逻辑: 从文件名中找到数字, 然后按照数字排序, 最后取最后一个, 也就是最新的模型, 例如: checkpoint_100.pkl, checkpoint_200.pkl, checkpoint_300.pkl, 则取checkpoint_300.pkl
            #     weight_file = os.path.join(self.log_dir, weights[-1]) # 最新的模型的路径
            # except IndexError:
            #     weight_file = os.path.join(self.log_dir, "checkpoint_last.pkl") # 如果没有找到最新的模型, 则使用checkpoint_last.pkl
            
            #### new logic
            # for file in os.listdir(self.log_dir):
            #     if file.endswith(".pkl"):
            #         weight_file = os.path.join(self.log_dir, file) # 识别self.log_dir目录里后缀为.pkl的文件，作为weight_file
            #         self.logger.info(f"[TEST] Test with model {weight_file}")
            # 目录下有多个pkl文件(e.g. checkpoint_fold1_acc0.6667_ep0.pkl)，识别每个pkl文件的acc数值，选取最大的那个pkl,如果有2个pkl的acc一样，那么就选择fold数值更大的那个
            # self.log_dir = '/home/zl525/code/DeepPersonality/saved_model/deeppersonality/bimodal_resnet_noxi/03-29_15-58_sp2_bs16_ep2' # temp test
            if self.cfg.DATA_LOADER.DATASET_NAME == "UDIVA":
                for file in os.listdir(self.log_dir):
                    # e.g. file: checkpoint_acc0.6667_ep0.pkl or checkpoint_acc0.6667_last.pkl
                    if file.endswith(".pkl"):
                        acc_str = file.split("_")[1]
                        acc_float = float(acc_str[3:])
                        if acc_float > max_acc:
                            max_acc = acc_float
                            max_acc_file = file
                            weight_file = os.path.join(self.log_dir, max_acc_file)
            elif self.cfg.DATA_LOADER.DATASET_NAME == "NOXI":
                for file in os.listdir(self.log_dir):
                    # e.g. file: checkpoint_fold1_acc0.6667_ep0.pkl or checkpoint_fold1_acc0.6667_last.pkl
                    if file.endswith(".pkl"):
                        fold_str = file.split("_")[1]
                        fold_int = int(fold_str[4:])
                        acc_str = file.split("_")[2]
                        acc_float = float(acc_str[3:])
                        if acc_float > max_acc or (acc_float == max_acc and fold_int > max_fold):
                            max_acc = acc_float
                            max_fold = fold_int
                            max_acc_file = file
                            # print('max_acc_file: ', max_acc_file, 'max_acc: ', max_acc, 'max_fold: ', max_fold)
                            weight_file = os.path.join(self.log_dir, max_acc_file)
            
            if weight_file is None:
                self.logger.warning(f"[TEST] No saved model found in {self.log_dir}")
                return
            else:
                self.logger.info(f"[TEST] Test with model {weight_file}")
            
            # 如果load_model报错, 则打印警告，退出程序
            try:
                self.model = load_model(self.model, weight_file)
            except Exception as e:
                self.logger.warning(f"[TEST] Error when loading model {weight_file}: {e}")
                return
        
        if self.cfg.DATA_LOADER.DATASET_NAME == "UDIVA":
            if not self.cfg.TEST.FULL_TEST: # "FULL_TEST":false
                test_acc = self.trainer.test(self.data_loader["test"], self.model)
            else:
                ocean_acc_avg, ocean_acc, dataset_output, dataset_label = self.trainer.full_test(self.data_loader["full_test"], self.model)
        elif self.cfg.DATA_LOADER.DATASET_NAME == "NOXI":
            dataloader = build_dataloader(self.cfg, fold_id=0)
            test_acc = self.trainer.test(dataloader["test"], self.model)

        # if cfg.SAVE_DATASET_OUTPUT: # "SAVE_DATASET_OUTPUT":"" 即默认不保存
        #     os.makedirs(cfg.SAVE_DATASET_OUTPUT, exist_ok=True)
        #     torch.save(dataset_output, os.path.join(cfg.SAVE_DATASET_OUTPUT, "pred.pkl"))
        #     torch.save(dataset_label, os.path.join(cfg.SAVE_DATASET_OUTPUT, "label.pkl"))
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