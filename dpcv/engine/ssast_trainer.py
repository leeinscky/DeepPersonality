from cmath import e
import torch
import numpy as np
from .build import TRAINER_REGISTRY
import time
from torchmetrics.functional import auroc
import pickle
import datetime
from dpcv.tools.ssast_utils import *
import argparse
import os
import sys
import time
import wandb
from tqdm import tqdm

@TRAINER_REGISTRY.register()
class SSASTTrainer(object):
    """modified from audio model SSAST: https://github.com/YuanGongND/ssast
    Args:
        object (_type_): _description_
    """
    def __init__(self, cfg, collector, logger):
        self.cfg = cfg.TRAIN
        self.cfg_model = cfg.MODEL
        self.cfg_solver = cfg.SOLVER
        self.clt = collector
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args_dict = {'dataset': 'udiva', 
                     'num_mel_bins': 128, 
                     'exp_dir': '', 
                    #  'lr': 0.0001,  # initial learning rate
                     'optim': 'adam', 
                     'batch_size': 32, 
                     'n_epochs': 1, 
                     'lr_patience': 2, 
                     'n_print_steps': 100, 
                     'save_model': False, 
                     'freqm': 0, 
                     'timem': 0, 
                     'mixup': 0.0, 
                     'bal': 'none', 
                     'fstride': 16, 
                     'tstride': 16, 
                     'fshape': 16, 
                     'tshape': 16, 
                     'model_size': 'tiny', 
                     'task': 'pretrain_mpc', 
                     'mask_patch': 400, 
                     'cluster_factor': 3, 
                     'epoch_iter': 4000,  # for pretraining, how many iterations to verify and save models
                     'pretrained_mdl_path': None, 
                     'head_lr': 1, 
                     'noise': None, 
                     'metrics': 'mAP', 
                     'lrscheduler_start': 10, 
                     'lrscheduler_step': 5, 
                     'lrscheduler_decay': 0.5, 
                     'wa': None, 
                     'wa_start': 16, 
                     'wa_end': 30,
                     'loss': 'BCE'}
        self.exp_dir=f"/home/zl525/code/DeepPersonality/pre_trained_weights/SSAST/mask01-{args_dict['model_size']}-f{args_dict['fshape']}-t{args_dict['tshape']}-epo{self.cfg.MAX_EPOCH}-lr{self.cfg_solver.LR_INIT}-m{args_dict['mask_patch']}-{args_dict['task']}-{args_dict['dataset']}"
        args_dict['exp_dir'] = self.exp_dir
        self.args = argparse.Namespace(**args_dict)
        print('args:', self.args, ' type:', type(self.args))
        
        if os.path.exists("%s/models" % self.args.exp_dir) == False:
            os.makedirs("%s/models" % self.args.exp_dir)
        self.optimizer = None
        self.global_step = 0
        self.best_val_acc = 0
    
    def train(self, train_loader, audio_model, loss_f, optimizer, epoch_idx, args=None):
        # print(f'*********** SSASTTrainer.train start {datetime.datetime.now()} ***********')
        # print('training optimizer:', optimizer)
        lr = optimizer.param_groups[0]['lr']
        self.optimizer = optimizer
        self.logger.info(f"Training learning rate:{lr}")
        
        # print('[SSASTTrainer] train_loader:', train_loader, ' type:', type(train_loader))
        args = self.args
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize all of the statistics we want to keep track of
        batch_time = AverageMeter()
        per_sample_time = AverageMeter()
        data_time = AverageMeter()
        per_sample_data_time = AverageMeter()
        loss_meter = AverageMeter()
        per_sample_dnn_time = AverageMeter()
        train_acc_meter = AverageMeter()
        train_nce_meter = AverageMeter()
        exp_dir = args.exp_dir

        # def _save_progress():
        #     progress.append([epoch, self.global_step, best_epoch, time.time() - start_time])
        #     with open("%s/progress.pkl" % exp_dir, "wb") as f:
        #         pickle.dump(progress, f)

        if not isinstance(audio_model, nn.DataParallel):
            audio_model = nn.DataParallel(audio_model)

        audio_model = audio_model.to(device)
        # Set up the optimizer
        audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
        if epoch_idx == 0:
            print('Total parameter number is : {:.9f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
            print('Total trainable parameter number is : {:.9f} million'.format(sum(p.numel() for p in audio_trainables) / 1e6))
        # trainables = audio_trainables
        # self.optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        audio_model.train()
        end_time = time.time()
        audio_model.train()

        # save from-scratch models before the first epoch
        if epoch_idx == 0:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, self.global_step+1))

        # 打印train_loader的属性
        # print('[SSASTTrainer] train_loader: ', train_loader.__dict__)
        for i, data in enumerate(train_loader):
            inputs, labels = self.data_fmt(data)
            audio_input = inputs[0].to(device)
            
            # measure data loading time
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up  学习率从0开始逐渐增加到args.lr，每隔50步就乘以一个系数，这个系数是global_step/1000, 直到global_step=1000，学习率就是args.lr*1
            # 因为论文里使用的数据集的训练集音频数量很多(例如FSD50K 训练集有3.7万个音频clips)，bs=16时，一个完整的epoch有2312个step，所以用了1000，和50这个系数，代表lr在第一个epoch内完成warm up，过程中每50个step就增长；对于我们的数据集，可以适当调整小一些。udiva训练集有115个session, 即完整的一个epoch就是(115/bs)个step, 如果bs=16, 一共115/16=7.2=8个step。也就是说一个epoch只有8个step。我们计划每4个step就增长一次，一直到第5个epoch即step=40时，学习率就是args.lr*1 所以我们设置的系数是40，和4
            if self.global_step <= 40 and self.global_step % 4 == 0: # 如果global_step小40，每4步打印一次 当bs=16时，即每半个epoch进行warm up，一直到第5个epoch即step=40时，学习率就是args.lr*1
                warm_lr = (self.global_step / 1000) * self.cfg_solver.LR_INIT # 0.001*(2/1000) = 0.0002
                print('[warm_lr compute], step: {}, LR_INIT: {}, factor: {}, warm_lr = LR_INIT*factor = {:.2e}'.format(self.global_step, self.cfg_solver.LR_INIT, (self.global_step / 1000), warm_lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                lr = optimizer.param_groups[0]['lr']
                # print('\n[Train] current warm-up learning rate is {:.2e}'.format(optimizer.param_groups[0]['lr']))

            # use cluster masking only when masking patches, not frames
            cluster = (args.num_mel_bins != args.fshape) # num_mel_bins = 128, fshape = 16 cluster = True
            # if pretrain with discriminative objective
            if args.task == 'pretrain_mpc':
                acc, loss = audio_model(audio_input, args.task, mask_patch=args.mask_patch, cluster=cluster)
                # this is for multi-gpu support, in our code, loss is calculated in the model
                # pytorch concatenates the output of each gpu, we thus get mean of the losses of each gpu
                acc, loss = acc.mean(), loss.mean()
            # if pretrain with generative objective
            elif args.task == 'pretrain_mpg':
                loss = audio_model(audio_input, args.task, mask_patch=args.mask_patch, cluster=cluster)
                loss = loss.mean()
                # dirty code to make the code report mse loss for generative objective
                acc = loss
            # if pretrain with joint discriminative and generative objective
            elif args.task == 'pretrain_joint':
                acc, loss1 = audio_model(audio_input, 'pretrain_mpc', mask_patch=args.mask_patch, cluster=cluster)
                acc, loss1 = acc.mean(), loss1.mean()
                loss2 = audio_model(audio_input, 'pretrain_mpg', mask_patch=args.mask_patch, cluster=cluster)
                loss2 = loss2.mean()
                loss = loss1 + 10 * loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            train_acc_meter.update(acc.detach().cpu().item())
            train_nce_meter.update(loss.detach().cpu().item())
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time) # 单位：s
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0]) # 单位：s
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0]) # 单位：s

            # print_step = self.global_step % args.n_print_steps == 0 # n_print_steps = 100
            # early_print_step = epoch == 0 and self.global_step % (args.n_print_steps/10) == 0
            # print_step = print_step or early_print_step

            # # if print_step and self.global_step != 0:
            # print('Epoch: [{0}][{1}/{2}]\t'
            #     'Per Sample Total Time {per_sample_time.avg:.5f}\t'
            #     'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
            #     'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
            #     'Train Loss {loss_meter.val:.4f}\t'.format(
            #     epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
            #         per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
            
            if np.isnan(loss_meter.avg):
                print("[Train] training diverged...")
                return

            if self.cfg.USE_WANDB:
                wandb.log({
                    "train_loss":  float(loss_meter.val),
                    "train_acc": float(train_acc_meter.val), # 当前batch的acc
                    "train_nce": float(train_nce_meter.val), # 当前batch的loss
                    "learning rate": lr,
                    "epoch": epoch_idx + 1,
                })

            self.logger.info(
                "[Train-{}]: Epo[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] Global_step:{} lr:{:.2e} Total_Time:{:.5f} Data_Time:{:.5f} DNN_Time:{:.5f} Nce Loss:{:.6f} Acc:{:.6f}".format(
                    epoch_idx + 1, epoch_idx + 1, self.cfg.MAX_EPOCH,   # Epo 
                    i + 1, len(train_loader),            # Iter
                    self.global_step,                    # global_step
                    lr,                                  # lr
                    per_sample_time.avg,                 # Total Time
                    per_sample_data_time.avg,            # Data Time
                    per_sample_dnn_time.avg,             # DNN Time
                    loss_meter.val,                      # Nce Loss
                    train_acc_meter.val,                 # Acc
                ))

            end_time = time.time()
            self.global_step += 1

        """
            # pretraining data is usually very large, save model every epoch is too sparse.
            # save the model every args.epoch_iter steps. 
            # 因为ssast代码库中训练代码只有一行：trainmask(audio_model, train_loader, val_loader, args)，没有包含valid()，所以把valid()放在train()中，每4000步验证一次
            epoch_iteration = args.epoch_iter # epoch_iter = 4000
            if global_step % epoch_iteration == 0: 
                equ_epoch = int(global_step/epoch_iteration) + 1 # 如果global_step = 4000, equ_epoch = 2，如果global_step = 8000, equ_epoch = 3，以此类推
                print(f"---------------- step {str(global_step)} evaluation, equ_epoch:{equ_epoch} ----------------")
                acc_eval, nce_eval = self.valid(val_loader=test_loader, audio_model=audio_model, args=args, epoch=equ_epoch)

                print("masked acc train: {:.6f}".format(acc))
                print("nce loss train: {:.6f}".format(loss))
                print("masked acc eval: {:.6f}".format(acc_eval))
                print("nce loss eval: {:.6f}".format(nce_eval))
                result.append([train_acc_meter.avg, train_nce_meter.avg, acc_eval, nce_eval, optimizer.param_groups[0]['lr']]) # 将训练集准确率、nce loss、验证集准确率、nce loss和学习率等信息添加到结果列表中。
                np.savetxt(exp_dir + '/result.csv', result, delimiter=',') # 将result写入result.csv

                if acc > best_acc: # 如果当前模型在验证集上的准确率比历史最佳准确率更高，则保存当前模型为最佳模型。
                    best_acc = acc
                    torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))

                #  每隔一定步数保存一次模型。
                torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, equ_epoch))
                
                if len(train_loader.dataset) > 2e5: # 2e5 = 200000
                    torch.save(optimizer.state_dict(), "%s/models/optim_state.pth" % (exp_dir))

                # if the task is generation, stop after eval mse loss stop improve
                if args.task == 'pretrain_mpg':
                    # acc_eval is in fact the mse loss, it is dirty code
                    scheduler.step(-acc_eval)
                else:
                    scheduler.step(acc_eval)

                print('# {:d}, step {:d}-{:d}, lr: {:e}'.format(equ_epoch, global_step-epoch_iteration, global_step, optimizer.param_groups[0]['lr']))

                _save_progress() # 保存当前训练进度信息。 可以用wandb替代

                finish_time = time.time()
                print('# {:d}, step {:d}-{:d}, training time: {:.3f}'.format(equ_epoch, global_step-epoch_iteration, global_step, finish_time-begin_time))
                begin_time = time.time()

                train_acc_meter.reset()
                train_nce_meter.reset()
                batch_time.reset()
                per_sample_time.reset()
                data_time.reset()
                per_sample_data_time.reset()
                loss_meter.reset()
                per_sample_dnn_time.reset()

                # change the models back to train mode
                audio_model.train()
                print('---------------- evaluation finished ----------------')
        """

    def valid(self, val_loader, audio_model, loss_f, scheduler, epoch_idx, args=None, epoch=2):
        # print('*********** SSASTTrainer.valid start ***********')
        # print('input scheduler:', scheduler) # <torch.optim.lr_scheduler.ReduceLROnPlateau object at 0x151a34a984c0>
        args = self.args
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(audio_model, nn.DataParallel):
            audio_model = nn.DataParallel(audio_model)
        audio_model = audio_model.to(device)
        audio_model.eval()

        A_acc = []
        A_nce = []
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = self.data_fmt(data)
                # print('audio_input[0] type:', type(audio_input[0]), 'shape:', audio_input[0].shape)
                audio_input = inputs[0].to(device)
                
                # use cluster masking only when masking patches, not frames
                cluster = (args.num_mel_bins != args.fshape)
                # always use mask_patch=400 for evaluation, even the training mask patch number differs.
                if args.task == 'pretrain_mpc':
                    acc, nce = audio_model(audio_input, args.task, mask_patch=400, cluster=cluster)
                    A_acc.append(torch.mean(acc).cpu())
                    A_nce.append(torch.mean(nce).cpu())
                elif args.task == 'pretrain_mpg':
                    mse = audio_model(audio_input, args.task, mask_patch=400, cluster=cluster)
                    # this is dirty code to track mse loss, A_acc and A_nce now track mse, not the name suggests
                    A_acc.append(torch.mean(mse).cpu())
                    A_nce.append(torch.mean(mse).cpu())
                elif args.task == 'pretrain_joint':
                    acc, _ = audio_model(audio_input, 'pretrain_mpc', mask_patch=400, cluster=cluster)
                    mse = audio_model(audio_input, 'pretrain_mpg', mask_patch=400, cluster=cluster)

                    A_acc.append(torch.mean(acc).cpu())
                    # A_nce then tracks the mse loss
                    A_nce.append(torch.mean(mse).cpu())

            acc_eval = np.mean(A_acc)
            nce_eval = np.mean(A_nce)

            if acc_eval > self.best_val_acc: # 如果当前模型在验证集上的准确率比历史最佳准确率更高，则保存当前模型为最佳模型。
                print(f'[Valid-{epoch_idx + 1}]:acc_eval:{acc_eval:.9f} > best_val_acc:{self.best_val_acc:.9f}, gap:{(acc_eval-self.best_val_acc):.9f}, will save model: {self.exp_dir}/models/best_audio_model.pth')
                self.best_val_acc = acc_eval
                torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (self.exp_dir))
            
            if acc_eval > self.clt.best_valid_acc: # 如果当前的epoch_summary_acc大于之前的最好的epoch_summary_acc
                # print(f'[Valid-{epoch_idx + 1}]: Current val epoch summary acc:{acc_eval:.7f} > best val epoch summary acc: {self.clt.best_valid_acc:.7f}, will save model')
                self.clt.update_best_acc(acc_eval)
                self.clt.update_model_save_flag(1)   # 1表示需要保存模型
            else:
                # print(f'[Valid-{epoch_idx + 1}]: Current val epoch summary acc:{acc_eval:.7f} <= best val epoch summary acc: {self.clt.best_valid_acc:.7f}, not save model')
                self.clt.update_model_save_flag(0)  # 0表示不需要保存模型

            # if the task is generation, stop after eval mse loss stop improve
            if args.task == 'pretrain_mpg':
                # acc_eval is in fact the mse loss, it is dirty code
                scheduler.step(-acc_eval)
                # 打印调整后的学习率
            else:
                scheduler.step(acc_eval)

            if self.cfg.USE_WANDB:
                wandb.log({
                    "val_epoch_summary_acc": float("{:.6f}".format(acc_eval)),
                    "val_epoch_summary_loss": float("{:.6f}".format(nce_eval)),
                    "val_epoch_summary_lr": self.optimizer.param_groups[0]['lr'],
                    "epoch": epoch_idx + 1})

            self.logger.info(
                "[Valid-{}]: Epo[{:0>3}/{:0>3}] lr:{:.4f} Epo Summary Acc:{:.6f} (best:{:.6f}) Nce loss:{:.6f}\n".
                format(
                    epoch_idx + 1, epoch_idx + 1, self.cfg.MAX_EPOCH, # Epoch
                    self.optimizer.param_groups[0]['lr'], # lr
                    acc_eval,       # Epo Summary Acc
                    self.best_val_acc,   # best Epo Summary Acc
                    nce_eval,       # Epo Summary Nce loss
                ))
        return acc, nce

    def test(self, test_loader, audio_model, epoch_idx, args=None, epoch=2):
        # print('*********** SSASTTrainer.test start ***********')
        args = self.args
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(audio_model, nn.DataParallel):
            audio_model = nn.DataParallel(audio_model)
        audio_model = audio_model.to(device)
        # switch to evaluate mode
        audio_model.eval()

        A_acc = []
        A_nce = []
        with torch.no_grad():
            for data in tqdm(test_loader): 
                inputs, labels = self.data_fmt(data)
                audio_input = inputs[0].to(device)

                # use cluster masking only when masking patches, not frames
                cluster = (args.num_mel_bins != args.fshape)
                # always use mask_patch=400 for evaluation, even the training mask patch number differs.
                if args.task == 'pretrain_mpc':
                    acc, nce = audio_model(audio_input, args.task, mask_patch=400, cluster=cluster)
                    A_acc.append(torch.mean(acc).cpu())
                    A_nce.append(torch.mean(nce).cpu())
                elif args.task == 'pretrain_mpg':
                    mse = audio_model(audio_input, args.task, mask_patch=400, cluster=cluster)
                    # this is dirty code to track mse loss, A_acc and A_nce now track mse, not the name suggests
                    A_acc.append(torch.mean(mse).cpu())
                    A_nce.append(torch.mean(mse).cpu())
                elif args.task == 'pretrain_joint':
                    acc, _ = audio_model(audio_input, 'pretrain_mpc', mask_patch=400, cluster=cluster)
                    mse = audio_model(audio_input, 'pretrain_mpg', mask_patch=400, cluster=cluster)

                    A_acc.append(torch.mean(acc).cpu())
                    # A_nce then tracks the mse loss
                    A_nce.append(torch.mean(mse).cpu())

            acc = np.mean(A_acc)
            nce = np.mean(A_nce)

            if self.cfg.USE_WANDB:
                wandb.log({
                    "test_acc": float("{:.6f}".format(acc)),
                    "test_nce_loss": float("{:.6f}".format(nce)),
                    "epoch": epoch_idx + 1})

            self.logger.info(
                "[TEST-{}]: Epo[{:0>3}/{:0>3}] Test Acc:{:.6f} Nce loss:{:.6f}\n".
                format(
                    epoch_idx + 1, epoch_idx + 1, self.cfg.MAX_EPOCH, # Epoch
                    acc, # Epo Summary Acc
                    nce, # Epo Summary Nce loss
                ))

        return acc

    def data_fmt(self, data):
        # print('[bi_modal_trainer] type of data: ', type(data))
        if isinstance(data, dict): 
            # 1、如果data_loader中没有使用RandomOverSampler data就是一个dict，dict里有image audio label 这几个key，分别对应image,audio,label的数据
            for k, v in data.items(): # Python 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组。
                data[k] = v.to(self.device)
            if self.cfg.BIMODAL_OPTION == 1:
                aud_in = None
                img_in, labels = data["image"], data["label"]
            elif self.cfg.BIMODAL_OPTION == 2:
                aud_in, labels = data["audio"], data["label"]
                img_in = None
            elif self.cfg.BIMODAL_OPTION == 3:
                img_in, aud_in, labels = data["image"], data["audio"], data["label"]
            else:
                raise ValueError("BIMODAL_OPTION should be 1, 2 or 3. not {}".format(self.cfg.BIMODAL_OPTION))
        elif isinstance(data, list):
            # 2、如果data_loader中使用了RandomOverSampler，那么这里得到的data就是一个list，list里有两个元素，分别是image和audio，image的shape:[batch_size, sample_size, c, h, w] e.g.[8, 16, 6, 224, 224]  label.shape: [batch_size, 2]
            for i, v in enumerate(data): # Python enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                data[i] = v.to(self.device)
            if self.cfg.BIMODAL_OPTION == 1:
                aud_in = None
                img_in, labels = data[0], data[1] # img_in.shape: [batch_size, sample_size, c, h, w] e.g.[8, 16, 6, 224, 224]  label.shape: [batch_size, 2]
                # print('[bi_modal_trainer] img_in.shape: ', img_in.shape, ' labels.shape: ', labels.shape)
            elif self.cfg.BIMODAL_OPTION == 2:
                aud_in, labels = data[0], data[1]
                img_in = None
            elif self.cfg.BIMODAL_OPTION == 3:
                img_in, aud_in, labels = data[0], data[1], data[2]
            else:
                raise ValueError("BIMODAL_OPTION should be 1, 2 or 3. not {}".format(self.cfg.BIMODAL_OPTION))
        else:
            raise ValueError("data type should be dict or list, not {}".format(type(data)))

        # 不同的模型需要不同的维度格式，这里根据不同的模型进行数据格式的转换
        if self.cfg_model.NAME == "resnet50_3d_model_udiva":
            img_in = img_in.permute(0, 2, 1, 3, 4) # 将输入的数据从 [batch, time, channel, height, width] 转换为 [batch, channel, time, height, width] e.g. 4 * 16 * 6 * 224 * 224 -> 4 * 6 * 16 * 224 * 224
            return (img_in, ), labels
        elif self.cfg_model.NAME == "vivit_model_udiva":
            return (img_in, ), labels # img_in: [batch, time, channel, height, width]
        elif self.cfg_model.NAME == "vivit_model3_udiva":
            img_in = img_in.permute(0, 2, 1, 3, 4) # img_in: [batch, channel, time, height, width], # 将输入的数据从 [batch, time, channel, height, width] 转换为 [batch, channel, time, height, width] e.g. 4 * 16 * 6 * 224 * 224 -> 4 * 6 * 16 * 224 * 224
            # print('[data_fmt] img_in.device: ', img_in.device, ', labels.device: ', labels.device)
            return (img_in, ), labels 
        elif self.cfg_model.NAME == "timesformer_udiva":
            img_in = img_in.permute(0, 2, 1, 3, 4) # 将输入的数据从 [batch, time, channel, height, width] 转换为 [batch, channel, time, height, width] e.g. 4 * 16 * 6 * 224 * 224 -> 4 * 6 * 16 * 224 * 224
            return (img_in, ), labels
        elif self.cfg_model.NAME == "ssast_udiva":
            # print('[data_fmt] aud_in.shape: ', aud_in.shape) # [4, 1598, 256] 
            return(aud_in, ), labels
        else:
            return (aud_in, img_in), labels
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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




