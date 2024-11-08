from cmath import e
import torch
from tqdm import tqdm
import numpy as np
import math
import os
from .build import TRAINER_REGISTRY
from torch.utils.tensorboard import SummaryWriter
import time
import wandb
from torchmetrics.functional import auroc
from torchmetrics.classification import BinaryF1Score
torch.set_printoptions(sci_mode=False, linewidth=800) # 使得print时不用科学计数法
# torch.set_printoptions(sci_mode=False) # 使得print时不用科学计数法
import json
import torch.cuda.amp as amp


@TRAINER_REGISTRY.register()
class BiModalTrainer(object):
    """base trainer for bi-modal input"""
    def __init__(self, cfg, collector, logger):
        # print('[DeepPersonality/dpcv/engine/bi_modal_trainer.py] 开始执行BiModal模型的初始化 BiModalTrainer.__init__() ')
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clt = collector
        self.logger = logger
        self.tb_writer = SummaryWriter(cfg.OUTPUT_DIR)
        # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] 结束执行BiModal模型的初始化 BiModalTrainer.__init__()')

    def train(self, data_loader, model, loss_f, optimizer, epoch_idx):
        '''
        # data_loader: 上一层级调用该函数时传入了: self.data_loader["train"], self.data_loader = self.build_dataloader(),  build_dataloader 函数返回的是 data_loader_dicts，
        然后：
        data_loader_dicts = {
            "train": dataloader(cfg, mode="train"),
            "valid": dataloader(cfg, mode="valid"),
            "test": dataloader(cfg, mode="test"),
        }
        然后:
        dataloader = DATA_LOADER_REGISTRY.get(name) #'NAME': 'bimodal_resnet_data_loader', 在 dpcv/data/datasets/audio_visual_data.py里有 def bimodal_resnet_data_loader(cfg, mode) 函数定义
        然后:
        data_loader = DataLoader(  # torch.utils.data.DataLoader, Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
            dataset=dataset, 
            batch_size=batch_size,  # batch_size = cfg.DATA_LOADER.TRAIN_BATCH_SIZE = 8
            shuffle=cfg.DATA_LOADER.SHUFFLE, # cfg.DATA_LOADER.SHUFFLE = True
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,  # 'NUM_WORKERS': 0,
            drop_last=cfg.DATA_LOADER.DROP_LAST, # 'DROP_LAST': True,
        )
        # torch.utils.data.DataLoader 
        其中:
        dataset = AudioVisualData(
            cfg.DATA.ROOT,
            cfg.DATA.TRAIN_IMG_DATA, # 'TRAIN_IMG_DATA': 'ChaLearn2016_tiny/train_data',
            cfg.DATA.TRAIN_AUD_DATA, # 'TRAIN_AUD_DATA': 'ChaLearn2016_tiny/voice_data/voice_librosa/train_data',
            cfg.DATA.TRAIN_LABEL_DATA, # 'TRAIN_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_training.pkl',
            transforms
        )
        然后:
        AudioVisualData 类里有个函数:def __getitem__(self, idx). 该函数返回的是: sample = {"image": img, "audio": wav, "label": label}, 其中: img, wav, label 都是 torch.Tensor 类型, img.shape() =  torch.Size([3, 224, 224]) wav.shape() =  torch.Size([1, 1, 50176]) label.shape() =  torch.Size([5])
        '''
        
        # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 开始执行BiModal模型的train方法')
        lr = optimizer.param_groups[0]['lr']
        self.logger.info(f"Training: learning rate:{lr}")
        self.tb_writer.add_scalar("lr", lr, epoch_idx)

        model.train()
        loss_list = []
        acc_avg_list = []
        # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... len(data_loader)=', len(data_loader), 'type(data_loader): ', type(data_loader)) # len(data_loader)= 7 type(data_loader):  <class 'torch.utils.data.dataloader.DataLoader'>
        final_i = 0 
        # 通过 看日志发现当执行下面这行 for i, data in enumerate(data_loader)语句时，会调用 AudioVisualData(VideoData)类里的 __getitem__ 函数，紧接着调用def get_ocean_label()函数， 具体原因参考：https://www.geeksforgeeks.org/how-to-use-a-dataloader-in-pytorch/
        for i, data in enumerate(data_loader): # len(data_loader)= 7 type(data_loader):  <class 'torch.utils.data.dataloader.DataLoader'> 一共有7个batch, 每个batch的大小是8, 所以一共有56个样本。
            final_i = i
            # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在训练... i=', i)
            # if i == 0:
                # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i, '  data.keys()=', data.keys()) # data.keys()= dict_keys(['image', 'audio', 'label'])
                # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i,  'data[image]的size: ', data['image'].size()) # torch.Size([8, 3, 224, 224]) 8对应config里的batch_size，即8张帧image，3对应RGB，224对应图片的大小，3x224x224代表图片的大小
                # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i,  'data[audio]的size: ', data['audio'].size()) # torch.Size([8, 1, 1, 50176]) 8对应config里的batch_size，即8个wav音频，1对应1个channel，50176对应音频的长度，1x1x50176代表音频的大小，1代表channel，1x50176代表音频的长度
                # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i,  'data[label]的size: ', data['label'].size(),'  data[label]=', data['label']) # torch.Size([8, 5]) 8对应config里的batch_size，5对应5个维度的personality
                # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() i=', i, '  data=', data)
            epo_iter_num = len(data_loader)
            iter_start_time = time.time()

            # self.data_fmt(data) 代表将data里的image, audio, label分别取出来，放到inputs，label里
            inputs, labels = self.data_fmt(data) # inputs是一个元组，包含了image (data['image']: torch.Size([8, 3, 224, 224])) 和audio(data['audio']: torch.Size([8, 1, 1, 50176]))的输入，labels是label
            outputs = model(*inputs)
            optimizer.zero_grad()
            loss = loss_f(outputs.cpu(), labels.cpu())
            self.tb_writer.add_scalar("loss", loss.item(), i)
            loss.backward()
            optimizer.step()

            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time

            loss_list.append(loss.item())
            acc_avg = (1 - torch.abs(outputs.cpu() - labels.cpu())).mean().clip(min=0)
            acc_avg = acc_avg.detach().numpy()
            acc_avg_list.append(acc_avg)

            # print loss and training info for an interval
            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1:
                remain_iter = epo_iter_num - i
                remain_epo = self.cfg.MAX_EPOCH - epoch_idx
                eta = (epo_iter_num * iter_time) * remain_epo + (remain_iter * iter_time)
                eta = int(eta)
                eta_string = f"{eta // 3600}h:{eta % 3600 // 60}m:{eta % 60}s"
                self.logger.info(
                    "Train: Epo[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] IterTime:[{:.2f}s] LOSS: {:.4f} ACC:{:.4f} ETA:{} ".format(
                        epoch_idx + 1, self.cfg.MAX_EPOCH,   # Epo
                        i + 1, epo_iter_num,                     # Iter  
                        iter_time,                      # IterTime
                        float(loss.item()), float(acc_avg),  # LOSS ACC ETA
                        eta_string,    
                    )
                )
        # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] 训练结束, data_loader里的元素个数为: final_i=', final_i)
        self.clt.record_train_loss(loss_list)
        self.clt.record_train_acc(acc_avg_list)

    def valid(self, data_loader, model, loss_f, epoch_idx):
        model.eval()
        with torch.no_grad():
            loss_batch_list = []
            acc_batch_list = []
            ocean_acc_epoch = []
            for i, data in enumerate(data_loader):
                # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i)
                # if i == 0:
                    # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i, '  data.keys()=', data.keys()) # data.keys()= dict_keys(['image', 'audio', 'label'])
                    # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i,  'data[image]的size: ', data['image'].size()) # torch.Size([4, 3, 224, 224]) 4对应config里的BATCH_SIZE
                    # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i,  'data[audio]的size: ', data['audio'].size()) # torch.Size([4, 1, 1, 50176]) 4对应config里的BATCH_SIZE
                    # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i,  'data[label]的size: ', data['label'].size(),'  data[label]=', data['label']) # torch.Size([4, 5]) 4对应config里的BATCH_SIZE
                inputs, labels = self.data_fmt(data)
                outputs = model(*inputs)
                loss = loss_f(outputs.cpu(), labels.cpu())
                loss_batch_list.append(loss.item())
                ocean_acc_batch = (1 - torch.abs(outputs.cpu().detach() - labels.cpu().detach())).mean(dim=0).clip(min=0)
                ocean_acc_epoch.append(ocean_acc_batch)
                acc_batch_avg = ocean_acc_batch.mean()
                acc_batch_list.append(acc_batch_avg)
            ocean_acc = torch.stack(ocean_acc_epoch, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_acc_avg = ocean_acc.mean()
            self.tb_writer.add_scalar("valid_acc", ocean_acc_avg, epoch_idx)
        self.clt.record_valid_loss(loss_batch_list)
        self.clt.record_valid_acc(acc_batch_list)  # acc over batches
        self.clt.record_valid_ocean_acc(ocean_acc)
        if ocean_acc_avg > self.clt.best_valid_acc:
            self.clt.update_best_acc(ocean_acc_avg)
            self.clt.update_model_save_flag(1)
        else:
            self.clt.update_model_save_flag(0)

        self.logger.info(
            "Valid: Epoch[{:0>3}/{:0>3}] Train Mean_Acc: {:.2%} Valid Mean_Acc:{:.2%} OCEAN_ACC:{}\n".
            format(
                epoch_idx + 1, self.cfg.MAX_EPOCH,
                float(self.clt.epoch_train_acc),
                float(self.clt.epoch_valid_acc),
                self.clt.valid_ocean_acc)
        )

    def test(self, data_loader, model):
        mse_func = torch.nn.MSELoss(reduction="none")
        model.eval()
        with torch.no_grad():
            mse_ls = []
            ocean_acc = []
            label_list = []
            output_list = []
            for data in tqdm(data_loader):
                inputs, labels = self.data_fmt(data)
                outputs = model(*inputs)

                outputs = outputs.cpu().detach()
                labels = labels.cpu().detach()
                output_list.append(outputs)
                label_list.append(labels)
                mse = mse_func(outputs, labels).mean(dim=0)
                ocean_acc_batch = (1 - torch.abs(outputs - labels)).mean(dim=0).clip(min=0)
                mse_ls.append(mse)
                ocean_acc.append(ocean_acc_batch)
            ocean_mse = torch.stack(mse_ls, dim=0).mean(dim=0).numpy()
            ocean_acc = torch.stack(ocean_acc, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_mse_mean = ocean_mse.mean()
            ocean_acc_avg = ocean_acc.mean()
            dataset_output = torch.cat(output_list, dim=0).numpy()
            dataset_label = torch.cat(label_list, dim=0).numpy()
        ocean_mse_mean_rand = np.round(ocean_mse_mean, 4)
        ocean_acc_avg_rand = np.round(ocean_acc_avg.astype("float64"), 4)
        self.tb_writer.add_scalar("test_acc", ocean_acc_avg_rand)
        keys = ["O", "C", "E", "A", "N"]
        ocean_mse_dict, ocean_acc_dict = {}, {}
        for i, k in enumerate(keys):
            ocean_mse_dict[k] = np.round(ocean_mse[i], 4)
            ocean_acc_dict[k] = np.round(ocean_acc[i], 4)
        # ocean_acc_avg_rand 遍历data_loader时每次迭代得到的预测准确率，然后计算得到平均值
        # ocean_acc_dict 遍历data_loader时每次迭代得到的预测准确率所组成的字典
        # dataset_output 模型具体的预测输出结果
        # dataset_label 标签
        # ocean_mse_dict 遍历data_loader时得到的各个均方误差值所组成的字典
        # ocean_mse_mean_rand 遍历data_loader时得到的均方误差，然后计算得到平均值
        return ocean_acc_avg_rand, ocean_acc_dict, dataset_output, dataset_label, (ocean_mse_dict, ocean_mse_mean_rand)

    def full_test(self, data_set, model):
        model.eval()
        out_ls, label_ls = [], []
        with torch.no_grad():
            for data in tqdm(data_set):
                inputs, label = self.full_test_data_fmt(data)
                out = model(*inputs)
                out_ls.append(out.mean(0).cpu().detach())
                label_ls.append(label)
        all_out = torch.stack(out_ls, 0)
        all_label = torch.stack(label_ls, 0)
        ocean_acc = (1 - torch.abs(all_out - all_label)).mean(0).numpy()
        ocean_acc_avg = ocean_acc.mean(0)

        ocean_acc_avg_rand = np.round(ocean_acc_avg, 4)
        ocean_acc_dict = {k: np.round(ocean_acc[i], 4) for i, k in enumerate(["O", "C", "E", "A", "N"])}

        dataset_output = all_out.numpy()
        dataset_label = all_label.numpy()

        return ocean_acc_avg_rand, ocean_acc_dict, dataset_output, dataset_label

    def data_extract(self, model, data_set, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for idx, data in enumerate(tqdm(data_set)):
                inputs, label = self.full_test_data_fmt(data)
                # mini_batch = 64
                out_ls, feat_ls = [], []
                for i in range(math.ceil(len(inputs[0]) / 64)):
                    mini_batch_1 = inputs[0][(i * 64): (i + 1) * 64]
                    mini_batch = (mini_batch_1,)
                    try:
                        mini_batch_2 = inputs[1][(i * 64): (i + 1) * 64]
                        mini_batch = (mini_batch_1, mini_batch_2)
                    except IndexError:
                        pass

                    # mini_batch = (mini_batch_1, mini_batch_2)
                    if model.return_feature:
                        out, feat = model(*mini_batch)
                        out_ls.append(out.cpu())
                        feat_ls.append(feat.cpu())
                    else:
                        out = model(*mini_batch)
                        out_ls.append(out.cpu())
                        feat_ls.append(torch.tensor([0]))
                out_pred, out_feat = torch.cat(out_ls, dim=0), torch.cat(feat_ls, dim=0)
                video_extract = {
                    "video_frames_pred": out_pred,
                    "video_frames_feat": out_feat,
                    "video_label": label.cpu()
                }
                save_to_file = os.path.join(output_dir, "{:04d}.pkl".format(idx))
                torch.save(video_extract, save_to_file)

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        img_in, aud_in, labels = data["image"], data["audio"], data["label"]
        return (aud_in, img_in), labels

    def full_test_data_fmt(self, data):
        images, wav, label = data["image"], data["audio"], data["label"]
        images_in = torch.stack(images, 0).to(self.device)
        # wav_in = torch.stack([wav] * 100, 0).to(self.device)
        wav_in = wav.repeat(len(images), 1, 1, 1).to(self.device)
        return (wav_in, images_in), label


@TRAINER_REGISTRY.register()
class BiModalTrainerUdiva(object):
    """trainer for Udiva bi-modal input"""
    def __init__(self, cfg, collector, logger):
        self.cfg = cfg.TRAIN
        self.cfg_model = cfg.MODEL
        self.loss_name = cfg.LOSS.NAME
        self.dataset_name = cfg.DATA_LOADER.DATASET_NAME
        self.non_blocking = cfg.DATA_LOADER.NON_BLOCKING
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clt = collector
        self.logger = logger
        # 在 cfg.OUTPUT_DIR 路径后加上 /tensorboard_events
        # tb_writer_dir = os.path.join(self.cfg.OUTPUT_DIR, "tensorboard_events")
        # self.tb_writer = SummaryWriter(tb_writer_dir)
        self.f1_metric = BinaryF1Score()
        self.best_valid_loss = 1e10
        self.LOG_INTERVAL_VALID = 10

    def train(self, data_loader, model, loss_f, optimizer, epoch_idx, scaler=None):
        ''' data_loader的调用逻辑如下
        # data_loader: 上一层级调用该函数时传入了: self.data_loader["train"], self.data_loader = self.build_dataloader(),  build_dataloader 函数返回的是 data_loader_dicts，
        然后：
        data_loader_dicts = {
            "train": dataloader(cfg, mode="train"),
            "valid": dataloader(cfg, mode="valid"),
            "test": dataloader(cfg, mode="test"),
        }
        然后:
        dataloader = DATA_LOADER_REGISTRY.get(name) #'NAME': 'bimodal_resnet_data_loader', 在 dpcv/data/datasets/audio_visual_data.py里有 def bimodal_resnet_data_loader(cfg, mode) 函数定义
        然后:
        data_loader = DataLoader(  # torch.utils.data.DataLoader, Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
            dataset=dataset, 
            batch_size=batch_size,  # batch_size = cfg.DATA_LOADER.TRAIN_BATCH_SIZE = 8
            shuffle=cfg.DATA_LOADER.SHUFFLE, # cfg.DATA_LOADER.SHUFFLE = True
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,  # 'NUM_WORKERS': 0,
            drop_last=cfg.DATA_LOADER.DROP_LAST, # 'DROP_LAST': True,
        )
        # torch.utils.data.DataLoader 
        其中:
        dataset = AudioVisualData(
            cfg.DATA.ROOT,
            cfg.DATA.TRAIN_IMG_DATA, # 'TRAIN_IMG_DATA': 'ChaLearn2016_tiny/train_data',
            cfg.DATA.TRAIN_AUD_DATA, # 'TRAIN_AUD_DATA': 'ChaLearn2016_tiny/voice_data/voice_librosa/train_data',
            cfg.DATA.TRAIN_LABEL_DATA, # 'TRAIN_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_training.pkl',
            transforms
        )
        然后:
        AudioVisualData 类里有个函数:def __getitem__(self, idx). 该函数返回的是: sample = {"image": img, "audio": wav, "label": label}, 其中: img, wav, label 都是 torch.Tensor 类型, img.shape() =  torch.Size([3, 224, 224]) wav.shape() =  torch.Size([1, 1, 50176]) label.shape() =  torch.Size([5])
        '''
        lr = optimizer.param_groups[0]['lr']
        self.logger.info(f"Training: learning rate:{lr}")
        optimizer.zero_grad() # 梯度清零，即将梯度变为0
        model.train()
        loss_list = []
        batch_loss_sum = 0
        acc_avg_list = []
        epoch_total_acc = 0
        epoch_total_num = 0
        epo_iter_num = len(data_loader)
        print('Train epo_iter_num = ', epo_iter_num)
        f1, f1_2 = -1, -1
        pred_list, label_list = [], []
        for i, data in enumerate(data_loader): # i代表第几个batch, data代表第i个batch的数据 # 通过看日志，当执行下面这行 for i, data in enumerate(data_loader)语句时，会调用 AudioVisualData(VideoData)类里的 __getitem__ 函数，紧接着调用def get_ocean_label()函数， 具体原因参考：https://www.geeksforgeeks.org/how-to-use-a-dataloader-in-pytorch/
            iter_start_time = time.time()
            iter_start_time_print = time.strftime("%H:%M:%S", time.localtime())
            # print('[bi_modal_trainer.py] train... type(data)=', type(data), ', len(data)=', len(data)) # type(data)= <class 'list'> , len(data)= 2
            # print('[bi_modal_trainer.py] train... data[0].shape=',  data[0].shape, ', data[1].shape=', data[1].shape, ' type(data[0]):', type(data[0]), ', type(data[1]):', type(data[1])) # type(data[0]): <class 'torch.Tensor'> , type(data[1]): <class 'torch.Tensor'>
            """ 1、如果data_loader中使用了RandomOverSampler，那么这里得到的data就是一个list，list里有两个元素，分别是image和audio，image的shape:[batch_size, sample_size, c, h, w] e.g.[8, 16, 6, 224, 224]  label.shape: [batch_size, 2]
                2、如果没有使用RandomOverSampler，那么这里得到的data就是一个dict，dict里有image audio label 这几个key，分别对应image,audio,label的数据 """
            inputs, labels, session_id, segment_id, is_continue = self.data_fmt(data) # self.data_fmt(data) 代表将data里的image, audio, label分别取出来，放到inputs，label里
            # print('Train: is_continue:', is_continue)
            # if torch.any(is_continue == False):
            #     print('Train: part of data is not continuous, continue to next batch, i:', i)
            #     continue
            '''
            # print('[bi_modal_trainer.py] train... labels.shape=', labels.shape, ', session_id.shape=', session_id.shape, ', segment_id.shape=', segment_id.shape, ', type(session_id)=', type(session_id), ', type(segment_id)=', type(segment_id))
            # print('[bi_modal_trainer.py] train... model.device=', model.device, 'inputs[0].device=', inputs[0].device)
            # print('[bi_modal_trainer.py] train... inputs[0].shape=', inputs[0].shape, ', inputs[1]=', inputs[1], ', labels.shape=', labels.shape)
            # if i in [0, 1] and torch.cuda.is_available():
            #     print('before model, i:', i, ', CUDA memory_summary:\n', torch.cuda.memory_summary())
            
            # try: # refer: https://zhuanlan.zhihu.com/p/497192910
            #     # inputs加一个*星号：表示参数数量不确定，将传入的参数存储为元组（https://blog.csdn.net/qq_42951560/article/details/112006482）。*inputs意思是将inputs里的元素分别取出来，作为model的输入参数，这里的inputs是一个元组，包含了image和audio。models里的forward函数里的参数是image和audio，所以这里的*inputs就是将image和audio分别取出来，作为model的输入参数。为什么是forward函数的参数而不是__init__函数的参数？因为forward函数是在__init__函数里被调用的，所以forward函数的参数就是__init__函数的参数。forward 会自动被调用，调用时会传入输入数据，所以forward函数的参数就是输入数据。
            #     outputs = model(*inputs)
            # except RuntimeError as exception:
            #     if "out of memory" in str(exception):
            #         print(exception)
            #         print('Train WARNING: out of memory')
            #         if hasattr(torch.cuda, 'empty_cache'):
            #             torch.cuda.empty_cache()
            #             if i in [0, 1] and torch.cuda.is_available():
            #                 print('after model, i:', i, ', CUDA memory_summary:\n', torch.cuda.memory_summary())
            #         else:
            #             raise exception
            '''
            if self.cfg_model.NUM_CLASS == 2: 
                temp_labels = labels.float()
            else:
                temp_labels = labels.argmax(dim=-1)
            if self.cfg.USE_AMP: # use AMP:automatic mixed precision
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(*inputs)
                        outputs = outputs.float()
                        if torch.isnan(outputs).any():
                            continue
                        # print('[bi_modal_trainer.py] train... outputs=', outputs, 'labels=', temp_labels, ' outputs.size()', outputs.size(),  '  labels.size()=', labels.size())
                        loss = loss_f(outputs.cpu(), temp_labels.cpu())
                else:
                    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                        outputs = model(*inputs)
                        outputs = outputs.float()
                        loss = loss_f(outputs.cpu(), temp_labels.cpu())
            else:
                outputs = model(*inputs) # print('[bi_modal_trainer.py] train... outputs=', outputs, 'labels=', labels, ' outputs.size()', outputs.size(),  '  labels.size()=', labels.size())
                # print('[bi_modal_trainer.py] train... outputs=', outputs, 'labels=', labels, ' outputs.size()', outputs.size(),  '  labels.size()=', labels.size())
                if self.cfg.USE_HALF and torch.cuda.is_available(): # use half precision
                    outputs = outputs.float() # avoid RuntimeError: "binary_cross_entropy" not implemented for 'Half'
                loss = loss_f(outputs.cpu(), temp_labels.cpu())
            del temp_labels
            outputs = outputs.detach().cpu()
            labels = labels.detach().cpu()
            if epoch_idx % 10 == 0 and i in [0]: # epoch_idx % 10 == 0 代表每10个epoch打印一次
                # if self.loss_name == "binary_cross_entropy":
                #     print_labels = labels.argmax(dim=-1).unsqueeze(1) # purpose: match shape for printing cat tensor
                # else:
                #     print_labels = labels.unsqueeze(1)
                # print_labels = labels.argmax(dim=-1).unsqueeze(1) # purpose: match shape for printing cat tensor
                if session_id is not None and segment_id is not None:
                    session_id = session_id.detach().cpu()
                    segment_id = segment_id.detach().cpu()
                    print(f'{torch.cat([outputs, labels.argmax(dim=-1).unsqueeze(1), session_id.unsqueeze(1), segment_id.unsqueeze(1)], dim=1)}, *********** Epo[{(epoch_idx+1):0>3}/{self.cfg.MAX_EPOCH:0>3}] Iter[{(i + 1):0>3}/{epo_iter_num:0>3}], train loss:{loss.item():.4f} TIME:{time.strftime("%H:%M:%S", time.localtime())}***********')
                else:
                    print(f'{torch.cat([outputs, labels.argmax(dim=-1).unsqueeze(1)], dim=1)}, *********** Epo[{(epoch_idx+1):0>3}/{self.cfg.MAX_EPOCH:0>3}] Iter[{(i + 1):0>3}/{epo_iter_num:0>3}], train loss:{loss.item():.4f} TIME:{time.strftime("%H:%M:%S", time.localtime())}***********')
                
            '''
            # batch_size = 3时
            # tensor([[0.4946, 0.5024],
            #         [0.4945, 0.5026],
            #         [0.4944, 0.5025]], grad_fn=<MeanBackward1>)
            # -----
            # tensor([[0., 1.],
            #         [0., 1.],
            #         [1., 0.]])
            # loss= tensor(0.2487, grad_fn=<MseLossBackward0>)
            
            # tensor([[0.2553, 0.7423],
            #         [0.2523, 0.7545],
            #         [0.2578, 0.7498]], grad_fn=<MeanBackward1>)
            # -----
            # tensor([[0., 1.],
            #         [1., 0.],
            #         [0., 1.]])
            # loss= tensor(0.2315, grad_fn=<MseLossBackward0>) 
            
            # batch_size = 1时
            # tensor([[0.5033, 0.4948]], grad_fn=<MeanBackward1>)
            # -----
            # tensor([[0., 1.]])
            # loss= tensor(0.2542, grad_fn=<MseLossBackward0>) 计算过程: 0.5033-0=0.5033, 0.4948-1=-0.5052, 0.5033^2=0.25331089, -0.5052^2=0.25522704, 0.25331089+0.25522704=0.50853793, 0.50853793/2=0.254268965
                        
            # tensor([[0.3009, 0.6966]], grad_fn=<MeanBackward1>)
            # -----
            # tensor([[1., 0.]])
            # loss= tensor(0.4870, grad_fn=<MseLossBackward0>)
            
            # 使用BCELoss时，batch_size = 3时
            # tensor([[0.5133, 0.4925],
            #         [0.5132, 0.4929],
            #         [0.5132, 0.4926]], grad_fn=<MeanBackward1>)
            # -----
            # tensor([[0., 1.],
            #         [1., 0.],
            #         [1., 0.]])
            # outputs.size()= torch.Size([3, 2]) , labels.size()= torch.Size([3, 2])
            # loss= tensor(0.6867, grad_fn=<BinaryCrossEntropyBackward0>)
            
            # 使用BCELoss时，batch_size = 1时
            # tensor([[0.4929, 0.4921]], grad_fn=<MeanBackward1>)
            # -----
            # tensor([[0., 1.]])
            # outputs.size()= torch.Size([1, 2]) , labels.size()= torch.Size([1, 2])
            # loss= tensor(0.6941, grad_fn=<BinaryCrossEntropyBackward0>) # 计算过程: 0.4929*log(0.4929)+(1-0.4929)*log(1-0.4929)=0.6941
            '''
            if self.cfg.USE_AMP and scaler is not None:
                scaler.scale(loss).backward() # 为了梯度放大. Multiplies (‘scales’) a tensor or list of tensors by the scale factor.
                scaler.step(optimizer) # 首先把梯度的值unscale回来. 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重, 否则，忽略step调用，从而保证权重不更新（不被破坏）
                scaler.update() # Updates the scale factor. 如果没有出现inf或NaN,那么权重正常更新，并且当连续多次(growth_interval指定)没有出现inf或NaN，则scaler.update()会将scaler的大小增加(乘上growth_factor)
            else:
                loss.backward()
                optimizer.step()
            
            iter_end_time = time.time()
            iter_end_time_print = time.strftime("%H:%M:%S", time.localtime())
            iter_time = iter_end_time - iter_start_time

            loss_list.append(loss.item())
            batch_loss_sum += loss.item()
            # acc_avg = (1 - torch.abs(outputs.cpu() - labels.cpu())).mean().clip(min=0) # acc_avg的公式里用1来减去是因为 we use 1 - |y_pred - y_true| as the accuracy metric, beacause we want to penalize the model more when the prediction is far from the ground truth. # torch.abs 计算 tensor 的每个元素的绝对值, torch.abs(outputs.cpu() - labels.cpu())是预测值与真实值的差的绝对值, 即预测值与真实值的差的绝对值的平均值即为acc_avg, 即acc_avg = (1 - torch.abs(outputs.cpu() - labels.cpu())).mean(), clip(min=0)是将acc_avg的最小值限制在0以上
            ''' 
            acc_avg计算示例(batch_size=1): acc_avg = (1 - torch.abs(outputs.cpu() - labels.cpu())).mean().clip(min=0)
                1. outputs: [[0.4764987  0.52350134]] labels: [[1. 0.]] acc_avg: 0.4764987,
                    abs(outputs-labels): [[0.5235013  0.52350134]]  1-abs: [[0.47649872 0.47649866]], 平均值:0.47649872+0.47649866=0.9529974/2=0.4764987
                2. outputs: [[0.73084384 0.26915607]] labels: [[0. 1.]] acc_avg: 0.26915613,
                    abs(outputs-labels): [[0.73084384 0.7308439 ]]  1-abs: [[0.26915616 0.2691561 ]], 平均值:0.26915616+0.2691561=0.53831226/2=0.26915613
                3. outputs: [[0.22885753 0.7711425 ]] labels: [[0. 1.]] acc_avg: 0.7711425,
                    abs(outputs-labels): [[0.22885753 0.22885752]]  1-abs: [[0.7711425 0.7711425]]
            
            acc_avg计算示例(batch_size=2):
                1. 
                outputs: [[0.31670436 0.68329555]
                          [0.3105736  0.6894264 ]]
                labels: [[0. 1.] [0. 1.]] 
                acc_avg: 0.686361 , 
                abs(outputs-labels): [[0.31670436 0.31670445]
                                      [0.3105736  0.31057358]]  
                1-abs:  [[0.6832956  0.68329555]
                        [0.6894264  0.6894264 ]] # acc平均值: 0.6832956+0.68329555+0.6894264+0.6894264/4=2.74544395/4=0.6863609875
                2. 
                outputs: [[0.29868743 0.70131254]
                        [0.275699   0.72430104]] 
                labels: [[1. 0.]
                        [1. 0.]] 
                acc_avg: 0.28719324 , 
                abs(outputs-labels): [[0.70131254 0.70131254]
                                      [0.724301   0.72430104]]  
                1-abs: [[0.29868746 0.29868746]
                        [0.27569902 0.27569896]] # acc平均值: 0.29868746+0.29868746+0.27569902+0.27569896/4=1.1487729/4=0.287193225

                3. 
                outputs: [[0.39302874 0.60697126]
                        [0.34609628 0.6539037 ]] 
                labels: [[1. 0.]
                        [0. 1.]] 
                acc_avg: 0.5234662 , 
                abs(outputs-labels): [[0.60697126 0.60697126]
                                      [0.34609628 0.34609628]]  
                1-abs: [[0.39302874 0.39302874]
                        [0.6539037  0.6539037 ]] # acc平均值: 0.39302874+0.39302874+0.6539037+0.6539037/4=2.09386488/4=0.52346622
            '''
            #### 计算指标：acc
            # print('before acc, outputs.shape:', outputs.shape, 'labels.shape:', labels.shape, outputs.argmax(dim=-1), labels.argmax(dim=-1))
            batch_total_acc = (outputs.argmax(dim=-1) == labels.argmax(dim=-1)).sum() # 当前batch里分类预测正确的样本数
            batch_total_num = outputs.shape[0] # 当前batch的样本总数
            batch_acc = batch_total_acc / batch_total_num # 当前batch的acc
            epoch_total_acc += batch_total_acc
            epoch_total_num += batch_total_num
            epoch_current_acc = epoch_total_acc / epoch_total_num
            
            acc_avg = batch_acc.numpy()
            acc_avg_list.append(acc_avg)
            acc_time = time.strftime("%H:%M:%S", time.localtime())
            
            outputs = torch.round(outputs * 1000000) / 1000000 # 小数点后保留6位
            labels = labels.type(torch.int64)
            
            # if self.loss_name == "binary_cross_entropy":
                # labels = labels.argmax(dim=-1)
            # print('outputs:', outputs, outputs.shape, '\nlabels:', labels, labels.shape)
            # print('outputs[:, 0]:', outputs[:, 0], ', outputs.argmax(dim=-1)=', outputs.argmax(dim=-1)) # [:, 0] 表示在张量的第一个维度上选择所有的元素，并在第二个维度上选择索引为 0 的元素。即选取张量的所有行（第一个维度）和第一列（第二个维度）的元素。
            # print('labels[:, 0]:', labels[:, 0], 'labels.argmax(dim=-1):', labels.argmax(dim=-1))
            # pred_list.extend(outputs[:, 0].tolist())
            
            # pred_list.extend(outputs.argmax(dim=-1).tolist())
            # pred_list2.extend(outputs.tolist()) # pred_list2.extend(outputs.argmax(dim=-1).tolist())
            # if labels.dim() == 1:
            #     label_list.extend(labels.tolist())
            #     label_list2=label_list
            # elif labels.dim() == 2:
            #     label_list.extend(labels[:, 0].tolist())
            #     label_list2.extend(labels.argmax(dim=-1).tolist())
            # label_list.extend(labels.tolist())
            
            # print('[bi_modal_trainer.py] before AUC, outputs=', outputs, 'labels=', labels, ' outputs.size()=', outputs.size(),  'labels.size()=', labels.size())
            #### 计算指标：AUC
            if self.cfg_model.NUM_CLASS == 2: # Udiva 二分类，因此task="binary"
                # batch_auc = auroc(outputs[:, 0], labels[:, 0], task="binary")
                # batch_auc2 = auroc(outputs.argmax(dim=-1).type(torch.float32), labels.argmax(dim=-1), task="binary")
                batch_auc = auroc(outputs.argmax(dim=-1).type(torch.float32), labels.argmax(dim=-1), task="binary")
                pred_list.extend(outputs.argmax(dim=-1).tolist())
                label_list.extend(labels.argmax(dim=-1).tolist())
            elif self.cfg_model.NUM_CLASS == 4: # NoXi 四分类，因此task="multiclass"
                # batch_auc = auroc(outputs[:, 0], labels[:, 0], task="multiclass")
                batch_auc = auroc(outputs.type(torch.float32), labels.argmax(dim=-1), task='multiclass', num_classes=4)
                pred_list.extend(outputs.tolist())
                label_list.extend(labels.argmax(dim=-1).tolist())
            auc_time = time.strftime("%H:%M:%S", time.localtime())
            #### 计算指标：F1
            # f1 = self.f1_metric(outputs[:, 0], labels[:, 0]) # 最好用这个，详见：https://www.notion.so/MPhil-Project-b3de240fa9d64832b26795439d0142d9?pvs=4#3820c783cbc947cb83c1c5dd0d91434b
            # f1_2 = self.f1_metric(outputs.argmax(dim=-1), labels.argmax(dim=-1))

            if self.cfg.USE_WANDB:
                wandb.log({
                    "train_loss":  float(loss.item()),
                    "train_acc": float(acc_avg), # 当前batch的acc
                    "train_epoch_current_acc": float(epoch_current_acc), # 当前epoch的持续记录的acc
                    "train_batch_auc": float(batch_auc), # 当前batch的auc
                    # "train_batch_auc2": float(batch_auc2),
                    "train_batch_f1": float(f1), # 当前batch的f1
                    "train_batch_f1_2": float(f1_2), # 当前batch的f1
                    "learning rate": lr,
                    "epoch": epoch_idx + 1,
                })
            wandb_time = time.strftime("%H:%M:%S", time.localtime())

            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1: #  self.cfg.LOG_INTERVAL = 10，即每10个batch打印一次loss和acc_avg
                remain_iter = epo_iter_num - i # epo_iter_num(一个epoch里的batch数)-当前的batch数=剩余的batch数
                remain_epo = self.cfg.MAX_EPOCH - epoch_idx 
                eta = (epo_iter_num * iter_time) * remain_epo + (remain_iter * iter_time) # eta是剩余的时间，即剩余的epoch数乘以一个epoch的时间加上剩余的batch数乘以一个batch的时间
                eta = int(eta) # 将eta转换成int类型
                eta_string = f"{eta // 3600}h:{eta % 3600 // 60}m:{eta % 60}s"  # 将eta转换成时分秒的形式
                self.logger.info(
                    "Train: Epo[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] IterTime:[{:.2f}s] Batch Loss:{:.4f} Acc:{:.4f} ({}/{}) Epo current ACC:{:.4f} ({}/{}) AUC:{:.4f} F1:{:.4f} ({:.4f}) ETA:{} TIME:{}".format(
                        epoch_idx + 1, self.cfg.MAX_EPOCH,   # Epo 
                        i + 1, epo_iter_num,                 # Iter
                        iter_time,                           # IterTime
                        float(loss.item()),                  # LOSS
                        float(acc_avg), batch_total_acc, batch_total_num,    # Batch ACC
                        epoch_current_acc, epoch_total_acc, epoch_total_num, # Epo current ACC
                        batch_auc,                           # AUC
                        f1, f1_2,                            # F1
                        eta_string,                          # ETA
                        time.strftime("%H:%M:%S", time.localtime()), # Current TIME
                    ))
            log_time = time.strftime("%H:%M:%S", time.localtime())
            if i % 100 == 99: # 每100个batch打印一次
                print("Train: Iter[{:0>3}/{:0>3}] IterTime:[{:.2f}s] iter_start_time:{} iter_end_time:{} acc_time:{} auc_time:{} wandb_time:{} log_time:{}".format(
                    i + 1, epo_iter_num,
                    iter_time,
                    iter_start_time_print, iter_end_time_print,
                    acc_time, auc_time, 
                    wandb_time, log_time,
                    ))
        #### 计算指标：epoch loss
        epoch_summary_loss = batch_loss_sum / epo_iter_num
        # print('loss_list:', loss_list, ', len(loss_list):', len(loss_list), ', train_loss:', train_loss, ', batch_loss_sum:', batch_loss_sum, ', epo_iter_num:', epo_iter_num)
        print('Train: len(loss_list):', len(loss_list), ', epoch_summary_loss:', epoch_summary_loss, ', batch_loss_sum:', batch_loss_sum, ', epo_iter_num:', epo_iter_num)
        
        #### 计算指标：acc
        epoch_summary_acc = epoch_total_acc / epoch_total_num
        
        #### 计算指标：AUC
        # print('pred_list:', pred_list, 'label_list:', label_list, '\npred_list2:', pred_list2, 'label_list2:', label_list2, '\nlen(pred_list):', len(pred_list), ', len(label_list):', len(label_list), ', len(pred_list2):', len(pred_list2), ', len(label_list2):', len(label_list2))
        pred = torch.tensor(pred_list).type(torch.float32)
        label = torch.tensor(label_list)
        # print('pred.shape:', pred.shape, ', pred2.shape:', pred2.shape, ', label.shape:', label.shape)
        if self.cfg_model.NUM_CLASS == 2: # Udiva 二分类，因此task="binary"
            epoch_auc = auroc(pred, label, task="binary")
        elif self.cfg_model.NUM_CLASS > 2: # NoXi 四分类，因此task="multiclass"
            epoch_auc = auroc(pred, label, task="multiclass", num_classes=self.cfg_model.NUM_CLASS)
        else:
            raise Exception('Invalid NUM_CLASS')
        
        #### 计算指标：F1 score
        # f1 = self.f1_metric(torch.tensor(pred_list), torch.tensor(label_list))
        # f1_2 = self.f1_metric(torch.tensor(pred_list2), torch.tensor(label_list2))

        self.logger.info(
            "Train: Epo[{:0>3}/{:0>3}] Epo Summary Loss:{:.4f} ACC:{:.4f} ({}/{}) AUC:{:.4f} F1_Score: {:.4f} ({:.4f}) TIME:{}".
            format(
                epoch_idx + 1, self.cfg.MAX_EPOCH, # Epoch
                epoch_summary_loss, # Epo Loss
                epoch_summary_acc, epoch_total_acc, epoch_total_num, # Epo Summary Acc
                epoch_auc, # Epo AUC
                f1, f1_2, # Epo F1_Score
                time.strftime("%H:%M:%S", time.localtime()), # Current TIME
            ))
        
        if self.cfg.USE_WANDB:
            wandb.log({
                "train_epoch_summary_loss": float(epoch_summary_loss), # 记录当前epoch全部batch遍历完后的总体loss
                "train_epoch_summary_acc": float(epoch_summary_acc),  # 记录当前epoch全部batch遍历完后的总体acc
                "train_epoch_summary_auc": float(epoch_auc), # auc
                "train_epoch_summary_f1_score": float(f1), # 比f1_2更准确的f1
                "train_epoch_summary_f1_score2": float(f1_2),
                "epoch": epoch_idx + 1})
        
        self.clt.record_train_loss(loss_list) # 将loss_list里的loss值记录到self.clt里
        self.clt.record_train_acc(acc_avg_list) # 将acc_avg_list里的acc_avg值记录到self.clt里

    def valid(self, data_loader, model, loss_f, scheduler, epoch_idx):
        model.eval()
        with torch.no_grad():
            loss_batch_list = []
            batch_loss_sum = 0
            acc_batch_list = []
            ocean_acc_epoch = []
            epoch_total_acc = 0
            epoch_total_num = 0
            epo_iter_num = len(data_loader)
            pred_list, label_list, pred_list2 = [], [], []
            f1, f1_2= -1, -1
            for i, data in enumerate(data_loader):
                inputs, labels, session_id, segment_id, is_continue = self.data_fmt(data) # labels:[batch_size, 2], session_id:[batch_size], segment_id:[batch_size] e.g. session_id: tensor([1080, ... 1080]) segment_id: tensor([ 1,  2, 3, ... 32])
                # print('Valid: is_continue:', is_continue)
                # if torch.any(is_continue == False):
                #     print('Valid: part of data is not continuous, continue to next batch')
                #     continue
                outputs = model(*inputs)
                if self.cfg_model.NUM_CLASS == 2:
                    labels = labels.float()
                    loss = loss_f(outputs.cpu(), labels.cpu())
                else:
                    loss = loss_f(outputs.cpu(), labels.argmax(dim=-1).cpu())
                """ if self.cfg.USE_AMP:
                    if torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            outputs = model(*inputs)
                            outputs = outputs.float()
                            print('[valid] outputs:', outputs, ', labels:', labels) # 出现了nan
                            loss = loss_f(outputs.cpu(), labels.cpu().float())
                    else:
                        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                            outputs = model(*inputs)
                            outputs = outputs.float()
                            loss = loss_f(outputs.cpu(), labels.cpu().float())
                else:
                    outputs = model(*inputs) 
                    if self.cfg.USE_HALF and torch.cuda.is_available():
                        outputs = outputs.float() # avoid RuntimeError: "binary_cross_entropy" not implemented for 'Half'
                    loss = loss_f(outputs.cpu(), labels.cpu().float()) """
                
                outputs = outputs.detach().cpu()
                labels = labels.detach().cpu()
                session_id = session_id.detach().cpu()
                segment_id = segment_id.detach().cpu()
                
                if epoch_idx % 10 == 0 and i in [0]:
                    # if self.loss_name == "binary_cross_entropy":
                    #     print_labels = labels.argmax(dim=-1).unsqueeze(1) # purpose: match shape for printing cat tensor
                    # else:
                    #     print_labels = labels.unsqueeze(1)
                    # print(f'{torch.cat([outputs, labels.argmax(dim=-1).unsqueeze(1), session_id.unsqueeze(1), segment_id.unsqueeze(1)], dim=1)}, *********** Epo[{(epoch_idx+1):0>3}/{self.cfg.MAX_EPOCH:0>3}] Iter[{(i + 1):0>3}/{epo_iter_num:0>3}], val loss:{loss.item():.4f} ***********')
                    print(f'{torch.cat([outputs, labels.argmax(dim=-1).unsqueeze(1), session_id.unsqueeze(1), segment_id.unsqueeze(1)], dim=1)}, *********** Epo[{(epoch_idx+1):0>3}/{self.cfg.MAX_EPOCH:0>3}] Iter[{(i + 1):0>3}/{epo_iter_num:0>3}], val loss:{loss.item():.4f} ***********')
                
                loss_batch_list.append(loss.item())
                batch_loss_sum += loss.item()
                
                #### 计算acc
                # print('before acc, outputs.shape:', outputs.shape, 'labels.shape:', labels.shape)
                batch_total_acc = (outputs.argmax(dim=-1) == labels.argmax(dim=-1)).sum() # 当前batch里分类预测正确的样本数
                batch_total_num = outputs.shape[0] # 当前batch的样本总数
                batch_acc = batch_total_acc / batch_total_num # 当前batch的acc
                epoch_total_acc += batch_total_acc
                epoch_total_num += batch_total_num
                epoch_current_acc = epoch_total_acc / epoch_total_num
                
                # if self.loss_name == "binary_cross_entropy":
                    # labels = labels.argmax(dim=-1)
                # labels = labels.argmax(dim=-1)
                
                #### 计算AUC
                labels = labels.type(torch.int64)
                # pred_list.extend(outputs.argmax(dim=-1).tolist())
                # pred_list2.extend(outputs.tolist()) # pred_list2.extend(outputs.argmax(dim=-1).tolist())
                # label_list.extend(labels[:, 0].tolist())
                # label_list2.extend(labels.argmax(dim=-1).tolist())
                label_list.extend(labels.argmax(dim=-1).tolist())
                # batch_auc = auroc(outputs[:, 0], labels[:, 0], task="binary")
                # batch_auc2 = auroc(outputs.argmax(dim=-1).type(torch.float32), labels.argmax(dim=-1), task="binary")
                if self.cfg_model.NUM_CLASS == 2:
                    pred_list.extend(outputs.argmax(dim=-1).tolist())
                else:
                    pred_list.extend(outputs.tolist())
                
                #### 计算指标：F1
                # f1 = self.f1_metric(outputs[:, 0], labels[:, 0]) # 最好用这个，详见：https://www.notion.so/MPhil-Project-b3de240fa9d64832b26795439d0142d9?pvs=4#3820c783cbc947cb83c1c5dd0d91434b
                # f1_2 = self.f1_metric(outputs.argmax(dim=-1), labels.argmax(dim=-1))  # 不准确
                
                if i % self.LOG_INTERVAL_VALID == self.LOG_INTERVAL_VALID - 1:
                    self.logger.info(
                        "Valid: Epo[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] Batch Loss: {:.4f} Batch Acc:{:.4f} ({}/{}) Epo Current Acc:{:.4f} ({}/{}) TIME:{}".format(
                            epoch_idx + 1, self.cfg.MAX_EPOCH,   # Epo
                            i + 1, epo_iter_num,                 # Iter
                            float(loss.item()),                  # LOSS
                            float(batch_acc), batch_total_acc, batch_total_num,  # Batch ACC
                            epoch_current_acc, epoch_total_acc, epoch_total_num, # Epo Current ACC
                            time.strftime("%H:%M:%S", time.localtime()), # Current TIME
                            # batch_auc, batch_auc2, # AUC  AUC:{:.4f} ({:.4f})  由于valid时shuffle=False，所以是按照segment从1递增顺序进行测试，每个batch里label的值要么都是1，或者都是0，导致batch的AUC一直=0，因此不计算batch里的AUC，只看整个epoch的AUC
                            # f1, f1_2, # F1 F1:{:.4f} ({:.4f}) 
                        ))
                if self.cfg.USE_WANDB:
                    wandb.log({
                        "valid_loss": float(loss.item()),
                        "valid_acc": float(batch_acc),
                        "valid_epoch_current_acc": float(epoch_current_acc),
                        # "valid_batch_auc": float(batch_auc),
                        # "valid_batch_auc2": float(batch_auc2),
                        # "valid_batch_f1": float(f1),
                        # "valid_batch_f1_2": float(f1_2),
                        "epoch": epoch_idx + 1,
                    })
                acc_batch_list.append(batch_acc)
            # self.tb_writer.add_scalar("valid_acc", batch_acc, epoch_idx)
        #### 计算指标：valid loss
        epoch_summary_loss = batch_loss_sum / epo_iter_num # 当前epoch的总体loss
        if epoch_summary_loss < self.best_valid_loss:
            # print(f'Valid: Current epoch summary loss:{epoch_summary_loss:.4f} < best epoch summary loss: {self.best_valid_loss:.4f}, will update best loss')
            self.best_valid_loss = round(epoch_summary_loss, 4) # 保存当前最佳的loss
        
        #### 计算指标：Epoch ACC (Video segment level)
        epoch_summary_acc = (epoch_total_acc / epoch_total_num).item() # 当前epoch的总体acc  type(epoch_summary_acc): <class 'torch.Tensor'> , type(epoch_total_acc): <class 'torch.Tensor'> , type(epoch_total_num): <class 'int'>
        epoch_summary_acc = round(epoch_summary_acc, 4) # type(epoch_summary_acc): <class 'float'>
        self.clt.record_valid_loss(loss_batch_list) # loss over batches
        self.clt.record_valid_acc(acc_batch_list)  # acc over batches
        if epoch_summary_acc > self.clt.best_valid_acc: # 如果当前的epoch_summary_acc大于之前的最好的epoch_summary_acc #TODO 0.6667 > 0.6667, need atol=1e-4
            print(f'Valid: Current epoch summary acc:{epoch_summary_acc:.4f} > best_valid_acc: {self.clt.best_valid_acc:.4f}, will save model, TIME:{time.strftime("%H:%M:%S", time.localtime())}') # type(self.clt.best_valid_acc): <class 'int'>
            self.clt.update_best_acc(epoch_summary_acc)
            self.clt.update_model_save_flag(1)   # 1表示需要保存模型
        else:
            # print(f'Valid: Current epoch summary acc:{epoch_summary_acc:.4f} <= best_valid_acc: {self.clt.best_valid_acc:.4f}, not save model')
            self.clt.update_model_save_flag(0)  # 0表示不需要保存模型
        
        #### 计算指标：AUC
        # print('Valid pred_list:', pred_list, 'label_list:', label_list, '\nValid pred_list2:', pred_list2, '\nValid label_list2:', label_list2, '\nValid len(pred_list):', len(pred_list), ', len(label_list):', len(label_list), ', len(pred_list2):', len(pred_list2), ', len(label_list2):', len(label_list2))
        pred = torch.tensor(pred_list).type(torch.float32)
        label = torch.tensor(label_list)
        if self.cfg_model.NUM_CLASS == 2: # Udiva 二分类，因此task="binary"
            epoch_auc = auroc(pred, label, task="binary")
        else: # NoXi 四分类，因此task="multiclass"
            epoch_auc = auroc(pred, label, task="multiclass", num_classes=self.cfg_model.NUM_CLASS)
        
        #### 计算指标：F1 score
        # f1 = self.f1_metric(torch.tensor(pred_list), torch.tensor(label_list))
        # f1_2 = self.f1_metric(torch.tensor(pred_list2), torch.tensor(label_list2))
        
        self.logger.info(
            "Valid: Epo[{:0>3}/{:0>3}] Val Epo Summary LOSS:{:.4f} ACC:{:.4f} ({}/{}) AUC:{:.4f} TIME:{}".
            format(
                epoch_idx + 1, self.cfg.MAX_EPOCH, # Epoch
                epoch_summary_loss,                # Epo Loss
                epoch_summary_acc, epoch_total_acc, epoch_total_num, # Epo Summary Acc
                epoch_auc, # Epo AUC
                # f1, f1_2, # Epo F1_Score
                time.strftime("%H:%M:%S", time.localtime()), # Current TIME
            ))
        if self.cfg.USE_WANDB:
            wandb.log({
                # "valid_acc_batch_avg": acc_batch_avg,
                # "valid_ocean_acc_avg": ocean_acc_avg,
                # "Train Mean_Acc": float(self.clt.epoch_train_acc),
                # "Valid Mean_Acc": float(self.clt.epoch_valid_acc),
                "val_epoch_summary_loss": float(epoch_summary_loss),
                "val_epoch_summary_acc": float(epoch_summary_acc),
                "val_epoch_summary_auc": float(epoch_auc),
                # "val_epoch_summary_f1": float(f1),
                # "val_epoch_summary_f1_2": float(f1_2),
                "epoch": epoch_idx + 1,
            })
        return self.best_valid_loss, self.clt.best_valid_acc

    def test(self, data_loader, model, epoch_idx=None):
        model.eval()
        with torch.no_grad(): # 不计算梯度 也不更新参数，因为是测试阶段，不需要更新参数，只需要计算loss，计算acc
            # label_list = [] # 用于存储真实标签label
            # output_list = [] # 用于存储模型的输出output
            epo_iter_num = len(data_loader)
            total_acc = 0
            total_num = 0
            test_acc = 0
            pred_list, label_list, pred_list2 = [], [], []
            f1, f1_2 = -1, -1
            session_result = {}
            # for data in tqdm(data_loader): # 遍历data_loader
            for i, data in enumerate(data_loader):
                inputs, labels, session_id, segment_id, is_continue = self.data_fmt(data) # labels: [batch_size, 2], session_id: torch.Size([batch_size]), segment_id: torch.Size([batch_size]), e.g. when bs=6, session_id: tensor([8105., 8105., 8105., 8105., 8105., 8105.]) , segment_id: tensor([1., 2., 3., 4., 5., 6.])
                # print('[bi_modal_trainer.test]. labels.shape=', labels.shape, ', session_id.shape=', session_id.shape, ', segment_id.shape=', segment_id.shape, ', type(session_id)=', type(session_id), ', type(segment_id)=', type(segment_id), ', session_id:', session_id, ', segment_id:', segment_id)
                # print('Test: is_continue:', is_continue)
                
                outputs = model(*inputs)
                # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainerUdiva.test() 正在test... outputs=', outputs, 'labels=', labels, ' inputs[0].size()=', inputs[0].size(), ' inputs[1].size()=', inputs[1].size())
                outputs = outputs.detach().cpu()
                labels = labels.detach().cpu()
                session_id = session_id.detach().cpu()
                segment_id = segment_id.detach().cpu()
                
                if epoch_idx is not None:
                    if epoch_idx % 10 == 0 and i in [0]:
                        if self.loss_name == "binary_cross_entropy":
                            print_labels = labels.argmax(dim=-1).unsqueeze(1) # purpose: match shape for printing cat tensor
                        else:
                            print_labels = labels.unsqueeze(1)
                        print(f'\n{torch.cat([outputs, print_labels, session_id.unsqueeze(1), segment_id.unsqueeze(1)], dim=1)}, ***********  Epo[{(epoch_idx+1):0>3}/{self.cfg.MAX_EPOCH:0>3}] Iter[{(i + 1):0>3}/{epo_iter_num:0>3}] [Test] outputs,labels,session_id,segment_id ***********')
            
                #### 计算session ACC，即每个session层面的ACC，分母是测试集中session的个数，分子是预测正确的session的个数
                for idx in range(len(session_id)):
                    # 如果当前循环的session_id已经是session_result中的key，则将当前循环的outputs和segment_id拼接到session_result[session_id]["predict"]中，且拼接时在列上拼接 并且将该session_id对应的唯一一个label存入session_result[session_id]["label"]中，且只在第一次存入
                    # 否则将当前循环的outputs和segment_id作为value，session_id作为key，存入session_result中，且将该session_id对应的唯一一个label存入session_result[session_id]["label"]中，且只在第一次存入
                    new_predict = torch.cat([outputs[idx], segment_id[idx].unsqueeze(0)], dim=0).unsqueeze(0)
                    # print('new_predict=', new_predict, ', shape=', new_predict.shape)
                    if session_id[idx].item() in session_result:
                        session_result[session_id[idx].item()]["predict"] =  torch.cat([session_result[session_id[idx].item()]["predict"], new_predict], dim=0) # 拼接时：如果原先的向量为[6,2], 拼接的向量为[3,2], 则拼接后的向量为[9,2], 即在第0维上拼接，dim=0
                        # print('idx:', idx, ', after concat predict shape=', session_result[session_id[idx].item()]["predict"].shape)
                        if "label" not in session_result[session_id[idx].item()]:
                            session_result[session_id[idx].item()]["label"] = labels[idx]
                    else:
                        session_result[session_id[idx].item()] = {"predict": new_predict, "label": labels[idx]}
                # print('session_result=\n', session_result)
                #### 计算batch ACC, 即每个视频片段segment层面的ACC计算，分母是测试集中视频片段的个数(session个数 x 每个session的segments个数)，分子是预测正确的视频片段个数
                # print('before acc, outputs.shape:', outputs.shape, 'labels.shape:', labels.shape)
                total_acc += (outputs.argmax(dim=-1) == labels.argmax(dim=-1)).sum()
                total_num += outputs.shape[0]
                test_acc = total_acc / total_num

                if self.loss_name == "binary_cross_entropy":
                    labels = labels.argmax(dim=-1)
                    
                #### 计算AUC
                labels = labels.type(torch.int64)
                pred_list.extend(outputs.argmax(dim=-1).tolist())
                pred_list2.extend(outputs.tolist())
                label_list.extend(labels.tolist()) # 比label_list2更准确 labels.shape:[batch_size, 2], labels[:, 0].shape: [batch_size]
                
                # if epoch_idx is not None:
                #     self.logger.info(f"Test: Epo[{(epoch_idx + 1):0>3}/{ self.cfg.MAX_EPOCH:0>3}] Iter[{(i + 1):0>3}/{epo_iter_num:0>3}] Current test_acc:{test_acc} ({total_acc}/{total_num})")
                # else:
                #     self.logger.info(f"Test: Current test_acc:{test_acc:.4f} ({total_acc}/{total_num})")
                
                # output_list.append(outputs)
                # label_list.append(labels)
            
            # dataset_output = torch.cat(output_list, dim=0).numpy() # [val_batch_size, 2]
            # dataset_label = torch.cat(label_list, dim=0).numpy() # [val_batch_size, 2]] 
            # dataset_output= [[0.5023817  0.49859118]
            #                 [0.502379   0.49847925]]
            # dataset_label= [[1. 0.]
            #                 [0. 1.]]
        # self.tb_writer.add_scalar("test_acc", test_acc)
        
        # print('session_result=', session_result)
        #### 计算指标：session ACC  即每个session层面的ACC，分母是测试集中session的个数，分子是预测正确的session的个数。如果测试集有N个session，每个session有M个segment，对于某个session，将所有segments的outputs的第0列概率值求平均作为Known的概率值，第1列概率值求平均作为Unknown的概率值，如果Known的概率值大于Unknown的概率值，则认为该session的预测值为Known，否则为Unknown，最后与该session的真实label进行比较，如果相同则认为该session的预测正确，否则认为该session的预测错误，最后计算所有session的预测正确率
        if epoch_idx is not None:
            if epoch_idx % 3 == 0:
                print(f'Test: Epo[{(epoch_idx + 1):0>3}/{ self.cfg.MAX_EPOCH:0>3}] final session predict:{session_result} TIME:{time.strftime("%H:%M:%S", time.localtime())}')
        # 遍历session_result，计算每个session对应的ACC。计算方式：对于每个session的predict，将其所有segment的outputs的第0列概率值求平均作为Known的概率值，将其所有segment的outputs的第1列概率值求平均作为Unknown的概率值，如果Known的概率值大于Unknown的概率值，则认为该session的label为Known，否则认为该session的label为Unknown，最后与该session的真实label进行比较，如果相同则认为该session的预测正确，否则认为该session的预测错误，最后计算所有session的预测正确率
        session_total_acc, session_total_num, session_acc = 0, 0, 0
        session_pred_list, session_label_list = [], [] 
        quarter_acc_results = {}
        for session_id in session_result:
            session_predict = session_result[session_id]["predict"][:, :-1] # 将 shape=[segments_num, 3] 变为 [segments_num, 2], 去掉session_id对应的predict中的segment_id列, :-1 表示从第0列到倒数第2列，即去掉最后一列 
            session_label = session_result[session_id]["label"] # shape=[2] ，即[Known的概率值, Unknown的概率值]

            ### 1. compute quarter acc
            # 将当前session_id对应的所有segments_num个预测值分成四等分（四个quarter），每个quarter的shape=[segments_num/4, 2]，然后将四个quarter的第0列概率值求平均作为Known的概率值，第1列概率值求平均作为Unknown的概率值。如果Known的概率值大于Unknown的概率值，则认为该session的label为Known，否则认为该session的label为Unknown，最后与该session的真实label进行比较，如果相同则认为该session的预测正确，否则认为该session的预测错误，最后计算q1、q2、q3、q4四个quarter各自的预测正确率
            # print('[bi_modal_trainer.test] len(session_predict):', len(session_predict), ', len(session_predict)//4=', len(session_predict)//4, ', len(session_predict)//2=', len(session_predict)//2, ', len(session_predict)//4*3=', len(session_predict)//4*3) # e.g. len(session_predict): 25 , len(session_predict)//4= 6 , len(session_predict)//2= 12 , len(session_predict)//4*3= 18, then quarter1_predict.shape= torch.Size([6, 2]) , quarter2_predict.shape= torch.Size([6, 2]) , quarter3_predict.shape= torch.Size([6, 2]) , quarter4_predict.shape= torch.Size([7, 2])    e.g.2.len(session_predict): 29 , len(session_predict)//4= 7 , len(session_predict)//2= 14 , len(session_predict)//4*3= 21, quarter1_predict.shape= torch.Size([7, 2]) , quarter2_predict.shape= torch.Size([7, 2]) , quarter3_predict.shape= torch.Size([7, 2]) , quarter4_predict.shape= torch.Size([8, 2]) 
            quarter1_predict = session_predict[:len(session_predict)//4] # shape=[segments_num/4, 2]
            quarter2_predict = session_predict[len(session_predict)//4: len(session_predict)//2] # shape=[segments_num/4, 2]
            quarter3_predict = session_predict[len(session_predict)//2: len(session_predict)//4*3] # shape=[segments_num/4, 2]
            quarter4_predict = session_predict[len(session_predict)//4*3:] # shape=[segments_num/4, 2]
            # print('[bi_modal_trainer.test] quarter1_predict.shape=', quarter1_predict.shape, ', quarter2_predict.shape=', quarter2_predict.shape, ', quarter3_predict.shape=', quarter3_predict.shape, ', quarter4_predict.shape=', quarter4_predict.shape, ', \nquarter1_predict=', quarter1_predict, '\nquarter2_predict=', quarter2_predict, '\nquarter3_predict=', quarter3_predict, '\nquarter4_predict=', quarter4_predict)
            
            quarter1_predict = quarter1_predict.mean(dim=0) # shape=[segments_num/4, 2] -> [segments_num/4] 
            quarter2_predict = quarter2_predict.mean(dim=0)
            quarter3_predict = quarter3_predict.mean(dim=0)
            quarter4_predict = quarter4_predict.mean(dim=0)
            
            quarter_label = session_label
            # print('[bi_modal_trainer.test] after mean, quarter1_predict=', quarter1_predict, ', quarter1_predict.argmax(dim=-1)=', quarter1_predict.argmax(dim=-1), ', quarter_label.argmax(dim=-1)=', quarter_label.argmax(dim=-1), ', quarter2_predict=', quarter2_predict, ', quarter2_predict.argmax(dim=-1)=', quarter2_predict.argmax(dim=-1), ', quarter_label.argmax(dim=-1)=', quarter_label.argmax(dim=-1), ', quarter3_predict=', quarter3_predict, ', quarter3_predict.argmax(dim=-1)=', quarter3_predict.argmax(dim=-1), ', quarter_label.argmax(dim=-1)=', quarter_label.argmax(dim=-1), ', quarter4_predict=', quarter4_predict, ', quarter4_predict.argmax(dim=-1)=', quarter4_predict.argmax(dim=-1), ', quarter_label.argmax(dim=-1)=', quarter_label.argmax(dim=-1))
            quarter1_acc = (quarter1_predict.argmax(dim=-1) == quarter_label.argmax(dim=-1)).sum()
            quarter2_acc = (quarter2_predict.argmax(dim=-1) == quarter_label.argmax(dim=-1)).sum()
            quarter3_acc = (quarter3_predict.argmax(dim=-1) == quarter_label.argmax(dim=-1)).sum()
            quarter4_acc = (quarter4_predict.argmax(dim=-1) == quarter_label.argmax(dim=-1)).sum()
            # print('[bi_modal_trainer.test] session_id:', session_id, ', quarter1_acc=', quarter1_acc.item(), ', quarter2_acc=', quarter2_acc.item(), ', quarter3_acc=', quarter3_acc.item(), ', quarter4_acc=', quarter4_acc.item())
            quarter_acc_results[str(session_id)] = [quarter1_acc.item(), quarter2_acc.item(), quarter3_acc.item(), quarter4_acc.item()]
            
            ### 2. compute session acc
            session_predict = session_predict.mean(dim=0) # 将所有segments的outputs求平均，得到[2]的向量，即[Known的概率值, Unknown的概率值]
            # print('[bi_modal_trainer.test] session_id:', session_id, ', session_predict=', session_predict, ', session_label=', session_label, ', session_predict.argmax=', session_predict.argmax(dim=-1), ', session_label.argmax=', session_label.argmax(dim=-1), ', session_predict==session_label:', (session_predict.argmax(dim=-1) == session_label.argmax(dim=-1)).sum())
            session_total_acc += (session_predict.argmax(dim=-1) == session_label.argmax(dim=-1)).sum()
            session_total_num += 1
            session_acc = session_total_acc / session_total_num
            
            ### 3. used for session AUC
            session_label = session_label.type(torch.int64)
            # print('[bi_modal_trainer.test]. session_predict.shape=', session_predict.shape, ', session_label.shape=', session_label.shape, session_label, session_label.unsqueeze(0), session_label.unsqueeze(0).dim())
            if session_label.unsqueeze(0).dim() == 1: # self.dataset_name == "NOXI"
                session_pred_list.extend(session_predict.unsqueeze(0).tolist())
                session_label_list.extend(session_label.unsqueeze(0).tolist())
            else:
                session_pred_list.extend(session_predict.unsqueeze(0)[:, 0].tolist()) # session_predict.shape=torch.Size([2]) 分别是Known和Unknown的概率值，session_predict.unsqueeze(0).shape=torch.Size([1, 2])，session_predict.unsqueeze(0)[:, 0].shape=torch.Size([1])，session_predict.unsqueeze(0)[:, 0].tolist()=[0.5023816823959351]
                session_label_list.extend(session_label.unsqueeze(0)[:, 0].tolist())
            
        quarter_acc_results = json.dumps(quarter_acc_results) # 将dict转换为str
        if epoch_idx is not None:
            print(f'Test: Epo[{(epoch_idx + 1):0>3}/{ self.cfg.MAX_EPOCH:0>3}] quarter_acc_results:{quarter_acc_results}')
            if self.cfg.USE_WANDB:
                wandb.log({"quarter_acc_results": quarter_acc_results})
        
        ### 计算指标：session AUC
        print('[bi_modal_trainer] session AUC, session_pred_list=', session_pred_list, ', session_label_list=', session_label_list)
        if self.cfg_model.NUM_CLASS == 2:
            session_auc = auroc(torch.tensor(session_pred_list), torch.tensor(session_label_list), task="binary") # session_pred_list= [0.579576849937439, 0.6079692244529724, ...] , session_label_list= [1, 0, ...]
        elif self.cfg_model.NUM_CLASS > 2:
            session_auc = auroc(torch.tensor(session_pred_list), torch.tensor(session_label_list), task="multiclass", num_classes=self.cfg_model.NUM_CLASS)
        else:
            raise ValueError(f"Invalid NUM_CLASS={self.cfg_model.NUM_CLASS}")
        
        ### 计算指标：epoch AUC
        print('[bi_modal_trainer] segment AUC, pred_list=', pred_list, ', label_list=', label_list)
        pred, pred2 = torch.tensor(pred_list).type(torch.float32), torch.tensor(pred_list2).type(torch.float32)
        label = torch.tensor(label_list)
        if self.cfg_model.NUM_CLASS == 2: # Udiva 二分类，因此task="binary"
            epoch_auc = auroc(pred, label, task="binary")
        elif self.cfg_model.NUM_CLASS > 2: # NoXi 四分类，因此task="multiclass"
            epoch_auc = auroc(pred2, label, task="multiclass", num_classes=self.cfg_model.NUM_CLASS)
        else:
            raise Exception('Invalid NUM_CLASS')
        
        #### 计算指标：F1 score
        # f1 = self.f1_metric(torch.tensor(pred_list), torch.tensor(label_list))
        # f1_2 = self.f1_metric(torch.tensor(pred_list2), torch.tensor(label_list2))
        
        if epoch_idx is not None:
            self.logger.info("Test: Epo[{:0>3}/{:0>3}] Test Epo Summary Acc:{:.4f} ({}/{}) Epo AUC: {:.4f} Epo F1_Score: {:.4f} ({:.4f}) Session Acc:{:.4f} ({}/{}) Session AUC: {:.4f} TIME:{}".format(
                                epoch_idx + 1, self.cfg.MAX_EPOCH, # Epoch
                                test_acc, total_acc, total_num,    # Epo ACC
                                epoch_auc, # Epo AUC
                                f1, f1_2, # Epo F1_Score
                                session_acc, session_total_acc, session_total_num, # Session ACC
                                session_auc, # Session AUC
                                time.strftime("%H:%M:%S", time.localtime()), # Current TIME
                            ))
            if self.cfg.USE_WANDB:
                wandb.log({
                    "test_acc": float(test_acc),
                    "test_auc": float(epoch_auc),
                    "test_f1": float(f1),
                    "test_f1_2": float(f1_2),
                    "test_session_acc": float(session_acc),
                    "test_session_auc": float(session_auc),
                    "epoch": epoch_idx + 1})
        else:
            self.logger.info("Test only: Final Acc:{:.4f} ({}/{}) Final AUC:{:.4f} Final Session Acc:{:.4f} ({}/{}) Final Session AUC:{:.4f} TIME:{}\n".format(
                test_acc, total_acc, total_num,  # Epo ACC
                epoch_auc,           # Epo AUC
                session_acc, session_total_acc, session_total_num, # Session ACC
                session_auc,                     # Session AUC
                time.strftime("%H:%M:%S", time.localtime()), # Current TIME
                ))
            if self.cfg.USE_WANDB:
                wandb.log({
                    "test_final_acc": float(test_acc),
                    "test_final_auc": float(epoch_auc),
                    "test_final_session_acc": float(session_acc),
                    "test_final_session_auc": float(session_auc)
                    })

        return test_acc

    def full_test(self, data_set, model):
        model.eval()
        out_ls, label_ls = [], []
        with torch.no_grad():
            for data in tqdm(data_set):
                inputs, label = self.full_test_data_fmt(data)
                out = model(*inputs)
                out_ls.append(out.mean(0).cpu().detach())
                label_ls.append(label)
        all_out = torch.stack(out_ls, 0)
        all_label = torch.stack(label_ls, 0)
        ocean_acc = (1 - torch.abs(all_out - all_label)).mean(0).numpy()
        ocean_acc_avg = ocean_acc.mean(0)

        ocean_acc_avg_rand = np.round(ocean_acc_avg, 4)
        ocean_acc_dict = {k: np.round(ocean_acc[i], 4) for i, k in enumerate(["O", "C", "E", "A", "N"])}

        dataset_output = all_out.numpy()
        dataset_label = all_label.numpy()

        return ocean_acc_avg_rand, ocean_acc_dict, dataset_output, dataset_label

    def data_extract(self, model, data_set, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for idx, data in enumerate(tqdm(data_set)):
                inputs, label = self.full_test_data_fmt(data)
                # mini_batch = 64
                out_ls, feat_ls = [], []
                for i in range(math.ceil(len(inputs[0]) / 64)):
                    mini_batch_1 = inputs[0][(i * 64): (i + 1) * 64]
                    mini_batch = (mini_batch_1,)
                    try:
                        mini_batch_2 = inputs[1][(i * 64): (i + 1) * 64]
                        mini_batch = (mini_batch_1, mini_batch_2)
                    except IndexError:
                        pass

                    # mini_batch = (mini_batch_1, mini_batch_2)
                    if model.return_feature:
                        out, feat = model(*mini_batch)
                        out_ls.append(out.cpu())
                        feat_ls.append(feat.cpu())
                    else:
                        out = model(*mini_batch)
                        out_ls.append(out.cpu())
                        feat_ls.append(torch.tensor([0]))
                out_pred, out_feat = torch.cat(out_ls, dim=0), torch.cat(feat_ls, dim=0)
                video_extract = {
                    "video_frames_pred": out_pred,
                    "video_frames_feat": out_feat,
                    "video_label": label.cpu()
                }
                save_to_file = os.path.join(output_dir, "{:04d}.pkl".format(idx))
                torch.save(video_extract, save_to_file)

    def data_fmt(self, data):
        # print('[bi_modal_trainer] type of data: ', type(data))
        session_id, segment_id = None, None
        if isinstance(data, dict): 
            # 1、如果data_loader中没有使用RandomOverSampler data就是一个dict，dict里有image audio label 这几个key，分别对应image,audio,label的数据
            for k, v in data.items(): # Python 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组。
                if k in ["image", "audio"]:
                    data[k] = v.to(self.device, non_blocking=self.non_blocking)
            if self.cfg.BIMODAL_OPTION == 1:
                aud_in = None
                img_in, labels, session_id, segment_id, is_continue = data["image"], data["label"], data["session_id"], data["segment_id"], data["is_continue"]
            elif self.cfg.BIMODAL_OPTION == 2:
                aud_in, labels, session_id, segment_id, is_continue = data["audio"], data["label"], data["session_id"], data["segment_id"], data["is_continue"]
                img_in = None
            elif self.cfg.BIMODAL_OPTION == 3:
                img_in, aud_in, labels, session_id, segment_id, is_continue = data["image"], data["audio"], data["label"], data["session_id"], data["segment_id"], data["is_continue"]
            else:
                raise ValueError("BIMODAL_OPTION should be 1, 2 or 3. not {}".format(self.cfg.BIMODAL_OPTION))
        elif isinstance(data, list):
            # 2、如果data_loader中使用了RandomOverSampler，那么这里得到的data就是一个list，list里有两个元素，分别是image和audio，image的shape:[batch_size, sample_size, c, h, w] e.g.[8, 16, 6, 224, 224]  label.shape: [batch_size, 2]
            for i, v in enumerate(data): # Python enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                data[i] = v.to(self.device)
            if self.cfg.BIMODAL_OPTION == 1:
                aud_in = None
                img_in, labels= data[0], data[1] # img_in.shape: [batch_size, sample_size, c, h, w] e.g.[8, 16, 6, 224, 224]  label.shape: [batch_size, 2]
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
        
        # if self.loss_name == "cross_entropy": # refer: https://stackoverflow.com/a/57327968
            # labels = torch.argmax(labels, dim=1) # convert one-hot tensor: [bs, num_classes] to index: [bs]
        # labels = torch.argmax(labels, dim=1) # convert one-hot tensor: [bs, num_classes] to index: [bs]
        
        # 不同的模型需要不同的维度格式，这里根据不同的模型进行数据格式的转换
        if self.cfg_model.NAME == "audiovisual_resnet_lstm_udiva":
            img_in = img_in.permute(0, 2, 1, 3, 4) # 将输入的数据从 [batch, time, channel, height, width] 转换为 [batch, channel, time, height, width] e.g. 4 * 16 * 6 * 224 * 224 -> 4 * 6 * 16 * 224 * 224
            return (aud_in, img_in), labels, session_id, segment_id, is_continue
        elif self.cfg_model.NAME in ["resnet50_3d_model_udiva"]:
            img_in = img_in.permute(0, 2, 1, 3, 4) # 将输入的数据从 [batch, time, channel, height, width] 转换为 [batch, channel, time, height, width] e.g. 4 * 16 * 6 * 224 * 224 -> 4 * 6 * 16 * 224 * 224
            return (img_in, ), labels, session_id, segment_id, is_continue
        elif self.cfg_model.NAME == "vivit_model_udiva":
            return (img_in, ), labels, session_id, segment_id, is_continue # img_in: [batch, time, channel, height, width]
        elif self.cfg_model.NAME == "vivit_model3_udiva":
            img_in = img_in.permute(0, 2, 1, 3, 4) # img_in: [batch, channel, time, height, width], # 将输入的数据从 [batch, time, channel, height, width] 转换为 [batch, channel, time, height, width] e.g. 4 * 16 * 6 * 224 * 224 -> 4 * 6 * 16 * 224 * 224
            # print('[data_fmt] img_in.device: ', img_in.device, ', labels.device: ', labels.device)
            return (img_in, ), labels, session_id, segment_id, is_continue
        elif self.cfg_model.NAME == "timesformer_udiva":
            img_in = img_in.permute(0, 2, 1, 3, 4) # 将输入的数据从 [batch, time, channel, height, width] 转换为 [batch, channel, time, height, width] e.g. 4 * 16 * 6 * 224 * 224 -> 4 * 6 * 16 * 224 * 224
            return (img_in, ), labels, session_id, segment_id, is_continue
        elif self.cfg_model.NAME == "ssast_udiva":
            return(aud_in, self.cfg.TASK), labels, session_id, segment_id, is_continue
        elif self.cfg_model.NAME == "visual_graph_representation_learning":
            if self.cfg.USE_HALF and torch.cuda.is_available():
                img_in = img_in.half()
            return (img_in, ), labels, session_id, segment_id, is_continue
        elif self.cfg_model.NAME == "audio_graph_representation_learning":
            return (aud_in, ), labels, session_id, segment_id, is_continue
        else:
            return (aud_in, img_in), labels, session_id, segment_id, is_continue
    
    def full_test_data_fmt(self, data):
        images, wav, label = data["image"], data["audio"], data["label"]
        images_in = torch.stack(images, 0).to(self.device)
        # wav_in = torch.stack([wav] * 100, 0).to(self.device)
        wav_in = wav.repeat(len(images), 1, 1, 1).to(self.device)
        return (wav_in, images_in), label


@TRAINER_REGISTRY.register()
class BimodalLSTMTrain(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        img_in, aud_in, labels = data["image"], data["audio"], data["label"]
        img_in = img_in.view(-1, 3, 112, 112)
        aud_in = aud_in.view(-1, 68)
        return (aud_in, img_in), labels


@TRAINER_REGISTRY.register()
class BimodalLSTMTrainVisual(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        img_in,  labels = data["image"],  data["label"]
        # img_in的维度是[batch_size, 6(帧序列), 3, 112, 112]，经过view后变成[batch_size*6, 3, 112, 112]
        img_in = img_in.view(-1, 3, 112, 112) 
        # aud_in = aud_in.view(-1, 68)
        return (img_in,), labels


@TRAINER_REGISTRY.register()
class ImgModalLSTMTrain(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        img_in, _, labels = data["image"], data["audio"], data["label"]
        img_in = img_in.view(-1, 3, 112, 112)
        return (img_in,), labels


@TRAINER_REGISTRY.register()
class AudModalLSTMTrain(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        _, aud_in, labels = data["image"], data["audio"], data["label"]
        aud_in = aud_in.view(-1, 68)
        return (aud_in,), labels


@TRAINER_REGISTRY.register()
class DeepBimodalTrain(BimodalLSTMTrain):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs, labels = data["image"], data["label"]
        return (inputs,), labels


@TRAINER_REGISTRY.register()
class ImageModalTrainer(BiModalTrainer):
    """
    for model only image data used
    """
    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs, labels = data["image"], data["label"]
        return (inputs,), labels

    def full_test_data_fmt(self, data):
        images, label = data["all_images"], data["label"]
        images_in = torch.stack(images, 0).to(self.device)
        return (images_in, ), label


@TRAINER_REGISTRY.register()
class MultiModalTrainer(BiModalTrainer):
    """
    for model only image data used
    """
    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs, labels = data["feature"], data["label"]
        return (inputs,), labels


@TRAINER_REGISTRY.register()
class ImageListTrainer(BiModalTrainer):
    """
    for interpret cnn model, only image data used
    """
    def data_fmt(self, data):
        inputs, labels = data["image"], data["label"]
        inputs = [item.to(self.device) for item in inputs]
        labels = labels.to(self.device)
        return (inputs,), labels

    def full_test_data_fmt(self, data):
        images, label = data["all_images"], data["label"]
        # short_sque, long_sque = zip(*images)
        inputs = [torch.stack(sque, 0).to(self.device) for sque in zip(*images)]
        return (inputs,), label


@TRAINER_REGISTRY.register()
class TPNTrainer(BiModalTrainer):
    """
    for interpret cnn model, only image data used
    """
    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        inputs, labels = data["image"], data["label"]
        data_input = {"num_modalities": [1], "img_group_0": inputs, "img_meta": None, "gt_label": labels}

        return data_input, labels

    def train(self, data_loader, model, loss_f, optimizer, epoch_idx):
        model.train() # model.train() is necessary for dropout and batchnorm,model. train()的作用是启用 Batch Normalization 和 Dropout, model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。 https://cloud.tencent.com/developer/article/1819853
        self.logger.info(f"Training: learning rate:{optimizer.param_groups[0]['lr']}")
        loss_list = []
        acc_avg_list = []
        for i, data in enumerate(data_loader):
            inputs, labels = self.data_fmt(data)
            loss, outputs = model(**inputs)
            optimizer.zero_grad()
            # loss = loss_f(outputs.cpu(), labels.cpu())
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            acc_avg = (1 - torch.abs(outputs.cpu() - labels.cpu())).mean().clip(min=0)
            acc_avg = acc_avg.detach().numpy()
            acc_avg_list.append(acc_avg)
            # print loss info for an interval
            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1:
                self.logger.info(
                    "Train: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.4f}".format(
                        epoch_idx + 1, self.cfg.MAX_EPOCH,
                        i + 1, len(data_loader),
                        float(loss.item()), float(acc_avg)
                    )
                )

        self.clt.record_train_loss(loss_list)
        self.clt.record_train_acc(acc_avg_list)

    def valid(self, data_loader, model, loss_f, epoch_idx):
        model.eval()
        with torch.no_grad():
            loss_batch_list = []
            acc_batch_list = []
            ocean_acc_epoch = []
            for i, data in enumerate(data_loader):
                inputs, labels = self.data_fmt(data)
                loss, outputs = model(**inputs)
                # loss = loss_f(outputs.cpu(), labels.cpu())
                loss_batch_list.append(loss.item())
                ocean_acc_batch = (1 - torch.abs(outputs.cpu().detach() - labels.cpu().detach())).mean(dim=0)
                ocean_acc_epoch.append(ocean_acc_batch)
                acc_batch_avg = ocean_acc_batch.mean()
                acc_batch_list.append(acc_batch_avg)
            ocean_acc = torch.stack(ocean_acc_epoch, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_acc_avg = ocean_acc.mean()

        self.clt.record_valid_loss(loss_batch_list)
        self.clt.record_valid_acc(acc_batch_list)  # acc over batches
        self.clt.record_valid_ocean_acc(ocean_acc)
        if ocean_acc_avg > self.clt.best_valid_acc:
            self.clt.update_best_acc(ocean_acc_avg)
            self.clt.update_model_save_flag(1)
        else:
            self.clt.update_model_save_flag(0)

        self.logger.info(
            "Valid: Epoch[{:0>3}/{:0>3}] Train Mean_Acc: {:.2%} Valid Mean_Acc:{:.2%} OCEAN_ACC:{}\n".
            format(
                epoch_idx + 1, self.cfg.MAX_EPOCH,
                float(self.clt.epoch_train_acc),
                float(self.clt.epoch_valid_acc),
                self.clt.valid_ocean_acc)
        )

    def test(self, data_loader, model):
        model.eval() # model.eval()的作用是不启用 Batch Normalization 和 Dropout。  model.eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode
        mse = torch.nn.MSELoss(reduction="none")
        with torch.no_grad():
            ocean_acc = []
            ocean_mse = []
            label_list = []
            output_list = []
            for data in tqdm(data_loader):
                inputs, labels = self.data_fmt(data)
                loss, outputs = model(**inputs)

                outputs = outputs.cpu().detach()
                labels = labels.cpu().detach()
                output_list.append(outputs)
                label_list.append(labels)
                ocean_mse_batch = mse(outputs, labels).mean(dim=0)
                ocean_acc_batch = (1 - torch.abs(outputs - labels)).mean(dim=0)
                ocean_mse.append(ocean_mse_batch)
                ocean_acc.append(ocean_acc_batch)
            ocean_mse = torch.stack(ocean_mse, dim=0).mean(dim=0).numpy()
            ocean_acc = torch.stack(ocean_acc, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_mse_avg = ocean_mse.mean()
            ocean_acc_avg = ocean_acc.mean()

            dataset_output = torch.cat(output_list, dim=0).numpy()
            dataset_label = torch.cat(label_list, dim=0).numpy()
        ocean_mse_avg_rand = np.round(ocean_mse_avg.astype("float64"), 4)
        ocean_acc_avg_rand = np.round(ocean_acc_avg.astype("float64"), 4)
        keys = ["O", "C", "E", "A", "N"]
        ocean_mse_dict, ocean_acc_dict = {}, {}
        for i, k in enumerate(keys):
            ocean_mse_dict[k] = np.round(ocean_mse[i], 4)
            ocean_acc_dict[k] = np.round(ocean_acc[i], 4)
        return ocean_acc_avg_rand, ocean_acc_dict, dataset_output, dataset_label, (ocean_mse_dict, ocean_mse_avg_rand)

    def full_test(self, data_loader, model):
        model.eval()
        with torch.no_grad():
            ocean_acc = []
            label_list = []
            output_list = []
            for data in tqdm(data_loader):
                inputs, labels = self.full_test_data_fmt(data)
                loss, outputs = model(**inputs)

                outputs = outputs.cpu().detach()
                labels = labels.cpu().detach()
                output_list.append(outputs)
                label_list.append(labels)
                ocean_acc_batch = (1 - torch.abs(outputs - labels)).mean(dim=0)
                ocean_acc.append(ocean_acc_batch)
            ocean_acc = torch.stack(ocean_acc, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_acc_avg = ocean_acc.mean()
            dataset_output = torch.cat(output_list, dim=0).numpy()
            dataset_label = torch.cat(label_list, dim=0).numpy()

        ocean_acc_avg_rand = np.round(ocean_acc_avg.astype("float64"), 4)
        keys = ["O", "C", "E", "A", "N"]
        ocean_acc_dict = {}
        for i, k in enumerate(keys):
            ocean_acc_dict[k] = np.round(ocean_acc[i], 4)
        return ocean_acc_avg_rand, ocean_acc_dict, dataset_output, dataset_label

    def full_test_data_fmt(self, data):
        inputs, labels = data["all_images"], data["label"]
        inputs = torch.stack(inputs, 0).to(self.device)
        labels_repeats = labels.repeat(6, 1).to(self.device)
        data_input = {"num_modalities": [1], "img_group_0": inputs, "img_meta": None, "gt_label": labels_repeats}
        return data_input, labels_repeats


@TRAINER_REGISTRY.register()
class PersEmoTrainer(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.squeeze().to(self.device)
        per_inputs, emo_inputs = data["per_img"], data["emo_img"],
        per_labels, emo_labels = data["per_label"], data["emo_label"]
        return (per_inputs, emo_inputs), per_labels, emo_labels

    def train(self, data_loader, model, loss_f, optimizer, epoch_idx):
        model.train()
        self.logger.info(f"Training: learning rate:{optimizer.param_groups[0]['lr']}")
        loss_list = []
        acc_avg_list = []
        for i, data in enumerate(data_loader):
            inputs, p_labels, e_labels = self.data_fmt(data)
            p_score, p_co, e_score, e_co, x_ep = model(*inputs)
            optimizer.zero_grad()
            loss = loss_f(p_score, p_labels, e_score, e_labels, p_co, e_co, x_ep)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            acc_avg = (1 - torch.abs(p_score.cpu() - p_labels.cpu())).mean().clip(min=0)
            acc_avg = acc_avg.detach().numpy()
            acc_avg_list.append(acc_avg)
            # print loss info for an interval
            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1:
                self.logger.info(
                    "Train: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.4f}".format(
                        epoch_idx + 1, self.cfg.MAX_EPOCH,
                        i + 1, len(data_loader),
                        float(loss.item()), float(acc_avg)
                    )
                )

        self.clt.record_train_loss(loss_list)
        self.clt.record_train_acc(acc_avg_list)

    def valid(self, data_loader, model, loss_f, epoch_idx):
        model.eval()
        with torch.no_grad():
            loss_batch_list = []
            acc_batch_list = []
            ocean_acc_epoch = []
            for i, data in enumerate(data_loader):
                inputs, p_labels, e_labels = self.data_fmt(data)
                p_score, p_co, e_score, e_co, x_ep = model(*inputs)
                loss = loss_f(p_score, p_labels, e_score, e_labels, p_co, e_co, x_ep)
                loss_batch_list.append(loss.item())
                ocean_acc_batch = (1 - torch.abs(p_score.cpu().detach() - p_labels.cpu().detach())).mean(dim=0)
                ocean_acc_epoch.append(ocean_acc_batch)
                acc_batch_avg = ocean_acc_batch.mean()
                acc_batch_list.append(acc_batch_avg)
            ocean_acc = torch.stack(ocean_acc_epoch, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_acc_avg = ocean_acc.mean()

        self.clt.record_valid_loss(loss_batch_list)
        self.clt.record_valid_acc(acc_batch_list)  # acc over batches
        self.clt.record_valid_ocean_acc(ocean_acc)
        if ocean_acc_avg > self.clt.best_valid_acc:
            self.clt.update_best_acc(ocean_acc_avg)
            self.clt.update_model_save_flag(1)
        else:
            self.clt.update_model_save_flag(0)

        self.logger.info(
            "Valid: Epoch[{:0>3}/{:0>3}] Train Mean_Acc: {:.2%} Valid Mean_Acc:{:.2%} OCEAN_ACC:{}\n".
            format(
                epoch_idx + 1, self.cfg.MAX_EPOCH,
                float(self.clt.epoch_train_acc),
                float(self.clt.epoch_valid_acc),
                self.clt.valid_ocean_acc)
        )

    def test(self, data_loader, model):
        model.eval()
        mse = torch.nn.MSELoss(reduction="none")
        with torch.no_grad():
            ocean_acc = []
            ocean_mse = []
            label_list = []
            output_list = []
            for data in tqdm(data_loader):
                inputs, p_labels, e_labels = self.data_fmt(data)
                p_score, p_co, e_score, e_co, x_ep = model(*inputs)
                p_score = p_score.cpu().detach()
                p_labels = p_labels.cpu().detach()
                output_list.append(p_score)
                label_list.append(p_labels)
                ocean_mse_batch = mse(p_score, p_labels).mean(dim=0)
                ocean_acc_batch = (1 - torch.abs(p_score - p_labels)).mean(dim=0)
                ocean_mse.append(ocean_mse_batch)
                ocean_acc.append(ocean_acc_batch)
            ocean_mse = torch.stack(ocean_mse, dim=0).mean(dim=0).numpy()
            ocean_acc = torch.stack(ocean_acc, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_mse_avg = ocean_mse.mean()
            ocean_acc_avg = ocean_acc.mean()

            dataset_output = torch.stack(output_list, dim=0).view(-1, 5).numpy()
            dataset_label = torch.stack(label_list, dim=0).view(-1, 5).numpy()

        keys = ["O", "C", "E", "A", "N"]
        ocean_mse_dict, ocean_acc_dict = {}, {}
        for i, k in enumerate(keys):
            ocean_mse_dict[k] = np.round(ocean_mse[i], 4)
            ocean_acc_dict[k] = np.round(ocean_acc[i], 4)
        return ocean_acc_avg, ocean_acc_dict, dataset_output, dataset_label, (ocean_mse_dict, ocean_mse_avg)

    def full_test(self, data_loader, model):
        return self.test(data_loader, model)

    def full_test_data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.squeeze().to(self.device)
        per_inputs, emo_inputs = data["per_img"], data["emo_img"],
        per_labels, emo_labels = data["per_label"], data["emo_label"]
        return (per_inputs, emo_inputs), per_labels[0]

    def data_extract(self, model, data_set, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for idx, data in enumerate(tqdm(data_set)):
                inputs, label = self.full_test_data_fmt(data)
                mini_batch = 64
                out_ls, feat_ls = [], []
                for i in range(math.ceil(len(inputs[0]) / mini_batch)):
                    mini_batch_1 = inputs[0][(i * mini_batch): (i + 1) * mini_batch]
                    mini_batch_2 = inputs[1][i * 3: (i * 3 + mini_batch)]  # jump 3 images every time
                    mini_batch_input = (mini_batch_1, mini_batch_2)
                    out, *_, feat = model(*mini_batch_input)
                    out_ls.append(out.cpu())
                    feat_ls.append(feat.cpu())

                out_pred, out_feat = torch.cat(out_ls, dim=0), torch.cat(feat_ls, dim=0)
                video_extract = {
                    "video_frames_pred": out_pred,
                    "video_frames_feat": out_feat,
                    "video_label": label.cpu()
                }
                save_to_file = os.path.join(output_dir, "{:04d}.pkl".format(idx))
                torch.save(video_extract, save_to_file)


@TRAINER_REGISTRY.register()
class AudioTrainer(BiModalTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        return (data["aud_data"],), data["aud_label"]


@TRAINER_REGISTRY.register()
class StatisticTrainer(BiModalTrainer):

    def data_fmt(self, data):
        return (data["data"].to(self.device),), data["label"].to(self.device)


@TRAINER_REGISTRY.register()
class SpectrumTrainer(BiModalTrainer):

    def data_fmt(self, data):
        return (data["data"].to(self.device),), data["label"].to(self.device)
