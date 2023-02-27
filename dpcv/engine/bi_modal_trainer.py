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
                ocean_acc_batch = (
                    1 - torch.abs(outputs.cpu().detach() - labels.cpu().detach())
                ).mean(dim=0).clip(min=0)
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clt = collector
        self.logger = logger
        # 在 cfg.OUTPUT_DIR 路径后加上 /tensorboard_events
        tb_writer_dir = os.path.join(self.cfg.OUTPUT_DIR, "tensorboard_events")
        self.tb_writer = SummaryWriter(tb_writer_dir)
        self.f1_metric = BinaryF1Score()

    def train(self, data_loader, model, loss_f, optimizer, epoch_idx):
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
        self.tb_writer.add_scalar("lr", lr, epoch_idx)
        optimizer.zero_grad() # 梯度清零，即将梯度变为0
        model.train()
        loss_list = []
        acc_avg_list = []
        epoch_total_acc = 0
        epoch_total_num = 0
        epo_iter_num = len(data_loader)
        pred_list, label_list, pred_list2, label_list2 = [], [], [], []
        for i, data in enumerate(data_loader): # i代表第几个batch, data代表第i个batch的数据 # 通过看日志，当执行下面这行 for i, data in enumerate(data_loader)语句时，会调用 AudioVisualData(VideoData)类里的 __getitem__ 函数，紧接着调用def get_ocean_label()函数， 具体原因参考：https://www.geeksforgeeks.org/how-to-use-a-dataloader-in-pytorch/
            iter_start_time = time.time()
            # print('[bi_modal_trainer.py] train... type(data)=', type(data), ', len(data)=', len(data)) # type(data)= <class 'list'> , len(data)= 2
            # print('[bi_modal_trainer.py] train... data[0].shape=',  data[0].shape, ', data[1].shape=', data[1].shape, ' type(data[0]):', type(data[0]), ', type(data[1]):', type(data[1])) # type(data[0]): <class 'torch.Tensor'> , type(data[1]): <class 'torch.Tensor'>
            """ 1、如果data_loader中使用了RandomOverSampler，那么这里得到的data就是一个list，list里有两个元素，分别是image和audio，image的shape:[batch_size, sample_size, c, h, w] e.g.[8, 16, 6, 224, 224]  label.shape: [batch_size, 2]
                2、如果没有使用RandomOverSampler，那么这里得到的data就是一个dict，dict里有image audio label 这几个key，分别对应image,audio,label的数据 """
            inputs, labels = self.data_fmt(data) # self.data_fmt(data) 代表将data里的image, audio, label分别取出来，放到inputs，label里
            # print('[bi_modal_trainer.py] train... model.device=', model.device, 'inputs[0].device=', inputs[0].device)
            # if i in [0, 1] and torch.cuda.is_available():
            #     print('before model, i:', i, ', CUDA memory_summary:\n', torch.cuda.memory_summary())
            try: # refer: https://zhuanlan.zhihu.com/p/497192910
                # inputs加一个*星号：表示参数数量不确定，将传入的参数存储为元组（https://blog.csdn.net/qq_42951560/article/details/112006482）。*inputs意思是将inputs里的元素分别取出来，作为model的输入参数，这里的inputs是一个元组，包含了image和audio。models里的forward函数里的参数是image和audio，所以这里的*inputs就是将image和audio分别取出来，作为model的输入参数。为什么是forward函数的参数而不是__init__函数的参数？因为forward函数是在__init__函数里被调用的，所以forward函数的参数就是__init__函数的参数。forward 会自动被调用，调用时会传入输入数据，所以forward函数的参数就是输入数据。
                outputs = model(*inputs)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print(exception)
                    # print('Train WARNING: out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                        if i in [0, 1] and torch.cuda.is_available():
                            print('after model, i:', i, ', CUDA memory_summary:\n', torch.cuda.memory_summary())
                    else:
                        raise exception
            # print('[bi_modal_trainer.py] train... outputs=', outputs, 'labels=', labels, ' outputs.size()', outputs.size(),  '  labels.size()=', labels.size())
            loss = loss_f(outputs.cpu(), labels.cpu().float())

            outputs = outputs.detach().cpu()
            labels = labels.detach().cpu()

            if i in [0, 1, 2, 3, 4]:
                print(torch.cat([outputs, labels], dim=1))
                print(f'*********** Epo[{(epoch_idx+1):0>3}/{self.cfg.MAX_EPOCH:0>3}] Iter[{(i + 1):0>3}/{epo_iter_num:0>3}], train loss:{loss.item():.4f} ***********')
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
            self.tb_writer.add_scalar("loss", loss.item(), i) # loss.item()是将loss转换成float类型，即tensor转换成float类型，add_scalar()是将loss写入到tensorboard里, i是第
            loss.backward() # loss.backward()是将loss反向传播，即计算loss对每个参数的梯度，即loss对每个参数的偏导数
            optimizer.step() # step()是将梯度更新到参数上，即将loss对每个参数的偏导数更新到参数上， 即 w = w - lr * gradient, 其中lr是学习率，gradient是loss对每个参数的偏导数，即loss对每个参数的梯度， w是模型的参数

            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time

            loss_list.append(loss.item())
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
            
            # 因为我们需要预测的是认识/不认识，属于分类问题而不是回归问题，所以计算acc的方式为：acc = (y_pred.argmax(dim=1) == y_true).float()
            # outputs = outputs.detach().cpu()
            # labels = labels.detach().cpu()
            
            #### 计算指标：acc
            batch_total_acc = (outputs.argmax(dim=-1) == labels.argmax(dim=-1)).sum() # 当前batch里分类预测正确的样本数
            batch_total_num = outputs.shape[0] # 当前batch的样本总数
            batch_acc = batch_total_acc / batch_total_num # 当前batch的acc
            epoch_total_acc += batch_total_acc
            epoch_total_num += batch_total_num
            epoch_current_acc = epoch_total_acc / epoch_total_num
            
            acc_avg = batch_acc.numpy()
            acc_avg_list.append(acc_avg)
            
            outputs = torch.round(outputs * 1000000) / 1000000 # 小数点后保留6位
            labels = labels.type(torch.int64)
            
            # print('outputs[:, 0]:', outputs[:, 0], 'outputs.shape:', outputs.shape, ', outputs.argmax(dim=-1)=', outputs.argmax(dim=-1))
            # print('labels[:, 0]:', labels[:, 0], 'labels.shape:', labels.shape)
            pred_list.extend(outputs[:, 0].tolist())
            label_list.extend(labels[:, 0].tolist())
            pred_list2.extend(outputs.argmax(dim=-1).tolist())
            label_list2.extend(labels.argmax(dim=-1).tolist())
            
            #### 计算指标：AUC
            batch_auc = auroc(outputs[:, 0], labels[:, 0], task="binary")
            batch_auc2 = auroc(outputs.argmax(dim=-1).type(torch.float32), labels.argmax(dim=-1), task="binary")
            
            #### 计算指标：F1
            f1 = self.f1_metric(outputs[:, 0], labels[:, 0]) # 最好用这个，详见：https://www.notion.so/MPhil-Project-b3de240fa9d64832b26795439d0142d9?pvs=4#3820c783cbc947cb83c1c5dd0d91434b
            f1_2 = self.f1_metric(outputs.argmax(dim=-1), labels.argmax(dim=-1))

            if self.cfg.USE_WANDB:
                wandb.log({
                    "train_loss":  float(loss.item()),
                    "train_acc": float(acc_avg), # 当前batch的acc
                    "train_epoch_current_acc": float(epoch_current_acc), # 当前epoch的持续记录的acc
                    "train_batch_auc": float(batch_auc), # 当前batch的auc
                    "train_batch_auc2": float(batch_auc2),
                    "train_batch_f1": float(f1), # 当前batch的f1
                    "train_batch_f1_2": float(f1_2), # 当前batch的f1
                    "learning rate": lr,
                    "epoch": epoch_idx + 1,
                })

            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1: #  self.cfg.LOG_INTERVAL = 10，即每10个batch打印一次loss和acc_avg
                remain_iter = epo_iter_num - i # epo_iter_num(一个epoch里的batch数)-当前的batch数=剩余的batch数
                remain_epo = self.cfg.MAX_EPOCH - epoch_idx 
                eta = (epo_iter_num * iter_time) * remain_epo + (remain_iter * iter_time) # eta是剩余的时间，即剩余的epoch数乘以一个epoch的时间加上剩余的batch数乘以一个batch的时间
                eta = int(eta) # 将eta转换成int类型
                eta_string = f"{eta // 3600}h:{eta % 3600 // 60}m:{eta % 60}s"  # 将eta转换成时分秒的形式
                self.logger.info(
                    "Train: Epo[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] IterTime:[{:.2f}s] LOSS: {:.4f} Batch ACC:{:.4f} ({}/{}) Epo current ACC:{:.4f} ({}/{}) AUC:{:.4f} ({:.4f}) F1:{:.4f} ({:.4f}) ETA:{} \n".format(
                        epoch_idx + 1, self.cfg.MAX_EPOCH,   # Epo 
                        i + 1, epo_iter_num,                 # Iter
                        iter_time,                           # IterTime
                        float(loss.item()),                  # LOSS
                        float(acc_avg), batch_total_acc, batch_total_num,    # Batch ACC
                        epoch_current_acc, epoch_total_acc, epoch_total_num, # Epo current ACC
                        batch_auc, batch_auc2, # AUC
                        f1, f1_2,                            # F1
                        eta_string,                          # ETA
                    ))
        #### 计算指标：acc
        epoch_summary_acc = epoch_total_acc / epoch_total_num
        
        #### 计算指标：AUC
        # print('pred_list:', pred_list, 'label_list:', label_list, 'pred_list2:', pred_list2, ', len(pred_list):', len(pred_list), ', len(label_list):', len(label_list), ', len(pred_list2):', len(pred_list2))
        epoch_auc = auroc(torch.tensor(pred_list), torch.tensor(label_list), task="binary")
        epoch_auc2 = auroc(torch.tensor(pred_list2).type(torch.float32), torch.tensor(label_list2), task="binary")
        
        #### 计算指标：F1 score
        f1 = self.f1_metric(torch.tensor(pred_list), torch.tensor(label_list))
        f1_2 = self.f1_metric(torch.tensor(pred_list2), torch.tensor(label_list2))

        self.logger.info(
            "Train: Epo[{:0>3}/{:0>3}] Epo Summary Acc:{:.4f} ({}/{}) Epo AUC: {:.4f} ({:.4f}) Epo F1_Score: {:.4f} ({:.4f})\n".
            format(
                epoch_idx + 1, self.cfg.MAX_EPOCH, # Epoch
                epoch_summary_acc, epoch_total_acc, epoch_total_num, # Epo Summary Acc
                epoch_auc, epoch_auc2, # Epo AUC
                f1, f1_2, # Epo F1_Score
            ))
        
        if self.cfg.USE_WANDB:
            wandb.log({
                "train_epoch_summary_acc": float(epoch_summary_acc),  # 记录当前epoch全部batch遍历完后的总体acc
                "train_epoch_summary_auc": float(epoch_auc), # 比epoch_auc2更准确的auc
                "train_epoch_summary_auc2": float(epoch_auc2),
                "train_epoch_summary_f1_score": float(f1), # 比f1_2更准确的f1
                "train_epoch_summary_f1_score2": float(f1_2),
                "epoch": epoch_idx + 1})
        
        self.clt.record_train_loss(loss_list) # 将loss_list里的loss值记录到self.clt里
        self.clt.record_train_acc(acc_avg_list) # 将acc_avg_list里的acc_avg值记录到self.clt里

    def valid(self, data_loader, model, loss_f, epoch_idx):
        model.eval()
        with torch.no_grad():
            loss_batch_list = []
            acc_batch_list = []
            ocean_acc_epoch = []
            epoch_total_acc = 0
            epoch_total_num = 0
            epo_iter_num = len(data_loader)
            pred_list, label_list, pred_list2, label_list2 = [], [], [], []
            for i, data in enumerate(data_loader):
                inputs, labels = self.data_fmt(data)
                outputs = model(*inputs)
                loss = loss_f(outputs.cpu(), labels.cpu().float())
                
                print(torch.cat([outputs, labels], dim=1))
                print(f'*********** Epo[{(epoch_idx+1):0>3}/{self.cfg.MAX_EPOCH:0>3}] Iter[{(i + 1):0>3}/{epo_iter_num:0>3}], val loss:{loss.item():.4f} *********** \n')
                
                loss_batch_list.append(loss.item())
                
                outputs = outputs.detach().cpu()
                labels = labels.detach().cpu()
                
                #### 计算acc
                batch_total_acc = (outputs.argmax(dim=-1) == labels.argmax(dim=-1)).sum() # 当前batch里分类预测正确的样本数
                batch_total_num = outputs.shape[0] # 当前batch的样本总数
                batch_acc = batch_total_acc / batch_total_num # 当前batch的acc
                epoch_total_acc += batch_total_acc
                epoch_total_num += batch_total_num
                epoch_current_acc = epoch_total_acc / epoch_total_num
                
                #### 计算AUC
                labels = labels.type(torch.int64)
                pred_list.extend(outputs[:, 0].tolist()) # 比pred_list2更准确
                label_list.extend(labels[:, 0].tolist()) # 比label_list2更准确
                pred_list2.extend(outputs.argmax(dim=-1).tolist())
                label_list2.extend(labels.argmax(dim=-1).tolist())
                batch_auc = auroc(outputs[:, 0], labels[:, 0], task="binary")
                batch_auc2 = auroc(outputs.argmax(dim=-1).type(torch.float32), labels.argmax(dim=-1), task="binary")
                
                #### 计算指标：F1
                f1 = self.f1_metric(outputs[:, 0], labels[:, 0]) # 最好用这个，详见：https://www.notion.so/MPhil-Project-b3de240fa9d64832b26795439d0142d9?pvs=4#3820c783cbc947cb83c1c5dd0d91434b
                f1_2 = self.f1_metric(outputs.argmax(dim=-1), labels.argmax(dim=-1)) # 不准确

                if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1:
                    self.logger.info(
                        "Valid: Epo[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] LOSS: {:.4f} Batch ACC:{:.4f} ({}/{}) Epo Current ACC:{:.4f} ({}/{}) AUC:{:.4f} ({:.4f}) F1:{:.4f} ({:.4f}) \n".format(
                            epoch_idx + 1, self.cfg.MAX_EPOCH,   # Epo
                            i + 1, epo_iter_num,                 # Iter
                            float(loss.item()),                  # LOSS
                            float(batch_acc), batch_total_acc, batch_total_num,  # Batch ACC
                            epoch_current_acc, epoch_total_acc, epoch_total_num, # Epo Current ACC
                            batch_auc, batch_auc2, # AUC
                            f1, f1_2, # F1
                        ))
                if self.cfg.USE_WANDB:
                    wandb.log({
                        "valid_loss": float(loss.item()),
                        "valid_acc": float(batch_acc),
                        "valid_epoch_current_acc": float(epoch_current_acc),
                        "valid_batch_auc": float(batch_auc),
                        "valid_batch_auc2": float(batch_auc2),
                        "valid_batch_f1": float(f1),
                        "valid_batch_f1_2": float(f1_2),
                        "epoch": epoch_idx + 1,
                    })
                acc_batch_list.append(batch_acc)
            self.tb_writer.add_scalar("valid_acc", batch_acc, epoch_idx)
        
        epoch_summary_acc = epoch_total_acc / epoch_total_num # 当前epoch的总体acc
        self.clt.record_valid_loss(loss_batch_list) # loss over batches
        self.clt.record_valid_acc(acc_batch_list)  # acc over batches
        if epoch_summary_acc > self.clt.best_valid_acc: # 如果当前的epoch_summary_acc大于之前的最好的epoch_summary_acc
            self.clt.update_best_acc(epoch_summary_acc)
            self.clt.update_model_save_flag(1)   # 1表示需要保存模型
            print(f'Valid: Current epoch summary acc:{epoch_summary_acc:.4f} > best epoch summary acc: {self.clt.best_valid_acc:.4f}, will save model')
        else:
            print(f'Valid: Current epoch summary acc:{epoch_summary_acc:.4f} <= best epoch summary acc: {self.clt.best_valid_acc:.4f}, not save model')
            self.clt.update_model_save_flag(0)  # 0表示不需要保存模型
        
        #### 计算指标：AUC
        epoch_auc = auroc(torch.tensor(pred_list), torch.tensor(label_list), task="binary")
        epoch_auc2 = auroc(torch.tensor(pred_list2).type(torch.float32), torch.tensor(label_list2), task="binary")
        
        #### 计算指标：F1 score
        f1 = self.f1_metric(torch.tensor(pred_list), torch.tensor(label_list))
        f1_2 = self.f1_metric(torch.tensor(pred_list2), torch.tensor(label_list2))
        
        self.logger.info(
            "Valid: Epo[{:0>3}/{:0>3}] Epo Summary Acc:{:.4f} ({}/{}) Epo AUC: {:.4f} ({:.4f}) Epo F1_Score: {:.4f} ({:.4f})\n".
            format(
                epoch_idx + 1, self.cfg.MAX_EPOCH, # Epoch
                epoch_summary_acc, epoch_total_acc, epoch_total_num, # Epo Summary Acc
                epoch_auc, epoch_auc2, # Epo AUC
                f1, f1_2, # Epo F1_Score
            ))
        if self.cfg.USE_WANDB:
            wandb.log({
                # "valid_acc_batch_avg": acc_batch_avg,
                # "valid_ocean_acc_avg": ocean_acc_avg,
                # "Train Mean_Acc": float(self.clt.epoch_train_acc),
                # "Valid Mean_Acc": float(self.clt.epoch_valid_acc),
                "val_epoch_summary_acc": float(epoch_summary_acc),
                "val_epoch_summary_auc": float(epoch_auc),
                "val_epoch_summary_auc2": float(epoch_auc2),
                "val_epoch_summary_f1": float(f1),
                "val_epoch_summary_f1_2": float(f1_2),
                "epoch": epoch_idx + 1,
            })

    def test(self, data_loader, model, epoch_idx=None):
        model.eval()
        with torch.no_grad(): # 不计算梯度 也不更新参数，因为是测试阶段，不需要更新参数，只需要计算loss，计算acc
            # label_list = [] # 用于存储真实标签label
            # output_list = [] # 用于存储模型的输出output
            total_acc = 0
            total_num = 0
            test_acc = 0
            pred_list, label_list, pred_list2, label_list2 = [], [], [], []
            for data in tqdm(data_loader): # 遍历data_loader
                inputs, labels = self.data_fmt(data)
                outputs = model(*inputs)
                # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainerUdiva.test() 正在test... outputs=', outputs, 'labels=', labels, ' inputs[0].size()=', inputs[0].size(), ' inputs[1].size()=', inputs[1].size())
                outputs = outputs.detach().cpu()
                labels = labels.detach().cpu()
                
                print(torch.cat([outputs, labels], dim=1))
                print(f'*********** Test outputs and labels ***********')
                
                #### 计算ACC
                total_acc += (outputs.argmax(dim=-1) == labels.argmax(dim=-1)).sum()
                total_num += outputs.shape[0]
                test_acc = total_acc / total_num 
                
                #### 计算AUC
                labels = labels.type(torch.int64)
                pred_list.extend(outputs[:, 0].tolist()) # 比pred_list2更准确
                label_list.extend(labels[:, 0].tolist()) # 比label_list2更准确
                pred_list2.extend(outputs.argmax(dim=-1).tolist())
                label_list2.extend(labels.argmax(dim=-1).tolist())
                
                if epoch_idx is not None:
                    self.logger.info(f"Test: Epo of Train[{(epoch_idx + 1):0>3}/{ self.cfg.MAX_EPOCH:0>3}] Tqdm Current test_acc:{test_acc} ({total_acc}/{total_num})")
                else:
                    self.logger.info(f"Test: Tqdm Current test_acc:{test_acc:.4f} ({total_acc}/{total_num})")
                
                # output_list.append(outputs)
                # label_list.append(labels)
            
            # dataset_output = torch.cat(output_list, dim=0).numpy() # [val_batch_size, 2]
            # dataset_label = torch.cat(label_list, dim=0).numpy() # [val_batch_size, 2]] 
            # dataset_output= [[0.5023817  0.49859118]
            #                 [0.502379   0.49847925]]
            # dataset_label= [[1. 0.]
            #                 [0. 1.]]
        self.tb_writer.add_scalar("test_acc", test_acc)
        
        #### 计算指标：AUC
        epoch_auc = auroc(torch.tensor(pred_list), torch.tensor(label_list), task="binary")
        epoch_auc2 = auroc(torch.tensor(pred_list2).type(torch.float32), torch.tensor(label_list2), task="binary")
        
        #### 计算指标：F1 score
        f1 = self.f1_metric(torch.tensor(pred_list), torch.tensor(label_list))
        f1_2 = self.f1_metric(torch.tensor(pred_list2), torch.tensor(label_list2))
        
        if epoch_idx is not None:
            self.logger.info("Test: Epo of Train[{:0>3}/{:0>3}] Epo Summary Acc:{:.4f} ({}/{}) Epo AUC: {:.4f} ({:.4f}) Epo F1_Score: {:.4f} ({:.4f})\n".format(
                                epoch_idx + 1, self.cfg.MAX_EPOCH, # Epoch
                                test_acc, total_acc, total_num,    # Epo ACC
                                epoch_auc, epoch_auc2, # Epo AUC
                                f1, f1_2, # Epo F1_Score
                            ))
            if self.cfg.USE_WANDB:
                wandb.log({
                    "test_acc": float(test_acc),
                    "test_auc": float(epoch_auc),
                    "test_auc2": float(epoch_auc2),
                    "test_f1": float(f1),
                    "test_f1_2": float(f1_2),
                    "epoch": epoch_idx + 1})
        else:
            self.logger.info("Test only: Final Acc:{:.4f} ({}/{}) Final AUC:{:.4f} ({:.4f})\n".format(test_acc, total_acc, total_num, epoch_auc, epoch_auc2))
            if self.cfg.USE_WANDB:
                wandb.log({
                    "test_final_acc": float(test_acc),
                    "test_final_auc": float(epoch_auc),
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
        else:
            return (aud_in, img_in), labels
    
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
