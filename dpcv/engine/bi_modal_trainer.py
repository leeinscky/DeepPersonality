import torch
from tqdm import tqdm
import numpy as np
import math
import os
from .build import TRAINER_REGISTRY
from torch.utils.tensorboard import SummaryWriter
import time
import wandb


@TRAINER_REGISTRY.register()
class BiModalTrainer(object):
    """base trainer for bi-modal input"""
    def __init__(self, cfg, collector, logger):
        print('[DeepPersonality/dpcv/engine/bi_modal_trainer.py] 开始执行BiModal模型的初始化 BiModalTrainer.__init__() ')
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clt = collector
        self.logger = logger
        self.tb_writer = SummaryWriter(cfg.OUTPUT_DIR)
        print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] 结束执行BiModal模型的初始化 BiModalTrainer.__init__()')

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
        
        print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 开始执行BiModal模型的train方法')
        lr = optimizer.param_groups[0]['lr']
        self.logger.info(f"Training: learning rate:{lr}")
        self.tb_writer.add_scalar("lr", lr, epoch_idx)

        model.train()
        loss_list = []
        acc_avg_list = []
        print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... len(data_loader)=', len(data_loader), 'type(data_loader): ', type(data_loader)) # len(data_loader)= 7 type(data_loader):  <class 'torch.utils.data.dataloader.DataLoader'>
        final_i = 0 
        # 通过 看日志发现当执行下面这行 for i, data in enumerate(data_loader)语句时，会调用 AudioVisualData(VideoData)类里的 __getitem__ 函数，紧接着调用def get_ocean_label()函数， 具体原因参考：https://www.geeksforgeeks.org/how-to-use-a-dataloader-in-pytorch/
        for i, data in enumerate(data_loader): # len(data_loader)= 7 type(data_loader):  <class 'torch.utils.data.dataloader.DataLoader'> 一共有7个batch, 每个batch的大小是8, 所以一共有56个样本。
            final_i = i
            print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在训练... i=', i)
            if i == 0:
                print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i, '  data.keys()=', data.keys()) # data.keys()= dict_keys(['image', 'audio', 'label'])
                print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i,  'data[image]的size: ', data['image'].size()) # torch.Size([8, 3, 224, 224]) 8对应config里的batch_size，即8张帧image，3对应RGB，224对应图片的大小，3x224x224代表图片的大小
                print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i,  'data[audio]的size: ', data['audio'].size()) # torch.Size([8, 1, 1, 50176]) 8对应config里的batch_size，即8个wav音频，1对应1个channel，50176对应音频的长度，1x1x50176代表音频的大小，1代表channel，1x50176代表音频的长度
                print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i,  'data[label]的size: ', data['label'].size(),'  data[label]=', data['label']) # torch.Size([8, 5]) 8对应config里的batch_size，5对应5个维度的personality
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
        print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] 训练结束, data_loader里的元素个数为: final_i=', final_i)
        self.clt.record_train_loss(loss_list)
        self.clt.record_train_acc(acc_avg_list)

    def valid(self, data_loader, model, loss_f, epoch_idx):
        model.eval()
        with torch.no_grad():
            loss_batch_list = []
            acc_batch_list = []
            ocean_acc_epoch = []
            for i, data in enumerate(data_loader):
                print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i)
                if i == 0:
                    print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i, '  data.keys()=', data.keys()) # data.keys()= dict_keys(['image', 'audio', 'label'])
                    print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i,  'data[image]的size: ', data['image'].size()) # torch.Size([4, 3, 224, 224]) 4对应config里的BATCH_SIZE
                    print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i,  'data[audio]的size: ', data['audio'].size()) # torch.Size([4, 1, 1, 50176]) 4对应config里的BATCH_SIZE
                    print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i,  'data[label]的size: ', data['label'].size(),'  data[label]=', data['label']) # torch.Size([4, 5]) 4对应config里的BATCH_SIZE
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
    """base trainer for bi-modal input"""
    def __init__(self, cfg, collector, logger):
        print('[DeepPersonality/dpcv/engine/bi_modal_trainer.py] 开始执行BiModal模型的初始化 BiModalTrainer.__init__() ')
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clt = collector
        self.logger = logger
        self.tb_writer = SummaryWriter(cfg.OUTPUT_DIR)
        print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] 结束执行BiModal模型的初始化 BiModalTrainer.__init__()')

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
        
        print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 开始执行BiModal模型的train方法')
        lr = optimizer.param_groups[0]['lr']
        self.logger.info(f"Training: learning rate:{lr}")
        self.tb_writer.add_scalar("lr", lr, epoch_idx)

        model.train()
        loss_list = []
        acc_avg_list = []
        print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... len(data_loader)=', len(data_loader), 'type(data_loader): ', type(data_loader)) # len(data_loader)= 7 type(data_loader):  <class 'torch.utils.data.dataloader.DataLoader'>
        final_i = 0 
        # 通过 看日志发现当执行下面这行 for i, data in enumerate(data_loader)语句时，会调用 AudioVisualData(VideoData)类里的 __getitem__ 函数，紧接着调用def get_ocean_label()函数， 具体原因参考：https://www.geeksforgeeks.org/how-to-use-a-dataloader-in-pytorch/
        for i, data in enumerate(data_loader): # len(data_loader)= 7 type(data_loader):  <class 'torch.utils.data.dataloader.DataLoader'> 一共有7个batch, 每个batch的大小是8, 所以一共有56个样本。
            final_i = i
            print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i)
            if i == 0:
                print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i, '  data.keys()=', data.keys()) # data.keys()= dict_keys(['image', 'audio', 'label'])
                print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i,  'data[image]的size: ', data['image'].size()) # torch.Size([batch_size, 3*2(由于torch.cat)=6, 224, 224])
                print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i,  'data[audio]的size: ', data['audio'].size()) # torch.Size([batch_size, 1, 1*2(由于torch.cat)=2, 50176]) 8对应config里的batch_size，即8个wav音频，1对应1个channel，50176对应音频的长度，1x1x50176代表音频的大小，1代表channel，1x50176代表音频的长度
                print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i,  'data[label]的size: ', data['label'].size(),'  data[label]=', data['label']) # torch.Size([1(batch_size), 1]) 8对应config里的batch_size，1对应1个关系的标签
                # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() i=', i, '  data=', data)
            epo_iter_num = len(data_loader)
            iter_start_time = time.time()

            # self.data_fmt(data) 代表将data里的image, audio, label分别取出来，放到inputs，label里
            inputs, labels = self.data_fmt(data) # inputs是一个元组，包含了image (data['image']: torch.Size([batchsize, 6, 224, 224])) 和audio(data['audio']: torch.Size([batchsize, 1, 2, 50176]))的输入，labels: torch.Size([1(batch_size), 1])
            
            # ret = self.data_fmt(data)
            # fc1_input = ret['fc1_in']
            # fc2_input = ret['fc2_in']
            # labels = ret['label']

            # 这里该怎么处理一对视频融合？我们在数据预处理时就将一对视频的数据进行了早期融合，所以这里不需要再对两个视频分支进行融合了，模型输出的结果是融合后的结果
            outputs = model(*inputs) # 加一个*星号：表示参数数量不确定，将传入的参数存储为元组（https://blog.csdn.net/qq_42951560/article/details/112006482）。*inputs意思是将inputs里的元素分别取出来，作为model的输入参数，这里的inputs是一个元组，包含了image和audio。models里的forward函数里的参数是image和audio，所以这里的*inputs就是将image和audio分别取出来，作为model的输入参数。为什么是forward函数的参数而不是__init__函数的参数？因为forward函数是在__init__函数里被调用的，所以forward函数的参数就是__init__函数的参数。forward 会自动被调用，调用时会传入输入数据，所以forward函数的参数就是输入数据。
            print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i, '  outputs=', outputs, 'labels=', labels, ' outputs.size()', outputs.size(),  '  labels.size()=', labels.size())
            # fc1_output = model(*fc1_input)
            # fc2_output = model(*fc2_input)
            # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i, '  fc1_output=', fc1_output, '  fc2_output', fc2_output, '  labels.size()=', labels.size())
            
            optimizer.zero_grad() # 梯度清零，即将梯度变为0  # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005), SGD即随机梯度下降, lr是学习率，momentum是动量，weight_decay是权重衰减, 全称是Stochastic Gradient Descent，即随机梯度下降，即每次迭代时，随机选取一个batch的样本来进行梯度下降，而不是像梯度下降那样，每次迭代时，使用所有的样本来进行梯度下降，这样会使得每次迭代的计算量变大，而且每次迭代的结果也不稳定，而随机梯度下降则可以避免这个问题，但是随机梯度下降的结果也不稳定，因此，可以结合动量和权重衰减来使得随机梯度下降的结果更加稳定，即动量是为了使得每次迭代的结果更加稳定，而权重衰减是为了防止过拟合。
            loss = loss_f(outputs.cpu(), labels.cpu()) # loss_f = nn.MSELoss()  即 mean_square_error，即均方误差，即预测值与真实值的差的平方的均值，即 (y_pred - y_true)^2，其中y_pred是预测值，y_true是真实值
            print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i, '  loss=', loss)
            self.tb_writer.add_scalar("loss", loss.item(), i) # loss.item()是将loss转换成float类型，即tensor转换成float类型，add_scalar()是将loss写入到tensorboard里, i是第
            loss.backward() # loss.backward()是将loss反向传播，即计算loss对每个参数的梯度，即loss对每个参数的偏导数
            print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i, '  经过backward函数后, loss=', loss, '  loss.item()=', loss.item())
            optimizer.step() # step()是将梯度更新到参数上，即将loss对每个参数的偏导数更新到参数上， 即 w = w - lr * gradient, 其中lr是学习率，gradient是loss对每个参数的偏导数，即loss对每个参数的梯度， w是模型的参数

            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time

            loss_list.append(loss.item())
            acc_avg = (1 - torch.abs(outputs.cpu() - labels.cpu())).mean().clip(min=0) # torch.abs 计算 tensor 的每个元素的绝对值, torch.abs(outputs.cpu() - labels.cpu())是预测值与真实值的差的绝对值, 即预测值与真实值的差的绝对值的平均值即为acc_avg, 即acc_avg = (1 - torch.abs(outputs.cpu() - labels.cpu())).mean(), clip(min=0)是将acc_avg的最小值限制在0以上
            print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i, '  acc_avg=', acc_avg)
            acc_avg = acc_avg.detach().numpy() # detach()是将acc_avg从计算图中分离出来后，再转换成numpy类型的float类型，即tensor转换成float类型
            print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i, ' 经过acc_avg.detach().numpy()后, acc_avg=', acc_avg)
            acc_avg_list.append(acc_avg)
            print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.train() 正在训练... i=', i, '  acc_avg_list=', acc_avg_list)

            # log to wandb
            wandb.log({
                "train_loss":  float(loss.item()),
                "train_acc": float(acc_avg),
                "epoch": epoch_idx + 1,
            })

            # print loss and training info for an interval
            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1: #  self.cfg.LOG_INTERVAL = 10，即每10个batch打印一次loss和acc_avg
                remain_iter = epo_iter_num - i # epo_iter_num(一个epoch里的batch数)-当前的batch数=剩余的batch数
                remain_epo = self.cfg.MAX_EPOCH - epoch_idx 
                eta = (epo_iter_num * iter_time) * remain_epo + (remain_iter * iter_time) # eta是剩余的时间，即剩余的epoch数乘以一个epoch的时间加上剩余的batch数乘以一个batch的时间
                eta = int(eta) # 将eta转换成int类型
                eta_string = f"{eta // 3600}h:{eta % 3600 // 60}m:{eta % 60}s"  # 将eta转换成时分秒的形式
                self.logger.info(
                    "Train: Epo[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] IterTime:[{:.2f}s] LOSS: {:.4f} ACC:{:.4f} ETA:{} ".format(
                        epoch_idx + 1, self.cfg.MAX_EPOCH,   # Epo 
                        i + 1, epo_iter_num,                 # Iter
                        iter_time,                           # IterTime
                        float(loss.item()), float(acc_avg),  # LOSS ACC 
                        eta_string,                          # ETA
                    )
                )
        print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] 训练结束, data_loader里的元素个数为: final_i=', final_i)
        self.clt.record_train_loss(loss_list) # 将loss_list里的loss值记录到self.clt里
        self.clt.record_train_acc(acc_avg_list) # 将acc_avg_list里的acc_avg值记录到self.clt里

    def valid(self, data_loader, model, loss_f, epoch_idx):
        model.eval()
        with torch.no_grad():
            loss_batch_list = []
            acc_batch_list = []
            ocean_acc_epoch = []
            for i, data in enumerate(data_loader):
                print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i)
                if i == 0:
                    print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i,  'data.keys()=', data.keys()) # data.keys()= dict_keys(['image', 'audio', 'label'])
                    print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i,  'data[image]的size: ', data['image'].size()) # torch.Size([4, 3, 224, 224]) 4对应config里的BATCH_SIZE
                    print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i,  'data[audio]的size: ', data['audio'].size()) # torch.Size([4, 1, 1, 50176]) 4对应config里的BATCH_SIZE
                    print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainer.valid() 正在valid... i=', i,  'data[label]的size: ', data['label'].size(),'  data[label]=', data['label']) # torch.Size([4, 5]) 4对应config里的BATCH_SIZE
                inputs, labels = self.data_fmt(data)
                outputs = model(*inputs)
                loss = loss_f(outputs.cpu(), labels.cpu())
                loss_batch_list.append(loss.item())
                ocean_acc_batch = ( # ocean acc on a batch
                    1 - torch.abs(outputs.cpu().detach() - labels.cpu().detach())
                ).mean(dim=0).clip(min=0)
                ocean_acc_epoch.append(ocean_acc_batch)
                acc_batch_avg = ocean_acc_batch.mean()
                acc_batch_list.append(acc_batch_avg)
            ocean_acc = torch.stack(ocean_acc_epoch, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images
            ocean_acc_avg = ocean_acc.mean()
            self.tb_writer.add_scalar("valid_acc", ocean_acc_avg, epoch_idx)
        self.clt.record_valid_loss(loss_batch_list) # loss over batches
        self.clt.record_valid_acc(acc_batch_list)  # acc over batches
        self.clt.record_valid_ocean_acc(ocean_acc) # ocean acc over all valid data
        if ocean_acc_avg > self.clt.best_valid_acc: # 如果当前的ocean_acc_avg大于之前的最好的ocean_acc_avg
            self.clt.update_best_acc(ocean_acc_avg)
            self.clt.update_model_save_flag(1)   # 1表示需要保存模型
        else:
            self.clt.update_model_save_flag(0)  # 0表示不需要保存模型

        self.logger.info(
            "Valid: Epoch[{:0>3}/{:0>3}] Train Mean_Acc: {:.2%} Valid Mean_Acc:{:.2%} OCEAN_ACC:{}\n".
            format(
                epoch_idx + 1, self.cfg.MAX_EPOCH, # Epoch
                float(self.clt.epoch_train_acc),   # Train Mean_Acc
                float(self.clt.epoch_valid_acc),   # Valid Mean_Acc
                self.clt.valid_ocean_acc)          # OCEAN_ACC
        )

        # log to wandb
        wandb.log({
            "valid_loss": loss.item(),
            "valid_acc_batch_avg": acc_batch_avg,
            "valid_ocean_acc_avg": ocean_acc_avg,
            "Train Mean_Acc": float(self.clt.epoch_train_acc),
            "Valid Mean_Acc": float(self.clt.epoch_valid_acc),
            "epoch": epoch_idx + 1,
        })

    def test(self, data_loader, model):
        mse_func = torch.nn.MSELoss(reduction="none") # MSE损失函数, reduction="none"表示不对损失求均值
        model.eval()
        with torch.no_grad(): # 不计算梯度 也不更新参数，因为是测试阶段，不需要更新参数，只需要计算loss，计算acc，计算ocean_acc，计算ocean_mse... 
            mse_ls = []  # 用于存储mse
            ocean_acc = [] # 用于存储ocean_acc
            label_list = [] # 用于存储label
            output_list = [] # 用于存储output
            count = 0
            for data in tqdm(data_loader): # 遍历data_loader
                inputs, labels = self.data_fmt(data)
                outputs = model(*inputs)
                print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainerUdiva.test() 正在test... outputs=', outputs, 'labels=', labels, ' inputs[0].size()=', inputs[0].size(), ' inputs[1].size()=', inputs[1].size())

                outputs = outputs.cpu().detach()
                labels = labels.cpu().detach()
                output_list.append(outputs)
                label_list.append(labels)
                mse = mse_func(outputs, labels).mean(dim=0) #torch.nn.MSELoss(reduction="none") 表示不对损失求均值，所以这里需要再求一次均值  MSE全称是均方误差，是回归问题中常用的损失函数，其计算公式为：MSE=1/n∑(y-y')^2，其中y是真实值，y'是预测值，n是样本的个数， MSE越小，说明模型的预测效果越好，MSE越大，说明模型的预测效果越差，MSE的取值范围是[0,+∞)，MSE越接近0，说明模型的预测效果越好，MSE越接近+∞，说明模型的预测效果越差。
                ocean_acc_batch = (1 - torch.abs(outputs - labels)).mean(dim=0).clip(min=0)
                print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainerUdiva.test() 正在test... mse=', mse, 'ocean_acc_batch=', ocean_acc_batch)
                mse_ls.append(mse)
                ocean_acc.append(ocean_acc_batch)
                count += 1
                # # log to wandb
                # wandb.log({
                #     "test_mse": mse,
                #     "test_ocean_acc_batch": ocean_acc_batch,
                #     "test_iteration": count,
                # })
            print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainerUdiva.test() 正在test... ocean_acc =', ocean_acc) # ocean_acc = [tensor([0.5024, 0.5014]), tensor([0.4976, 0.4985])]
            ocean_mse = torch.stack(mse_ls, dim=0).mean(dim=0).numpy() # torch.stack() 将mse_ls中的每个元素都堆叠起来，dim=0表示按照第0维进行堆叠，即按照行堆叠，最后的结果是一个矩阵，矩阵的行数是mse_ls中元素的个数，列数是每个元素的列数
            ocean_acc = torch.stack(ocean_acc, dim=0).mean(dim=0).numpy()  # ocean acc on all valid images 
            print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainerUdiva.test() 正在test... after .mean(dim=0).numpy(), ocean_mse=', ocean_mse, ' ocean_acc=', ocean_acc) # after .mean(dim=0).numpy(), ocean_mse= [0.25000432 0.2500581 ]  ocean_acc= [0.5000013  0.49994403]
            ocean_mse_mean = ocean_mse.mean() 
            ocean_acc_avg = ocean_acc.mean()
            dataset_output = torch.cat(output_list, dim=0).numpy()
            dataset_label = torch.cat(label_list, dim=0).numpy()
            # print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainerUdiva.test() 正在test... ocean_mse=', ocean_mse, '  ocean_mse_mean=',ocean_mse_mean, '  ocean_acc=', ocean_acc, ' ocean_acc_avg=', ocean_acc_avg, '\n dataset_output=', dataset_output, '\n dataset_label=', dataset_label)
            # ocean_mse= [0.25000432 0.2500581] ocean_mse_mean= 0.25003123   ocean_acc= [0.5000013  0.49994403]  ocean_acc_avg= 0.49997267
            # dataset_output= [[0.5023817  0.49859118]
            #                 [0.502379   0.49847925]]
            # dataset_label= [[1. 0.]
            #                 [0. 1.]]
        ocean_mse_mean_rand = np.round(ocean_mse_mean, 4) # round to 4 decimal places , 保留4位小数, 参考：https://www.geeksforgeeks.org/numpy-round_-python/
        ocean_acc_avg_rand = np.round(ocean_acc_avg.astype("float64"), 4)
        print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainerUdiva.test() 正在test... ocean_mse_mean_rand=', ocean_mse_mean_rand, '  ocean_acc_avg_rand=', ocean_acc_avg_rand) # ocean_mse_mean_rand= 0.25   ocean_acc_avg_rand= 0.5

        self.tb_writer.add_scalar("test_acc", ocean_acc_avg_rand)
        # wandb.log({
        #     "test_ocean_mse_mean_rand": ocean_mse_mean_rand, # 遍历data_loader时每次迭代得到的所有均方误差，然后求平均计算得到平均值
        #     "test_ocean_acc_avg_rand": ocean_acc_avg_rand, # 遍历data_loader时每次迭代得到的所有预测准确率，然后求平均计算得到平均值
        # })
        keys = ["known_label", "unknown_label"]
        ocean_mse_dict, ocean_acc_dict = {}, {}
        for i, k in enumerate(keys):
            print('当前循环 i=', i, '  k=', k, '  ocean_mse[i]=', ocean_mse[i], '  ocean_acc[i]=', ocean_acc[i])
            ocean_mse_dict[k] = np.round(ocean_mse[i], 4)
            ocean_acc_dict[k] = np.round(ocean_acc[i], 4)
        print('[deeppersonality/dpcv/engine/bi_modal_trainer.py] BiModalTrainerUdiva.test() 正在test... ocean_mse_dict=', ocean_mse_dict, '  ocean_acc_dict=', ocean_acc_dict)
        
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
        # # 处理audio_visual_data_udiva.py的__getitem__返回的sample
        # fc1_img_in, fc1_aud_in = data["fc1_image"], data["fc1_audio"]
        # fc2_img_in, fc2_aud_in = data["fc2_image"], data["fc2_audio"]

        # ret = {}
        # ret['fc1_in'] = (fc1_aud_in, fc1_img_in)
        # ret['fc2_in'] = (fc2_aud_in, fc2_img_in)
        # ret['label'] = data["label"]
        # return ret

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
