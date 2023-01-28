# Udiva测试命令
    mac本地：
        conda activate DeepPersonality && cd /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality && python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18.yaml
        # test阶段中断的解决办法：find . -name ".DS_Store" -delete  # https://github.com/fastai/fastai/issues/488

    远程：
        conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/ && python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva.yaml

    img_dir_ls 打印结果：
        [DeepPersonality/dpcv/data/datasets/bi_modal_data.py] self.img_dir_ls:  
        [
            'datasets/ChaLearn2016_tiny/train_data/-AmMDnVl4s8.003', 
            'datasets/ChaLearn2016_tiny/train_data/2kqPuht5jTg.002', 
            'datasets/ChaLearn2016_tiny/train_data/4CSV8L7aVik.000', 
            'datasets/ChaLearn2016_tiny/train_data/50gokPvvMs8.000', 
            'datasets/ChaLearn2016_tiny/train_data/6KKNrufnL80.000', 
            'datasets/ChaLearn2016_tiny/train_data/83cmR2fkyy8.005', 
            'datasets/ChaLearn2016_tiny/train_data/98fnGDVky00.005', 
            'datasets/ChaLearn2016_tiny/train_data/9KAqOrdiZ4I.002', 
            'datasets/ChaLearn2016_tiny/train_data/9hqH1PJ6cG8.001', 
            'datasets/ChaLearn2016_tiny/train_data/A3StIKMjn4k.002', 
            'datasets/ChaLearn2016_tiny/train_data/BWAEpai6FK0.003', 
            'datasets/ChaLearn2016_tiny/train_data/C_NtwmmF2Ys.000', 
            'datasets/ChaLearn2016_tiny/train_data/DnTtbAR_Qyw.004', 
            'datasets/ChaLearn2016_tiny/train_data/F0_EI_X5JVk.003', 
            'datasets/ChaLearn2016_tiny/train_data/HegkSmkiBos.005', 
            'datasets/ChaLearn2016_tiny/train_data/JBdLI6AhJrw.000', 
            'datasets/ChaLearn2016_tiny/train_data/JIYZTruMpiI.003', 
            'datasets/ChaLearn2016_tiny/train_data/JiXJeI5_jGM.000', 
            'datasets/ChaLearn2016_tiny/train_data/KJ643kfjqLY.003', 
            'datasets/ChaLearn2016_tiny/train_data/L9sG80PI1Gw.003', 
            'datasets/ChaLearn2016_tiny/train_data/L_gmlaz-0s4.003', 
            'datasets/ChaLearn2016_tiny/train_data/MOXPVzRBDPo.002', 
            'datasets/ChaLearn2016_tiny/train_data/NDBCrVvp0Vg.003', 
            'datasets/ChaLearn2016_tiny/train_data/OWZ-qVZG14A.002', 
            'datasets/ChaLearn2016_tiny/train_data/Q2AI4XpApFs.002', 
            'datasets/ChaLearn2016_tiny/train_data/Qz_cjgCtDcM.003', 
            'datasets/ChaLearn2016_tiny/train_data/RlUuWWWFrhM.005', 
            'datasets/ChaLearn2016_tiny/train_data/T6CMGXdPUTA.001', 
            'datasets/ChaLearn2016_tiny/train_data/TPk6KiHuPag.004', 
            'datasets/ChaLearn2016_tiny/train_data/Tr3A7WODEuM.001', 
            'datasets/ChaLearn2016_tiny/train_data/Uu-NbXUPr-A.001', 
            'datasets/ChaLearn2016_tiny/train_data/W0FCCk0a0tg.001', 
            'datasets/ChaLearn2016_tiny/train_data/WT1YjeADatU.001', 
            'datasets/ChaLearn2016_tiny/train_data/Yj36y7ELRZE.004', 
            'datasets/ChaLearn2016_tiny/train_data/_uNup91ZYw0.002', 
            'datasets/ChaLearn2016_tiny/train_data/bt-ev53zZWE.004', 
            'datasets/ChaLearn2016_tiny/train_data/dd0z9mErfSo.003', 
            'datasets/ChaLearn2016_tiny/train_data/eD4b8sM-Tpw.000', 
            'datasets/ChaLearn2016_tiny/train_data/eI_7SimPnnQ.001', 
            'datasets/ChaLearn2016_tiny/train_data/geXpIfaFzF4.001', 
            'datasets/ChaLearn2016_tiny/train_data/in-HuMgiDCE.001', 
            'datasets/ChaLearn2016_tiny/train_data/jDdRrqRcSzM.002', 
            'datasets/ChaLearn2016_tiny/train_data/jTkEWnuDnbA.001', 
            'datasets/ChaLearn2016_tiny/train_data/jwcSbw4NDn0.005', 
            'datasets/ChaLearn2016_tiny/train_data/myhEW1aZRg4.000', 
            'datasets/ChaLearn2016_tiny/train_data/n8IiQJyqjiE.003', 
            'datasets/ChaLearn2016_tiny/train_data/nGGtTu6dSJE.000', 
            'datasets/ChaLearn2016_tiny/train_data/nOFHZ_s7Et4.005', 
            'datasets/ChaLearn2016_tiny/train_data/nZz1hK90gwA.004', 
            'datasets/ChaLearn2016_tiny/train_data/okSmKH2k5lE.002', 
            'datasets/ChaLearn2016_tiny/train_data/om-9kFEKJIs.004', 
            'datasets/ChaLearn2016_tiny/train_data/opEoJBrcmbI.002', 
            'datasets/ChaLearn2016_tiny/train_data/vMtF0akNUK4.000', 
            'datasets/ChaLearn2016_tiny/train_data/vr5FWHUkYRM.001', 
            'datasets/ChaLearn2016_tiny/train_data/vrMlwwTLWIE.005', 
            'datasets/ChaLearn2016_tiny/train_data/wTo1uZns2X8.000', 
            'datasets/ChaLearn2016_tiny/train_data/x0CZuHnJ0Hs.005', 
            'datasets/ChaLearn2016_tiny/train_data/yOzHZOg95Ug.003', 
            'datasets/ChaLearn2016_tiny/train_data/yOzHZOg95Ug.005', 
            'datasets/ChaLearn2016_tiny/train_data/yftfxiDNXko.002'
        ]


# 创建数据集软链接
    ln -s [源文件或目录] [目标文件或目录] # https://www.cnblogs.com/sueyyyy/p/10985443.html
    例子：当前路径创建test 引向/var/www/test 文件夹: ln –s  /var/www/test  test
    ln -s /home/zl525/rds/hpc-work/datasets/udiva_tiny /home/zl525/code/DeepPersonality/datasets/ 


# transformer 融入框架
    1.Multimodal-Transformer代码库
        调用模型的关键代码如下：src/train.py
        # 调用模型py文件
        from src import models  # 里面有class MULTModel(nn.Module):类的定义

        # hyp_params.model = 'MULT'  
        model = getattr(models, hyp_params.model+'Model')(hyp_params) # getattr is a built-in function that returns the value of the named attribute of an object. If not found, it returns the default value provided to the function. getattr(object, name[, default]), name is a string, object is the object whose attribute's value is to be returned. (hyp_params) is the argument to be passed to the model class.

        settings = {'model': model,
                    'optimizer': optimizer,
                    'criterion': criterion,
                    'ctc_a2l_module': ctc_a2l_module,
                    'ctc_v2l_module': ctc_v2l_module,
                    'ctc_a2l_optimizer': ctc_a2l_optimizer,
                    'ctc_v2l_optimizer': ctc_v2l_optimizer,
                    'ctc_criterion': ctc_criterion,
                    'scheduler': scheduler}
        model = settings['model']
        
        from torch.utils.data import DataLoader
        dataset = str.lower(args.dataset.strip())
        train_data = get_data(args, dataset, 'train')
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader): # batch_X is a tuple, batch_Y is the label of the sample, batch_META is the metadata of the sample, e.g. the length of the sample. 
            sample_ind, text, audio, vision = batch_X
        
            net = nn.DataParallel(model) if batch_size > 10 else model
            
            # 关键代码如下，通过模型得到预测值preds和隐藏层hiddens，net是一个nn.DataParallel对象，如果batch_size > 10, 则net会把模型复制多份，然后并行计算，最后把结果合并。
            preds, hiddens = net(text, audio, vision) 
            # text.shape= torch.Size([24, 50, 300]) 
            # audio.shape= torch.Size([24, 500, 74]) 
            # vision.shape= torch.Size([24, 500, 35])
        