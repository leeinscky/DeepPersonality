# Udiva测试命令
    mac本地：
        conda activate DeepPersonality && cd /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality && python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18.yaml
        # test阶段中断的解决办法：find . -name ".DS_Store" -delete  # https://github.com/fastai/fastai/issues/488

    远程：
        conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/ && python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva.yaml
        alias rundeep='conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/ && python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva.yaml'
        # 后台跑命令
        nohup python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva.yaml --max_epoch 100 >nohup.out 2>&1 &

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
        