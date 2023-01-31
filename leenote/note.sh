# Udiva测试命令
mac本地：
    conda activate DeepPersonality && cd /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality && python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18.yaml
    # test阶段中断的解决办法：find . -name ".DS_Store" -delete  # https://github.com/fastai/fastai/issues/488

远程：
    conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/
    conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/ && python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva.yaml
    
    #################### udiva tiny数据集 ####################
    alias rundeep='conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/ && python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva.yaml'
    # 后台跑命令
    nohup python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva.yaml --max_epoch 100 >nohup.out 2>&1 &

    #################### udiva full数据集 全量数据集 ####################
    alias rundeepfull='conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/ && python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml'

# epochs=10, batch_size=16
nohup python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml --max_epoch 10 --bs 16 >nohup_full_epo10_bs16.out 2>&1 &
sshcamhpc: [1] 64305 done https://wandb.ai/hyllbd-1009694687/DeepPersonality/runs/pep79rxl?workspace=user-1009694687

# epochs=100, batch_size=16
nohup python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml --max_epoch 100 --bs 16 >nohup_full_epo100_bs16_01302355.out 2>&1 &
# sshhpccpu login-e-10: [1] 22987 running
nohup python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml --max_epoch 100 --bs 16 >nohup_full_epo100_bs16_01310021.out 2>&1 &
    # COMPUTERLAB-SL3-CPU -p skylake: zl525@cpu-e-824 [1] 244128 running


# epochs=150, batch_size=16
nohup python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml --max_epoch 150 --bs 16 >nohup_full_epo150_bs16_01302355.out 2>&1 &
    # sshcamhpc: login-q-4 [1] 87489 running

# epochs=10, batch_size=32
nohup python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml --max_epoch 10 --bs 32 >nohup_full_epo10_bs32.out 2>&1 &

# epochs=10, batch_size=48
nohup python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml --max_epoch 10 --bs 48 >nohup_full_epo10_bs48.out 2>&1 &




####### 提交任务到HPC上 在GPU上跑 #######
    # 参考/home/zl525/code/DeepPersonality/leenote/slurm_submit_deep 文件的最后一行
    cd /home/zl525/code/DeepPersonality/leenote && sbatch slurm_submit_deep

    # 运行记录：
    Submitted batch job 13703460

####### 提交任务到HPC的CPU上 在CPU上跑 #######
    cd /home/zl525/code/DeepPersonality/leenote && sbatch slurm_submit_deep.peta4-skylake

    # 运行记录：
    Submitted batch job 13703745


# 替换print，将下面的两行的第一行替换成第二行即可注释所有print信息
  print('[
  # print('[

但是不要替换Exception这里的print信息
    except Exception as e:
        printxx
解决办法: 在print后面加上return None

# 创建数据集软链接
    ln -s [源文件或目录] [目标文件或目录] # https://www.cnblogs.com/sueyyyy/p/10985443.html
    例子：当前路径创建test 引向/var/www/test 文件夹: ln –s  /var/www/test  test
    ln -s /home/zl525/rds/hpc-work/datasets/udiva_tiny /home/zl525/code/DeepPersonality/datasets/
    ln -s /home/zl525/rds/hpc-work/datasets/udiva_full /home/zl525/code/DeepPersonality/datasets/ 


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
        