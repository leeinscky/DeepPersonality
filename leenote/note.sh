####### Udiva测试命令 #######
mac本地：
    conda activate DeepPersonality && cd /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality && python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18.yaml
    # test阶段中断的解决办法：find . -name ".DS_Store" -delete  # https://github.com/fastai/fastai/issues/488

远程：
    参考：/home/zl525/code/DeepPersonality/leenote/train.sh
    conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/
    conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/ && python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva.yaml
    
    #################### udiva tiny数据集 ####################
    alias rundeep='conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/ && python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva.yaml'
    # 后台跑命令
    nohup python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva.yaml --max_epoch 100 >nohup.out 2>&1 &

    #################### udiva full数据集 全量数据集 ####################
    alias rundeepfull='conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/ && python ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml'
    cd /home/zl525/code/DeepPersonality/ && python3 ./script/run_exp.py --cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml --max_epoch 1 --bs 32

# 命令模板
conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/ && nohup python3 ./script/run_exp.py \
--cfg_file ./config/demo/bimodal_resnet18_udiva_full.yaml --max_epoch 50 --bs 16 --lr 0.001 >nohup_full_epo50_bs16_`date +'%m-%d-%H:%M:%S'`.out 2>&1 &

####### 提交任务到HPC的CPU上 在CPU上跑 #######
    cd /home/zl525/code/DeepPersonality/leenote && sbatch slurm_submit_deep.peta4-skylake

    # 运行记录：
    Submitted batch job 13891140

####### 提交任务到HPC上 在GPU上跑 #######
    # 参考/home/zl525/code/DeepPersonality/leenote/slurm_submit_deep 文件的最后一行
    cd /home/zl525/code/DeepPersonality/leenote && sbatch slurm_submit_deep

    # 运行记录：

    # slurm_submit_deep_4nodes 提交记录：
        14589749:
            sample_size=144
            sample_size2=160
            sample_size3=176
            sample_size4=192
            sample_size5=16
            sample_size6=32

            batch_size=8
            batch_size2=8
            batch_size3=8
            batch_size4=8
            batch_size5=32
            batch_size6=32

        14590606:
            sample_size=48
            sample_size2=64
            sample_size3=80
            sample_size4=96
            sample_size5=112
            sample_size6=128

            batch_size=32
            batch_size2=32
            batch_size3=32
            batch_size4=32
            batch_size5=32
            batch_size6=32


        14595310: 
            sample_size=8
            sample_size2=16
            sample_size3=32
            sample_size4=48
            sample_size5=64
            sample_size6=80

            batch_size=64
            batch_size2=64
            batch_size3=64
            batch_size4=64
            batch_size5=64
            batch_size6=64

####### 调参总结 #######
参考:
    train loss与test loss结果分析 原文链接：https://blog.csdn.net/ytusdc/article/details/107738749
    train loss 不断下降，test loss不断下降，说明网络仍在学习;
    train loss 不断下降，test loss趋于不变，说明网络过拟合;
    train loss 趋于不变，test loss不断下降，说明数据集100%有问题;
    train loss 趋于不变，test loss趋于不变，说明学习遇到瓶颈，需要减小学习率或批量数目;
    train loss 不断上升，test loss不断上升，说明网络结构设计不当，训练超参数设置不当，数据集经过清洗等问题。

    验证集loss不断上升，训练集loss一直下降：
        - 一般情况下，训练集loss下降，测试集loss上升是因为过拟合问题。20个epoch就出现过拟合，应该是训练样本严重不足或者训练样本相似性过高，建议找一些比较大的开源数据集来跑。
        - 解决办法：打乱数据，增加数据，数据增广，加正则项等等都是方法 https://www.zhihu.com/question/399992175/answer/1271494711

    为什么会发生train loss上升？
        - 数据有坏点 起始学习率 batchsize太小 https://www.zhihu.com/question/396221084/answer/1236081752

    batch size
        - 过小，会导致模型损失波动大，难以收敛，过大时，模型前期由于梯度的平均，导致收敛速度过慢。来源：loss下降很慢……，训练曲线如图，有人帮忙分析一下嘛？卡这很长时间了、非常感谢各位 ? - 搬砖人的回答 - 知乎 https://www.zhihu.com/question/472162326/answer/2308198711

    sample_size=50 bs=8 lr=0.5  val loss 下降收敛了 val acc 上升趋势 auspicious-festival-241  且val acc准确率很高
    sample_size=50 bs=8 lr=0.2  train loss 震荡 train acc 震荡 val loss 下降收敛了 val acc 上升趋势 bright-monkey-253
    sample_size=50 bs=8 lr=0.1  val loss 上升 val acc 下降 但是train loss下降趋势不错 fortuitous-cake-240

    下一步: 重复 lr=0.1 - 0.5 之间的值，看看哪个效果最好

    发现：
    根据 dpcv/engine/bi_modal_trainer.py里关于acc的计算逻辑，
        - 如果batch_size=1，那么acc就是一个样本的平均准确率，从0.2-0.8都有可能，波动大
        - 如果batch_size=8，那么acc就是8个样本的平均准确率，但是这8个样本中有的样本acc接近1，有的样本acc接近0，所以acc的平均值为0.5，
    因此，可以发现，当batch_size越小，acc波动就越大，从0.2-0.8都有可能，当batch_size越大，acc越接近0.5，因为很多极值被平均掉了。
    请问，这种准确率的计算方式是否准确？？ batchsize越大越好还是越小越好？？
    关于准确率的计算方式，参考：https://colab.research.google.com/drive/1Ip20FMBtBDyvcRJJj_lXzozLyeimsN4d#scrollTo=ngDiEm6p1EqP
    即对于回归问题，acc的计算方式是：acc = (y_pred - y_true).abs().mean()， 对于分类问题，acc的计算方式是：acc = (y_pred.argmax(dim=1) == y_true).float().mean()， 因为分类问题的y_pred是一个one-hot向量，所以argmax之后，就是一个数值，即预测的类别，然后和真实的类别进行比较，如果相等，那么acc=1，否则acc=0，然后求平均值，就是准确率。

    关于batch size的解释：
        https://www.zhihu.com/question/32673260/answer/376970624
        
        Batch size会影响模型性能，过大或者过小都不合适。
        1. 是什么？设置过大的批次（batch）大小，可能会对训练时网络的准确性产生负面影响，因为它降低了梯度下降的随机性。
        2. 怎么做？要在可接受的训练时间内，确定最小的批次大小。一个能合理利用GPU并行性能的批次大小可能不会达到最佳的准确率，
            因为在有些时候，较大的批次大小可能需要训练更多迭代周期才能达到相同的正确率。在开始时，要大胆地尝试很小的批次大小，如16、8，甚至是1。
        3. 为什么？较小的批次大小能带来有更多起伏、更随机的权重更新。这有两个积极的作用，一是能帮助训练“跳出之前可能卡住它的局部最小值，
            二是能让训练在平坦的最小值结束，这通常会带来更好的泛化性能。

        太大的batch size 容易陷入sharp minima，泛化性不好
        
        研究表明大的batchsize收敛到sharp minimum，而小的batchsize收敛到flat minimum，后者具有更好的泛化能力。两者的区别就在于变化的趋势，一个快一个慢，
        如下图，造成这个现象的主要原因是小的batchsize带来的噪声有助于逃离sharp minimum。
        对此实际上是有两个建议：
            - 如果增加了学习率，那么batch size最好也跟着增加，这样收敛更稳定。
            - 尽量使用大的学习率，因为很多研究都表明更大的学习率有利于提高泛化能力。如果真的要衰减，可以尝试其他办法，比如增加batch size，学习率对模型的收敛影响真的很大，慎重调整。
        作者：初识CV 链接：https://www.zhihu.com/question/32673260/answer/1877223574

        batch_size设的大一些，收敛得快，也就是需要训练的次数少，准确率上升得也很稳定，但是实际使用起来精度不高。
        batch_size设的小一些，收敛得慢，而且可能准确率来回震荡，所以还要把基础学习速率降低一些；但是实际使用起来精度较高。
        一般我只尝试batch_size=64或者batch_size=1两种情况。
        作者：江何 链接：https://www.zhihu.com/question/32673260/answer/238851203


    怎么判断网络是否收敛，以loss为准还是acc为准？
        https://www.cnblogs.com/emanlee/p/14815390.html
    
    当使用 ResNet 3D模型时：
        学习率：根据实验smoldering-smooch-371，初始学习率为0.001时，loss上升，说明学习率过大，需要降低学习率。当学习率为0.0001时，loss下降，即初始学习率为0.0001时比较好
        但是发现验证集的acc不高，因此过拟合了，考虑进行数据增强，引入噪声，加入dropout等方法


# 一些结果相对比较好的保存模型
resume="results/demo/unified_frame_images/bimodal_resnet_udiva/02-03_21-46/checkpoint_0.pkl"
resume="/home/zl525/code/DeepPersonality/results/demo/unified_frame_images/bimodal_resnet_udiva/02-04_02-10/checkpoint_0.pkl"

    # 根据wandb的链接id来查找日志所在文件夹
    find . -name "d7lcr806"

# 验证 以下两种方法计算的auroc是否相同
    from torchmetrics.functional import auroc
    from sklearn.metrics import roc_auc_score
    结论： 在train val test三个数据集上都是一样的！ 所以只要用其中一种就行了

    那么train_batch_sklearn_auc和train_batch_sklearn_auc2的趋势对比呢？即使用 (labels.argmax(dim=-1), outputs.argmax(dim=-1)) 和 (labels[:, 0], outputs[:, 0]) 的区别
    结论：取值不一样，但是趋势大概是80%的相似，后续保留2的版本
    参考：https://www.notion.so/MPhil-Project-b3de240fa9d64832b26795439d0142d9?pvs=4#ce59a5d3e9174c028d1d87f69183abc7


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

# 查看输入数据inputs labels 和 模型models在哪个device上 
    参考：https://www.cnblogs.com/picassooo/p/13736843.html

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

######## wandb上的对比实验记录 #########
将wandb上每个实验的tag设置为experiment-x, x 是一个数字，代表对应的实验目的，不同id对应的对比实验目的不一样，用于周会展示

1. Project: DeepPersonality
    experiment-1: 控制其他超参数不变，递增sample_size，观察acc的变化
