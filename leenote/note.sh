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
    
    验证集loss上升，准确率却上升：
        - 验证集loss上升，acc也上升这种现象很常见，原因是过拟合或者训练验证数据分布不一致导致，即在训练后期，预测的结果趋向于极端，使少数预测错的样本主导了loss，但同时少数样本不影响整体的验证acc情况。
            在我把所有分类信息打印出来之后发现是模型过于极端导致的，即模型会出现在正确分类上给出0.00..x的概率值，导致loss异常的高，超过20，因此极大的提高了平均loss，导致出现了loss升高，acc也升高的奇怪现象。 说了多少次了，不要看loss，loss波动很正常，loss设的不够好导致部分上升占主导，掩盖了另一部分的下降也很正常。 https://www.cnblogs.com/emanlee/p/14815390.html
        - 可以明显看出训练200轮后结果趋于极端，而这些极端的负面Loss拉大了总体Loss导致验证集Loss飙升。出现这种情况大多是训练集验证集数据分布不一致，或者训练集过小，未包含验证集中所有情况，也就是过拟合导致的 验证集loss上升，准确率却上升该如何理解？ - 刘国洋的回答 - 知乎 https://www.zhihu.com/question/318399418/answer/1202932315
        解决办法：增加训练样本
                增加正则项系数权重，减小过拟合
                加入早停机制，ValLoss上升几个epoch直接停止
                采用Focal Loss
                加入Label Smoothing
        
        

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
backup: 
    sp16- ; sp32- ; sp48- ; sp64- ; sp80- ; sp96-

Tag解释:
    1. testjob: 提交正式slurm作业前的预测试，运行时间短，用于测试sbatch脚本是否能跑通
    2. experiment-x: 正式slurm作业，将wandb上每个实验的tag设置为experiment-x, x 是一个数字，代表对应的实验目的，不同id对应的对比实验目的不一样，用于周会展示

1. Project: DeepPersonality
    experiment-1: 
        【模型】：resnet50_3d_model_udiva 
        【分支】：视觉分支
        【变量】：控制其他超参数不变，递增sample_size，观察acc的变化
    experiment-2: 
        【相比于上一个实验的变化】：相比于experiment-1，增加了auc、f1score指标，
        【模型】： resnet50_3d_model_udiva 
        【分支】：视觉分支
        【变量】：控制其他超参数不变，递增sample_size，观察auc的变化
        【更新-experiment-2_1】 增加session_acc指标，quarter_acc指标, 用于周会展示
            30mins test: 15629279 15629280 15629281 15629401
            bs8: sp144(OOM❌)
            bs16:
            bs32:

            sample_size=32
            sample_size2=48
            sample_size3=64
            sample_size4=64
            sample_size5=64
            sample_size6=144

            batch_size=16
            batch_size2=16
            batch_size3=16
            batch_size4=8
            batch_size5=32
            batch_size6=8

            Submitted batch job 15630606
            Submitted batch job 15630608
            Submitted batch job 15630609
            Submitted batch job 15630610
            Submitted batch job 15630611
            Submitted batch job 15630612(OOM❌)

            第一轮跑的太慢，导致就跑了几个epoch，所以第二轮重新跑了一遍，把job运行时间增大到8小时：
            Submitted batch job 15767933
            Submitted batch job 15767935
            Submitted batch job 15767936
            Submitted batch job 15767937
            Submitted batch job 15767938
            Submitted batch job 15767940(OOM❌)


            quarter_acc_results 结果总结：(选取test_session_auc 从高到底排序的前几个来展示)
            sp, bs, quarter_acc_results, test_session_acc, test_session_auc:
            64 8 {"8105":[1,1,0,1],"56109":[1,1,1,1],"66067":[0,1,1,0],"86087":[0,0,0,0],"86089":[0,0,0,0],"87088":[0,0,0,0],"87089":[1,1,1,1],"88089":[1,1,1,1],"105117":[1,1,0,1],"111130":[0,1,0,0],"137138":[1,1,1,1]} 0.5455 0.8889
            48 16 {"8105":[0,0,0,0],"56109":[1,1,1,1],"66067":[0,0,0,0],"86087":[0,0,0,0],"86089":[0,0,0,0],"87088":[0,0,0,0],"87089":[0,0,0,0],"88089":[0,0,0,0],"105117":[0,0,0,0],"111130":[0,0,0,0],"137138":[1,1,1,1]} 0.1818 0.5
            32 16 {"8105":[1,1,1,1],"56109":[0,0,0,0],"66067":[1,1,1,1],"86087":[1,1,1,1],"86089":[1,1,1,1],"87088":[1,1,1,1],"87089":[0,0,0,0],"88089":[1,1,1,1],"105117":[0,1,0,0],"111130":[0,0,0,0],"137138":[0,0,0,0]} 0.5455 0.3333
            64 32 {"8105":[0,0,0,0],"56109":[0,0,0,0],"66067":[1,1,1,1],"86087":[0,0,0,0],"86089":[0,0,0,0],"87088":[0,0,0,0],"87089":[0,0,0,0],"88089":[0,0,0,0],"105117":[0,0,0,0],"111130":[0,0,0,0],"137138":[1,1,1,1]} 0.1818 0.3333

    experiment-3: 
        【相比于上一个实验的变化】：相比于experiment-2，从视觉分支改为音频分支，且模型改为audio_resnet_udiva, 且数据集进行了过采样避免不均衡
        【模型】： audio_resnet_udiva
        【分支】：音频分支
        【变量】：控制其他超参数不变，递增sample_size，观察acc的变化
        【GPU job id】 14970054(5mins test, success✅) 14956420 14956425 14956433 14956435 14956438 14956441 (全部job都success✅)
                    bs16: sp112-15025516 sp128-15025517 sp144-15025518 sp160-15025523 sp176-15025524 sp192-15025526
                    bs32: sp16-15025577 sp32-15025578 sp48-15025583 sp80-15025584 sp96-15025585 sp112-15025594
                    bs8:  sp16-15025605 sp32-15025607 sp48-15025610 sp80-15025611 sp96-15025612 sp112-15025614
                    bs48: sp16-15025617 sp32-15025620 sp48-15025621 sp64-15025622 sp80-15025627 sp96-15025628
        【CPU job id】 1+2: 14977081, 3+4: 14977083, 5+6: 14977084
        【总结】audio跑的很快，bs16_sp32:gpu8分钟跑完，cpu5h跑完
        【更新】要全部重新跑❎，因为代码里有bug，之前代码里是把2个相同的音频cat了，没有真正的把两个人的音频拼接起来
            bs8: sp16-15095034 ; sp32-15095409 ; sp48-15095411 ; sp64-15095425 ; sp80-15095426 ; sp96-15095427
            bs16:(sp192 不会OOM gpu 20mins 跑完) sp16-15096165 ; sp32-15096166 ; sp48-15096167 ; sp64-15096168 ; sp80-15096192 ; sp96-15096193
            bs32:(sp112 不会OOM gpu 12mins跑完) sp16-15096292 ; sp32-15096293 ; sp48-15096294 ; sp64-15096295 ; sp80-15096304 ; sp96-15096305
            bs48:(sp96 不会OOM gpu 12mins跑完) sp16-15096345 ; sp32-15096347 ; sp48-15096348 ; sp64-15096349 ; sp80-15096350 ; sp96-15096351

        【子实验-experiment-3_1】 增加session_acc指标，quarter_acc指标, 增加 train_epoch_summary_loss, val_epoch_summary_loss 用于周会展示
            test: 15652018
            sample_size=16
            sample_size2=32
            sample_size3=48
            sample_size4=64
            sample_size5=80
            sample_size6=96

            batch_size=48
            batch_size2=48
            batch_size3=48
            batch_size4=48
            batch_size5=48
            batch_size6=48

            Submitted batch job 15652020
            Submitted batch job 15652031
            Submitted batch job 15652040
            Submitted batch job 15652051
            Submitted batch job 15652213
            Submitted batch job 15652076

            sample_size=32
            sample_size2=48
            sample_size3=64
            sample_size4=80
            sample_size5=96
            sample_size6=112

            batch_size=64
            batch_size2=64
            batch_size3=64
            batch_size4=64
            batch_size5=64
            batch_size6=64

            Submitted batch job 15652450
            Submitted batch job 15652461
            Submitted batch job 15652472
            Submitted batch job 15652482
            Submitted batch job 15652492
            Submitted batch job 15652501

            sample_size=32
            sample_size2=48
            sample_size3=64
            sample_size4=80
            sample_size5=96
            sample_size6=112

            batch_size=128
            batch_size2=128
            batch_size3=128
            batch_size4=128
            batch_size5=128
            batch_size6=128

            Submitted batch job 15652646
            Submitted batch job 15652676
            Submitted batch job 15652700
            Submitted batch job 15652782
            Submitted batch job 15652831
            Submitted batch job 15652848


            quarter_acc_results 结果总结：(选取test_session_auc 从高到底排序的前几个来展示)

            sp, bs, quarter_acc_results, test_session_acc, test_session_auc:
            96 128 {"8105":[0,0,0,0],"56109":[1,1,1,1],"66067":[0,0,0,0],"86087":[0,1,1,1],"86089":[1,1,1,1],"87088":[1,1,1,1],"87089":[0,0,0,0],"88089":[0,0,0,0],"105117":[1,1,0,0],"111130":[0,0,0,0],"137138":[1,1,1,1]} 0.5455 0.8333
            32 128 {"8105":[0,0,0,0],"56109":[1,1,1,1],"66067":[0,0,0,1],"86087":[0,0,1,0],"86089":[1,1,1,1],"87088":[1,1,1,1],"87089":[0,0,0,0],"88089":[0,0,0,0],"105117":[1,0,0,0],"111130":[0,0,0,0],"137138":[1,1,1,1]} 0.3636 0.7778
            48 48 {"8105":[0,0,0,0],"56109":[1,1,1,1],"66067":[0,0,0,1],"86087":[1,0,1,1],"86089":[0,0,0,1],"87088":[1,1,1,1],"87089":[0,0,0,0],"88089":[0,0,0,0],"105117":[1,1,0,0],"111130":[0,0,0,0],"137138":[1,1,1,1]} 0.5455 0.7222
            64 48 {"8105":[0,0,0,0],"56109":[1,1,0,1],"66067":[1,0,1,0],"86087":[0,0,1,0],"86089":[0,0,0,1],"87088":[1,1,1,1],"87089":[0,0,0,0],"88089":[0,0,0,0],"105117":[1,1,0,0],"111130":[0,0,0,0],"137138":[1,1,1,1]} 0.5455 0.6667



    experiment-4: 
        【相比于上一个实验的变化】：相比于experiment-3，使用Transformer模型:timesformer_udiva 来处理视觉分支
        【模型】：timesformer_udiva
        【分支】：视觉分支
        【变量】：控制其他超参数不变，递增sample_size，观察acc的变化
        【job id】 
            bs16: 10mins test: 14976665(success✅); 5h: 1-14976675(success✅), 2-14976697(CUDA OOM❌), 3-14977044(CUDA OOM), 4-14977045(CUDA OOM), 5-14977046(CUDA OOM), 6-14977049(CUDA OOM)
            bs8 bs4: 8mins test: 14994245(success✅) ; 5h: 1-14994284(success✅) 2-14994289(success✅) 3-14994300(CUDA OOM❌) 4-14994301(success✅) 5-14994312(CUDA OOM❌) 6-14994315(CUDA OOM❌)
            bs4: sp16-14995294 sp32-14995308 sp48-14995324
        【总结】 
            当batch_size=16,sample_size>=32时，timesformer跑不起来，CUDA OOM
            当batch_size=8,sample_size>=48时，timesformer跑不起来，CUDA OOM
            当batch_size=4,sample_size>=80时，timesformer跑不起来，CUDA OOM
    experiment-5_1: 预训练
        【相比于上一个实验的变化】：相比于experiment-34，使用音频Transformer模型:ssast_udiva 来处理音频分支, 预训练阶段
        【模型】：ssast_udiva - 预训练阶段 https://github.com/YuanGongND/ssast
        【分支】：音频分支
        【变量】：无
        【GPU job id】30mins test: 15149208(finished✅) 
                bs16: sp16-15149222
        【CPU job id】bs16&8: 15149495
    experiment-5_2: 调优
        【相比于上一个实验的变化】：相比于experiment-34，使用音频Transformer模型:ssast_udiva 来处理音频分支, fine tunning调优阶段
        【模型】： ssast_udiva - fine tunning调优阶段 https://github.com/YuanGongND/ssast 和5_1区别，有pretrain参数
        【分支】：音频分支
        【变量】：无
        【GPU job id】10mins test: 15161247(success✅) 15166250
                bs8: sp16-15166513(success✅) ; sp32-15166928 (success✅); sp48-15166930(success✅) ; sp64-15166931(CUDA OOM❌) ; sp80-15166933 ; sp96-15167058; sp112-15169007 ; sp128-15169008 ; sp144-15169009 ; sp160-15169138 ; sp176-15169139 ; sp192-15169140
                bs16: sp16-15164654(success✅) ; sp32-15164988(success✅) ; sp48-15165009(CUDA OOM❌) ; sp64-15165010 ; sp80-15165111 ; sp96-15165455
                bs32: sp16-15167983(success✅) ; sp32-15168511(CUDA OOM❌) ; sp48- ; sp64- ; sp80- ; sp96-
                bs48: sp16-15168733(success✅) ; sp32-15168758(CUDA OOM❌) ; sp48- ; sp64- ; sp80- ; sp96-
        
        【子实验-experiment-5_3】 增加session_acc指标，quarter_acc指标, 增加 train_epoch_summary_loss, val_epoch_summary_loss 用于周会展示
            sample_size=16
            sample_size2=16
            sample_size3=16
            sample_size4=16
            sample_size5=32
            sample_size6=48

            batch_size=8
            batch_size2=16
            batch_size3=32
            batch_size4=48
            batch_size5=8
            batch_size6=8

            Submitted batch job 15767446
            Submitted batch job 15767447
            Submitted batch job 15767451
            Submitted batch job 15767452
            Submitted batch job 15767453
            Submitted batch job 15767458

            sample_size5=32
            batch_size5=16
            Submitted batch job 15767560
    
            quarter_acc_results 结果总结：(选取test_session_auc 从高到底排序的前几个来展示)
            sp, bs, quarter_acc_results, test_session_acc, test_session_auc:
            32 16 {"8105":[1,0,0,0],"56109":[1,1,1,1],"66067":[0,0,0,0],"86087":[0,0,0,0],"86089":[0,0,0,0],"87088":[0,0,0,0],"87089":[0,0,0,0],"88089":[0,0,0,0],"105117":[0,0,0,0],"111130":[0,0,0,0],"137138":[1,1,1,1]} 0.1818 0.8333
            32 8 {"8105":[1,0,0,0],"56109":[1,1,1,1],"66067":[0,0,0,0],"86087":[0,0,0,0],"86089":[0,0,0,0],"87088":[0,0,0,0],"87089":[0,0,0,0],"88089":[0,0,0,0],"105117":[0,0,0,0],"111130":[0,0,0,0],"137138":[1,1,1,1]} 0.1818 0.7778
            16 8 {"8105":[1,0,0,0],"56109":[1,1,1,1],"66067":[0,0,0,0],"86087":[0,0,0,0],"86089":[0,0,0,0],"87088":[0,0,0,0],"87089":[0,0,0,0],"88089":[0,0,0,0],"105117":[0,0,0,0],"111130":[0,0,0,0],"137138":[1,1,1,1]} 0.1818 0.7222
            16 32 {"8105":[1,0,0,0],"56109":[1,1,1,1],"66067":[0,0,0,0],"86087":[0,0,0,0],"86089":[0,0,0,0],"87088":[0,0,0,0],"87089":[0,0,0,0],"88089":[0,0,0,0],"105117":[0,0,0,0],"111130":[0,0,0,0],"137138":[1,1,1,1]} 0.1818 0.6667



    (DeepPersonality) [zl525@login-q-1 leenote]$ squeue -u zl525 --start
                JOBID PARTITION     NAME     USER ST          START_TIME  NODES SCHEDNODES           NODELIST(REASON)
            14976665    ampere   gpujob    zl525 PD 2023-02-26T08:25:00      1 gpu-q-69             (Priority)
            14956420    ampere   gpujob    zl525 PD 2023-02-26T11:52:34      1 gpu-q-64             (Priority)
            14956425    ampere   gpujob    zl525 PD 2023-02-26T11:52:34      1 gpu-q-14             (Priority)
            14956433    ampere   gpujob    zl525 PD 2023-02-26T11:52:34      1 gpu-q-21             (Priority)
            14956435    ampere   gpujob    zl525 PD 2023-02-26T11:52:34      1 gpu-q-24             (Priority)
            14956436    ampere   gpujob    zl525 PD 2023-02-26T12:06:40      1 gpu-q-57             (Priority)
            14956438    ampere   gpujob    zl525 PD 2023-02-26T12:06:40      1 gpu-q-73             (Priority)
            14956441    ampere   gpujob    zl525 PD 2023-02-26T12:16:05      1 gpu-q-25             (Priority)
            14976675    ampere   gpujob    zl525 PD 2023-02-26T19:09:12      1 gpu-q-13             (Priority)
            14976697    ampere   gpujob    zl525 PD 2023-02-27T04:00:00      1 gpu-q-73             (Priority)
            14977044    ampere   gpujob    zl525 PD 2023-02-27T04:10:00      1 gpu-q-25             (Priority)
            14977045    ampere   gpujob    zl525 PD 2023-02-27T04:25:00      1 gpu-q-26             (Priority)
            14977046    ampere   gpujob    zl525 PD 2023-02-27T04:25:00      1 gpu-q-28             (Priority)
            14977049    ampere   gpujob    zl525 PD 2023-02-27T04:25:00      1 gpu-q-53             (Priority)

####### SSAST 模型对比总结 # 

UDIVA:
    模型的构造函数：
                label_dim=2, 
                fshape=16, tshape=16, fstride=16, tstride=16,
                input_fdim=256, input_tdim=cfg.DATA.SAMPLE_SIZE,
                 model_size='tiny', pretrain_stage=True

    模型的输入数据：
        [ASTModel] input x.shape:  torch.Size([4, 1598, 256]) , task:  pretrain_mpc , cluster:  True , mask_patch:  400
    
    gen_maskid_patch:
        修改后：[ASTModel ****重要] 参数fstride: 16 tstride: 16 input_fdim: 256 input_tdim: 1598 fshape: 16 tshape: 16
        
        self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
        num_patches = self.p_f_dim * self.p_t_dim
        self.num_patches = num_patches
        即 sequence_len = self.p_f_dim * self.p_t_dim

        [ASTModel - gen_maskid_patch] sequence_len: 1584 mask_size: 400 cluster: 3 , cur_clus: 4


官方代码:
    模型的构造函数：
    【****重要】lee 打印构造的ASTModel模型的所有__init__参数 
        fshape:  16 tshape:  16 fstride:  16 tstride:  16 
        input_fdim:  128 input_tdim:  1024 
        model_size:  base pretrain_stage:  True


    模型的输入数据：
        1-input x.shape:  torch.Size([2, 1024, 128]) , task:  pretrain_mpc , cluster:  True , mask_patch:  400
        
        
    gen_maskid_patch:
        [lee ASTModel ****重要]计算num_patches的参数 fstride: 16 tstride: 16 input_fdim: 128 input_tdim: 1024 fshape: 16 tshape: 16
        [lee ASTModel - gen_maskid_patch] sequence_len: 512 mask_size: 400 cluster: 3 , cur_clus: 4
        
        sequence_len: 512 是怎么计算的：参数 
            fstride: 16 tstride: 16 
            input_fdim: 128 input_tdim: 1024 
            fshape: 16 tshape: 16

""" SSAST备份
Namespace(adaptschedule=False, bal='none', batch_size=2, 
          cluster_factor=3, data_eval=None, data_train='/home/zl525/code/Audio-Transformer/ssast/src/prep_data/esc50/data/datafiles/esc_train_data_1.json',
          data_val='/home/zl525/code/Audio-Transformer/ssast/src/prep_data/esc50/data/datafiles/esc_eval_data_1.json', 
          dataset='esc50', dataset_mean=-4.2677393, dataset_std=4.5689974, epoch_iter=4000, 
          exp_dir='./exp/mask01-base-f16-t16-b2-lr1e-4-m400-pretrain_mpc-esc50', 
          freqm=0, fshape=16, fstride=16, head_lr=1, 
          label_csv='/home/zl525/code/Audio-Transformer/ssast/src/finetune/esc50/data/esc_class_labels_indices.csv', 
          loss='BCE', lr=0.0001, lr_patience=2, lrscheduler_decay=0.5, lrscheduler_start=10, lrscheduler_step=5, 
          mask_patch=400, metrics='mAP', mixup=0.0, model_size='base', n_class=527, n_epochs=2, n_print_steps=100, 
          noise=None, num_mel_bins=128, num_workers=16, optim='adam', pretrained_mdl_path=None, save_model=False, 
          target_length=1024, task='pretrain_mpc', timem=0, tshape=16, tstride=16, wa=None, wa_end=30, wa_start=16, warmup=True)

args_dict备份
        args_dict = {'data_train': '/home/zl525/code/Audio-Transformer/ssast/src/prep_data/esc50/data/datafiles/esc_train_data_1.json', 
                     'data_val': '/home/zl525/code/Audio-Transformer/ssast/src/prep_data/esc50/data/datafiles/esc_eval_data_1.json', 
                     'data_eval': None, 
                     'label_csv': '/home/zl525/code/Audio-Transformer/ssast/src/finetune/esc50/data/esc_class_labels_indices.csv', 
                     'n_class': 2, 
                     'dataset': 'udiva', 
                     'dataset_mean': -4.2677393, 
                     'dataset_std': 4.5689974, 
                     'target_length': 1024, 
                     'num_mel_bins': 128, 
                     'exp_dir': '', 
                     'lr': 0.0001, 
                     'warmup': True, 
                     'optim': 'adam', 
                     'batch_size': 2, 
                     'num_workers': 16, 
                     'n_epochs': 2, 
                     'lr_patience': 2, 
                     'adaptschedule': False, 
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
                     'model_size': 'base', 
                     'task': 'pretrain_mpc', 
                     'mask_patch': 400, 
                     'cluster_factor': 3, 
                     'epoch_iter': 4000, 
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
""" 