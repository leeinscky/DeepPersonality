###
 # @Author: leeinscky 1009694687@qq.com
 # @Date: 2022-12-27 01:10:20
 # @LastEditors: leeinscky 1009694687@qq.com
 # @LastEditTime: 2023-02-02 09:51:24
 # @FilePath: /LAVISH/Users/lizejian/cambridge/mphil_project/note/文件传输命令.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
# 1. 
cd cambridge/mphil_project/learn/udiva/DeepPersonality/datasets

1.1 剑桥HPC
    nohup scp udiva_tiny_hpc.zip zl525@login-icelake.hpc.cam.ac.uk:/home/zl525/rds/hpc-work/temp

1.2 ACS GPU
    nohup scp udiva_tiny_hpc.zip zl525@dev-gpu-acs.cl.cam.ac.uk:/home/zl525/rds/hpc-work

# 2. 
cd /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny_hpc压缩文件

2.1 剑桥HPC
    scp train.zip zl525@login-icelake.hpc.cam.ac.uk:/home/zl525/rds/hpc-work/temp
    scp -C val.zip zl525@login-icelake.hpc.cam.ac.uk:/home/zl525/rds/hpc-work/temp
    scp test.zip zl525@login-icelake.hpc.cam.ac.uk:/home/zl525/rds/hpc-work/temp

2.2 ACS GPU
    scp train.zip zl525@dev-gpu-acs.cl.cam.ac.uk:/home/zl525/rds/hpc-work
    scp val.zip zl525@dev-gpu-acs.cl.cam.ac.uk:/home/zl525/rds/hpc-work
    scp test.zip zl525@dev-gpu-acs.cl.cam.ac.uk:/home/zl525/rds/hpc-work


# 3. 下载UDIVA数据集命令
zip文件解压: unzip metadata_train.zip

wget --user='ssong' --password='w3S&8u5wFc4T>)>+' -b https://data.chalearnlap.cvc.uab.es/UDIVA_Protected/UDIVA_v0.5/train/recordings/animals_recordings_train.z01
wget --user='ssong' --password='w3S&8u5wFc4T>)>+' https://data.chalearnlap.cvc.uab.es/UDIVA_Protected/UDIVA_v0.5/train/recordings/ghost_recordings_train.z01
wget --user='ssong' --password='w3S&8u5wFc4T>)>+' -b https://data.chalearnlap.cvc.uab.es/UDIVA_Protected/UDIVA_v0.5/train/metadata/metadata_train.zip
wget -c --user='ssong' --password='w3S&8u5wFc4T>)>+' https://data.chalearnlap.cvc.uab.es/UDIVA_Protected/UDIVA_v0.5/train/recordings/animals_recordings_train.z01

# wget 命令重要参数：
    -P: 使用 -P 参数来指定目录，如果指定的目录不存在，则会自动创建
    -b: 后台下载，使用wget -b + url # https://blog.csdn.net/wanglc7/article/details/85136418
    -x: 使用 -x 会强制建立服务器上一模一样的目录
    -c 断点续传 当文件特别大或者网络特别慢的时候，往往一个文件还没有下载完，连接就已经被切断，此时就需要断点续传
    -r: 遍历所有子目录
    -np: 不到上一层子目录去 如wget -c -r www.xianren.org/pub/path/, 没有加参数-np，就会同时下载path的上一级目录pub下的其它文件
    -nH: 不要将文件保存到主机名文件夹
    -R index.html : 不下载 index.html 文件
    

# 下载整个目录下的所有文件
    wget -c -r -np -nH -R index.html http://xxx
    参考链接：
        https://zhuanlan.zhihu.com/p/343117380
        https://blog.csdn.net/a751127122/article/details/103815175?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-103815175-blog-89879693.pc_relevant_3mothn_strategy_and_data_recovery&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-103815175-blog-89879693.pc_relevant_3mothn_strategy_and_data_recovery&utm_relevant_index=1
        https://www.v2ex.com/t/146724
# 将下载器挂起并在后台运行
    # https://www.cnblogs.com/bw98blogs/p/9404502.html
    nohup command >OutFile.out 2>&1 & 



# 正式在HPC机器上的下载命令，要求：
    # -c:断点续传 
    # -r: 遍历所有子目录
    # -np: 下载整个目录下的所有文件，不到上一层子目录去, 
    #  -R index.html: 不下载 index.html 文件
    nohup wget --user='ssong' --password='w3S&8u5wFc4T>)>+' \
    -c -r -np -R index.html \
    https://data.chalearnlap.cvc.uab.es/UDIVA_Protected/UDIVA_v0.5/ >OutFile.out 2>&1 &

    运行记录：[1] 631799

# 递归遍历当前文件夹，并删除所有的index.html文件
    # 参考：find . -name ".DS_Store" -delete
    find . -name "index.html.tmp" -type f -delete
    

# 下载UCF101数据集
    wget -b https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate



# rds剩余的存储空间(一共有1T空间)
/home/zl525/rds/hpc-work
du -lh --max-depth=1
31G	./.conda
266G	./datasets
297G	.

###### 处理UDIVA数据集，解压缩等 ######
## 一、合并多个zip part文件为一个大文件
官网举例：Big zip files have been split into max 4GB zip parts. To extract them, you first need to download 
all the parts and join them. For instance, lets say you have 9 zip parts called new.zip, new.z01, new.z02, etc. Using linux, run:
"zip -F new.zip --out existing.zip" or "zip -s0 new.zip --out existing.zip" 
to recreate an existing.zip. Then, run: `unzip existing.zip`

因此：我的命令
    # 后台合并所有zip文件 -train
    cd /home/zl525/rds/hpc-work/datasets/udiva_full/data.chalearnlap.cvc.uab.es/UDIVA_Protected/UDIVA_v0.5/train/recordings
    nohup zip -F animals_recordings_train.zip --out animals_recordings_train_img.zip >log_animals.log 2>&1 &
    nohup zip -F ghost_recordings_train.zip --out ghost_recordings_train_img.zip >log_ghost.log 2>&1 &
    nohup zip -F lego_recordings_train.zip --out lego_recordings_train_img.zip >log_lego.log 2>&1 &
    nohup zip -F talk_recordings_train.zip --out talk_recordings_train_img.zip >log_talk.log 2>&1 &

    进程号：[1] 732621 [2] 736668 [3] 739607 [4] 740180
    ps -ef | grep "732621"
    ps -ef | grep "736668"
    ps -ef | grep "739607"
    ps -ef | grep "740180"

    # 后台合并所有zip文件 -val
    cd /home/zl525/rds/hpc-work/datasets/udiva_full/val/recordings
    nohup zip -F animals_recordings_val.zip --out animals_recordings_val_img.zip >log_animals.log 2>&1 &
    nohup zip -F ghost_recordings_val.zip --out ghost_recordings_val_img.zip >log_ghost.log 2>&1 &
    nohup zip -F lego_recordings_val.zip --out lego_recordings_val_img.zip >log_lego.log 2>&1 &
    nohup zip -F talk_recordings_val.zip --out talk_recordings_val_img.zip >log_talk.log 2>&1 &

    nohup zip -F animals_recordings_val.zip --out animals_recordings_val_img.zip >log_animals.log 2>&1 &
    [1] 2299171
    nohup zip -F ghost_recordings_val.zip --out ghost_recordings_val_img.zip >log_ghost.log 2>&1 &
    [2] 2300077
    nohup zip -F lego_recordings_val.zip --out lego_recordings_val_img.zip >log_lego.log 2>&1 &
    [3] 2300850
    nohup zip -F talk_recordings_val.zip --out talk_recordings_val_img.zip >log_talk.log 2>&1 &
    [4] 2301748

    # 后台合并所有zip文件 -test
    cd /home/zl525/rds/hpc-work/datasets/udiva_full/test/recordings
    nohup zip -F animals_recordings_test.zip --out animals_recordings_test_img.zip >log_animals.log 2>&1 &
    nohup zip -F ghost_recordings_test.zip --out ghost_recordings_test_img.zip >log_ghost.log 2>&1 &
    nohup zip -F lego_recordings_test.zip --out lego_recordings_test_img.zip >log_lego.log 2>&1 &

    nohup zip -F animals_recordings_test.zip --out animals_recordings_test_img.zip >log_animals.log 2>&1 &
    [5] 2303580
    nohup zip -F ghost_recordings_test.zip --out ghost_recordings_test_img.zip >log_ghost.log 2>&1 &
    [6] 2304452
    nohup zip -F lego_recordings_test.zip --out lego_recordings_test_img.zip >log_lego.log 2>&1 &
    [7] 2305491



## 二、解压缩
    先执行： 
        # val
        mv recordings old_recordings && mkdir recordings
        mkdir -p recordings/animals_recordings_val_img && mkdir recordings/ghost_recordings_val_img && mkdir recordings/lego_recordings_val_img && mkdir recordings/talk_recordings_val_img
        
        mv old_recordings/animals_recordings_val_img.zip recordings/animals_recordings_val_img/
        mv old_recordings/ghost_recordings_val_img.zip recordings/ghost_recordings_val_img/
        mv old_recordings/lego_recordings_val_img.zip recordings/lego_recordings_val_img/
        mv old_recordings/talk_recordings_val_img.zip recordings/talk_recordings_val_img/

        mv animals_recordings_train_img animals_recordings_val_img
        mv ghost_recordings_train_img ghost_recordings_val_img;
        mv lego_recordings_train_img lego_recordings_val_img;
        mv talk_recordings_train_img talk_recordings_val_img;

        mv animals_recordings_val_img animals_recordings_val_img.zip
        mv ghost_recordings_val_img ghost_recordings_val_img.zip
        mv lego_recordings_val_img lego_recordings_val_img.zip
        mv talk_recordings_val_img talk_recordings_val_img.zip

        mv animals_recordings_val_img.zip animals_recordings_val_img/
        mv ghost_recordings_val_img.zip ghost_recordings_val_img/;
        mv lego_recordings_val_img.zip lego_recordings_val_img/;
        mv talk_recordings_val_img.zip talk_recordings_val_img/;

        # test
        mkdir -p recordings/animals_recordings_test_img && mkdir recordings/ghost_recordings_test_img && mkdir recordings/lego_recordings_test_img && mkdir recordings/talk_recordings_test_img

        mv old_recordings/animals_recordings_test_img.zip recordings/animals_recordings_test_img/;
        mv old_recordings/ghost_recordings_test_img.zip recordings/ghost_recordings_test_img/;
        mv old_recordings/lego_recordings_test_img.zip recordings/lego_recordings_test_img/;
        cp old_recordings/talk_recordings_test.zip recordings/talk_recordings_test_img/talk_recordings_test_img.zip;

        # test masked还没看 待办
    然后执行：
        unzip -P "?7W3WmJu{fNPVg<u" -d ../recordings/animals_recordings_train_img animals_recordings_train_img.zip
        unzip -P "#/WCDf6x+8}Ex%PR" -d ../recordings/ghost_recordings_train_img ghost_recordings_train_img.zip
        unzip -P "7>epBX>WRjG3_]9p" -d ../recordings/lego_recordings_train_img lego_recordings_train_img.zip
        unzip -P "t@6Gwvm%M^-M6M5-" -d ../recordings/talk_recordings_train_img talk_recordings_train_img.zip

    unzip会报错：error: invalid zip file with overlapped components (possible zip bomb)
    
    因此，替换方案： # https://unix.stackexchange.com/questions/634315/unzip-thinks-my-zip-file-is-a-zip-bomb
        mkdir animals_recordings_train_img ghost_recordings_train_img lego_recordings_train_img talk_recordings_train_img
        mv animals_recordings_train_img.zip ../recordings/animals_recordings_train_img
        mv ghost_recordings_train_img.zip ../recordings/ghost_recordings_train_img
        mv lego_recordings_train_img.zip ../recordings/lego_recordings_train_img
        mv talk_recordings_train_img.zip ../recordings/talk_recordings_train_img
        
        1. jar
            # 参考： https://unix.stackexchange.com/questions/634315/unzip-thinks-my-zip-file-is-a-zip-bomb
            jar xf animals_recordings_train_img.zip 
            实验后发现不行。。。

        2. 7zip（ok可以跑通）
            # 下载：https://bytexd.com/how-to-install-and-use-7zip-in-linux/ 密码用法：http://blog.itpub.net/8404772/viewspace-608234/  https://stackoverflow.com/questions/28160254/7-zip-command-to-create-and-extract-a-password-protected-zip-file-on-windows
            # 因为HPC机器上没有root权限，所以下载新软件可以通过python pip 或者 wget下载到自己的目录下，手动安装
            /home/zl525/tools/7zzs --help
            cd /home/zl525/rds/hpc-work/datasets/udiva_full/data.chalearnlap.cvc.uab.es/UDIVA_Protected/UDIVA_v0.5/train/recordings/

            # ------ train ------ #
            nohup /home/zl525/tools/7zzs x -p"?7W3WmJu{fNPVg<u" animals_recordings_train_img.zip >nohup.out 2>&1 &
            nohup /home/zl525/tools/7zzs x -p"#/WCDf6x+8}Ex%PR" ghost_recordings_train_img.zip >nohup.out 2>&1 &
            nohup /home/zl525/tools/7zzs x -p"7>epBX>WRjG3_]9p" lego_recordings_train_img.zip >nohup.out 2>&1 &
            nohup /home/zl525/tools/7zzs x -p"t@6Gwvm%M^-M6M5-" talk_recordings_train_img.zip >nohup.out 2>&1 &

            预览压缩包里的内容：
                unzip -l animals_recordings_train_img.zip # 232 files 即116个session文件夹
                unzip -l ghost_recordings_train_img.zip # 232 files 即116个session文件夹
                unzip -l lego_recordings_train_img.zip # 232 files 即116个session文件夹
                unzip -l talk_recordings_train_img.zip # 232 files 即116个session文件夹
            验证116个文件夹是否全都顺利产出：ls | grep -v nohup | grep -v .zip | wc -l 
                animals_recordings_train_img: ✅
                ghost_recordings_train_img: ✅
                lego_recordings_train_img: ✅
                talk_recordings_train_img: ✅
                结论：全都顺利产出mp4文件

            删除出错文件：rm -rf 0* 1* W* e* no*

            # -------- val -------- #
            nohup /home/zl525/tools/7zzs x -p"Jw5Q^'2y9N+<Qj>S" animals_recordings_val_img.zip >nohup.out 2>&1 &
            nohup /home/zl525/tools/7zzs x -p"5E5s?e?,N^;}_w(}" ghost_recordings_val_img.zip >nohup.out 2>&1 &
            # nohup /home/zl525/tools/7zzs x -p"Dd,e;G3!<YY6yjCE" lego_recordings_val_img.zip >nohup.out 2>&1 &
            /home/zl525/tools/7zzs x lego_recordings_val_img.zip
            nohup /home/zl525/tools/7zzs x -p"R2P5Jj2'*N32/M3=" talk_recordings_val_img.zip >nohup.out 2>&1 &

            预览压缩包里的内容：
                unzip -l animals_recordings_val_img/animals_recordings_val_img.zip;
                unzip -l ghost_recordings_val_img/ghost_recordings_val_img.zip;
                unzip -l lego_recordings_val_img/lego_recordings_val_img.zip;
                unzip -l talk_recordings_val_img/talk_recordings_val_img.zip;
                # 四个都是36个files，即18个session文件夹，每个session文件夹里有两个mp4文件

            验证116个文件夹是否全都顺利产出：ls | grep -v nohup | grep -v .zip | wc -l 
                animals_recordings_val_img: ✅
                ghost_recordings_val_img: ✅
                lego_recordings_val_img: ✅
                talk_recordings_val_img: ✅
                结论：全都顺利产出mp4文件


            # -------- test -------- #
            nohup /home/zl525/tools/7zzs x -p'-[6C7"bFm{*GaF<B' animals_recordings_test_img.zip >nohup.out 2>&1 &
            # 因为ghost_recordings_test_img的密码里有特殊字符，无法直接后台运行，需要用：/home/zl525/tools/7zzs x ghost_recordings_test_img.zip ，手动输入密码
            # nohup /home/zl525/tools/7zzs x -p"9jqb`Z{!Ld'N3p`7" ghost_recordings_test_img.zip >nohup.out 2>&1 & 
            nohup /home/zl525/tools/7zzs x -p'@J#3`P6#-Wrc"LLf' lego_recordings_test_img.zip >nohup.out 2>&1 &
            nohup /home/zl525/tools/7zzs x -p'fR@fpJY.A<6R<Uzj' talk_recordings_test_img.zip >nohup.out 2>&1 &
            

            预览压缩包里的内容：
                unzip -l animals_recordings_test_img/animals_recordings_test_img.zip;
                unzip -l ghost_recordings_test_img/ghost_recordings_test_img.zip;
                unzip -l lego_recordings_test_img/lego_recordings_test_img.zip;
                unzip -l talk_recordings_test_img/talk_recordings_test_img.zip;
                # 四个都是 22files，即11个session文件夹，每个session文件夹里有两个mp4文件

            验证116个文件夹是否全都顺利产出：ls | grep -v nohup | grep -v .zip | wc -l 
                animals_recordings_test_img: ✅
                ghost_recordings_test_img: ✅
                lego_recordings_test_img: ✅
                talk_recordings_test_img: ✅
                结论：全都顺利产出mp4文件

## 三、其他

# linux 查看当前目录下每个子目录的文件数量
    find . -maxdepth 1 -type d | while read dir; do count=$(find "$dir" -type f | wc -l); echo "$dir : $count"; done

    cd /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings && find . -maxdepth 1 -type d | while read dir; do count=$(find "$dir" -type f | wc -l); echo "$dir : $count"; done

./lego_recordings_train_wav : 464
./talk_recordings_train_img : 404934
./ghost_recordings_train_img : 51218
./animals_recordings_train_img : 232
./lego_recordings_train_img : 232
./talk_recordings_train_wav : 464
./animals_recordings_train_wav : 464

# UDIVA 全量数据集文件个数统计
    首先，需要看下一共有多少个mp4视频文件：

    - 训练集train
        - animals_recordings_train_img: 116个session文件夹=232个mp4文件
        - ghost_recordings_train_img：116个session文件夹=232个mp4文件
        - lego_recordings_train_img：116个session文件夹=232个mp4文件
        - talk_recordings_train_img：116个session文件夹=232个mp4文件
        - 总计：一共 232 * 4 = 928个 mp4 视频文件，session个数是116*4=464个
            - 如果batch_size=8, 完成1次epoch需要 464/8=58 个iteration
            - 如果batch_size=16, 完成1次epoch需要 464/16=29 个iteration
            - 如果batch_size=32, 完成1次epoch需要 464/32=14 个iteration
            - 如果batch_size=48, 完成1次epoch需要 464/48=9.6 个iteration
        - 如果只训练animals_recordings_train_img 116个session文件夹
            - 如果batch_size=2, 完成1次epoch需要 116/2=58 个iteration
            - 如果batch_size=4, 完成1次epoch需要 116/4=29 个iteration
            - 如果batch_size=8, 完成1次epoch需要 116/8=14 个iteration 余4个
            - 如果batch_size=12, 完成1次epoch需要 116/12=9 个iteration 余8个
            - 如果batch_size=16, 完成1次epoch需要 116/16=7 个iteration 余4个
            - 如果batch_size=32, 完成1次epoch需要 116/32=3 个iteration 余20个

    - 验证集val
        - animals_recordings_train_img: 18个session文件夹=36个mp4文件
        - ghost_recordings_train_img：18个session文件夹=36个mp4文件
        - lego_recordings_train_img：18个session文件夹=36个mp4文件
        - talk_recordings_train_img：18个session文件夹=36个mp4文件
        - 总计：一共 36 * 4 = 144 个 mp4 视频文件，session个数是18*4=72个
            - 如果batch_size=8, 完成1次epoch需要 72/8=9 个iteration
            - 如果batch_size=16, 完成1次epoch需要 72/16=4.5 个iteration
            - 如果batch_size=24, 完成1次epoch需要 72/24=3 个iteration
            - 如果batch_size=32, 完成1次epoch需要 72/32=2.25 个iteration
    
    - 测试集test
        - animals_recordings_train_img: 11个session文件夹=22个mp4文件
        - ghost_recordings_train_img：11个session文件夹=22个mp4文件
        - lego_recordings_train_img：11个session文件夹=22个mp4文件
        - talk_recordings_train_img：11个session文件夹=22个mp4文件
        - 总计：一共 22 * 4 = 88 个 mp4 视频文件，session个数是11*4=44个
    
    - 训练集+验证集+测试集总计：
        - 928 + 144 + 88 = 1160个mp4视频文件。每个视频文件的长度：6-10分钟，360秒-600秒
        - 464 + 72 + 44 = 580个session文件夹
    

# UDIVA 数据集一些异常数据集处理
cd /home/zl525/code/DeepPersonality/datasets/udiva_full/train/recordings/talk_recordings_train_img/
find . -maxdepth 2 -type d | while read dir; do count=$(find "$dir" -type f | wc -l); echo "$dir : $count"; done

cd /home/zl525/code/DeepPersonality/datasets/udiva_full/val/recordings/talk_recordings_val_img/
find . -maxdepth 2 -type d | while read dir; do count=$(find "$dir" -type f | wc -l); echo "$dir : $count"; done

cd /home/zl525/code/DeepPersonality/datasets/udiva_full/test/recordings/talk_recordings_test_img/
find . -maxdepth 2 -type d | while read dir; do count=$(find "$dir" -type f | wc -l); echo "$dir : $count"; done

# 异常1
/home/zl525/code/DeepPersonality/datasets/udiva_full/train/recordings/talk_recordings_train_img/025044
./025044 : 749
./025044/FC2_T : 367
./025044/FC1_T : 380

mv /home/zl525/code/DeepPersonality/datasets/udiva_full/train/recordings/talk_recordings_train_img/025044 /home/zl525/code/DeepPersonality/datasets/udiva_full/backup/


# linux 查找当前路径下所有包含某个字符串的文件夹名称
chatGPT:
grep -r -l '"MAX_EPOCH":2,' ./wandb | xargs -I{} dirname {} | sort -u














(DeepPersonality) [zl525@login-q-1 wandb]$ ls | grep "offline-run-20230224_22"
offline-run-20230224_222102-c1r7p1qk
offline-run-20230224_222102-k83vqfyf
offline-run-20230224_222102-kverz03z
offline-run-20230224_222103-hahg3gr2
offline-run-20230224_223232-5tqvudfz
offline-run-20230224_223730-yu46r4ja

wandb sync offline-run-20230224_222102-c1r7p1qk offline-run-20230224_222102-k83vqfyf offline-run-20230224_222102-kverz03z offline-run-20230224_222103-hahg3gr2 offline-run-20230224_223232-5tqvudfz offline-run-20230224_223730-yu46r4ja

