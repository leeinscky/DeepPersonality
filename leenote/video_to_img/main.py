'''
Author: leeinscky 1009694687@qq.com
Date: 2022-12-25 01:08:50
LastEditors: leeinscky 1009694687@qq.com
LastEditTime: 2022-12-27 14:31:43
FilePath: /DeepPersonality/lzjnote/video_to_img/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

""" 遍历 datasets/udiva_tiny/train/recordings/animals_recordings_train_img 文件夹下的所有文件夹，如果某个文件夹没有FC1_A和FC2_A两个子文件夹，
就执行shell 命令： 
python video_to_image.py \
--video-dir 'xxx'
--output-dir 'xxx'
其中video-dir和output-dir后面的路径都是当前遍历的文件夹绝对路径 """

import  os
import sys
sys.path.append('../../datasets/') # 将datasets路径添加到系统路径中，这样就可以直接导入datasets下的模块了
sys.path.append('../video_to_img/') # 将video_to_img路径添加到系统路径中，这样就可以直接导入video_to_img下的模块了

def method1(): # 用于遍历hpc上的tiny数据集文件夹: udiva_tiny
    # mac_animals_recordings_train_img = '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/recordings/animals_recordings_train_img'
    # mac_animals_recordings_val_img = '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/val/recordings/animals_recordings_val_img'
    # mac_animals_recordings_test_img = '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/test/recordings/animals_recordings_test_img'
    
    linux_animals_recordings_train_img = '/home/zl525/rds/hpc-work/datasets/udiva_tiny/train/recordings/animals_recordings_train_img'
    linux_animals_recordings_val_img = '/home/zl525/rds/hpc-work/datasets/udiva_tiny/val/recordings/animals_recordings_val_img'
    linux_animals_recordings_test_img = '/home/zl525/rds/hpc-work/datasets/udiva_tiny/test/recordings/animals_recordings_test_img'
    
    video_dir_list = [linux_animals_recordings_train_img, linux_animals_recordings_val_img, linux_animals_recordings_test_img]
    
    for video_dir in video_dir_list:
        # 遍历文件夹下的所有文件夹
        for session_id  in  os.listdir(video_dir):
            # 如果当前遍历到的文件夹里没有FC1_A和FC2_A两个子文件夹并且没有.DS_Store文件夹，就执行shell 命令
            if  (not os.path.exists(os.path.join(video_dir, session_id,  'FC1_A' )) and not os.path.exists(os.path.join(video_dir, session_id,  'FC2_A' ))) and session_id !=  '.DS_Store' :
                # 执行shell 命令
                print( '准备执行命令：python3 ./udiva_video_to_image.py --video-dir '  + os.path.join(video_dir, session_id) +  ' --output-dir '  + os.path.join(video_dir, session_id))
                os.system( 'python3 ./udiva_video_to_image.py --video-dir '  + os.path.join(video_dir, session_id) +  ' --output-dir '  + os.path.join(video_dir, session_id))
                print( 'session_id: ', session_id, '当前命令执行完毕' )
                # continue
            else:
                # 提示已经存在
                print('session_id: ', session_id, '已经存在FC1_ 前缀的文件夹 和 FC2_前缀开头的子文件夹，不需要执行命令')
            
                # # 删除FC1_ 前缀的文件夹 和 FC2_前缀开头的子文件夹
                # print('session_id: ', session_id, '已经存在FC1_ 前缀的文件夹 和 FC2_前缀开头的子文件夹，准备删除FC1_ 前缀的文件夹 和 FC2_前缀开头的子文件夹')
                # for dir_name in [ 'FC1_A/',  'FC2_A/',  'FC1_G/',  'FC2_G/',  'FC1_L/',  'FC2_L/',  'FC1_T/',  'FC2_T/']:
                #     if os.path.exists(os.path.join(video_dir, session_id, dir_name)):
                #         print('session_id: ', session_id, '执行命令： rm -rf ' + os.path.join(video_dir, session_id, dir_name))
                #         os.system('rm -rf ' + os.path.join(video_dir, session_id, dir_name))

def method2(): # 用于遍历hpc上的全量数据集文件夹: udiva_full
    frame_num = "5" #frame_num表示每秒抽取frame_num数量的帧。如果fps=25, 则每秒抽取25/frame_num=5帧, 每秒抽取的帧数为5帧
    hpc_recordings_train_img = '/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings'
    hpc_recordings_val_img = '/home/zl525/rds/hpc-work/datasets/udiva_full/val/recordings'
    hpc_recordings_test_img = '/home/zl525/rds/hpc-work/datasets/udiva_full/test/recordings'
    
    video_dir_list = [hpc_recordings_train_img, hpc_recordings_val_img, hpc_recordings_test_img]
    
    count_video = 0
    for video_dir in video_dir_list:
        # video_dir = hpc_recordings_train_img, hpc_recordings_val_img, hpc_recordings_test_img
        print('=============================================================================')
        print('video_dir: ', video_dir)
        count_dir = 0
        for dir in os.scandir(video_dir):
            # 如果 dir.name 以 _wav 结尾，就跳过
            if dir.name.endswith('_wav'):
                continue
            # 如果dir.name以ghost或者lego开头，就跳过 (避免删除ghost相关的文件夹) 即 ghost 相关文件夹目前还是每秒抽一帧
            if dir.name.startswith('ghost') or dir.name.startswith('lego') or dir.name.startswith('animals'):
            # if dir.name.startswith('ghost') or dir.name.startswith('talk'):
                continue
            print('*********************')
            print('task dir: ', dir) # dir: <DirEntry 'talk_recordings_train_img'> <DirEntry 'ghost_recordings_val_img'>
            count_session = 0
            for session_id in os.scandir(dir):
                # print('session_id: ', session_id)
                # 如果当前遍历到的文件夹里没有FC1_ 前缀的文件夹 和 FC2_前缀开头的子文件夹 就执行shell 命令
                if  (not os.path.exists(os.path.join(video_dir, dir, session_id, 'FC1_A')) 
                    and not os.path.exists(os.path.join(video_dir, dir, session_id, 'FC2_A')) \
                    and not os.path.exists(os.path.join(video_dir, dir, session_id, 'FC1_G')) \
                    and not os.path.exists(os.path.join(video_dir, dir, session_id, 'FC2_G')) \
                    and not os.path.exists(os.path.join(video_dir, dir, session_id, 'FC1_L')) \
                    and not os.path.exists(os.path.join(video_dir, dir, session_id, 'FC2_L')) \
                    and not os.path.exists(os.path.join(video_dir, dir, session_id, 'FC1_T')) \
                    and not os.path.exists(os.path.join(video_dir, dir, session_id, 'FC2_T')) \
                    ):
                    # continue
                
                    # # 处理方式1. 从mp4视频文件中提取帧图片，而不是人脸图片，即包含了背景
                    # print('session_id: ', session_id, ', 运行 python3 ./udiva_video_to_image.py --video-dir ' + os.path.join(video_dir, dir, session_id) + ' --output-dir ' + os.path.join(video_dir, dir, session_id), ' --frame-num ' + frame_num)
                    # os.system('python3 ./udiva_video_to_image.py --video-dir '  + os.path.join(video_dir, dir, session_id) +  ' --output-dir '  + os.path.join(video_dir, dir, session_id) + ' --frame-num ' + frame_num)
                    
                    # 处理方式2. 从mp4视频文件中提取人脸图片，即不包含背景
                    print('session_id: ', session_id, ', 运行 python3 ./udiva_video_to_face.py --video-dir ' + os.path.join(video_dir, dir, session_id) + ' --output-dir ' + os.path.join(video_dir, dir, session_id), ' --frame-num ' + frame_num)
                    os.system('python3 ./udiva_video_to_face.py --video-dir '  + os.path.join(video_dir, dir, session_id) +  ' --output-dir '  + os.path.join(video_dir, dir, session_id) + ' --frame-num ' + frame_num)
                    
                    # 跳过
                    # continue
                else:
                    continue
                    # print('session_id: ', session_id, '已经存在FC1_ 前缀 和 FC2_前缀开头的子文件夹，不需要执行命令')
                    
                    # 删除FC1_ 前缀的文件夹 和 FC2_前缀开头的子文件夹
                    print('session_id: ', session_id, '已经存在FC1_和FC2_前缀开头的子文件夹，准备删除')
                    for dir_name in [ 'FC1_A/',  'FC2_A/',  'FC1_G/',  'FC2_G/',  'FC1_L/',  'FC2_L/',  'FC1_T/',  'FC2_T/']:
                        if os.path.exists(os.path.join(video_dir, dir, session_id, dir_name)):
                            print('session_id: ', session_id, '执行命令： rm -rf ' + os.path.join(video_dir, dir, session_id, dir_name))
                            os.system('rm -rf ' + os.path.join(video_dir, dir, session_id, dir_name))
        #         count_session += 1
        #         if count_session == 1:
        #             break
        #     count_dir += 1
        #     if count_dir == 1:
        #         break
        # count_video += 1
        # if count_video == 1:
        #     break

if __name__ == '__main__':
    # method1()
    method2()
    print('main.py done')


# nohup python3 -u main.py >nohup_`date +'%m-%d-%H:%M:%S'`.out 2>&1 &
# [1] 3601098
# python nohup运行时print不输出显示，解决办法：https://blog.csdn.net/voidfaceless/article/details/106363925



""" debug
173179: 日志显示ok，生辰了图片
python3 ./video_to_image.py --video-dir /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/173179 --output-dir /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/173179

011034: 日志显示ok，但是没有生成图片
python3 ./video_to_image.py --video-dir /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/011034 --output-dir /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/011034

034133: 
python3 ./video_to_image.py --video-dir /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/034133 --output-dir /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/034133

"""