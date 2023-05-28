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
import datetime
import multiprocessing
from multiprocessing import Pool
import time

def process_udiva_tiny(): # 用于遍历hpc上的tiny数据集文件夹: udiva_tiny
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
                pass


def process_udiva_full(): # 用于遍历hpc上的全量数据集文件夹: udiva_full
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
        count_task_dir = 0
        for task_dir in os.scandir(video_dir):
            if task_dir.name.endswith('_wav'): # 如果 task_dir.name 以 _wav 结尾，就跳过
                continue
            # if task_dir.name.startswith('ghost') or task_dir.name.startswith('lego') or task_dir.name.startswith('animals'): # 如果dir.name以ghost或者lego开头，就跳过 (避免删除ghost相关的文件夹) 即 ghost 相关文件夹目前还是每秒抽一帧
            #     continue 
            if task_dir.name.split('_')[0] in ['animals', 'ghost', 'lego']:
                print('task_dir.name:', task_dir.name, ', will be skipped')
                continue
            print('*********************')
            print('task_dir: ', task_dir) # task_dir: <DirEntry 'talk_recordings_train_img'> <DirEntry 'ghost_recordings_val_img'>
            count_session = 0
            for session_id in os.scandir(task_dir):
                # print('session_id: ', session_id)
                session_dir = os.path.join(video_dir, task_dir, session_id)
                # 如果当前遍历到的文件夹里没有FC1_ 前缀的文件夹 和 FC2_前缀开头的子文件夹 就执行shell 命令
                if  (not os.path.exists(os.path.join(video_dir, task_dir, session_id, 'FC1_A')) 
                    and not os.path.exists(os.path.join(video_dir, task_dir, session_id, 'FC2_A')) \
                    and not os.path.exists(os.path.join(video_dir, task_dir, session_id, 'FC1_G')) \
                    and not os.path.exists(os.path.join(video_dir, task_dir, session_id, 'FC2_G')) \
                    and not os.path.exists(os.path.join(video_dir, task_dir, session_id, 'FC1_L')) \
                    and not os.path.exists(os.path.join(video_dir, task_dir, session_id, 'FC2_L')) \
                    and not os.path.exists(os.path.join(video_dir, task_dir, session_id, 'FC1_T')) \
                    and not os.path.exists(os.path.join(video_dir, task_dir, session_id, 'FC2_T')) \
                    ):
                    
                    ######## 处理方式0. 检查压缩包: 如果有FC1_和FC2_前缀开头的tar.gz文件，就执行解压命令或删除命令, 无需运行udiva_video_to_face.py提取人脸
                    # os.chdir(session_dir) # change to session_dir
                    # print('change dir: current dir is ', os.getcwd())
                    # for FC_name in [ 'FC1_A/',  'FC2_A/',  'FC1_G/',  'FC2_G/',  'FC1_L/',  'FC2_L/',  'FC1_T/',  'FC2_T/']:
                    #     if os.path.exists(session_dir + '/' + FC_name[:-1] + '.tar.gz'): # e.g. 如果存在 FC1_A.tar.gz 文件，就执行解压命令
                    #         ### 解压
                    #         # untar_command = 'tar -zxvf ' + FC_name[:-1] + '.tar.gz' + ' >/dev/null 2>&1'
                    #         # print('session_id: ', session_id.name, '执行解压命令：', untar_command)
                    #         # os.system(untar_command)
                            
                    #         ### 删除压缩包
                    #         rm_command = 'rm ' + FC_name[:-1] + '.tar.gz'
                    #         print('session_id: ', session_id.name, '执行删除命令：', rm_command)
                    #         os.system(rm_command)
                
                    ######## 处理方式1. 从mp4视频文件中提取完整帧图片，不是人脸图片，即包含了背景
                    # print('session_id: ', session_id, ', 运行 python3 ./udiva_video_to_image.py --video-dir ' + os.path.join(video_dir, task_dir, session_id) + ' --output-dir ' + os.path.join(video_dir, task_dir, session_id), ' --frame-num ' + frame_num)
                    # os.system('python3 ./udiva_video_to_image.py --video-dir '  + os.path.join(video_dir, task_dir, session_id) +  ' --output-dir '  + os.path.join(video_dir, task_dir, session_id) + ' --frame-num ' + frame_num)
                    
                    ######## 处理方式2. 从mp4视频文件中提取人脸图片，即不包含背景
                    # print('session_id: ', session_id, ', 运行 python3 ./udiva_video_to_face.py --video-dir ' + os.path.join(video_dir, task_dir, session_id) + ' --output-dir ' + os.path.join(video_dir, task_dir, session_id), ' --frame-num ' + frame_num)
                    # os.system('python3 ./udiva_video_to_face.py --video-dir '  + os.path.join(video_dir, task_dir, session_id) +  ' --output-dir '  + os.path.join(video_dir, task_dir, session_id) + ' --frame-num ' + frame_num)
                    
                    pass
                else:
                    # continue
                    # print('session_id: ', session_id, '已经存在FC1_ 前缀 和 FC2_前缀开头的子文件夹，不需要执行命令')
                    
                    ##### 处理方式1: 删除FC1_ 前缀的文件夹 和 FC2_前缀开头的子文件夹
                    # print('session_id: ', session_id, '已经存在FC1_和FC2_前缀开头的子文件夹，准备删除')
                    # for FC_name in [ 'FC1_A/',  'FC2_A/',  'FC1_G/',  'FC2_G/',  'FC1_L/',  'FC2_L/',  'FC1_T/',  'FC2_T/']:
                    #     FC_dir = os.path.join(video_dir, task_dir, session_id, FC_name)
                    #     if os.path.exists(FC_dir):
                    #         print('session_id: ', session_id, '执行命令： rm -rf ' + FC_dir)
                    #         os.system('rm -rf ' + FC_dir)
                    
                    ##### 处理方式2: 将 FC1_ 前缀和 FC2_ 前缀开头的子文件夹压缩成tar.gz文件, 并删除原来的FC1_ 前缀和 FC2_ 前缀开头的子文件夹
                    # # print('session_id: ', session_id, '已经存在FC1_和FC2_前缀开头的子文件夹，准备压缩成tar.gz文件并删除原先的文件夹')
                    # os.chdir(session_dir) # change to session_dir
                    # print('change dir: current dir is ', os.getcwd())
                    # for FC_name in [ 'FC1_A/',  'FC2_A/',  'FC1_G/',  'FC2_G/',  'FC1_L/',  'FC2_L/',  'FC1_T/',  'FC2_T/']:
                    #     FC_dir = os.path.join(video_dir, task_dir, session_id, FC_name)
                    #     if os.path.exists(FC_dir) and (os.path.exists(session_dir + '/' + FC_name[:-1] + '.tar.gz') == False): # e.g. 如果存在 FC1_A/ 文件夹 且 不存在FC1_A.tar.gz压缩包，就执行压缩命令
                    #         ### 压缩打包
                    #         # tar_command = 'tar -zcvf ' + session_dir + '/' + FC_name[:-1] + '.tar.gz ' + FC_dir
                    #         tar_command = 'tar -zcvf ' + FC_name[:-1] + '.tar.gz ' + FC_name + ' >/dev/null 2>&1'
                    #         print('session_id: ', session_id.name, '执行压缩命令：', tar_command)
                    #         os.system(tar_command)
                            
                    #         ### 删除原来的文件夹
                    #         rm_command = 'rm -rf ' + FC_dir
                    #         print('session_id: ', session_id.name, '执行删除命令：', rm_command)
                    #         os.system(rm_command)

                    ##### 处理方式3: 后置处理, 对齐FC1和FC2的人脸帧数id，使得两个文件夹中的人脸帧数id是一致的, 即FC1和FC2文件夹中的所有人脸都是一一对应的，时间上是一致的。
                    align_face_id(session_dir)
                    pass
        #         count_session += 1
        #         if count_session == 1:
        #             break
        #     count_task_dir += 1
        #     if count_task_dir == 1:
        #         break
        # count_video += 1
        # if count_video == 1:
        #     break
        pass


def process_noxi(): # 用于遍历hpc上的NoXI数据集文件夹: noxi  (/home/zl525/rds/hpc-work/datasets/noxi)
    frame_num = "5" #frame_num表示每秒抽取frame_num数量的帧。如果fps=25, 则每秒抽取25/frame_num=5帧, 每秒抽取的帧数为5帧
    noxi_full_dir = '/home/zl525/rds/hpc-work/datasets/noxi_full'
    noxi_tiny_dir = '/home/zl525/rds/hpc-work/datasets/noxi_tiny'
    noxi_temp_img_dir = '/home/zl525/rds/hpc-work/datasets/noxi_temp_test'
    
    video_dir_list = [noxi_full_dir]
    # video_dir_list = [noxi_temp_img_dir]
    
    use_pool = True # 是否使用多进程
    if use_pool:
        num_process = multiprocessing.cpu_count() - 5
        # num_process = 2
        num_process = print('num_process:', num_process) # login-icelake.hpc.cam.ac.uk: 71
        pool = Pool(num_process)

    count_video = 0
    for video_dir in video_dir_list:
        # video_dir = noxi_dir
        print('=============================================================================')
        print('video_dir: ', video_dir)
        count_dir = 0
        for modal_type_dir in os.scandir(video_dir): # modal_type_dir.name: img, wav, label
            if modal_type_dir.name == "img":
                count_session = 0
                for session_id in sorted(os.scandir(modal_type_dir), key=lambda x: x.name):
                    session_dir = os.path.join(video_dir, modal_type_dir, session_id)
                    # 如果当前遍历到的文件夹里没有 Expert 和 Novice 前缀开头的子文件夹 就执行shell 命令
                    if  (not os.path.exists(os.path.join(video_dir, modal_type_dir, session_id, 'Expert_video')) 
                        and not os.path.exists(os.path.join(video_dir, modal_type_dir,  session_id, 'Novice_video'))):
                        
                        # # 如果session文件夹名称包含下划线"_"，就重命名，只保留第一个下划线的前面部分，即只保留session的id, e.g. 004_2016-03-18_Paris -> 004
                        # if session_id.name.find('_') != -1:
                        #     session_id_new = session_id.name.split('_')[0]
                        #     # 将 session_id的name 重命名为 session_id_new
                        #     print('rename')
                        #     os.rename(os.path.join(video_dir, modal_type_dir, session_id), os.path.join(video_dir, modal_type_dir, session_id_new))
                        
                        ####### 处理方式0. 检查压缩包: 如果有Expert_和Novice_前缀开头的tar.gz文件，就执行解压命令或删除命令, 无需运行udiva_video_to_face.py提取人脸
                        # os.chdir(session_dir) # change to session_dir
                        # print('change dir: current dir is ', os.getcwd())
                        # use_pool = False
                        # for people_name in [ 'Expert_video/',  'Novice_video/']:
                        #     if os.path.exists(session_dir + '/' + people_name[:-1] + '.tar.gz') and (os.path.exists(os.path.join(session_dir, people_name)) == False): # e.g. 如果存在 Expert_video.tar.gz 压缩包 且 不存在 Expert_video/ 文件夹
                        #         ### 解压
                        #         untar_command = 'tar -xzvf ' + people_name[:-1] + '.tar.gz' + ' >/dev/null 2>&1'
                        #         print('session_id: ', session_id.name, '执行解压命令：', untar_command)
                        #         os.system(untar_command)
                                
                        #         ### 删除压缩包
                        #         rm_command = 'rm ' + people_name[:-1] + '.tar.gz'
                        #         print('session_id: ', session_id.name, '执行删除命令：', rm_command)
                        #         os.system(rm_command)
                            
                        ######## 处理方式1. 从mp4视频文件中提取完整帧图片，不是人脸图片，即包含了背景
                        # print('session_id: ', session_id.name, ', 运行 python3 ./udiva_video_to_image.py --video-dir ' + session_dir + ' --output-dir ' + session_dir, ' --frame-num ' + frame_num)
                        # os.system('python3 ./udiva_video_to_image.py --video-dir '  + session_dir +  ' --output-dir '  + session_dir + ' --frame-num ' + frame_num)
                        
                        ######## 处理方式2.1. 从mp4视频文件中提取人脸图片，即不包含背景，使用的模型：OpenCV’s deep neural network, Reference: https://towardsdatascience.com/extracting-faces-using-opencv-face-detection-neural-network-475c5cd0c260
                        # print('session_id: ', session_id.name, ', 运行 python3 ./udiva_video_to_face.py --video-dir ' + session_dir + ' --output-dir ' + session_dir, ' --frame-num ' + frame_num)
                        # os.system('python3 ./udiva_video_to_face.py --video-dir '  + session_dir +  ' --output-dir '  + session_dir + ' --frame-num ' + frame_num)

                        ####### 处理方式2.2. 从mp4视频文件中提取人脸图片，即不包含背景，使用的模型：MTCNN, Reference: Script for Video Face Extraction https://github.com/liaorongfan/DeepPersonality/blob/main/datasets/README.md
                        # print('session_id: ', session_id.name, ', 运行 python3 video_to_face/face_img_extractor.py --video-path ' + session_dir + ' --output-dir ' + session_dir)
                        # os.system('python3 video_to_face/face_img_extractor.py --video-path '  + session_dir +  ' --output-dir '  + session_dir)
                        pass
                    else:
                        # print('session_id: ', session_id.name, '已经存在Expert 和 Novice前缀开头的子文件夹')
                        
                        # 如果session文件夹名称包含下划线"_"，就重命名，只保留第一个下划线的前面部分，即只保留session的id, e.g. 004_2016-03-18_Paris -> 004
                        # if session_id.name.find('_') != -1:
                        #     session_id_new = session_id.name.split('_')[0]
                        #     # 将 session_id的name 重命名为 session_id_new
                        #     print('rename: ', session_id, ' -> ', session_id_new)
                        #     os.rename(os.path.join(video_dir, modal_type_dir, session_id), os.path.join(video_dir, modal_type_dir, session_id_new))
                        
                        ####### 1. 删除FC1_ 前缀的文件夹 和 FC2_前缀开头的子文件夹
                        # print('session_id: ', session_id.name, '已经存在Expert和Novice前缀开头的子文件夹，准备删除')
                        # for dir_name in ['Expert_video/', 'Novice_video/']:
                        #     processed_frame_dir = os.path.join(video_dir, session_id, dir_name)
                        #     if os.path.exists(processed_frame_dir):
                        #         print('session_id: ', session_id.name, '执行命令： rm -rf ' + processed_frame_dir)
                        #         # os.system('rm -rf ' + processed_frame_dir)
                        
                        ######## 2. 后置处理: 对齐Expert和Novice的人脸帧数id，使得两个文件夹中的人脸帧数id是一致的, 即Expert和Novice文件夹中的所有人脸都是一一对应的，时间上是一致的。
                        # align_face_id(session_dir) # session_dir e.g. DeepPersonality/datasets/noxi_full/img/001
                        
                        # ####### 3. 将 Expert_ 前缀和 Novice_ 前缀开头的子文件夹压缩成tar.gz文件, 并删除原来的Expert_ 前缀和 Novice_  前缀开头的子文件夹
                        print('session_id: ', session_id.name, '已经存在Expert_和Novice_前缀开头的子文件夹，准备压缩成tar.gz文件并删除原先Expert_和Novice_前缀开头的文件夹')
                        people_dir_list = ['Expert_video/', 'Novice_video/']
                        
                        ##### 处理方式3.1. 多线程并行压缩打包 (速度快)
                        if use_pool:
                            pool.apply_async(convert_img_dir_to_targz, args=(session_dir, people_dir_list)) # 多线程并行压缩打包
                        # convert_img_dir_to_targz(session_dir, people_dir_list) # 单线程串行压缩打包
                        
                        ##### 处理方式3.2. 单线程串行压缩打包（速度慢）
                        # os.chdir(session_dir) # change to session_dir
                        # print('change dir: current dir is ', os.getcwd())
                        # for people_name in ['Expert_video/', 'Novice_video/']:
                        #     # session_dir = os.path.join(video_dir, modal_type_dir, session_id)
                        #     people_dir = os.path.join(session_dir, people_name)
                        #     if os.path.exists(people_dir) and (os.path.exists(session_dir + '/' + people_name[:-1] + '.tar.gz') == False): # e.g. 如果存在 Expert_video/ 文件夹 且 不存在Expert_video.tar.gz压缩包，就执行压缩命令
                        #         ### 压缩打包
                        #         # tar_command = 'tar -zcvf ' + session_dir + '/' + people_name[:-1] + '.tar.gz ' + people_dir
                        #         tar_command = 'tar -zcvf ' + people_name[:-1] + '.tar.gz ' + people_name + ' >/dev/null 2>&1'
                        #         print('session_id: ', session_id.name, '执行压缩命令：', tar_command)
                        #         os.system(tar_command)
                                
                        #         ### 删除原来的文件夹
                        #         rm_command = 'rm -rf ' + people_dir
                        #         print('session_id: ', session_id.name, '执行删除命令：', rm_command)
                        #         os.system(rm_command)
                        pass
                #     count_session += 1
                #     if count_session == 2:
                #         break
                # count_dir += 1
                # if count_dir == 2:
                #     break
                pass
    if use_pool:
        print('Waiting for all subprocesses done...')
        pool.close()
        pool.join()
        print('All subprocesses done.')

def convert_img_dir_to_targz(session_dir, people_dir_list):
    """将包含人脸图片的文件夹压缩成tar.gz文件，并删除原来的文件夹
    例如：将datasets/noxi_temp_test/img/001/目录下的 Expert_video/文件夹压缩成 Expert_video.tar.gz文件，并删除原来的Expert_video/文件夹

    Args:
        session_dir (): e.g. /home/zl525/rds/hpc-work/datasets/noxi_full/img/001
        people_dir_list (): e.g. ['Expert_video/', 'Novice_video/']
    """
    os.chdir(session_dir) # change to session_dir
    print('\n******************** current dir: ', os.getcwd(), '********************')
    # print('[convert_img_dir_to_targz] current dir is ', os.getcwd())
    session_id = session_dir.split('/')[-1]
    print('[convert_img_dir_to_targz] session_id:', session_id, ', session_dir: ', session_dir, ', people_dir_list: ', people_dir_list)
    for people_name in people_dir_list:
        people_dir = os.path.join(session_dir, people_name)
        if os.path.exists(people_dir) and (os.path.exists(session_dir + '/' + people_name[:-1] + '.tar.gz') == False): # e.g. 如果存在 Expert_video/ 文件夹 且 不存在Expert_video.tar.gz压缩包，就执行压缩命令
            targz_name = os.path.join(session_dir, people_name[:-1] + '.tar.gz')
            ### 压缩打包
            # tar_command = 'tar -czvf ' + people_name[:-1] + '.tar.gz ' + people_dir + ' >/dev/null 2>&1'
            # tar_command = 'tar -czvf ' + targz_name + ' ' + people_dir + ' >/dev/null 2>&1' # 会导致压缩包中的文件夹层级过多，如何解决？ 可以使用 -C 参数，指定压缩包中的文件夹层级，例如：tar -zcvf test.tar.gz -C /home/zl525/rds/hpc-work/datasets/noxi_full/img/001/ Expert_video/ >/dev/null 2>&1
            tar_command = 'tar -czvf ' + targz_name + ' ' + people_name + ' >/dev/null 2>&1'
            # session_id = session_dir.split('/')[-1]
            print('session_id: ', session_id, '执行压缩命令：', tar_command)
            os.system(tar_command)
            
            ### 删除原来的文件夹
            rm_command = 'rm -rf ' + people_dir
            print('session_id: ', session_id, '执行删除命令：', rm_command)
            os.system(rm_command)
            # time.sleep(5)

def align_face_id(session_dir):
    """
    session_dir: e.g. DeepPersonality/datasets/noxi_full/img/001
    1、session_dir目录下有一对视频对应的帧文件夹，即两个包含有人脸帧的文件夹，例如Expert和Novice文件夹，文件夹中的文件名格式为：face_1.jpg, face_2.jpg, face_3.jpg, ...
    2、分别从Expert和Novice文件夹识别出所有的文件名中下划线_右边的后缀序号的列表, 例如从Expert_video目录中的face_18.jpg文件中提取出id:18
    3、然后找出Expert和Novice中非公有的序号（例: face_18.jpg只在Expert_video目录中出现但是没有出现在Novice_vide目录中），最后删除这些序号对应的文件(例:face_18.jpg)
    验证是否对齐成功: 
        cd /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/ && find . -maxdepth 2 -type d | while read dir; do count=$(find "$dir" -type f | wc -l); echo "$dir : $count"; done
        cd /home/zl525/rds/hpc-work/datasets/udiva_full/val/recordings/talk_recordings_val_img && find . -maxdepth 2 -type d | while read dir; do count=$(find "$dir" -type f | wc -l); echo "$dir : $count"; done
    经过测试，删除的速度非常快，只需要几秒钟，所以不需要考虑删除的效率问题，使用for循环遍历删除即可。
    """
    print('\n---------------------------------------------------------------------------------------')
    # 从session_dir目录下提取出两个子文件夹名称，作为dir_name1和dir_name2
    dir_names = [dir_name for dir_name in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, dir_name))]
    if len(dir_names) != 2:
        print('session_dir: ', session_dir, 'num of dirs is not 2, please check!')
        return
    dir_name1, dir_name2 = dir_names[0], dir_names[1] # dir_name1: Expert_video, dir_name2: Novice_video
    
    path1 = os.path.join(session_dir, dir_name1)
    path2 = os.path.join(session_dir, dir_name2)
    # print('session_dir: ', session_dir, '\npath1: ', path1, '\npath2: ', path2)
    
    frame_files1 = set(os.listdir(path1))
    frame_files2 = set(os.listdir(path2))
    
    ids_set1 = sorted([int(file.split('_')[1].split('.')[0]) for file in frame_files1])
    ids_set2 = sorted([int(file.split('_')[1].split('.')[0]) for file in frame_files2])
    # print('\n\nsession_dir: ', session_dir, '\nids_set1: ', ids_set1, '\nids_set2: ', ids_set2)
    
    if ids_set1 == ids_set2:
        print('session_dir: ', session_dir, 'ids_set1 == ids_set2, no need to align!')
        return
    
    unique_ids_1 = set(ids_set1) - set(ids_set2)
    unique_ids_2 = set(ids_set2) - set(ids_set1)
    print('session_dir: ', session_dir, '\nunique_ids_1: ', unique_ids_1, '\nunique_ids_2: ', unique_ids_2)
    
    for uid in unique_ids_1:
        file_to_delete = os.path.join(path1, 'face_' + str(uid) + '.jpg')
        if os.path.exists(file_to_delete):
            print('delete: ', file_to_delete)
            os.remove(file_to_delete)
    
    for uid in unique_ids_2:
        file_to_delete = os.path.join(path2, 'face_' + str(uid) + '.jpg')
        if os.path.exists(file_to_delete):
            print('delete: ', file_to_delete)
            os.remove(file_to_delete)
    pass


'''
def align_face_id(session_dir, dir_name1, dir_name2):
    """
    1、Expert和Novice文件夹 中的文件名格式为：face_1.jpg, face_2.jpg, face_3.jpg, ...
    2、分别从Expert和Novice文件夹识别出所有的文件名中下划线_右边的后缀序号的列表, 例如从Expert_video目录中的face_18.jpg文件中提取出id:18
    3、然后找出Expert和Novice中非公有的序号（例: face_18.jpg只在Expert_video目录中出现但是没有出现在Novice_vide目录中），最后删除这些序号对应的文件(例:face_18.jpg)
    """
    expert_path = os.path.join(session_dir, dir_name1)
    novice_path = os.path.join(session_dir, dir_name2)
    
    expert_files = set(os.listdir(expert_path))
    novice_files = set(os.listdir(novice_path))
    
    expert_ids = sorted([int(file.split('_')[1].split('.')[0]) for file in expert_files])
    novice_ids = sorted([int(file.split('_')[1].split('.')[0]) for file in novice_files])
    # print('\n\nsession_dir: ', session_dir, '\nexpert_ids: ', expert_ids, '\nnovice_ids: ', novice_ids)
    
    if expert_ids == novice_ids:
        print('session_dir: ', session_dir, '中的Expert和Novice文件夹中的人脸帧数id是一致的, 不需要删除文件')
        return
    
    expert_unique_ids = set(expert_ids) - set(novice_ids)
    novice_unique_ids = set(novice_ids) - set(expert_ids)
    print('session_dir: ', session_dir, '\nexpert_unique_ids: ', expert_unique_ids, '\nnovice_unique_ids: ', novice_unique_ids)
    
    for uid in expert_unique_ids:
        file_to_delete = os.path.join(expert_path, 'face_' + str(uid) + '.jpg')
        if os.path.exists(file_to_delete):
            print('删除文件：', file_to_delete)
            # os.remove(file_to_delete)
    
    for uid in novice_unique_ids:
        file_to_delete = os.path.join(novice_path, 'face_' + str(uid) + '.jpg')
        if os.path.exists(file_to_delete):
            print('删除文件：', file_to_delete)
            # os.remove(file_to_delete)
'''


def extract_face_parallel(part_id=1):
    """process noxi dataset, extract face images from videos, using multiprocess
       经过测试，如果上一次的nohup处理中断了，可以直接再次运行本函数，因为 face_img_extractor.py 中有相应的判断逻辑，即如果已经存在对应的人脸图片face_{cnt}.jpg，就不会再次提取保存到对应的文件夹中，详见def reduce_frame_rate(self, fps_new)函数的实现 
    """
    ###################################### some test commands: ######################################
    # 复制文件夹
    # cp -r /home/zl525/code/DeepPersonality/datasets/noxi_full/img/003 /home/zl525/rds/hpc-work/datasets/noxi_temp_test/img
    
    # 删除已经存在的Expert_video/ Novice_video/ 文件夹
    # cd /home/zl525/code/DeepPersonality/datasets/noxi_temp_test/img && rm -rf 001/Expert_video/ 002/Expert_video/ 003/Expert_video/ && tree
    # cd /home/zl525/code/DeepPersonality/datasets/noxi_temp_test/img && rm -rf 001/Expert_video/ 002/Expert_video/ 003/Expert_video/ 001/Novice_video/ 002/Novice_video/ 003/Novice_video/ && tree
    
    # 查看文件夹中的文件数量
    # ls 003/Expert_video/ | wc -l; ls 003/Novice_video/ | wc -l
    # ls 001/Expert_video/ -v; ls 001/Novice_video/ -v; ls 003/Expert_video/ -v; ls 003/Novice_video/ -v
    
    ###################################### 处理方式1: ######################################
    # noxi_full_img_dir = '/home/zl525/rds/hpc-work/datasets/noxi_full/img'
    # noxi_temp_img_dir = '/home/zl525/code/DeepPersonality/datasets/noxi_temp_test/img'
    # level = "dir"
    
    # video_path = noxi_full_img_dir # 正式
    # # video_path = noxi_temp_img_dir # 测试
    # print('运行 python3 video_to_face/face_img_extractor.py --video-path ' + video_path + ' --level ' + level)
    # os.system('python3 video_to_face/face_img_extractor.py --video-path ' + video_path + ' --level ' + level)
    
    ###################################### 处理方式2: 将dir分为part1, part2, part3, 分别使用多线程并行处理 ######################################
    #### 为了提高效率，将dir分为part1, part2, part3, 分别使用多线程并行处理
    ### NoXi 数据集上的Linux 命令:
        # cd /home/zl525/code/DeepPersonality/datasets/noxi_full/img
        # mkdir part1 part2 part3
        ## 当前目录下的所有文件夹一共有87个，分别为001到084文件夹以及 part1 part2 part3 文件夹. 84个文件节分为三个part: 84/3=28, 即将001-028文件夹移动到part1文件夹中，029-056文件夹移动到part2文件夹中，057-084文件夹移动到part3文件夹中
        # mv 0{01..28} part1/ ; mv 0{29..56} part2/  ; mv 0{57..84} part3/
        ## 人脸提取完成后，将part1, part2, part3文件夹中的文件夹合并到当前目录下, 恢复原来的文件结构
        # mv part1/* ./  ; mv part2/* ./  ; mv part3/* ./ ; rm -rf part1/ part2/ part3/
    ### UDIVA 数据集上的Linux 命令:
        # cd /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img
        ## 当前目录下的所有文件夹一共有115个，为002003到191192文件夹. 1-38个:002003-034133, 39-76个:035040-102176, 77-115个:106108-191192
        # 115个文件节分为三个part: 115/3=38, 即将002003-034133文件夹移动到part1文件夹中，035040-102176文件夹移动到part2文件夹中，106108-191192文件夹移动到part3文件夹中
        # mkdir part1 part2 part3
        # 是否可以: mv 0{02..34}0{03..133} part1/ ? 不建议，因为实际文件夹id名字不连续，这样做会将范围内的所有数字遍历。还是用脚本最好，见：leenote/video_to_img/split_session.sh
        ## 人脸提取完成后，将part1, part2, part3文件夹中的文件夹合并到当前目录下, 恢复原来的文件结构
        # mv part1/* ./  ; mv part2/* ./  ; mv part3/* ./ ; rm -rf part1/ part2/ part3/
    
    # NoXi 数据集(未划分训练集和测试集)
    noxi_full_part1 = '/home/zl525/rds/hpc-work/datasets/noxi_full/img/part1'
    noxi_full_part2 = '/home/zl525/rds/hpc-work/datasets/noxi_full/img/part2'
    noxi_full_part3 = '/home/zl525/rds/hpc-work/datasets/noxi_full/img/part3'
    # video_path_part1, video_path_part2, video_path_part3 = noxi_full_part1, noxi_full_part2, noxi_full_part3
    
    # UDIVA 训练集: 115个session 分为三个part
    udiva_full_part1 = '/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/part1'
    udiva_full_part2 = '/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/part2'
    udiva_full_part3 = '/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/part3'
    video_path_part1, video_path_part2, video_path_part3 = udiva_full_part1, udiva_full_part2, udiva_full_part3
    
    
    # UDIVA 验证集: 18个session  UDIVA 测试集: 11个session
    # udiva_val = '/home/zl525/rds/hpc-work/datasets/udiva_full/val/recordings/talk_recordings_val_img'
    # udiva_test = '/home/zl525/rds/hpc-work/datasets/udiva_full/test/recordings/talk_recordings_test_img'
    # video_path_part1, video_path_part2, video_path_part3 = udiva_val, udiva_test, None
    
    
    if int(part_id) == 1:
        video_path = video_path_part1
    elif int(part_id) == 2:
        video_path = video_path_part2
    elif int(part_id) == 3:
        video_path = video_path_part3
    else:
        raise ValueError('part_id should be 1, 2 or 3')
    level = "dir"
    
    print('运行 python3 video_to_face/face_img_extractor.py --video-path ' + video_path + ' --level ' + level)
    os.system('python3 video_to_face/face_img_extractor.py --video-path ' + video_path + ' --level ' + level)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print('main.py start, current time:', start_time.strftime('%Y-%m-%d %H:%M:%S'))
    #### UDIVA ####
    # process_udiva_tiny()
    process_udiva_full()
    
    #### NOXI ####
    ### 单线程循环处理提取人脸图片
    # process_noxi()
    
    ### 多线程并行处理提取人脸图片 (兼容 UDIVA 和 NOXI 数据集) ###
    # if len(sys.argv) > 1:
    #     extract_face_parallel(sys.argv[1])
    # else:
    #     extract_face_parallel() 
    
    print('main.py done, current time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ', duration:', datetime.datetime.now() - start_time, 'seconds')



# conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/leenote/video_to_img/
# 1. 测试脚本： python3 main.py > main.log
# 2. nohup后台跑脚本：nohup python3 main.py >nohup_`date +'%m-%d-%H:%M:%S'`.log 2>&1 &
# 备注: python nohup运行时print不输出显示，解决办法：https://blog.csdn.net/voidfaceless/article/details/106363925

######### 多线程并行处理part1, part2, part3 #########
# conda activate DeepPersonality && cd /home/zl525/code/DeepPersonality/leenote/video_to_img/
# 测试脚本: python3 main.py 3 > main.log
# 训练集正式脚本:
    # nohup python3 -u main.py 1 > log/main_part1_`date +'%m-%d-%H:%M:%S'`.log 2>&1 &
    # nohup python3 -u main.py 2 > log/main_part2_`date +'%m-%d-%H:%M:%S'`.log 2>&1 &
    # nohup python3 -u main.py 3 > log/main_part3_`date +'%m-%d-%H:%M:%S'`.log 2>&1 &
# 验证集和测试集正式脚本:
    # nohup python3 -u main.py 1 > log/main_val_`date +'%m-%d-%H:%M:%S'`.log 2>&1 &
    # nohup python3 -u main.py 2 > log/main_test_`date +'%m-%d-%H:%M:%S'`.log 2>&1 &

# part1: 1253553(login-q1) ; part2:1054363(login-q2); part3: 3454883(login-q3)
# ps -ef | grep "python3 -u main.py" | grep -v grep
# ps -ef | grep "python3 main.py" | grep -v grep
# ps -ef | grep "main.py" | grep -v grep
# ps -p 2577562

## 查看 UDIVA 训练集 part1 2 3 文件夹中的文件数量是否在变化
# cd /home/zl525/code/DeepPersonality/datasets/udiva_full/train/recordings/talk_recordings_train_img/ && find . -maxdepth 1 -type d | while read dir; do count=$(find "$dir" -type f | wc -l); echo "$dir : $count"; done
# cd /home/zl525/code/DeepPersonality/datasets/udiva_full/train/recordings/talk_recordings_train_img/part1 && find . -maxdepth 1 -type d | while read dir; do count=$(find "$dir" -type f | wc -l); echo "$dir : $count"; done
# find . -maxdepth 2 -type d | while read dir; do count=$(find "$dir" -type f | wc -l); echo "$dir : $count"; done

## 查看 UDIVA 验证集和测试集 文件夹中的文件数量是否在变化
# cd /home/zl525/rds/hpc-work/datasets/udiva_full/val/recordings/talk_recordings_val_img && find . -maxdepth 1 -type d | while read dir; do count=$(find "$dir" -type f | wc -l); echo "$dir : $count"; done
# cd /home/zl525/rds/hpc-work/datasets/udiva_full/test/recordings/talk_recordings_test_img && find . -maxdepth 1 -type d | while read dir; do count=$(find "$dir" -type f | wc -l); echo "$dir : $count"; done


""" debug
173179: 日志显示ok，生辰了图片
python3 ./video_to_image.py --video-dir /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/173179 --output-dir /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/173179

011034: 日志显示ok，但是没有生成图片
python3 ./video_to_image.py --video-dir /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/011034 --output-dir /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/011034

034133: 
python3 ./video_to_image.py --video-dir /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/034133 --output-dir /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/034133

"""