'''
Author: leeinscky 1009694687@qq.com
Date: 2022-12-25 01:08:50
LastEditors: leeinscky 1009694687@qq.com
LastEditTime: 2022-12-25 02:12:29
FilePath: /DeepPersonality/lzjnote/video_to_img/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import  os
import sys
sys.path.append('../../datasets/') # 将datasets路径添加到系统路径中，这样就可以直接导入datasets下的模块了
sys.path.append('../video_to_wav/') # 将video_to_img路径添加到系统路径中，这样就可以直接导入video_to_img下的模块了

def main ():
    hpc_recordings_train = '/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings'
    hpc_recordings_val = '/home/zl525/rds/hpc-work/datasets/udiva_full/val/recordings'
    hpc_recordings_test = '/home/zl525/rds/hpc-work/datasets/udiva_full/test/recordings'
    
    recordings_dir_list = [hpc_recordings_train, hpc_recordings_val, hpc_recordings_test]
    
    for recordings_dir in recordings_dir_list:
        print('recordings_dir: ', recordings_dir) # recordings_dir: /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings
        
        # 遍历recordings_dir时 如果遍历到的文件夹dir后缀是_img,就执行下面的代码
        for task_dir in os.scandir(recordings_dir):
            
            if task_dir.name.endswith('_img'):
                task_dir_img = task_dir
                print('task_dir_img: ', task_dir_img) # task_dir_img:  <DirEntry 'talk_recordings_train_img'>
                
                # wav目录的名称
                task_dir_wav = task_dir.name.replace('_img', '_wav') # animals_recordings_train_wav
                task_dir_wav_full_path = os.path.join(recordings_dir, task_dir_wav)
                # print('task_dir_wav_full_path: ', task_dir_wav_full_path) #  task_dir_wav_full_path: /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_wav
                # 如果wav目录不存在，就创建
                if not os.path.exists(task_dir_wav_full_path):
                    os.makedirs(task_dir_wav_full_path)
                
                # print('xxxx os.path.join(task_dir_wav_full_path, session_id): ', os.path.join(task_dir_wav_full_path, 'session_id'))
                
                for session_id in os.scandir(task_dir_img):
                    img_path = os.path.join(task_dir_img, session_id)
                    wav_path = os.path.join(task_dir_wav_full_path, session_id.name) # 注意需要用session_id.name 而不是session_id，因为session_id是一个对象，而不是字符串，所以需要用session_id.name来拼接路径，才能得到正确的路径，如果用session_id，拼接后打印路径会发现wav路径还是_img路径
                    # # 如果wav_path路径不存在，就创建
                    # if not os.path.exists(wav_path):
                    #     os.makedirs(wav_path)
                    # print('img_path: ', img_path, ', wav_path: ', wav_path)
                    ############# 方式1，处理还没生成的文件夹 开始 ##############
                    # 如果当前遍历到的文件夹没有.DS_Store文件夹,且session_id文件夹名称没有出现在recordings_wav文件夹中，就执行shell 命令
                    
                    # 判断task_dir_wav_full_path路径下是否存在session_id文件夹, 注意需要用session_id.name 而不是session_id，因为session_id是一个对象，而不是字符串
                    if session_id.name != '.DS_Store' and session_id.name not in os.listdir(task_dir_wav_full_path):
                        # 执行shell 命令
                        print('\n\nsession_id: ', session_id.name, ', 根据.mp4文件提取出.wav文件, 命令: python ./video_to_wave.py --video-dir ' + img_path + ' --output-dir ' + wav_path)
                        # os.system('python ./video_to_wave.py --video-dir ' + img_path + ' --output-dir ' + wav_path)
                        
                        print('\nsession_id: ', session_id.name, ', 根据.wav文件生成.wav.npy文件, 命令: python raw_audio_process.py --mode librosa --audio-dir ' + wav_path + ' --output-dir ' + wav_path)
                        # os.system('python raw_audio_process.py --mode librosa --audio-dir ' + wav_path + ' --output-dir ' + wav_path)
                    else:
                        print(f'session_id: {session_id.name} 已经存在于{task_dir_wav_full_path}，不需要再处理了')
                    ############# 方式1，处理没有的文件夹 结束 ##############
                    
                    
                    # ############# 方式2，强制处理所有文件夹 开始 ##############
                    # 执行shell 命令
                    # print('\n\n准备执行命令 生成.wav文件：python ./video_to_wave.py --video-dir ' + img_path + ' --output-dir ' + wav_path)
                    # os.system('python ./video_to_wave.py --video-dir ' + img_path + ' --output-dir ' + wav_path)
                    
                    # print('准备执行命令 生成.wav.npy文件：python raw_audio_process.py --mode librosa --audio-dir ' + wav_path + ' --output-dir ' + wav_path)
                    # os.system('python raw_audio_process.py --mode librosa --audio-dir ' + wav_path + ' --output-dir ' + wav_path)
                    # print( '当前命令执行完毕' )
                    # ############# 方式2，强制处理所有文件夹 结束 ##############

if  __name__   ==   '__main__' :
    main()


# 后台执行该脚本命令：
# nohup python -u main.py >nohup.log 2>&1 &
