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
    
    animals_recordings_train_img = '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/recordings/animals_recordings_train_img'
    animals_recordings_train_wav = '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/recordings/animals_recordings_train_wav'
    # 遍历文件夹下的所有文件夹
    for file  in  os.listdir(animals_recordings_train_img):
        # 如果当前遍历到的文件夹没有.DS_Store文件夹,且文件夹名称没有出现在animals_recordings_train_wav文件夹中，就执行shell 命令
        if  file  !=   '.DS_Store'   and  file  not   in  os.listdir(animals_recordings_train_wav):
            # 执行shell 命令
            print('\n\n准备执行命令：python ./video_to_wave.py --video-dir '  + os.path.join(animals_recordings_train_img, file) +  ' --output-dir '  + os.path.join(animals_recordings_train_wav, file))
            os.system('python ./video_to_wave.py --video-dir '  + os.path.join(animals_recordings_train_img, file) +  ' --output-dir '  + os.path.join(animals_recordings_train_wav, file))
            
            print('准备执行命令：python raw_audio_process.py --mode librosa --audio-dir '  + os.path.join(animals_recordings_train_wav, file) +  ' --output-dir '  + os.path.join(animals_recordings_train_wav, file))
            os.system('python raw_audio_process.py --mode librosa --audio-dir '  + os.path.join(animals_recordings_train_wav, file) +  ' --output-dir '  + os.path.join(animals_recordings_train_wav, file))
            print( '当前命令执行完毕' )

if  __name__   ==   '__main__' :
    main()