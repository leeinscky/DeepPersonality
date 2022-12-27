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

def main ():
    
    mac_animals_recordings_train_img = '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/recordings/animals_recordings_train_img'
    mac_animals_recordings_val_img = '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/val/recordings/animals_recordings_val_img'
    mac_animals_recordings_test_img = '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/test/recordings/animals_recordings_test_img'
    
    linux_animals_recordings_train_img = '/home/zl525/rds/hpc-work/udiva_tiny_hpc/train/recordings/animals_recordings_train_img'
    linux_animals_recordings_val_img = '/home/zl525/rds/hpc-work/udiva_tiny_hpc/val/recordings/animals_recordings_val_img'
    linux_animals_recordings_test_img = '/home/zl525/rds/hpc-work/udiva_tiny_hpc/test/recordings/animals_recordings_test_img'
    
    video_dir = mac_animals_recordings_train_img
    # 遍历文件夹下的所有文件夹
    for file  in  os.listdir(video_dir):
        # 如果当前遍历到的文件夹里没有FC1_A和FC2_A两个子文件夹并且没有.DS_Store文件夹，就执行shell 命令
        if  (not os.path.exists(os.path.join(video_dir, file,  'FC1_A' )) and not os.path.exists(os.path.join(video_dir, file,  'FC2_A' ))) and file !=  '.DS_Store' :
            # 执行shell 命令
            print( '准备执行命令：python ./video_to_image.py --video-dir '  + os.path.join(video_dir, file) +  ' --output-dir '  + os.path.join(video_dir, file))
            os.system( 'python ./video_to_image.py --video-dir '  + os.path.join(video_dir, file) +  ' --output-dir '  + os.path.join(video_dir, file))
            print( '当前命令执行完毕' )

if  __name__   ==   '__main__' :
    main()