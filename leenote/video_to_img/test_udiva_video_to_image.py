import  os
import sys
sys.path.append('../../datasets/') # 将datasets路径添加到系统路径中，这样就可以直接导入datasets下的模块了
sys.path.append('../video_to_img/') # 将video_to_img路径添加到系统路径中，这样就可以直接导入video_to_img下的模块了

if __name__ == '__main__':
    # video_dir = '/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/023191'
    # output_dir = '/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/023191'
    
    video_dir = '/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/animals_recordings_train_img/055125'
    output_dir = '/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/animals_recordings_train_img/055125'
    
    os.system('python3 ./udiva_video_to_image.py --video-dir '  + video_dir +  ' --output-dir '  + output_dir)

    # 打印结果：video:/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/023191/FC2_T.mp4 length = 8079.0, fps = 25. # 如果每隔1秒抽取一帧，那么总共抽取了8079/25=323.16帧
    # linux 命令验证文件后缀名： ls -v
    
    # 打印结果：video:/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/animals_recordings_train_img/055125/FC1_A.mp4 length = 9628.0, fps = 25.0 # 如果每隔1秒抽取一帧，那么总共抽取了9628/25=385.12帧
    # 为了验证，可以观察 
        # 伸出1根手指：/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/animals_recordings_train_img/055125/FC1_A/frame_7525.jpg  对应视频的时间为7525/25=301秒，即5分1秒
        # 伸出2跟手指： /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/animals_recordings_train_img/055125/FC1_A/frame_7575.jpg  对应视频的时间为7575/25=303秒，即5分3秒
        # 伸出3跟手指： /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/animals_recordings_train_img/055125/FC1_A/frame_7800.jpg  对应视频的时间为7800/25=312秒，即5分12秒
    