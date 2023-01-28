# 将视频文件转为音频文件 参考：/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/dpcv/data/utils/video_to_wave.py


################# 1. 将mp4视频文件转成wav音频文件 #################
##########################----train----##########################
# python video_to_wave.py \
# --video-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/recordings/animals_recordings_train_img/128129/' \
# --output-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/128129/'

##########################----val----##########################
# python video_to_wave.py \
# --video-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/val/recordings/animals_recordings_val_img/001080/' \
# --output-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/val/recordings/animals_recordings_val_wav/001080/'

python video_to_wave.py \
--video-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/val/recordings/animals_recordings_val_img/001081/' \
--output-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/val/recordings/animals_recordings_val_wav/001081/'

##########################----test----##########################
# python video_to_wave.py \
# --video-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/test/recordings/animals_recordings_test_img/008105/' \
# --output-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/test/recordings/animals_recordings_test_wav/008105/'

python video_to_wave.py \
--video-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/test/recordings/animals_recordings_test_img/056109/' \
--output-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/test/recordings/animals_recordings_test_wav/056109/'


################# 2. 将wav音频文件转成npy文件 #################

##########################----train----##########################
# python raw_audio_process.py --mode librosa \
#     --audio-dir /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/055128/ \
#     --output-dir /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/055128/

# python raw_audio_process.py --mode librosa \
#     --audio-dir /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/128129/ \
#     --output-dir /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/128129/

##########################----val----##########################
# python raw_audio_process.py --mode librosa \
#     --audio-dir /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/val/recordings/animals_recordings_val_wav/001080/ \
#     --output-dir /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/val/recordings/animals_recordings_val_wav/001080/

python raw_audio_process.py --mode librosa \
    --audio-dir /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/val/recordings/animals_recordings_val_wav/001081/ \
    --output-dir /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/val/recordings/animals_recordings_val_wav/001081/

##########################----test----##########################
# python raw_audio_process.py --mode librosa \
#     --audio-dir /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/test/recordings/animals_recordings_test_wav/008105/ \
#     --output-dir /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/test/recordings/animals_recordings_test_wav/008105/

python raw_audio_process.py --mode librosa \
    --audio-dir /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/test/recordings/animals_recordings_test_wav/056109/ \
    --output-dir /Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/test/recordings/animals_recordings_test_wav/056109/