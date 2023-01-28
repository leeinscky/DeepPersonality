# 将视频文件转为图片帧文件 参考：/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/dpcv/data/utils/video_to_image.py

# video_dir='/Users/lizejian/cambridge/mphil_project/relation_recognition/test/Expert_video.mp4'
# output_dir='/Users/lizejian/cambridge/mphil_project/relation_recognition/test/output'
# python video_to_image.py --video-dir '/Users/lizejian/cambridge/mphil_project/relation_recognition/test/video_test/' --output-dir '/Users/lizejian/cambridge/mphil_project/relation_recognition/test/output'

#################### train ####################
# python video_to_image.py \
# --video-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/recordings/animals_recordings_train_img/128129/' \
# --output-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/train/recordings/animals_recordings_train_img/128129/'

#################### val ####################
python video_to_image.py \
--video-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/val/recordings/animals_recordings_val_img/001081/'
--output-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/val/recordings/animals_recordings_val_img/001081/'

################### test ####################
python video_to_image.py \
--video-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/test/recordings/animals_recordings_test_img/056109/'
--output-dir '/Users/lizejian/cambridge/mphil_project/learn/udiva/DeepPersonality/datasets/udiva_tiny/test/recordings/animals_recordings_test_img/056109/'
