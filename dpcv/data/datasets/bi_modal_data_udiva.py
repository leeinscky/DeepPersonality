import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pickle
import os
from random import shuffle
import math

class VideoDataUdiva(Dataset):
    """base class for bi-modal input data"""
    def __init__(self, data_root, img_dir, label_file, audio_dir=None, parse_img_dir=True, parse_aud_dir=False, sample_size=16, mode='train', dataset_name="UDIVA", prefix1="FC1", prefix2="FC2", img_dir_ls=None):
        # print('[bi_modal_data_udiva.py] - 开始执行 def __init__')
        self.data_root = data_root
        self.img_dir = img_dir
        self.audio_dir = audio_dir
        self.sample_size = sample_size
        self.mode = mode
        self.dataset_name = dataset_name
        self.prefix1 = prefix1
        self.prefix2 = prefix2
        self.annotation = self.parse_annotation(label_file)
        self.sessionid_numsegements = {} # key: sessionid, value: num_segments 即每个session文件夹对应的的segment数量/视频片段数量
        if img_dir_ls is not None:
            self.img_dir_ls = img_dir_ls
            # print('[bi_modal_data_udiva.py]__init__函数 dataset type:', self.mode, ', input img_dir_ls != None, len(self.img_dir_ls): ', len(self.img_dir_ls), ', self.img_dir_ls[:3]: ', self.img_dir_ls[:3])
        else:
            if parse_img_dir:
                self.img_dir_ls = self.parse_data_dir_v2(img_dir)  # every directory name indeed a video # img_dir_ls是一个列表，包含了所有视频的路径，每个路径是一个字符串，例如'datasets/ChaLearn2016_tiny/train_data/-AmMDnVl4s8.003'
                # print('[bi_modal_data_udiva.py]__init__函数 dataset type:', self.mode, ',  input img_dir_ls == None, len(self.img_dir_ls):', len(self.img_dir_ls), ', self.img_dir_ls[:3]: ', self.img_dir_ls[:3]) # 打印结果在最下方，img_dir_ls是  self.img_dir_ls:  ['datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128', 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/128129'], NoXi: ['datasets/noxi/img/004_1', 'datasets/noxi/img/004_2', 'datasets/noxi/img/004_3']
            # print('=============================================================')
            if parse_aud_dir:
                self.aud_file_ls = self.parse_data_dir_v2(audio_dir)
                # print('[bi_modal_data_udiva.py]__init__函数 self.aud_file_ls: ', self.aud_file_ls)
        print('[bi_modal_data_udiva.py]__init__函数 dataset type:', self.mode, ', len(img_dir_ls): ', len(self.img_dir_ls), ', key(session_id)_value(num_segments): ', self.sessionid_numsegements)
    
    def parse_data_dir(self, data_dir):
        """

        Args:
            data_dir:(Str or List[Str, ]) training audio data directory or train and valid data directory

        Returns:
            img_dir_path:(List[Str, ]) a list contains the path of image files
        """
        # if file_type == 'img':
        #     # 将data_dir下所有以_img结尾的文件夹的路径存入img_dir_list
        #     img_data_dir_list = []
        #     for dir_i in os.listdir(os.path.join(self.data_root, data_dir)):
        #         print('img_dir_i=', dir_i)
        #         if dir_i.endswith('_img'):
        #             img_data_dir_list.append(dir_i)
        #     # print('[bi_modal_data_udiva.py]parse_data_dir函数 img_dir_list: ', img_data_dir_list)
        #     data_dir = img_data_dir_list
        # elif file_type == 'aud':
        #     # 将data_dir下所有以_aud结尾的文件夹的路径存入aud_data_dir_list
        #     aud_data_dir_list = []
        #     for dir_i in os.listdir(os.path.join(self.data_root, data_dir)):
        #         print('aud_dir_i=', dir_i)
        #         if dir_i.endswith('_aud'):
        #             aud_data_dir_list.append(dir_i)
        #     # print('[bi_modal_data_udiva.py]parse_data_dir函数 aud_dir_list: ', aud_data_dir_list)
        #     data_dir = aud_data_dir_list
        
        # print('[bi_modal_data_udiva.py]parse_data_dir函数 data_dir: ', data_dir) # data_dir:   udiva_tiny/train/recordings
        if isinstance(data_dir, list):
            data_dir_path = []
            for dir_i in data_dir:
                data_dir_ls = sorted(os.listdir(os.path.join(self.data_root, dir_i)))
                data_dir_path.extend([os.path.join(self.data_root, dir_i, item) for item in data_dir_ls])
        else:
            data_dir_ls = sorted(os.listdir(os.path.join(self.data_root, data_dir))) # data_dir_ls:  ['055125', '055128', '058110', '059134', '128129']
            data_dir_path = [os.path.join(self.data_root, data_dir, item) for item in data_dir_ls] 
        # print('[bi_modal_data_udiva.py]parse_data_dir函数 data_dir_ls: ', data_dir_ls)
        # print('[bi_modal_data_udiva.py]parse_data_dir函数 data_dir_path: ', data_dir_path)
        # 修改逻辑前 data_dir_path:  [
            # 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055125', 
            # 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128', 
            # 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/058110', 
            # 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/059134', 
            # 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/128129']
            
        # 修改逻辑后 data_dir_path: [
            # 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055125', 
            # 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128', 
            # 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/058110', 
            # 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/059134', 
            # 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/128129', 
            # 'datasets/udiva_tiny/train/recordings/ghost_recordings_train_img/055125', 
            # 'datasets/udiva_tiny/train/recordings/ghost_recordings_train_img/055128']
            
        return data_dir_path


    def parse_data_dir_v2(self, data_dir):
        """

        Args:
            data_dir:(Str or List[Str, ]) training audio data directory or train and valid data directory

        Returns:
            img_dir_path:(List[Str, ]) a list contains the path of image files
        """
        # print('[bi_modal_data_udiva.py]parse_data_dir函数 data_dir: ', data_dir) # data_dir: ['udiva_tiny/train/recordings/animals_recordings_train_img']
        if isinstance(data_dir, list):
            data_dir_path = []
            for dir_i in data_dir:
                data_dir_ls = sorted(os.listdir(os.path.join(self.data_root, dir_i)))
                data_dir_path.extend([os.path.join(self.data_root, dir_i, item) for item in data_dir_ls])
        else:
            data_dir_ls = sorted(os.listdir(os.path.join(self.data_root, data_dir))) # data_dir_ls:  ['055125', '055128', '058110', '059134', '128129']
            data_dir_path = [os.path.join(self.data_root, data_dir, item) for item in data_dir_ls] 
        
        # print('[bi_modal_data_udiva.py]parse_data_dir函数 data_dir_ls: ', data_dir_ls)
        # print('[bi_modal_data_udiva.py]parse_data_dir函数 data_dir_path: ', data_dir_path)
        
        ret_dir_path = []
        # 遍历data_dir_path里的每个session文件夹路径，计算当前遍历到的session文件夹下FC1_X 和 FC2_X两个文件夹各自含有的帧图片数量 nummber_of_frames_FC1_X 和 nummber_of_frames_FC2_X
        for session_dir_path in data_dir_path:
            fc1_img_dir_path, fc2_img_dir_path = '', ''
            if session_dir_path.endswith("031"): # TODO 临时处理，后续需要修改！！！
                # print('[bi_modal_data_udiva.py]-session_dir_path:', session_dir_path)
                continue
            for file in os.listdir(session_dir_path):
                # print('file:', file, 'type:', type(file))
                if os.path.isdir(os.path.join(session_dir_path, file)) and file.startswith(self.prefix1) and not file.endswith(".mp4"):
                    fc1_img_dir_path = os.path.join(session_dir_path, file)
                if os.path.isdir(os.path.join(session_dir_path, file)) and file.startswith(self.prefix2) and not file.endswith(".mp4"):
                    fc2_img_dir_path = os.path.join(session_dir_path, file)
            # print('[audio_visual_data_udiva.py]- get_image_data idx:', idx, 'fc1_img_dir_path:', fc1_img_dir_path, "fc2_img_dir_path:", fc2_img_dir_path)
            # 打印结果: get_image_data fc1_img_dir_path: datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128/FC1_A     fc2_img_dir_path: datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128/FC2_A
            
            num_frames_FC1 = len(os.listdir(fc1_img_dir_path))
            num_frames_FC2 = len(os.listdir(fc2_img_dir_path))
            min_len_frames = min(num_frames_FC1, num_frames_FC2) # 取两个视频的帧图片数量的最小值, 例如: 1000 和 2000, 则min_len_frames = 1000
            
            # 将一个长视频分成多个短视频片段(segments or video clips)，每个video clips 包含sample_size个帧图片, 这里需要得到segment的数量, 向下取整
            num_segments = int(math.floor(min_len_frames / self.sample_size)) # floor 向下取整, 例如: math.floor(3.5) = 3
            # print('min_len_frames:', min_len_frames, 'num_frames_FC1:', num_frames_FC1, 'num_frames_FC2:', num_frames_FC2, ', num_segments:', num_segments)
            self.sessionid_numsegements[session_dir_path.split('/')[-1]] = num_segments # key: sessionid, value: num_segments 即每个session文件夹对应的的segment数量/视频片段数量
            
            # 重新构造一个list:ret_dir_path，list里的每个元素是一个原先session文件夹路径，但是在原先session文件夹路径的结尾加上了一个下划线和一个数字，例如: datasets/udiva_tiny/train/recordings/ghost_recordings_train_img/055125_0，这个数字代表了当前session文件夹的第几个video clip，例如: /055125_0，代表了当前session文件夹的第1个video clip，055125_1，代表了当前session文件夹的第2个video clip
            for i in range(num_segments):
                # 将session_dir_path的结尾加上符号_, 然后加上i, 例如: datasets/udiva_tiny/train/recordings/ghost_recordings_train_img/055125_0
                ret_dir_path.append(session_dir_path + '_' + str(i+1))
            # print('[bi_modal_data_udiva.py]parse_data_dir函数 session_dir_path:', session_dir_path, ', num_frames_FC1:', num_frames_FC1, ', num_frames_FC2:', num_frames_FC2, ', num_segments:', num_segments)
        
        # print('[bi_modal_data_udiva.py]parse_data_dir函数 ret_dir_path: ', ret_dir_path)
        return ret_dir_path


    def parse_annotation(self, label_file):
        """
        load label file
            args:(srt / list[str, str]) annotation file path
        """
        if isinstance(label_file, list):
            assert len(label_file) == 2, "only support join train and validation data"
            anno_list = []
            for label_i in label_file:
                label_path = os.path.join(self.data_root, label_i)
                with open(label_path, "rb") as f:
                    anno_list.append(pickle.load(f, encoding="latin1"))
            for key in anno_list[0].keys():
                anno_list[0][key].update(anno_list[1][key])
            annotation = anno_list[0]
        else:
            label_path = os.path.join(self.data_root, label_file)
            with open(label_path, "rb") as f:
                annotation = pickle.load(f, encoding="latin1")
        # print('[bi_modal_data_udiva.py]parse_annotation函数 - annotation: ', annotation)
        return annotation


    def get_ocean_label(self, index): 
        # index是一个整数，表示video目录里的第几个video样本，从0开始，这个函数返回的是一个列表，包含了这个视频的所有5个个性标签值
        # print('[bi_modal_data_udiva.py]get_ocean_label函数 - 开始执行 def get_ocean_label, index: ', index, ', self.img_dir_ls: ', self.img_dir_ls) # index:  7
        session_path = self.img_dir_ls[index] # session_path: 
        session_path, segment_idx = session_path.rsplit("_", 1)
        session_id = f"{os.path.basename(session_path)}" # session_id: 055128, type(session_id):  <class 'str'>
        
        if self.dataset_name == "NoXi":
            session_id = session_id.lstrip('0') # remove leading zeros e.g. 004 -> 4 ; 005 -> 5
        
        # print('[bi_modal_data_udiva.py]get_ocean_label函数 - 开始执行 def get_ocean_label, session_path: ', session_path, ',session_id: ', session_id, ', segment_idx: ', segment_idx)
        # print('[bi_modal_data_udiva.py]get_ocean_label函数 - 开始执行 def get_ocean_label, self.annotation: ', self.annotation, 'type: ', type(self.annotation))

        # get the type of self.annotation[session_id]
        # print('[bi_modal_data_udiva.py]get_ocean_label函数 - 开始执行 def get_ocean_label, self.annotation[session_id]: ', self.annotation[session_id], 'type of self.annotation[session_id]: ', type(self.annotation[session_id]))
        
        # convert the type of self.annotation[session_id] to list
        self.annotation[session_id] = list(self.annotation[session_id])
        # print('[bi_modal_data_udiva.py]get_ocean_label函数 - 开始执行 def get_ocean_label, self.annotation[session_id]: ', self.annotation[session_id], 'type of self.annotation[session_id]: ', type(self.annotation[session_id]))
        
        relation = self.annotation[session_id]
        # print('[bi_modal_data_udiva.py]get_ocean_label函数 - 开始执行 def get_ocean_label, relation: ', relation)
        return relation


    def __getitem__(self, index):
        raise NotImplementedError

    def get_image_data(self, index):
        return self.img_dir_ls[index]

    def get_wave_data(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.img_dir_ls)


'''
img_dir_ls 打印结果：
self.img_dir_ls:  
    [
        'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128', 
        'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/128129'
    ]
'''