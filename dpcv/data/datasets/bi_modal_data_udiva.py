import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pickle
import os
from random import shuffle


class VideoDataUdiva(Dataset):
    """base class for bi-modal input data"""
    def __init__(self, data_root, img_dir, label_file, audio_dir=None, parse_img_dir=True, parse_aud_dir=False):
        # print('[bi_modal_data_udiva.py] - 开始执行 def __init__')
        self.data_root = data_root
        self.img_dir = img_dir
        self.audio_dir = audio_dir
        self.annotation = self.parse_annotation(label_file) # self.annotation是一个字典，包含了五个键值对（5种personality），每个键值对对应一个字典，字典的键是视频名，值是对应的分数
        if parse_img_dir:
            self.img_dir_ls = self.parse_data_dir(img_dir)  # every directory name indeed a video # img_dir_ls是一个列表，包含了所有视频的路径，每个路径是一个字符串，例如'datasets/ChaLearn2016_tiny/train_data/-AmMDnVl4s8.003'
            # print('[bi_modal_data_udiva.py]__init__函数 self.img_dir_ls: ', self.img_dir_ls) # 打印结果在最下方，img_dir_ls是  self.img_dir_ls:  ['datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128', 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/128129']
        # print('=============================================================')
        if parse_aud_dir:
            self.aud_file_ls = self.parse_data_dir(audio_dir)
            # print('[bi_modal_data_udiva.py]__init__函数 self.aud_file_ls: ', self.aud_file_ls)

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

    def parse_annotation(self, label_file):
        """
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
        session_id = f"{os.path.basename(session_path)}" # session_id:
        # print('[bi_modal_data_udiva.py]get_ocean_label函数 - 开始执行 def get_ocean_label, session_path: ', session_path) # session_path:  
        # print('[bi_modal_data_udiva.py]get_ocean_label函数 - 开始执行 def get_ocean_label, session_id: ', session_id, 'type: ', type(session_id)) #  session_id:  
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