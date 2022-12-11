import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pickle
import os
from random import shuffle


class VideoData(Dataset):
    """base class for bi-modal input data"""
    def __init__(self, data_root, img_dir, label_file, audio_dir=None, parse_img_dir=True, parse_aud_dir=False):
        print('[DeepPersonality/dpcv/data/datasets/bi_modal_data.py] - 开始执行 def __init__')
        self.data_root = data_root
        self.img_dir = img_dir
        self.audio_dir = audio_dir
        self.annotation = self.parse_annotation(label_file) # self.annotation是一个字典，包含了五个键值对（5种personality），每个键值对对应一个字典，字典的键是视频名，值是对应的分数
        if parse_img_dir:
            self.img_dir_ls = self.parse_data_dir(img_dir)  # every directory name indeed a video # img_dir_ls是一个列表，包含了所有视频的路径，每个路径是一个字符串，例如'datasets/ChaLearn2016_tiny/train_data/-AmMDnVl4s8.003'
            print('[DeepPersonality/dpcv/data/datasets/bi_modal_data.py] self.img_dir_ls: ', self.img_dir_ls) # 打印结果在最下方，img_dir_ls是  self.img_dir_ls:  ['datasets/ChaLearn2016_tiny/train_data/-AmMDnVl4s8.003', 'datasets/ChaLearn2016_tiny/train_data/2kqPuht5jTg.002', 'datasets/ChaLearn2016_tiny/train_data/4CSV8L7aVik.000', 'datasets/ChaLearn2016_tiny/train_data/50gokPvvMs8.000', 'datasets/ChaLearn2016_tiny/train_data/6KKNrufnL80.000', 'datasets/ChaLearn2016_tiny/train_data/83cmR2fkyy8.005', 'datasets/ChaLearn2016_tiny/train_data/98fnGDVky00.005', 'datasets/ChaLearn2016_tiny/train_data/9KAqOrdiZ4I.002', 'datasets/ChaLearn2016_tiny/train_data/9hqH1PJ6cG8.001', 'datasets/ChaLearn2016_tiny/train_data/A3StIKMjn4k.002', 'datasets/ChaLearn2016_tiny/train_data/BWAEpai6FK0.003', 'datasets/ChaLearn2016_tiny/train_data/C_NtwmmF2Ys.000', 'datasets/ChaLearn2016_tiny/train_data/DnTtbAR_Qyw.004', 'datasets/ChaLearn2016_tiny/train_data/F0_EI_X5JVk.003', 'datasets/ChaLearn2016_tiny/train_data/HegkSmkiBos.005', 'datasets/ChaLearn2016_tiny/train_data/JBdLI6AhJrw.000', 'datasets/ChaLearn2016_tiny/train_data/JIYZTruMpiI.003', 'datasets/ChaLearn2016_tiny/train_data/JiXJeI5_jGM.000', 'datasets/ChaLearn2016_tiny/train_data/KJ643kfjqLY.003', 'datasets/ChaLearn2016_tiny/train_data/L9sG80PI1Gw.003', 'datasets/ChaLearn2016_tiny/train_data/L_gmlaz-0s4.003', 'datasets/ChaLearn2016_tiny/train_data/MOXPVzRBDPo.002', 'datasets/ChaLearn2016_tiny/train_data/NDBCrVvp0Vg.003', 'datasets/ChaLearn2016_tiny/train_data/OWZ-qVZG14A.002', 'datasets/ChaLearn2016_tiny/train_data/Q2AI4XpApFs.002', 'datasets/ChaLearn2016_tiny/train_data/Qz_cjgCtDcM.003', 'datasets/ChaLearn2016_tiny/train_data/RlUuWWWFrhM.005', 'datasets/ChaLearn2016_tiny/train_data/T6CMGXdPUTA.001', 'datasets/ChaLearn2016_tiny/train_data/TPk6KiHuPag.004', 'datasets/ChaLearn2016_tiny/train_data/Tr3A7WODEuM.001', 'datasets/ChaLearn2016_tiny/train_data/Uu-NbXUPr-A.001', 'datasets/ChaLearn2016_tiny/train_data/W0FCCk0a0tg.001', 'datasets/ChaLearn2016_tiny/train_data/WT1YjeADatU.001', 'datasets/ChaLearn2016_tiny/train_data/Yj36y7ELRZE.004', 'datasets/ChaLearn2016_tiny/train_data/_uNup91ZYw0.002', 'datasets/ChaLearn2016_tiny/train_data/bt-ev53zZWE.004', 'datasets/ChaLearn2016_tiny/train_data/dd0z9mErfSo.003', 'datasets/ChaLearn2016_tiny/train_data/eD4b8sM-Tpw.000', 'datasets/ChaLearn2016_tiny/train_data/eI_7SimPnnQ.001', 'datasets/ChaLearn2016_tiny/train_data/geXpIfaFzF4.001', 'datasets/ChaLearn2016_tiny/train_data/in-HuMgiDCE.001', 'datasets/ChaLearn2016_tiny/train_data/jDdRrqRcSzM.002', 'datasets/ChaLearn2016_tiny/train_data/jTkEWnuDnbA.001', 'datasets/ChaLearn2016_tiny/train_data/jwcSbw4NDn0.005', 'datasets/ChaLearn2016_tiny/train_data/myhEW1aZRg4.000', 'datasets/ChaLearn2016_tiny/train_data/n8IiQJyqjiE.003', 'datasets/ChaLearn2016_tiny/train_data/nGGtTu6dSJE.000', 'datasets/ChaLearn2016_tiny/train_data/nOFHZ_s7Et4.005', 'datasets/ChaLearn2016_tiny/train_data/nZz1hK90gwA.004', 'datasets/ChaLearn2016_tiny/train_data/okSmKH2k5lE.002', 'datasets/ChaLearn2016_tiny/train_data/om-9kFEKJIs.004', 'datasets/ChaLearn2016_tiny/train_data/opEoJBrcmbI.002', 'datasets/ChaLearn2016_tiny/train_data/vMtF0akNUK4.000', 'datasets/ChaLearn2016_tiny/train_data/vr5FWHUkYRM.001', 'datasets/ChaLearn2016_tiny/train_data/vrMlwwTLWIE.005', 'datasets/ChaLearn2016_tiny/train_data/wTo1uZns2X8.000', 'datasets/ChaLearn2016_tiny/train_data/x0CZuHnJ0Hs.005', 'datasets/ChaLearn2016_tiny/train_data/yOzHZOg95Ug.003', 'datasets/ChaLearn2016_tiny/train_data/yOzHZOg95Ug.005', 'datasets/ChaLearn2016_tiny/train_data/yftfxiDNXko.002']
        if parse_aud_dir:
            self.aud_file_ls = self.parse_data_dir(audio_dir)
            print('[DeepPersonality/dpcv/data/datasets/bi_modal_data.py] self.aud_file_ls: ', self.aud_file_ls)

    def parse_data_dir(self, data_dir):
        """

        Args:
            data_dir:(Str or List[Str, ]) training audio data directory or train and valid data directory

        Returns:
            img_dir_path:(List[Str, ]) a list contains the path of image files
        """
        if isinstance(data_dir, list):
            data_dir_path = []
            for dir_i in data_dir:
                data_dir_ls = sorted(os.listdir(os.path.join(self.data_root, dir_i)))
                data_dir_path.extend([os.path.join(self.data_root, dir_i, item) for item in data_dir_ls])
        else:
            data_dir_ls = sorted(os.listdir(os.path.join(self.data_root, data_dir)))
            data_dir_path = [os.path.join(self.data_root, data_dir, item) for item in data_dir_ls]
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
        return annotation

    def get_ocean_label(self, index): 
        # index是一个整数，表示video目录里的第几个video样本，从0开始，这个函数返回的是一个列表，包含了这个视频的所有5个个性标签值
        print('[DeepPersonality/dpcv/data/datasets/bi_modal_data.py] - 开始执行 def get_ocean_label, index: ', index) # index:  7
        video_path = self.img_dir_ls[index] # video_path: datasets/ChaLearn2016_tiny/train_data/9KAqOrdiZ4I.002，结合最后的img_dir_ls 打印结果，找第7+1=8个视频，即 9KAqOrdiZ4I.002
        video_name = f"{os.path.basename(video_path)}.mp4" # video_name: 9KAqOrdiZ4I.002.mp4
        print('[DeepPersonality/dpcv/data/datasets/bi_modal_data.py] - 开始执行 def get_ocean_label, video_path: ', video_path) # video_path:  datasets/ChaLearn2016_tiny/train_data/9KAqOrdiZ4I.002 结合最后的img_dir_ls 打印结果，找第7+1=8个视频，即 9KAqOrdiZ4I.002
        print('[DeepPersonality/dpcv/data/datasets/bi_modal_data.py] - 开始执行 def get_ocean_label, video_name: ', video_name) #  video_name:  9KAqOrdiZ4I.002.mp4

        score = [
            self.annotation["openness"][video_name],
            self.annotation["conscientiousness"][video_name],
            self.annotation["extraversion"][video_name],
            self.annotation["agreeableness"][video_name],
            self.annotation["neuroticism"][video_name],
        ] # score是一个长度为5的列表，包含了五个personality的分数
        return score

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
    [DeepPersonality/dpcv/data/datasets/bi_modal_data.py] self.img_dir_ls:  
    [
        'datasets/ChaLearn2016_tiny/train_data/-AmMDnVl4s8.003', 
        'datasets/ChaLearn2016_tiny/train_data/2kqPuht5jTg.002', 
        'datasets/ChaLearn2016_tiny/train_data/4CSV8L7aVik.000', 
        'datasets/ChaLearn2016_tiny/train_data/50gokPvvMs8.000', 
        'datasets/ChaLearn2016_tiny/train_data/6KKNrufnL80.000', 
        'datasets/ChaLearn2016_tiny/train_data/83cmR2fkyy8.005', 
        'datasets/ChaLearn2016_tiny/train_data/98fnGDVky00.005', 
        'datasets/ChaLearn2016_tiny/train_data/9KAqOrdiZ4I.002', 
        'datasets/ChaLearn2016_tiny/train_data/9hqH1PJ6cG8.001', 
        'datasets/ChaLearn2016_tiny/train_data/A3StIKMjn4k.002', 
        'datasets/ChaLearn2016_tiny/train_data/BWAEpai6FK0.003', 
        'datasets/ChaLearn2016_tiny/train_data/C_NtwmmF2Ys.000', 
        'datasets/ChaLearn2016_tiny/train_data/DnTtbAR_Qyw.004', 
        'datasets/ChaLearn2016_tiny/train_data/F0_EI_X5JVk.003', 
        'datasets/ChaLearn2016_tiny/train_data/HegkSmkiBos.005', 
        'datasets/ChaLearn2016_tiny/train_data/JBdLI6AhJrw.000', 
        'datasets/ChaLearn2016_tiny/train_data/JIYZTruMpiI.003', 
        'datasets/ChaLearn2016_tiny/train_data/JiXJeI5_jGM.000', 
        'datasets/ChaLearn2016_tiny/train_data/KJ643kfjqLY.003', 
        'datasets/ChaLearn2016_tiny/train_data/L9sG80PI1Gw.003', 
        'datasets/ChaLearn2016_tiny/train_data/L_gmlaz-0s4.003', 
        'datasets/ChaLearn2016_tiny/train_data/MOXPVzRBDPo.002', 
        'datasets/ChaLearn2016_tiny/train_data/NDBCrVvp0Vg.003', 
        'datasets/ChaLearn2016_tiny/train_data/OWZ-qVZG14A.002', 
        'datasets/ChaLearn2016_tiny/train_data/Q2AI4XpApFs.002', 
        'datasets/ChaLearn2016_tiny/train_data/Qz_cjgCtDcM.003', 
        'datasets/ChaLearn2016_tiny/train_data/RlUuWWWFrhM.005', 
        'datasets/ChaLearn2016_tiny/train_data/T6CMGXdPUTA.001', 
        'datasets/ChaLearn2016_tiny/train_data/TPk6KiHuPag.004', 
        'datasets/ChaLearn2016_tiny/train_data/Tr3A7WODEuM.001', 
        'datasets/ChaLearn2016_tiny/train_data/Uu-NbXUPr-A.001', 
        'datasets/ChaLearn2016_tiny/train_data/W0FCCk0a0tg.001', 
        'datasets/ChaLearn2016_tiny/train_data/WT1YjeADatU.001', 
        'datasets/ChaLearn2016_tiny/train_data/Yj36y7ELRZE.004', 
        'datasets/ChaLearn2016_tiny/train_data/_uNup91ZYw0.002', 
        'datasets/ChaLearn2016_tiny/train_data/bt-ev53zZWE.004', 
        'datasets/ChaLearn2016_tiny/train_data/dd0z9mErfSo.003', 
        'datasets/ChaLearn2016_tiny/train_data/eD4b8sM-Tpw.000', 
        'datasets/ChaLearn2016_tiny/train_data/eI_7SimPnnQ.001', 
        'datasets/ChaLearn2016_tiny/train_data/geXpIfaFzF4.001', 
        'datasets/ChaLearn2016_tiny/train_data/in-HuMgiDCE.001', 
        'datasets/ChaLearn2016_tiny/train_data/jDdRrqRcSzM.002', 
        'datasets/ChaLearn2016_tiny/train_data/jTkEWnuDnbA.001', 
        'datasets/ChaLearn2016_tiny/train_data/jwcSbw4NDn0.005', 
        'datasets/ChaLearn2016_tiny/train_data/myhEW1aZRg4.000', 
        'datasets/ChaLearn2016_tiny/train_data/n8IiQJyqjiE.003', 
        'datasets/ChaLearn2016_tiny/train_data/nGGtTu6dSJE.000', 
        'datasets/ChaLearn2016_tiny/train_data/nOFHZ_s7Et4.005', 
        'datasets/ChaLearn2016_tiny/train_data/nZz1hK90gwA.004', 
        'datasets/ChaLearn2016_tiny/train_data/okSmKH2k5lE.002', 
        'datasets/ChaLearn2016_tiny/train_data/om-9kFEKJIs.004', 
        'datasets/ChaLearn2016_tiny/train_data/opEoJBrcmbI.002', 
        'datasets/ChaLearn2016_tiny/train_data/vMtF0akNUK4.000', 
        'datasets/ChaLearn2016_tiny/train_data/vr5FWHUkYRM.001', 
        'datasets/ChaLearn2016_tiny/train_data/vrMlwwTLWIE.005', 
        'datasets/ChaLearn2016_tiny/train_data/wTo1uZns2X8.000', 
        'datasets/ChaLearn2016_tiny/train_data/x0CZuHnJ0Hs.005', 
        'datasets/ChaLearn2016_tiny/train_data/yOzHZOg95Ug.003', 
        'datasets/ChaLearn2016_tiny/train_data/yOzHZOg95Ug.005', 
        'datasets/ChaLearn2016_tiny/train_data/yftfxiDNXko.002'
    ]
'''