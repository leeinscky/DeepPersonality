import os
from matplotlib import test
import torch
import glob
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from PIL import Image
import random
import pickle
import numpy as np
# from dpcv.data.datasets.bi_modal_data import VideoData
from dpcv.data.datasets.bi_modal_data_udiva import VideoDataUdiva
from dpcv.data.transforms.transform import set_audio_visual_transform
from dpcv.data.transforms.build import build_transform_spatial
from .build import DATA_LOADER_REGISTRY
from random import shuffle
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import wandb
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE, SMOTE
from collections import Counter
import torch.nn.functional as F
import torchaudio
from sklearn.model_selection import StratifiedKFold


class AudioVisualDataUdiva(VideoDataUdiva):

    def __init__(self, data_root, img_dir, audio_dir, label_file, transform=None, sample_size=100):
        # print('[audio_visual_data_udiva.py]- class AudioVisualDataUdiva(VideoDataUdiva) 开始执行 __init__')
        super().__init__(data_root, img_dir, label_file, audio_dir)
        self.transform = transform
        self.sample_size = sample_size
        # print('[audio_visual_data_udiva.py]- class AudioVisualDataUdiva(VideoDataUdiva) 结束执行 __init__')

    def __getitem__(self, idx): # idx means the index of video in the video directory
        # print('[audio_visual_data_udiva.py] - class AudioVisualDataUdiva(VideoDataUdiva) 开始执行 __getitem__ , idx = ', idx) # idx 和 get_ocean_label(self, index) 里的index含义一样，表示video目录里的第几个video样本
        label = self.get_ocean_label(idx)  # label是True或者False, 代表关系Known是True或者False
        
        # img = self.get_image_data(idx) # img是一个PIL.Image.Image对象, 代表该video的一帧图像, 例如：PIL.Image.Image object, mode=RGB, size=224x224, 代表该video的一帧图像的大小是224x224, 且是RGB三通道的, 也就是说该video的一帧图像是224x224x3的
        fc1_img, fc2_img = self.get_image_data(idx)
        
        # wav = self.get_wave_data(idx) # wav是一个numpy.ndarray对象, 代表该video的一帧音频, 例如：array([[[ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ]]], dtype=float32), 代表该video的一帧音频是50176维的, 也就是说该video的一帧音频是50176x1x1的, 且是float32类型的, 且是三维的
        fc1_wav, fc2_wav  = self.get_wave_data(idx) # fc1_wav是一个numpy.ndarray对象, 代表该video的一帧音频, 例如：array([[[ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ]]], dtype=float32), 代表该video的一帧音频是50176维的, 也就是说该video的一帧音频是50176x1x1的, 且是float32类型的, 且是三维的

        # print('[audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ - transform之前 type(img)=', type(img), 'img=', img) # type(img)= <class 'PIL.Image.Image'> img= <PIL.Image.Image image mode=RGB size=256x256 at 0x7FC8AA03EB20>
        if self.transform: # self.transform是一个Compose对象, 代表对img和wav的一系列变换, 对于bimodal_resnet_data_loader， transforms = build_transform_spatial(cfg)，'TRANSFORM': 'standard_frame_transform', 参考 dpcv/data/transforms/transform.py 里的def standard_frame_transform()
            # 原始逻辑: 将img单独进行transform
            # img = self.transform(img) # img是一个torch.Tensor对象, 代表该video的一帧图像, 例如：tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            
            # 修改后逻辑: 将fc1_img和fc2_img分别进行transform, 然后将两个transform后的tensor进行拼接
            fc1_img_tensor = self.transform(fc1_img) # shape: (3, 224, 224)
            fc2_img_tensor = self.transform(fc2_img) # shape: (3, 224, 224)
            # concatenate the fc1_img_tensor and fc2_img_tensor 
            img = torch.cat((fc1_img_tensor, fc2_img_tensor), 0) # shape: (6, 224, 224)  torch.cat 是将两个tensor进行拼接, 0表示按照第0维进行拼接, 拼接后的tensor的shape为(6, 224, 224)
        # # transform 的具体过程如下
        # @TRANSFORM_REGISTRY.register()
        # def standard_frame_transform():
        #     import torchvision.transforms as transforms # transfroms是一个类, 里面有很多函数, 比如Resize, CenterCrop, ToTensor, Normalize, Compose等, 这些函数都是用来对图片进行处理的, 例如Resize就是用来调整图片的大小的, CenterCrop就是用来裁剪图片的, ToTensor就是用来将图片转换成tensor的, Normalize就是用来对图片进行归一化的, Compose就是用来将多个函数组合成一个函数的, 这样就可以一次性对图片进行多种处理了
        #     # 例如这里就是先调整图片的大小, 然后再裁剪图片, 然后再将图片转换成tensor, 最后再对图片进行归一化
        #     transforms = transforms.Compose([
        #         transforms.Resize(256), # 调整图片的大小, 这里是调整成256*256, 也就是说图片的长和宽都是256, 如果只想调整图片的长或者宽, 可以使用transforms.Resize((256, None))或者transforms.Resize((None, 256)), 这样就只会调整图片的长或者宽, 不会调整图片的长和宽,
        #         transforms.RandomHorizontalFlip(0.5), # 随机水平翻转图片, 这里的0.5是指翻转的概率, 也就是说每张图片都有50%的概率会被翻转
        #         transforms.CenterCrop((224, 224)), # 裁剪图片, 这里是裁剪成224*224, 也就是说图片的长和宽都是224, 如果只想裁剪图片的长或者宽, 可以使用transforms.CenterCrop((224, None))或者transforms.CenterCrop((None, 224)), 这样就只会裁剪图片的长或者宽, 不会裁剪图片的长和宽
        #         transforms.ToTensor(), # 将图片转换成tensor, 这里的tensor是pytorch中的tensor, 也就是说图片的数据类型是torch.Tensor, 如果想要将图片转换成numpy中的array, 可以使用transforms.ToPILImage(), 这样就可以将图片转换成numpy中的array了, 但是这里要注意的是, 如果图片是灰度图, 那么转换成numpy中的array之后, 图片的数据类型是numpy.uint8, 如果图片是彩色图, 那么转换成numpy中的array之后, 图片的数据类型是numpy.float32
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 对图片进行归一化, 如果想要用自己的数据集的均值和方差对图片进行归一化, 可以使用transforms.Normalize(mean=[mean1, mean2, mean3], std=[std1, std2, std3]), 这里的mean1, mean2, mean3分别是图片的三个通道的均值, std1, std2, std3分别是图片的三个通道的方差
        #     ])
        #     return transforms


        # print('[deeppårsonality/dpcv/data/datasets/audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ - torch.as_tensor之前 type(label)=', type(label), 'label=', label) # type(label)= <class 'list'> label= [0.5111111111111111, 0.4563106796116505, 0.4018691588785047, 0.3626373626373627, 0.4166666666666667]
        # print('[audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ - torch.as_tensor之前 type(wav)=', type(wav), 'wav=', wav) # type(wav)= <class 'numpy.ndarray'> wav= [[[-1.0260781e-03 -2.3528279e-03 -2.2199661e-03 ...  9.4422154e-05 2.8776360e-04 -7.4460535e-05]]]
        # wav原始逻辑
        # wav = torch.as_tensor(wav, dtype=img.dtype) # torch.as_tensor - 将输入转换为张量, 并返回一个新的张量, 与输入共享内存, 但是不同的是, 如果输入是一个张量, 则返回的张量与输入不共享内存, 例如：tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        # wav修改后逻辑
        fc1_wav = torch.as_tensor(fc1_wav, dtype=img.dtype) # shape=(1,1,50176)
        fc2_wav = torch.as_tensor(fc1_wav, dtype=img.dtype) # shape=(1,1,50176)
        # 将两个tensor拼接在一起 concatenate the fc1_wav and fc2_wav, 拼接后的维度为(2,1,50176)，即将两个(1,1,50176)的tensor拼接在一起
        wav = torch.cat((fc1_wav, fc2_wav), 0) # shape=(2,1,50176) 
        # print('[audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ - torch.as_tensor之后 type(wav)=', type(wav), 'wav=', wav) # type(wav)= <class 'torch.Tensor'> wav= tensor([[[-1.0261e-03, -2.3528e-03, -2.2200e-03,  ...,  9.4422e-05, 2.8776e-04, -7.4461e-05]]])

        label = torch.as_tensor(label, dtype=img.dtype)
        # print('[audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ - torch.as_tensor之后 type(label)=', type(label), 'label=', label, ' label.shape=', label.shape) # type(label)= <class 'torch.Tensor'> label= tensor([0.5111, 0.4563, 0.4019, 0.3626, 0.4167])
        

        # 返回的sample原始逻辑（也适用于将2个视频的帧的tensor拼接后的逻辑）
        sample = {"image": img, "audio": wav, "label": label} # 非udiva的shape: img.shape()=torch.Size([3, 224, 224]) wav.shape()=torch.Size([1, 1, 50176]) label.shape()=torch.Size([5])  # udiva的shape: img.shape()= torch.Size([6, 224, 224]) wav.shape()= torch.Size([1, 2, 50176]) label.shape()= torch.Size([1])
        # print('[audio_visual_data_udiva.py] - class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ 函数返回的结果中 , img.shape()=', img.shape, 'wav.shape()=', wav.shape, 'label.shape()=', label.shape) # 
        # 因为__getitem__ 需要传入参数 idx，所以返回的sample也是一个视频对应的img，wav，label，至于为什么只有1帧image, 因为get_image_data函数只返回了1帧image，而没有返回多帧image
        
        # 返回的sample修改后逻辑
        # sample = {
        #     "fc1_image": fc1_img,
        #     "fc1_audio": fc1_wav,
        #     "fc2_image": fc2_img,
        #     "fc2_audio": fc2_wav,
        #     "label": label
        # }
        # print('[ audio_visual_data_udiva.py ] - class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ 函数返回的结果中 , fc1_img.shape() = ', fc1_img.shape, 'fc1_wav.shape() = ', fc1_wav.shape, 'fc2_img.shape() = ', fc2_img.shape, 'fc2_wav.shape() = ', fc2_wav.shape, 'label.shape() = ', label.shape) 
        
        return sample # sample是一个dict对象, 例如：{'image': tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],

    def get_image_data(self, idx):
        # print('[ audio_visual_data_udiva.py ] - get_image_data 函数开始执行')
        # get image data, return PIL.Image.Image object, for example: PIL.Image.Image object, mode=RGB, size=224x224, means the image size is 224x224, and is RGB three channels
        # print('[audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) 开始执行 get_image_data')
        img_dir_path = self.img_dir_ls[idx]
        # img_dir_path 是session的路径，例如 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128'
        # 对session路径下的FC1_A 文件夹和FC2_A文件夹分别进行提取帧，获得2个图片
        fc1_img_dir_path, fc2_img_dir_path = '', ''
        for file in os.listdir(img_dir_path):
            # print('file:', file, 'type:', type(file))
            # judge file is a directory and start with FC1 and not end with .mp4
            if os.path.isdir(os.path.join(img_dir_path, file)) and file.startswith("FC1") and not file.endswith(".mp4"):
                fc1_img_dir_path = os.path.join(img_dir_path, file)
            # judge file is a directory and start with FC2 and not end with .mp4
            if os.path.isdir(os.path.join(img_dir_path, file)) and file.startswith("FC2") and not file.endswith(".mp4"):
                fc2_img_dir_path = os.path.join(img_dir_path, file)
        # print('[audio_visual_data_udiva.py]- get_image_data fc1_img_dir_path:', fc1_img_dir_path, "fc2_img_dir_path:", fc2_img_dir_path)
        # 打印结果: get_image_data fc1_img_dir_path: datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128/FC1_A     fc2_img_dir_path: datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128/FC2_A

        # get fc1 image
        fc1_img_paths = glob.glob(fc1_img_dir_path + "/*.jpg") # fc1_img_paths是FC1_A目录下所有的jpg图像文件路径的集合。 例如：train session id=055128, len(fc1_img_paths): 7228
        # print('[audio_visual_data_udiva.py]- get_image_data fc1_img_paths:', fc1_img_paths, 'len(fc1_img_paths):', len(fc1_img_paths))
        #从所有的frame中按照，取出sample_size个frame，即随机取出sample_size个图片. 例如：一共有9293帧图片 随机取出sample_size = 100个图片. 那么公差=9293/100=92.93，即每隔92.93个图片取一个图片
        simple_fc1_frames = np.linspace(0, len(fc1_img_paths), self.sample_size, endpoint=False, dtype=np.int16) #从所有的frame中   np.linspace 返回等差数列，endpoint=False表示不包含最后一个数 例如：np.linspace(0, 10, 5, endpoint=False) 结果为：array([0., 2., 4., 6., 8.])，即不包含10，只包含0-8，共5个数，间隔为2，即等差数列
        # print('[audio_visual_data_udiva.py]- get_image_data simple_fc1_frames:', simple_fc1_frames) # self.sample_size = 100
        selected_fc1 = random.choice(simple_fc1_frames)
        # print('[audio_visual_data_udiva.py]- get_image_data selected_fc1:', selected_fc1)
        # show the selected fc1 image
        # show_selected_fc1 = mpimg.imread(fc1_img_paths[selected_fc1])
        # plt.imshow(show_selected_fc1) # show the image
        # plt.show()
        try:
            fc1_img = Image.open(fc1_img_paths[selected_fc1]).convert("RGB") # PIL.Image.open() 打开图片，返回一个Image对象，Image对象有很多方法，如：Image.show()，Image.save()，Image.convert()等，Image.convert()用于转换图片模式，如：RGB，L等，为了方便后续处理，这里转换为RGB模式，即3通道
        except:
            # print('[audio_visual_data_udiva.py]exception: fc1_img_paths:', fc1_img_paths)
            return None
        
        # get fc2 image
        fc2_img_paths = glob.glob(fc2_img_dir_path + "/*.jpg")
        # print('[audio_visual_data_udiva.py]- get_image_data fc2_img_paths:', fc2_img_paths)
        simple_fc2_frames = np.linspace(0, len(fc2_img_paths), self.sample_size, endpoint=False, dtype=np.int16) 
        # print('[audio_visual_data_udiva.py]- get_image_data simple_fc2_frames:', simple_fc2_frames) # self.sample_size = 100
        # selected_fc2 = random.choice(simple_fc2_frames)
        selected_fc2 = selected_fc1 # 保证fc1和fc2的图片对应于同一个时刻
        # print('[audio_visual_data_udiva.py]- get_image_data selected_fc2:', selected_fc2)
        # show the selected fc2 image
        # show_selected_fc2 = mpimg.imread(fc2_img_paths[selected_fc2])
        # plt.imshow(show_selected_fc2) # show the image
        # plt.show()
        try:
            fc2_img = Image.open(fc2_img_paths[selected_fc2]).convert("RGB")
        except:
            # print('[audio_visual_data_udiva.py]exception: fc2_img_paths:', fc2_img_paths)
            return None

        # # merge fc1 image and fc2 image ，merge是为了 生成一个2*img_size的图片，即将fc1和fc2的图片拼接在一起，方便后续处理
        # fc1_img = fc1_img.resize((self.img_size, self.img_size))
        # fc2_img = fc2_img.resize((self.img_size, self.img_size))
        # img = Image.new('RGB', (self.img_size * 2, self.img_size))
        # img.paste(fc1_img, (0, 0))
        # img.paste(fc2_img, (self.img_size, 0))

        return fc1_img, fc2_img

        # img_paths = glob.glob(img_dir_path + "/*.jpg") # glob.glob()返回所有匹配的文件路径列表
        # sample_frames = np.linspace(0, len(img_paths), self.sample_size, endpoint=False, dtype=np.int16) # np.linspace()返回在指定的间隔内返回均匀间隔的数字, np.int16是指定返回的数据类型是int16
        # selected = random.choice(sample_frames) # random.choice()从序列中获取一个随机元素, 这里的序列是sample_frames, 也就是说从sample_frames中随机选取一个元素, 例如：sample_frames=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 那么selected就是0, 1, 2, 3, 4, 5, 6, 7, 8, 9中的一个
        # # img_path = random.choice(img_paths)
        # try:
        #     img = Image.open(img_paths[selected]).convert("RGB") # Image.open()打开一个图像文件, convert("RGB")将图像转换为RGB模式, 也就是说将图像转换为三通道, 例如：PIL.Image.Image object, mode=RGB, size=224x224, means the image size is 224x224, and is RGB three channels, 如果不转换为RGB模式, 那么就是单通道, 例如：PIL.Image.Image object, mode=L, size=224x224, means the image size is 224x224, and is L one channel, L means luminance, 亮度, 也就是灰度图, 也就是黑白图.
        #     return img
        # except:
        #     print(img_paths)

    def get_wave_data(self, idx):
        # get wave data, return numpy.ndarray object, for example: numpy.ndarray object, shape=(1, 1, 128), means the wave data is 1x1x128, 音频数据是1x1x128，1代表1个声道，1代表1个采样点，128代表128个采样点，也就是说音频数据是128个采样点，每个采样点是1个声道，每个声道是1个采样点。
        # print('[audio_visual_data_udiva.py]-get_wave_data 函数开始执行')
        
        img_dir_path = self.img_dir_ls[idx] # img_dir_path 是训练集图像帧session的路径，例如 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128'
        session_id = os.path.basename(img_dir_path) # session_id 是训练集图像帧session的名称，例如 '055128'
        wav_dir_path = os.path.join(self.data_root, self.audio_dir, session_id) # wav_dir_path 是训练集音频帧session的路径，例如 'datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/055128'
        fc1_wav_path, fc2_wav_path = '', ''
        for file in os.listdir(wav_dir_path):
            # print('file:', file, 'type:', type(file)) # datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/128129/FC1_A.wav.npy 
            # datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/055128/FC1_A.wav.npy
            # judge file is a file and start with FC1 and end with .wav.npy
            if os.path.isfile(os.path.join(wav_dir_path, file)) and file.startswith('FC1') and file.endswith('.wav.npy'):
                fc1_wav_path = os.path.join(wav_dir_path, file)
                # print('[audio_visual_data_udiva.py]-get_wave_data函数 fc1_wav_path:', fc1_wav_path)
            # judge file is a file and start with FC2 and end with .wav.npy
            if os.path.isfile(os.path.join(wav_dir_path, file)) and file.startswith('FC2') and file.endswith('.wav.npy'):
                fc2_wav_path = os.path.join(wav_dir_path, file)
                # print('[audio_visual_data_udiva.py]-get_wave_data函数 fc2_wav_path:', fc2_wav_path)
        
        # process fc1 wave data
        fc1_wav_ft = np.load(fc1_wav_path) # fc1_wav_ft.shape: (1, 1, 4626530) len(fc1_wav_ft):  1
        # print('[audio_visual_data_udiva.py]-get_wave_data函数 fc1_wav_ft:', fc1_wav_ft, 'fc1_wav_ft.shape:', fc1_wav_ft.shape, '  len(fc1_wav_ft): ', len(fc1_wav_ft))
        try:
            n = np.random.randint(0, len(fc1_wav_ft) - 50176) # np.random.randint 生成一个随机整数，范围是[0, len(fc1_wav_ft) - 50176)
        except:
            # 看日志发现程序在这里执行了，说明len(fc1_wav_ft) - 50176 < 0
            n = 0
        fc1_wav_tmp = fc1_wav_ft[..., n: n + 50176] # 日志：fc1_wav_tmp.shape: (1, 1, 50176)  n: n + 50176 表示从n开始，取50176个采样点，每个采样点是1个声道，每个声道是1个采样点。
        # print('[audio_visual_data_udiva.py]-get_wave_data函数 fc1_wav_tmp:', fc1_wav_tmp, 'fc1_wav_tmp.shape:', fc1_wav_tmp.shape)
        if fc1_wav_tmp.shape[-1] < 50176: # 如果采样点数小于50176，就用0填充  实际程序没有进入这个if语句
            fc1_wav_fill = np.zeros((1, 1, 50176)) # fc1_wav_fill.shape: (1, 1, 50176) np.zeros((1, 1, 50176)) 表示生成一个1行1列50176个元素的矩阵，每个元素都是0
            fc1_wav_fill[..., :fc1_wav_tmp.shape[-1]] = fc1_wav_tmp # fc1_wav_tmp.shape[-1] 表示取fc1_wav_tmp的最后一个维度的元素个数，也就是50176，也就是取fc1_wav_tmp的前50176个元素，然后赋值给fc1_wav_fill的前50176个元素
            fc1_wav_tmp = fc1_wav_fill # 赋值给fc1_wav_tmp
        
        
        # process fc2 wave data
        fc2_wav_ft = np.load(fc2_wav_path)
        try:
            n = np.random.randint(0, len(fc2_wav_ft) - 50176)
        except:
            n = 0
        fc2_wav_tmp = fc2_wav_ft[..., n: n + 50176]
        if fc2_wav_tmp.shape[-1] < 50176:
            fc2_wav_fill = np.zeros((1, 1, 50176))
            fc2_wav_fill[..., :fc2_wav_tmp.shape[-1]] = fc2_wav_tmp
            fc2_wav_tmp = fc2_wav_fill
    
        # return fc1_wav_tmp

        # wav_ft = np.load(wav_path) # np.load()读取.npy文件, 返回的是一个numpy.ndarray对象, 例如：wav_ft.shape=(1, 128, 128), wav_ft.dtype=float32, wav_ft.max()=0.99999994, wav_ft.min()=-0.99999994, wav_ft.mean()=0.000101, wav_ft.std()=0.000101, 
        # try:
        #     n = np.random.randint(0, len(wav_ft) - 50176)
        # except:
        #     n = 0
        # wav_tmp = wav_ft[..., n: n + 50176] 
        # if wav_tmp.shape[-1] < 50176:
        #     wav_fill = np.zeros((1, 1, 50176))
        #     wav_fill[..., :wav_tmp.shape[-1]] = wav_tmp
        #     wav_tmp = wav_fill
        # return wav_tmp
        return fc1_wav_tmp, fc2_wav_tmp


class AudioVisualLstmDataUdiva(VideoDataUdiva): # 基于AudioVisualDataUdiva 增加针对LSTM的数据处理
    def __init__(self, data_root, img_dir, audio_dir, label_file, transform=None, sample_size=16, config=None, mode=None, dataset_name="UDIVA", prefix1="FC1", prefix2="FC2", img_dir_ls=None):
        super().__init__(data_root, img_dir, label_file, audio_dir, sample_size=sample_size, mode=mode, dataset_name=dataset_name, prefix1=prefix1, prefix2=prefix2, img_dir_ls=img_dir_ls)
        self.transform = transform
        self.sample_size = sample_size # 表示从一个视频中采样sample_size个连续的帧图片
        self.frame_idx = 0
        if config.TRAIN.USE_WANDB:
            wandb.config.sample_size = sample_size
        self.cfg = config
        self.img_dtype = None
        self.mode = mode
        self.prefix1 = prefix1
        self.prefix2 = prefix2
        self.is_continue = True

    def __getitem__(self, idx): # idx means the index of session in the directory
        sample = {} # 因为__getitem__ 需要传入参数 idx，所以返回的sample也是一个session对应的img，wav，label
        # cfg.TRAIN.BIMODAL_OPTION == 1 表示仅仅使用视觉数据，cfg.TRAIN.BIMODAL_OPTION == 2 表示仅仅使用语音数据，cfg.TRAIN.BIMODAL_OPTION == 3 表示同时使用视觉和语音数据
        if self.cfg.TRAIN.BIMODAL_OPTION == 1:
            img, session_id, segment_id = self.get_visual_input(idx)
            sample['image'] = img
        elif self.cfg.TRAIN.BIMODAL_OPTION == 2:
            wav, session_id, segment_id = self.get_audio_input(idx)
            sample['audio'] = wav # [2, 1, (sample_size / 5) x 16000 = duration_seconds x 16000]
        elif self.cfg.TRAIN.BIMODAL_OPTION == 3:
            img, session_id_img, segment_id_img = self.get_visual_input(idx)
            wav, session_id_wav, segment_id_wav = self.get_audio_input(idx)
            assert session_id_img == session_id_wav and segment_id_img == segment_id_wav, 'session_id_img != session_id_wav or segment_id_img != segment_id_wav'
            sample['image'] = img
            sample['audio'] = wav
            session_id = session_id_img
            segment_id = segment_id_img

        ### get label data ###
        label = self.get_ocean_label(idx)  # label是True或者False, 代表关系Known是True或者False
        sample['label'] = torch.as_tensor(label, dtype=self.img_dtype)

        # get session_id and segment_id
        sample['session_id'] = session_id
        sample['segment_id'] = segment_id
        
        sample['is_continue'] = self.is_continue
        # print('[__getitem__] dataset:', self.mode, ', idx:', idx, ', session_id:', session_id, ', segment_id:', segment_id, ', label:', label, ', label.shape:', sample['label'].shape)
        return sample  # sample是一个dict字典, {'image': [sample_size, c=6, h=224, w=224], 'audio': [2, 1, sample_size*16000], 'label.shape': [2], 'session_id': '55025', 'segment_id': '1'}

    def get_visual_input(self, idx):
        ### get image data ###
        # print('[get_visual_input] dataloader random idx = ', idx) # idx 和 get_ocean_label(self, index) 里的index含义一样，表示video目录里的第几个video样本
        fc1_img_tensor_list, fc2_img_tensor_list, session_id, segment_id = self.get_image_data(idx)
        img_tensor_list = []
        # print('[get_visual_input] len(fc1_img_tensor_list) = ', len(fc1_img_tensor_list), 'len(fc2_img_tensor_list) = ', len(fc2_img_tensor_list))
        for i in range(len(fc1_img_tensor_list)):
            fc1_img_tensor = fc1_img_tensor_list[i]
            fc2_img_tensor = fc2_img_tensor_list[i]
            img = torch.cat((fc1_img_tensor, fc2_img_tensor), 0) # concatenate the fc1_img_tensor and fc2_img_tensor
            # print('[get_visual_input] img.shape = ', img.shape) # img.shape =  torch.Size([6, 224, 224])
            img_tensor_list.append(img)
        img = torch.stack(img_tensor_list) # 将img_tensor_list中的tensor拼接在一起, 拼接后的维度为(16,6,224,224)，即将16个6x224x224的tensor拼接在一起
        # print('[get_visual_input] len(img_tensor_list) = ', len(img_tensor_list), ', img.shape = ', img.shape, ', img.dtype=', img.dtype)  # len(img_tensor_list) =  16 , img.shape =  torch.Size([sample_size=16, 3*2=6, 224, 224]) img.dtype= torch.float32
        self.img_dtype = img.dtype
        
        session_id = torch.as_tensor(session_id) # convert into tensor
        segment_id = torch.as_tensor(segment_id)
        return img, session_id, segment_id

    def get_image_data(self, idx):
        session_dir_path = self.img_dir_ls[idx] # session的路径
        # print('[audio_visual_data_udiva.py]- session_dir_path=', session_dir_path) # e.g. datasets/udiva_tiny/train/recordings/animals_recordings_train_img/058110_17
        # 对session路径下的FC1_A 文件夹和FC2_A文件夹分别提取sample_size个连续的帧图片
        session_dir_path, segment_idx = session_dir_path.rsplit("_", 1) # segment_idx表示session的第几个segment，例如058110_17中的17表示第17个segment，每个segment包含sample_size张图片
        # print('[get_image_data] session_dir_path:', session_dir_path, ', segment_idx:', segment_idx)
        session_id = session_dir_path.split('/')[-1]
        
        fc1_img_dir_path, fc2_img_dir_path = '', ''
        for file in os.listdir(session_dir_path):
            if os.path.isdir(os.path.join(session_dir_path, file)) and file.startswith(self.prefix1) and not file.endswith(".mp4"): # judge file is a directory and start with FC1 and not end with .mp4
                fc1_img_dir_path = os.path.join(session_dir_path, file)
            if os.path.isdir(os.path.join(session_dir_path, file)) and file.startswith(self.prefix2) and not file.endswith(".mp4"): # judge file is a directory and start with FC2 and not end with .mp4
                fc2_img_dir_path = os.path.join(session_dir_path, file)
        # print('[audio_visual_data_udiva.py]- get_image_data idx:', idx, 'fc1_img_dir_path:', fc1_img_dir_path, "fc2_img_dir_path:", fc2_img_dir_path) # fc1_img_dir_path: datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128/FC1_A     fc2_img_dir_path: datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128/FC2_A

        fc1_img_paths = glob.glob(fc1_img_dir_path + "/*.jpg") # fc1_img_paths是FC1_A目录下所有的jpg图像文件路径的集合。 例如：train session id=055128, len(fc1_img_paths): 7228
        fc2_img_paths = glob.glob(fc2_img_dir_path + "/*.jpg") # fc2_img_paths是FC2_A目录下所有的jpg图像文件路径的集合。 例如：train session id=055128, len(fc2_img_paths): 7228
        # print('[get_image_data] len(fc1_img_paths):', len(fc1_img_paths), 'len(fc2_img_paths):', len(fc2_img_paths))
        
        # ************************* get fc1 image - start *************************
        fc1_img_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0])) # 将fc1_img_paths中所有的图片按照后缀数字升序排序
        # print('[get_image_data] after sort, fc1_img_paths:', fc1_img_paths, 'len(fc1_img_paths):', len(fc1_img_paths))
        
        """ 采样方式一：随机取sample_size个连续的帧图片
        # smaller_len = min(len(fc1_img_paths), len(fc2_img_paths)) # 由于 fc1_img_paths 和 fc2_img_paths 里的帧个数有时候不完全一样(因为只提取人脸，视频里并非每一帧都能检测到完整的人脸)，所以这里要取两者中较小的那个值
        # if self.sample_size > smaller_len: # 如果 self.sample_size 比 smaller_len还大，那么就把 self.sample_size 设置为 smaller_len，即取这个session里的所有帧作为输入
        #     self.sample_size = smaller_len
        # self.frame_idx = random.randint(0, (smaller_len - self.sample_size)) # 在0，len(fc1_img_paths)-sample_size 之间，生成一个随机数frame_idx，表示索引为frame_idx的图片帧 # 如果一共16帧,索引值候选范围:从index=0到15, 共16个, sample_size=4, 那么frame_index=随机数，且取值范围是[0,12], 12=16-4
        # sample_fc1_frames = fc1_img_paths[self.frame_idx:self.frame_idx + self.sample_size] # 在 fc1_img_paths 取出从frame_idx开始的sample_size个图片帧，从所有的frame中按照self.frame_idx开始，取出sample_size个frame，即随机取出sample_size个图片，包含self.frame_idx，不包含self.frame_idx+sample_size，左闭右开 """
        
        # 采样方式二：按照segment_idx取sample_size个连续的帧图片
        start_frame_id = int(self.sample_size) * (int(segment_idx) - 1) # 例如，如果self.sample_size=4，segment_idx=2，那么start_frame_id=4*(2-1)=4
        end_frame_id = start_frame_id + self.sample_size # 例如，如果self.sample_size=4，segment_idx=2，那么end_frame_id=4+4=8, 即[4:8]，共4个frame, 是该视频的第2个segment，第一个segment是[0:3], 第二个segment是[4:7], 第三个segment是[8:11]
        sample_fc1_frames = fc1_img_paths[start_frame_id:end_frame_id]
        # print('[get_image_data] start_frame_id:', start_frame_id, ', end_frame_id:', end_frame_id, ', segment_idx:', segment_idx, ', self.sample_size:', self.sample_size, ', len(sample_fc1_frames):', len(sample_fc1_frames), sample_fc1_frames)
        if not self.check_continue(sample_fc1_frames):
            # print('[get_image_data] sample_fc1_frames is not continuous')
            self.is_continue = False
        
        # 遍历sample_fc1_frames里的每个图片帧，将其转换为RGB模式
        fc1_img_tensor_list = []
        for i, fc1_frame_path in enumerate(sample_fc1_frames):
            # print('[get_image_data] enumerate(sample_fc1_frames), fc1_frame_path:', fc1_frame_path, 'i:', i)
            try:
                fc1_img = Image.open(fc1_frame_path).convert("RGB") # PIL.Image.open() 打开图片，返回一个Image对象，Image对象有很多方法，如：Image.show()，Image.save()，Image.convert()等，Image.convert()用于转换图片模式，如：RGB，L等，为了方便后续处理，这里转换为RGB模式，即3通道
            except Exception as e:
                # print('[audio_visual_data_udiva.py]exception:', e, 'fc1_frame_path:', fc1_frame_path)
                return None
            # 将图片转换为tensor
            if self.transform:
                fc1_img_tensor = self.transform(fc1_img)
                # print('\n[get_image_data] before transform, fc1_img.size:', fc1_img.size, 'type(fc1_img):', type(fc1_img), ', \nafter transform, fc1_img_tensor.size():', fc1_img_tensor.size(), 'type(fc1_img_tensor):', type(fc1_img_tensor)) if i == 0 else None
            # 将图片tensor添加到fc1_imgs中
            fc1_img_tensor_list.append(fc1_img_tensor)
        # print('[get_image_data] fc1_img_tensor_list:', fc1_img_tensor_list, 'len(fc1_img_tensor_list):', len(fc1_img_tensor_list))
        # ************************* get fc1 image - done *************************
        
        
        # ************************* get fc2 image - start*************************
        fc2_img_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0])) # 将fc2_img_paths中所有的图片按照后缀数字升序排序，然后随机取出连续的sample_size个图片帧
        # sample_fc2_frames = fc2_img_paths[self.frame_idx:self.frame_idx + self.sample_size] # 采样方式一：随机取sample_size个连续的帧图片。从所有的frame中按照self.frame_idx开始，取出sample_size个frame，即随机取出sample_size个图片
        sample_fc2_frames = fc2_img_paths[start_frame_id:end_frame_id] # 采样方式二：按照segment_idx取sample_size个连续的帧图片
        # print('[get_image_data] sample_fc2_frames:', sample_fc2_frames, 'len(sample_fc2_frames):', len(sample_fc2_frames))
        if not self.check_continue(sample_fc2_frames):
            # print('[get_image_data] sample_fc2_frames is not continuous')
            self.is_continue = False

        fc2_img_tensor_list = []
        for i, fc2_frame_path in enumerate(sample_fc2_frames): # 遍历sample_fc2_frames里的每个图片帧，将其转换为RGB模式
            # print('[get_image_data] enumerate(sample_fc2_frames), fc2_frame_path:', fc2_frame_path, 'i:', i)
            try:
                fc2_img = Image.open(fc2_frame_path).convert("RGB") # PIL.Image.open() 打开图片，返回一个Image对象，Image对象有很多方法，如：Image.show()，Image.save()，Image.convert()等，Image.convert()用于转换图片模式，如：RGB，L等，为了方便后续处理，这里转换为RGB模式，即3通道
            except Exception as e:
                # print('[audio_visual_data_udiva.py]exception:', e, 'fc2_frame_path:', fc2_frame_path)
                return None
            if self.transform: # 将图片转换为tensor
                fc2_img_tensor = self.transform(fc2_img)
            fc2_img_tensor_list.append(fc2_img_tensor) # 将图片tensor添加到fc2_imgs中
        # print('[audio_visual_data_udiva.py]- get_image_data len(fc1_img_tensor_list):', len(fc1_img_tensor_list), 'len(fc2_img_tensor_list):', len(fc2_img_tensor_list))
        # ************************* get fc2 image - done *************************
        
        return fc1_img_tensor_list, fc2_img_tensor_list, int(session_id), int(segment_idx)

    def get_audio_input(self, idx):
        if self.cfg.MODEL.NAME == 'ssast_udiva':
            return self.ssast_audio_preprocess(idx) 
        
        ### get audio data ###
        fc1_wav, fc2_wav, session_id, segment_id = self.get_wave_data(idx) # fc1_wav是一个numpy.ndarray对象
        fc1_wav = torch.as_tensor(fc1_wav, dtype=self.img_dtype) # shape=(1,1,50176)
        fc2_wav = torch.as_tensor(fc2_wav, dtype=self.img_dtype) # shape=(1,1,50176)
        wav = torch.cat((fc1_wav, fc2_wav), 0) # shape=(2,1,50176)  # 将两个tensor拼接在一起 concatenate the fc1_wav and fc2_wav, 拼接后的维度为(2,1,50176)，即将两个(1,1,50176)的tensor拼接在一起
        # print('[audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ - torch.as_tensor之后 wav.shape=', wav.shape, ', fc1_wav.shape=', fc1_wav.shape, ', fc2_wav.shape=', fc2_wav.shape) # wav.shape= torch.Size([2, 1, 256000]) fc1_wav.shape= torch.Size([1, 1, 256000]) fc2_wav.shape= torch.Size([1, 1, 256000])
        return wav, session_id, segment_id

    def get_wave_data(self, idx):
        session_dir_path = self.img_dir_ls[idx] # session_dir_path 是训练集图像帧session的路径，例如 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128'
        # print('[get_wave_data] idx:', idx, ', session_dir_path:', session_dir_path) # 验证每次dataloader调用时，idx的值是否是shuffle后的, 即idx的值是否是随机的
        session_dir_path, segment_idx = session_dir_path.rsplit("_", 1) # segment_idx表示session的第几个segment，例如058110_17中的17表示第17个segment，每个segment包含sample_size张图片
        segment_idx = int(segment_idx) # 将segment_idx从str转换为int类型
        session_id = session_dir_path.split("/")[-1]
        # print('[audio_visual_data_udiva.py]-get_wave_data, session_id:', os.path.basename(session_dir_path)) # session_id 是训练集图像帧session的名称，例如 '055128'
        
        # 构造wav所在的路径，需要和img的路径保持一致，为同一task的同一个session，即如果 session_dir_path: datasets/udiva_tiny/train/recordings/animals_recordings_train_img/059134，那么 wav_dir_path: datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/059134
        # wav_dir_path = session_dir_path.replace('_img', '_wav') # 将 session_dir_path 路径里的 _img 替换为 _wav，例如 datasets/udiva_tiny/train/recordings/animals_recordings_train_img/059134 替换为 datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/059134
        wav_dir_path = session_dir_path.replace('img', 'wav') # 将 session_dir_path 路径里的 _img 替换为 _wav，例如 datasets/udiva_tiny/train/recordings/animals_recordings_train_img/059134 替换为 datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/059134
        # print('[audio_visual_data_udiva.py]-get_wave_data, session_dir_path:', session_dir_path, 'session_id:', session_id, 'wav_dir_path:', wav_dir_path)
        
        fc1_wav_path, fc2_wav_path = '', ''
        for file in os.listdir(wav_dir_path):
            # print('file:', file, 'type:', type(file)) # datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/128129/FC1_A.wav.npy 
            if os.path.isfile(os.path.join(wav_dir_path, file)) and file.startswith(self.prefix1) and file.endswith('.wav.npy'): # judge file is a file and start with prefix1 and end with .wav.npy
                fc1_wav_path = os.path.join(wav_dir_path, file)
            if os.path.isfile(os.path.join(wav_dir_path, file)) and file.startswith(self.prefix2) and file.endswith('.wav.npy'): # judge file is a file and start with prefix2 and end with .wav.npy
                fc2_wav_path = os.path.join(wav_dir_path, file)
        
        # ************************* process fc1 wave data *************************
        fc1_wav_ft = np.load(fc1_wav_path) # fc1_wav_ft.shape: (1, 1, 4626530) len(fc1_wav_ft):  1
        # print('[audio_visual_data_udiva.py]-get_wave_data函数 fc1_wav_ft:', fc1_wav_ft, 'fc1_wav_ft.shape:', fc1_wav_ft.shape, '  len(fc1_wav_ft): ', len(fc1_wav_ft))
        # 举例：animals_recordings_test_wav/008105/FC1_A.wav   时长:6分20秒=380秒    其对应的fc1_wav_ft.shape: (1, 1, 6073598)   6073598/380=15983.1526316，即每秒采样15983.1526316个采样点，为什么不完全等于 16000 呢？通过运行soxi -D FC1_A.wav，得到其精确的时长=379.599819秒，因此 379.599819*16000=6073597.104，四舍五入为6073598，能匹配！！
        # 举例：animals_recordings_val_wav/001081/FC1_A.wav    时长:10分07秒=607秒   其对应的fc1_wav_ft.shape: (1, 1, 9707799)   9707799/607=15993.0790774，即每秒采样15993.0790774个采样点, 精确时间: 606.737415秒, 因此 606.737415*16000=9707798.64，四舍五入为9707799，能匹配！！
        # 举例：animals_recordings_val_wav/001080/FC1_A.wav    时长:8分59秒=539秒    其对应的fc1_wav_ft.shape: (1, 1, 8621477)   8621477/539=15995.3191095，即每秒采样15995.3191095个采样点
        # 举例：animals_recordings_train_wav/055125/FC1_A.wav  时长:6分25秒=385秒    其对应的fc1_wav_ft.shape: (1, 1, 6161648)   6161648/385=16004.2805195，即每秒采样16004.2805195个采样点
        # 举例：animals_recordings_train_wav/058110/FC1_A.wav  时长:7分07秒=427秒    其对应的fc1_wav_ft.shape: (1, 1, 6824438)   6824438/427=15982.2903981，即每秒采样15982.2903981个采样点
        # 结合 leenote/video_to_wav/raw_audio_process.py 里的librosa_extract函数，wav_ft = librosa.load(wav_file_path, 16000)[0][None, None, :]，即当wav转为npy时，采样率为16000，所以每秒采样16000个点，所以每个npy文件的第三个维度值就是时长*16000
        
        """采样方式一
        # 采样方式一：随机取sample_size个连续的帧图片
        # 从音频中的第self.frame_idx秒开始，取时长为self.sample_size秒的音频片段, 即从第self.frame_idx*16000个采样点开始，取连续的self.sample_size*16000个采样点
        # start_point = self.frame_idx * 16000
        # end_point = start_point + self.sample_size * 16000
        """
        
        """采样方式二 （代码已验证）
        采样方式二：与图片的采样方式保持一致，图片采样了sample_size个连续的帧图片，对应的耗时是sample_size/5秒的视频时长，因此音频也需要采样同等时长的音频片段
        一个视频片段的时长，即sample_size个连续的帧图片的时长，为sample_size/5秒，因为每秒钟有5帧图片（5是由数据预处理阶段的降采样决定的）
        当前需要取第segment_idx个视频片段，即音频的取样开始时间为segment_idx*sample_size秒
        举例：如果sample_size=16, 当前segment_idx=30,即需要取第30个视频片段，说明采样的开始时间需要在前29个视频片段结束后，前29个视频片段的时长为 29*segment_duration=29*(16/5)=29*3.2=92.8秒, 对应的有 92.8*16000=1487360个采样点，因此音频的采样开始点为 1487360
        为了和图片的采样时长保持一样，我们需要采样的时长为：sample_size/5 秒，对应的有 sample_size/5*16000 个采样点，因此音频的采样结束点为： 1487360 + sample_size/5*16000 = 1487360 + 16/5*16000 = 1487360 + 3200 = 1490560
        """
        segment_duration = self.sample_size / 5  # 单位：秒，例如sample_size=5，即每个视频片段的时长为1秒
        start_point = int((segment_idx-1) * segment_duration * 16000)
        end_point = int(start_point + segment_duration * 16000)
        # print('[get_wave_data] sample_size:', self.sample_size, ', segment_idx:', segment_idx, ', start_point:', start_point, ', end_point:', end_point)
        
        if end_point > fc1_wav_ft.shape[-1]: # 如果end_point > fc1_wav_ft.shape[-1]，则说明从采样的那一秒往后加sample_size秒会超过音频的最后一秒，因此只需要取到音频的最后一秒（最后一个采样点）即可
            end_point = fc1_wav_ft.shape[-1]
        # print('[dpcv/data/datasets/audio_visual_data_udiva.py] fc1_wav_ft.shape:', fc1_wav_ft.shape, ', fc1 start_point:', start_point, 'end_point:', end_point)
        fc1_wav_tmp = fc1_wav_ft[..., start_point: end_point] # fc1_wav_tmp.shape: 

        # print('[audio_visual_data_udiva.py]-get_wave_data函数 fc1_wav_tmp.shape:', fc1_wav_tmp.shape)
        # if fc1_wav_tmp.shape[-1] < self.sample_size * 16000: # 如果采样点数小于self.sample_size * 16000，就用0填充剩余的采样点
        #     fc1_wav_fill = np.zeros((1, 1, self.sample_size * 16000))
        #     fc1_wav_fill[..., :fc1_wav_tmp.shape[-1]] = fc1_wav_tmp # 将fc1_wav_tmp的采样点填充到fc1_wav_fill的前fc1_wav_tmp.shape[-1]个采样点
        #     fc1_wav_tmp = fc1_wav_fill # 赋值给fc1_wav_tmp
        
        # ************************* process fc2 wave data *************************
        fc2_wav_ft = np.load(fc2_wav_path)
        
        """ 采样方式一：随机取sample_size个连续的帧图片
        start_point = self.frame_idx * 16000
        end_point = start_point + self.sample_size * 16000 """
        
        if end_point > fc2_wav_ft.shape[-1]: 
            end_point = fc2_wav_ft.shape[-1]
        # print('[dpcv/data/datasets/audio_visual_data_udiva.py] fc2_wav_ft.shape:', fc2_wav_ft.shape, ', fc2 start_point:', start_point, 'end_point:', end_point)
        fc2_wav_tmp = fc2_wav_ft[..., start_point: end_point]
        
        # if fc2_wav_tmp.shape[-1] < self.sample_size * 16000:
        #     fc2_wav_fill = np.zeros((1, 1, self.sample_size * 16000))
        #     fc2_wav_fill[..., :fc2_wav_tmp.shape[-1]] = fc2_wav_tmp
        #     fc2_wav_tmp = fc2_wav_fill

        # print('[get_wave_data] fc1_wav_tmp.shape:', fc1_wav_tmp.shape, 'fc2_wav_tmp.shape:', fc2_wav_tmp.shape) # fc1_wav_tmp.shape: (1, 1, 256000) fc2_wav_tmp.shape: (1, 1, 256000)
        return fc1_wav_tmp, fc2_wav_tmp, int(session_id), segment_idx

    def ssast_audio_preprocess(self, idx):
        """audio preprocess for ssast model: https://github.com/YuanGongND/ssast
        Args:
            idx (int): index of the data
            dataset_mean (float): mean of the dataset, default: -4.2677393
            dataset_std (float): std of the dataset, default: 4.5689974
        Returns:
            _type_: _description_
        """
        
        fc1_wav, fc2_wav, session_id, segment_id = self.get_wave_data(idx) # 分别从2个视频的完整音频中截取时长为self.sample_size秒的音频片段，共有连续的self.sample_size*16000个采样点，得到fc1_wav和fc2_wav, 2个wav都是一个numpy.ndarray对象
        # fc1_wav.shape: (1, 1, 256000) fc2_wav.shape: (1, 1, 256000), sample_size*16000=256000
        fc1_fbank = self.ssast_wav_process(fc1_wav) 
        fc2_fbank = self.ssast_wav_process(fc2_wav)
        wav = torch.cat([fc1_fbank, fc2_fbank], dim=1) # 将fc1_fbank和fc2_fbank沿着列方向拼接，得到wav, shape, e.g (1598, 128+128) -> (1598, 256)
        # print('[audio_visual_data_udiva.py]-ssast_audio_preprocess wav.shape:', wav.shape) # wav.shape: (1598, 256)
        return wav, session_id, segment_id

    def ssast_wav_process(self, wav, dataset_mean=-4.2677393, dataset_std=4.5689974):
        """audio preprocess for ssast model: https://github.com/YuanGongND/ssast
        Args:
            wav (torch.tensor): shape=(1,1,256000), 256000=16000*self.sample_size
            dataset_mean (float): mean of the dataset, default: -4.2677393
            dataset_std (float): std of the dataset, default: 4.5689974
        Returns:
            _type_: _description_
        """
        wav = torch.as_tensor(wav, dtype=self.img_dtype) # 将wav从numpy.ndarray对象转为torch.tensor对象，shape=(1,1,256000), 256000=16000*self.sample_size
        wav = wav.squeeze(0) # 将wav的shape从(1,1,256000)转为(1,256000)
        wav_fbank = self._wav2fbank(wav) # [n_frames(帧数), num_mel_bins] e.g. [1598, 128]
        wav_fbank = (wav_fbank - dataset_mean) / (dataset_std * 2) # use dataset mean and std to normalize the input.
        return wav_fbank

    def _wav2fbank(self, waveform, sr=16000, num_mel_bins=128):
        """convert wav to fbank
        Args:
            wav (tensor): input wave tensor, e.g (1,256000) 256000=16000*self.sample_size
            sr (int): sample rate of wav, default 16000
            num_mel_bins (int): 滤波器的数量 num of mel bins, default 128 num_mel_bins 是 torchaudio.compliance.kaldi.fbank() 函数的一个参数，
                        用于指定计算梅尔频率倒谱系数（Mel Frequency Cepstral Coefficients，简称MFCC）时使用的梅尔滤波器的数量。梅尔滤波器是一组三角形的滤波器，它们在频域上重叠并覆盖整个频谱，用于模拟人耳的听觉特性。
                        MFCC是一种常用的语音特征提取方法，它可以将语音信号转换为一个维度较低的特征向量，常用于语音识别、说话人识别等任务。
                        通常，num_mel_bins 的值可以设置为 20 ~ 40 之间的整数，用于计算一段语音信号的 MFCC 特征向量。这个值不是越大越好，过大会导致计算量增加而且可能会引入噪声，过小会导致丢失一些重要的语音特征。可以根据具体任务的需要和数据集的特点进行调整。
        Returns:
            _type_: _description_
        """
        waveform = waveform - waveform.mean() # 对fc1_wav的每个采样点减去fc1_wav的均值
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=num_mel_bins, dither=0.0, frame_shift=10)
        # print('[audio_visual_data_udiva.py]-_wav2fbank函数 waveform.shape:', waveform.shape, ', fbank.shape:', fbank.shape)
        # fbank函数返回的shape为[帧数, 特征维度], 如果fbank.shape: torch.Size([1598, 128]), 那么帧数就是1598约等于16秒*100帧，正好和sample_size*100对应, 即一秒100帧。在fbank.shape中，帧数frames的计算方式为：将处理后的音频数据按照固定的帧长和帧移进行分割，帧长和帧移一般是预先设定的参数，比如在本代码中的帧移是10毫秒。分割后，每个帧就对应一个音频特征向量，frames即为分割后的帧数。
        
        # old sample strategy:
        # sample_size=16秒: waveform.shape: torch.Size([1, 256000]) , fbank.shape: torch.Size([1598, 128]) 根据 1598计算公式：(256000-400)/160=1598, 400=16000*0.025, 160=16000*0.01, 0.025和0.01是kaldi的默认参数, 0.025是窗口长度，0.01是窗口移动步长, 0.025和0.01都是以秒为单位
        # sample_size=32秒: waveform.shape: torch.Size([1, 512000]) , fbank.shape: torch.Size([3198, 128]) 根据SSAST代码库里1024 frames (i.e., 10.24s)的解释，3198帧对应31.98秒 https://github.com/YuanGongND/ssast#Self-Supervised-Pretraining 
        # sample_size=48秒: waveform.shape: torch.Size([1, 768000]) , fbank.shape: torch.Size([4798, 128])
        # sample_size=64秒: waveform.shape: torch.Size([1, 1024000]) , fbank.shape: torch.Size([6398, 128])
        # 规律: fbank.shape[0] = sample_size * 100 - 2
        
        # new sample strategy:
        # sample_size = 16, time = 16/5 = 3.2s, waveform.shape: torch.Size([1, 51200]) , fbank.shape: torch.Size([318, 128])
        # sample_size = 32, time = 32/5 = 6.4s, waveform.shape: torch.Size([1, 102400]) , fbank.shape: torch.Size([638, 128])
        # 规律: fbank.shape[0] = (sample_size/5) * 100 - 2
        return fbank

    def check_continue(self, frames_list):
        """ Check if the suffix ids of all frames in the list are consecutive (check if the frames_list is continuous)
        Args:
            frames_list (list): list of frames
            e.g. ['xxx/face_19.jpg', 'xxx/face_20.jpg', 'xxx/face_21.jpg', 'xxx/face_22.jpg'] return True
            e.g. ['xxx/face_19.jpg', 'xxx/face_20.jpg', 'xxx/face_21.jpg', 'xxx/face_23.jpg', ....] return False
        """
        # frames_list_temp1 = ['datasets/noxi_tiny/img/003/Novice_video/face_19.jpg', 'datasets/noxi_tiny/img/003/Novice_video/face_20.jpg', 'datasets/noxi_tiny/img/003/Novice_video/face_23.jpg']
        # frames_list_temp2 = ['datasets/noxi_tiny/img/003/Novice_video/face_19.jpg', 'datasets/noxi_tiny/img/003/Novice_video/face_20.jpg', 'datasets/noxi_tiny/img/003/Novice_video/face_21.jpg']
        # 依次frames_list_temp1，frames_list_temp2中选取一个作为frames_list
        # frames_list = frames_list_temp1 if random.random() > 0.5 else frames_list_temp2
        
        frames_list = [int(frame.split('/')[-1].split('.')[0].split('_')[-1]) for frame in frames_list]
        # print('[check_continue] frames_list:', frames_list)
        # frames_list.sort()
        for i in range(len(frames_list)-1):
            if frames_list[i+1] - frames_list[i] != 1:
                return False
        return True


class ALLSampleAudioVisualDataUdiva(AudioVisualDataUdiva):

    def __getitem__(self, idx):
        label = self.get_ocean_label(idx)
        imgs = self.get_image_data(idx)
        wav = self.get_wave_data(idx)

        if self.transform:
            imgs = [self.transform(img) for img in imgs]

        wav = torch.as_tensor(wav, dtype=imgs[0].dtype)
        label = torch.as_tensor(label, dtype=imgs[0].dtype)

        sample = {"image": imgs, "audio": wav, "label": label}
        return sample

    def get_image_data(self, idx):
        img_dir_path = self.img_dir_ls[idx]
        img_path_ls = glob.glob(f"{img_dir_path}/*.jpg")
        sample_frames = np.linspace(0, len(img_path_ls), self.sample_size, endpoint=False, dtype=np.int16)
        img_path_ls_sampled = [img_path_ls[ind] for ind in sample_frames]
        img_obj_ls = [Image.open(path) for path in img_path_ls_sampled]
        return img_obj_ls


class ALLSampleAudioVisualDataUdiva2(AudioVisualDataUdiva):

    def __getitem__(self, idx):
        label = self.get_ocean_label(idx)
        imgs = self.get_image_data(idx)
        wav = self.get_wave_data(idx)

        if self.transform:
            imgs = [self.transform(img) for img in imgs]

        wav = torch.as_tensor(wav, dtype=imgs[0].dtype)
        label = torch.as_tensor(label, dtype=imgs[0].dtype)

        sample = {"image": imgs, "audio": wav, "label": label}
        return sample

    def get_image_data(self, idx):
        img_dir_path = self.img_dir_ls[idx]
        img_path_ls = sorted(glob.glob(f"{img_dir_path}/*.jpg"))
        # sample_frames = np.linspace(0, len(img_path_ls), self.sample_size, endpoint=False, dtype=np.int16)
        # img_path_ls_sampled = [img_path_ls[ind] for ind in sample_frames]
        img_obj_ls = [Image.open(path) for path in img_path_ls]
        return img_obj_ls


def make_data_loader(cfg, mode):
    trans = set_audio_visual_transform()
    if mode == "train":
        data_set = AudioVisualDataUdiva(
            cfg.DATA_ROOT,  # "/home/ssd500/personality_data",
            cfg.TRAIN_IMG_DATA,  # "image_data/train_data",
            cfg.TRAIN_AUD_DATA,  # "voice_data/train_data",
            cfg.TRAIN_LABEL_DATA,  # "annotation/annotation_training.pkl",
            trans
        )
    elif mode == "valid":
        data_set = AudioVisualDataUdiva(
            cfg.DATA_ROOT,  # "/home/ssd500/personality_data",
            cfg.VALID_IMG_DATA,  # "image_data/valid_data",
            cfg.VALID_AUD_DATA,  # "voice_data/valid_data",
            cfg.VALID_LABEL_DATA,  # annotation/annotation_validation.pkl",
            trans
        )
    elif mode == "trainval":
        data_set = AudioVisualDataUdiva(
            cfg.DATA_ROOT,  # "../datasets",
            cfg.TRAINVAL_IMG_DATA,  # ["image_data/training_data_01", "image_data/validation_data_01"],
            cfg.TRANIVAL_AUD_DATA,  # ["voice_data/trainingData", "voice_data/validationData"],
            cfg.TRAINVAL_LABEL_DATA,  # ["annotation/annotation_training.pkl", "annotation/annotation_validation.pkl"],
            trans,
        )
    elif mode == "test":
        data_set = AudioVisualDataUdiva(
            cfg.DATA_ROOT,  # "/home/ssd500/personality_data",
            cfg.TEST_IMG_DATA,  # "image_data/test_data",
            cfg.TEST_AUD_DATA,  # "voice_data/test_data",
            cfg.TEST_LABEL_DATA,  # "annotation/annotation_test.pkl",
            trans
        )
    else:
        raise ValueError("mode must in one of [train, valid, trianval, test]")

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKS,
        drop_last=True
    )

    return data_loader


""" 
@DATA_LOADER_REGISTRY.register()
def bimodal_resnet_data_loader_udiva(cfg, mode):
    assert (mode in ["train", "valid", "test", "full_test"]), " 'mode' only supports 'train' 'valid' 'test' "
    transforms = build_transform_spatial(cfg) # TRANSFORM: "standard_frame_transform"
    # TRANSFORM: "standard_frame_transform" 定义如下
    # def standard_frame_transform():
    #     import torchvision.transforms as transforms # torchvision.transforms 作用是对图像进行预处理, 例如: 转换为tensor, 归一化, 裁剪, 缩放等, 详见: 
    #     transforms = transforms.Compose([
    #         transforms.Resize(256),  # 缩放图像, 保持长宽比不变, 最短边为256像素
    #         transforms.RandomHorizontalFlip(0.5), # 随机水平翻转, 概率为0.5
    #         transforms.CenterCrop((224, 224)), # 从中心裁剪, 裁剪后的图像大小为(224, 224)
    #         transforms.ToTensor(), # 将图像转换为tensor, 并且将图像的通道数放在第一维
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 归一化
    #     ])
    #     return transforms

    if mode == "train":
        # print('[audio_visual_data_udiva.py]- bimodal_resnet_data_loader_udiva 开始进行训练集的dataloader初始化...')
        dataset = AudioVisualDataUdiva(
            cfg.DATA.ROOT,
            cfg.DATA.TRAIN_IMG_DATA, # 'TRAIN_IMG_DATA': 'ChaLearn2016_tiny/train_data',
            cfg.DATA.TRAIN_AUD_DATA, # 'TRAIN_AUD_DATA': 'ChaLearn2016_tiny/voice_data/voice_librosa/train_data',
            cfg.DATA.TRAIN_LABEL_DATA, # 'TRAIN_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_training.pkl',
            transforms
        )
        batch_size = cfg.DATA_LOADER.TRAIN_BATCH_SIZE
    elif mode == "valid":
        # print('[audio_visual_data_udiva.py]- bimodal_resnet_data_loader_udiva 开始进行验证集的dataloader初始化...')
        dataset = AudioVisualDataUdiva(
            cfg.DATA.ROOT,
            cfg.DATA.VALID_IMG_DATA, #'VALID_IMG_DATA': 'ChaLearn2016_tiny/valid_data',
            cfg.DATA.VALID_AUD_DATA, # 'VALID_AUD_DATA': 'ChaLearn2016_tiny/voice_data/voice_librosa/valid_data',
            cfg.DATA.VALID_LABEL_DATA, # 'VALID_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_validation.pkl',
            transforms
        )
        batch_size = cfg.DATA_LOADER.VALID_BATCH_SIZE
    elif mode == "full_test":
        # print('[audio_visual_data_udiva.py]- bimodal_resnet_data_loader_udiva 开始进行测试集的dataloader初始化...')
        return ALLSampleAudioVisualDataUdiva(
            cfg.DATA.ROOT,
            cfg.DATA.TEST_IMG_DATA, # 'TEST_IMG_DATA': 'ChaLearn2016_tiny/test_data',
            cfg.DATA.TEST_AUD_DATA, # 'TEST_AUD_DATA': 'ChaLearn2016_tiny/voice_data/voice_librosa/test_data',
            cfg.DATA.TEST_LABEL_DATA, # 'TEST_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_test.pkl',
            transforms
        )
    else:
        dataset = AudioVisualDataUdiva(
            cfg.DATA.ROOT,
            cfg.DATA.TEST_IMG_DATA,
            cfg.DATA.TEST_AUD_DATA,
            cfg.DATA.TEST_LABEL_DATA,
            transforms
        )
        batch_size = cfg.DATA_LOADER.VALID_BATCH_SIZE
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=cfg.DATA_LOADER.SHUFFLE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,  # cfg.NUM_WORKS
        drop_last=cfg.DATA_LOADER.DROP_LAST,
    )
    return data_loader

 """


@DATA_LOADER_REGISTRY.register()
def bimodal_resnet_lstm_data_loader_noxi(cfg, mode, fold_id=None): # NoXi dataset
    # print('[audio_visual_data_udiva.py] input fold_id=', fold_id)
    assert (mode in ["train", "valid", "test", "full_test"]), " 'mode' only supports 'train' 'valid' 'test' "
    transforms = build_transform_spatial(cfg) # TRANSFORM: "standard_frame_transform"
    
    img_data = cfg.DATA.NOXI_IMG_DATA
    aud_data = cfg.DATA.NOXI_AUD_DATA
    # noxi 全量数据集
    noxi_full_dataset = AudioVisualLstmDataUdiva(
        cfg.DATA.ROOT,
        img_data,
        aud_data,
        cfg.DATA.NOXI_LABEL_DATA,
        transforms,
        sample_size=cfg.DATA.SAMPLE_SIZE,
        config=cfg,
        mode="noxi_full_dataset",
        dataset_name="NoXi",
        prefix1="Expert",
        prefix2="Novice"
    )
    # print('[audio_visual_data_udiva.py]noxi_full_dataset len=', len(noxi_full_dataset), ' img_dir_ls type:', type(noxi_full_dataset.img_dir_ls)) # img_dir_ls type: <class 'list'>
    session_dir_ls = noxi_full_dataset.img_dir_ls
    
    # video_data_udiva = VideoDataUdiva(data_root=cfg.DATA.ROOT, img_dir=img_data, label_file=cfg.DATA.NOXI_LABEL_DATA)
    # session_dir_ls = video_data_udiva.parse_data_dir_v2(img_data)
    # print('[audio_visual_data_udiva.py]session_dir_ls len=', len(session_dir_ls), ' type:', type(session_dir_ls)) # session_dir_ls len= 100  type: <class 'list
    X = [] # data
    y = [] # label
    for session_dir_path in session_dir_ls:
        idx = session_dir_ls.index(session_dir_path) # session_dir_path 在session_dir_ls中的索引
        # session_id, segment_idx = session_dir_path.rsplit("_", 1)
        label = noxi_full_dataset.get_ocean_label(idx)
        # 获得元素1在label中的索引, 索引0到3分别代表: 'N/A', 'Acquaintances', 'Friends', 'Very good friends'
        for i, x in enumerate(label):
            if x == 1:
                relation_index = i
                break
        
        # print('idx:', idx, ', session_dir_path=', session_dir_path, ', label=', label, ', relation_index:', relation_index) # e.g. idx: 0 , session_dir_path= datasets/noxi/img/004_1 , label= [0, 0, 1, 0] , relation_index: 2
        X.append(session_dir_path)
        y.append(relation_index)

    # print('[audio_visual_data_udiva.py]before fold, X len=', len(X), ', y len=', len(y), ', X[:3]=', X[:3], ', y[:3]=', y[:3], ', X[-3:]=', X[-3:], ', y[-3:]=', y[-3:])
    
    # 将 X 和 y 转换为 numpy array
    X = np.array(X)
    y = np.array(y)
    skf = StratifiedKFold(n_splits=cfg.DATA_LOADER.NUM_FOLD)
    # 遍历每一折
    X_train_list, X_valid_list, y_train_list, y_valid_list = [], [], [], []
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        # 根据索引划分训练集和验证集
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        # print('[audio_visual_data_udiva.py]fold_idx:', fold_idx) # when K=5, fold_idx=0, 1, 2, 3, 4
        # print('[audio_visual_data_udiva.py]train_idx:', train_idx, 'len(train_idx): ', len(train_idx))
        # print('[audio_visual_data_udiva.py]valid_idx:', valid_idx, 'len(valid_idx): ', len(valid_idx))
        # [重要] 经过对比日志，不同时间段运行脚本时 train_idx valid_idx 一致！！因此，可以在这里进行for循环; 如果不一致，会导致每次运行脚本时，训练集和验证集的划分不一致
        print('[audio_visual_data_udiva.py]after fold, fold_idx:', fold_idx, ', len(X_train):', len(X_train), 'len(y_train): ', len(y_train), 'len(X_valid): ', len(X_valid), 'len(y_valid): ', len(y_valid))
        # print('[audio_visual_data_udiva.py]after fold, fold_idx:', fold_idx, ', X_train[:3]:', list(X_train[:3]), 'y_train[:3]: ', y_train[:3])
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_valid_list.append(X_valid)
        y_valid_list.append(y_valid)
    
    # print('[audio_visual_data_udiva.py]len(X_train_list): ', len(X_train_list), 'len(y_train_list): ', len(y_train_list), 'len(X_valid_list): ', len(X_valid_list), 'len(y_valid_list): ', len(y_valid_list)) # 5, 5, 5, 5
    # print('[audio_visual_data_udiva.py]list(X_train_list[0]):', list(X_train_list[0]))
    
    if mode == "train":
        img_dir_ls = list(X_train_list[fold_id]) if fold_id is not None else None
        # print('[audio_visual_data_udiva.py]train mode, len(img_dir_ls)=', len(img_dir_ls), ', img_dir_ls[:3]=', img_dir_ls[:3]) if img_dir_ls is not None else print('[audio_visual_data_udiva.py] train mode, img_dir_ls is None')
        dataset = AudioVisualLstmDataUdiva(
            cfg.DATA.ROOT,
            img_data,
            aud_data,
            cfg.DATA.NOXI_LABEL_DATA,
            transforms,
            sample_size=cfg.DATA.SAMPLE_SIZE,
            config=cfg,
            mode=mode,
            dataset_name="NoXi",
            prefix1="Expert",
            prefix2="Novice",
            img_dir_ls = img_dir_ls
        )
        batch_size = cfg.DATA_LOADER.TRAIN_BATCH_SIZE
        shuffle=cfg.DATA_LOADER.SHUFFLE
        
        # 过采样
        if cfg.DATA.SAMPLING_NAME != "":
            dataset = over_sampling(dataset, cfg.DATA.SAMPLING_NAME, cfg.TRAIN.BIMODAL_OPTION)
            if cfg.TRAIN.BIMODAL_OPTION == 1 or cfg.TRAIN.BIMODAL_OPTION == 2:
                image_or_audio, label = dataset[0]
                print('[oversampling]- final return dataset: len(dataset)=', len(dataset), ', image or audio, label = dataset[0], image or audio.shape=', image_or_audio.shape, ', label=', label)
                # image.shape= torch.Size([sample_size, 6, 224, 224]) , label= tensor([1., 0.])
            else:
                image, audio, label = dataset[0]
                print('[oversampling]- final return dataset: len(dataset)=', len(dataset), ', image, audio, label = dataset[0], image.shape=', image.shape, ', audio.shape=', audio.shape, ', label=', label)
    elif mode == "valid":
        img_dir_ls = list(X_valid_list[fold_id]) if fold_id is not None else None
        # print('[audio_visual_data_udiva.py]valid mode, len(img_dir_ls)=', len(img_dir_ls), ', img_dir_ls[:3]=', img_dir_ls[:3]) if img_dir_ls is not None else print('[audio_visual_data_udiva.py] valid mode, img_dir_ls is None')
        dataset = AudioVisualLstmDataUdiva(
            cfg.DATA.ROOT,
            img_data,
            aud_data,
            cfg.DATA.NOXI_LABEL_DATA,
            transforms,
            sample_size=cfg.DATA.SAMPLE_SIZE,
            config=cfg,
            mode=mode,
            dataset_name="NoXi",
            prefix1="Expert",
            prefix2="Novice",
            img_dir_ls = img_dir_ls
        )
        batch_size = cfg.DATA_LOADER.VALID_BATCH_SIZE
        shuffle=False
    elif mode == "test":
        img_dir_ls = list(X_valid_list[fold_id]) if fold_id is not None else None
        # print('[audio_visual_data_udiva.py]test mode, len(img_dir_ls)=', len(img_dir_ls), ', img_dir_ls[:3]=', img_dir_ls[:3]) if img_dir_ls is not None else print('[audio_visual_data_udiva.py] valid mode, img_dir_ls is None')
        dataset = AudioVisualLstmDataUdiva(
            cfg.DATA.ROOT,
            img_data,
            aud_data,
            cfg.DATA.NOXI_LABEL_DATA,
            transforms,
            sample_size=cfg.DATA.SAMPLE_SIZE,
            config=cfg,
            mode=mode,
            dataset_name="NoXi",
            prefix1="Expert",
            prefix2="Novice",
            img_dir_ls = img_dir_ls
        )
        batch_size = cfg.DATA_LOADER.TEST_BATCH_SIZE
        shuffle=False
    elif mode == "full_test":
        test_img_data = img_data
        test_aud_data = aud_data
        return ALLSampleAudioVisualDataUdiva(
            cfg.DATA.ROOT,
            test_img_data,
            test_aud_data,
            cfg.DATA.NOXI_LABEL_DATA,
            transforms,
        )
    else:
        dataset = noxi_full_dataset # 使用全量数据集，因为NoXi数据集后续会进行K折交叉验证来训练，所以这里不需要再进行划分，具体划分逻辑由交叉验证完成
        batch_size = cfg.DATA_LOADER.TEST_BATCH_SIZE
        shuffle=False
    # print('[audio_visual_data_udiva.py] batch_size: ', batch_size, ' mode: ', mode)
    data_loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,  # cfg.NUM_WORKS
        drop_last=cfg.DATA_LOADER.DROP_LAST,
        prefetch_factor=cfg.DATA_LOADER.PREFETCH_FACTOR,
    )
    return data_loader


@DATA_LOADER_REGISTRY.register()
def bimodal_resnet_lstm_data_loader_udiva(cfg, mode): # 基于AudioVisualDataUdiva 增加针对LSTM的数据处理
    assert (mode in ["train", "valid", "test", "full_test"]), " 'mode' only supports 'train' 'valid' 'test' "
    transforms = build_transform_spatial(cfg) # TRANSFORM: "standard_frame_transform"
    if mode == "train":
        # 如果 cfg.DATA.SESSION 字符串里包含了 'ANIMALS' 关键词，那么就在list里添加cfg.DATA.ANIMALS_TRAIN_IMG_DATA, 否则就不添加 # 如果 cfg.DATA.SESSION 字符串里包含了 'GHOST' 关键词，那么就在list里添加cfg.DATA.GHOST_TRAIN_IMG_DATA, 否则就不添加 # 如果 cfg.DATA.SESSION 字符串里包含了 'LEGO' 关键词，那么就在list里添加cfg.DATA.LEGO_TRAIN_IMG_DATA, 否则就不添加 # 如果 cfg.DATA.SESSION 字符串里包含了 'TALK' 关键词，那么就在list里添加cfg.DATA.TALK_TRAIN_IMG_DATA, 否则就不添加
        train_img_data = [
            cfg.DATA.ANIMALS_TRAIN_IMG_DATA if 'ANIMALS' in cfg.DATA.SESSION else None,
            cfg.DATA.GHOST_TRAIN_IMG_DATA if 'GHOST' in cfg.DATA.SESSION else None,
            cfg.DATA.LEGO_TRAIN_IMG_DATA if 'LEGO' in cfg.DATA.SESSION else None,
            cfg.DATA.TALK_TRAIN_IMG_DATA if 'TALK' in cfg.DATA.SESSION else None,]
        train_aud_data = [
            cfg.DATA.ANIMALS_TRAIN_AUD_DATA if 'ANIMALS' in cfg.DATA.SESSION else None,
            cfg.DATA.GHOST_TRAIN_AUD_DATA if 'GHOST' in cfg.DATA.SESSION else None,
            cfg.DATA.LEGO_TRAIN_AUD_DATA if 'LEGO' in cfg.DATA.SESSION else None,
            cfg.DATA.TALK_TRAIN_AUD_DATA if 'TALK' in cfg.DATA.SESSION else None,]
        train_img_data = [x for x in train_img_data if x is not None] # 去掉list里的None元素
        train_aud_data = [x for x in train_aud_data if x is not None]
        # print('[audio_visual_data_udiva.py]- train_img_data = ', train_img_data, ', train_aud_data = ', train_aud_data, ', len(train_img_data)=', len(train_img_data), ' len(train_aud_data)=', len(train_aud_data))
        dataset = AudioVisualLstmDataUdiva(
            cfg.DATA.ROOT,
            train_img_data, 
            train_aud_data, 
            cfg.DATA.TRAIN_LABEL_DATA,
            transforms,
            sample_size=cfg.DATA.SAMPLE_SIZE,
            config=cfg,
            mode=mode
        )
        
        batch_size = cfg.DATA_LOADER.TRAIN_BATCH_SIZE
        shuffle=cfg.DATA_LOADER.SHUFFLE
        
        # 过采样
        if cfg.DATA.SAMPLING_NAME != "":
            dataset = over_sampling(dataset, cfg.DATA.SAMPLING_NAME, cfg.TRAIN.BIMODAL_OPTION)
            if cfg.TRAIN.BIMODAL_OPTION == 1 or cfg.TRAIN.BIMODAL_OPTION == 2:
                image_or_audio, label = dataset[0]
                print('[oversampling]- final return dataset: len(dataset)=', len(dataset), ', image or audio, label = dataset[0], image or audio.shape=', image_or_audio.shape, ', label=', label)
                # image.shape= torch.Size([sample_size, 6, 224, 224]) , label= tensor([1., 0.])
            else:
                image, audio, label = dataset[0]
                print('[oversampling]- final return dataset: len(dataset)=', len(dataset), ', image, audio, label = dataset[0], image.shape=', image.shape, ', audio.shape=', audio.shape, ', label=', label)
    elif mode == "valid":
        # print('[audio_visual_data_udiva.py]- bimodal_resnet_lstm_data_loader_udiva 开始进行验证集的dataloader初始化...')
        val_img_data = [
            cfg.DATA.ANIMALS_VAL_IMG_DATA if 'ANIMALS' in cfg.DATA.SESSION else None,
            cfg.DATA.GHOST_VAL_IMG_DATA if 'GHOST' in cfg.DATA.SESSION else None,
            cfg.DATA.LEGO_VAL_IMG_DATA if 'LEGO' in cfg.DATA.SESSION else None,
            cfg.DATA.TALK_VAL_IMG_DATA if 'TALK' in cfg.DATA.SESSION else None,
        ]
        val_aud_data = [
            cfg.DATA.ANIMALS_VAL_AUD_DATA if 'ANIMALS' in cfg.DATA.SESSION else None,
            cfg.DATA.GHOST_VAL_AUD_DATA if 'GHOST' in cfg.DATA.SESSION else None,
            cfg.DATA.LEGO_VAL_AUD_DATA if 'LEGO' in cfg.DATA.SESSION else None,
            cfg.DATA.TALK_VAL_AUD_DATA if 'TALK' in cfg.DATA.SESSION else None,
        ]
        val_img_data = [x for x in val_img_data if x is not None]
        val_aud_data = [x for x in val_aud_data if x is not None]
        
        dataset = AudioVisualLstmDataUdiva(
            cfg.DATA.ROOT,
            val_img_data,
            val_aud_data,
            cfg.DATA.VALID_LABEL_DATA, # 'VALID_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_validation.pkl',
            transforms,
            sample_size=cfg.DATA.SAMPLE_SIZE,
            config=cfg,
            mode=mode,
        )
        batch_size = cfg.DATA_LOADER.VALID_BATCH_SIZE
        shuffle=False
    elif mode == "full_test":
        # print('[audio_visual_data_udiva.py]- bimodal_resnet_lstm_data_loader_udiva 开始进行测试集的dataloader初始化...')
        test_img_data = [
            cfg.DATA.ANIMALS_TEST_IMG_DATA if 'ANIMALS' in cfg.DATA.SESSION else None,
            cfg.DATA.GHOST_TEST_IMG_DATA if 'GHOST' in cfg.DATA.SESSION else None,
            cfg.DATA.LEGO_TEST_IMG_DATA if 'LEGO' in cfg.DATA.SESSION else None,
            cfg.DATA.TALK_TEST_IMG_DATA if 'TALK' in cfg.DATA.SESSION else None,
        ]
        test_aud_data = [
            cfg.DATA.ANIMALS_TEST_AUD_DATA if 'ANIMALS' in cfg.DATA.SESSION else None,
            cfg.DATA.GHOST_TEST_AUD_DATA if 'GHOST' in cfg.DATA.SESSION else None,
            cfg.DATA.LEGO_TEST_AUD_DATA if 'LEGO' in cfg.DATA.SESSION else None,
            cfg.DATA.TALK_TEST_AUD_DATA if 'TALK' in cfg.DATA.SESSION else None,
        ]
        test_img_data = [x for x in test_img_data if x is not None]
        test_aud_data = [x for x in test_aud_data if x is not None]
        
        return ALLSampleAudioVisualDataUdiva(
            cfg.DATA.ROOT,
            test_img_data,
            test_aud_data,
            cfg.DATA.TEST_LABEL_DATA, # 'TEST_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_test.pkl',
            transforms,
        )
    else:
        test_img_data = [
            cfg.DATA.ANIMALS_TEST_IMG_DATA if 'ANIMALS' in cfg.DATA.SESSION else None,
            cfg.DATA.GHOST_TEST_IMG_DATA if 'GHOST' in cfg.DATA.SESSION else None,
            cfg.DATA.LEGO_TEST_IMG_DATA if 'LEGO' in cfg.DATA.SESSION else None,
            cfg.DATA.TALK_TEST_IMG_DATA if 'TALK' in cfg.DATA.SESSION else None,
        ]
        test_aud_data = [
            cfg.DATA.ANIMALS_TEST_AUD_DATA if 'ANIMALS' in cfg.DATA.SESSION else None,
            cfg.DATA.GHOST_TEST_AUD_DATA if 'GHOST' in cfg.DATA.SESSION else None,
            cfg.DATA.LEGO_TEST_AUD_DATA if 'LEGO' in cfg.DATA.SESSION else None,
            cfg.DATA.TALK_TEST_AUD_DATA if 'TALK' in cfg.DATA.SESSION else None,
        ]
        test_img_data = [x for x in test_img_data if x is not None]
        test_aud_data = [x for x in test_aud_data if x is not None]
        
        dataset = AudioVisualLstmDataUdiva(
            cfg.DATA.ROOT,
            test_img_data,
            test_aud_data,
            cfg.DATA.TEST_LABEL_DATA,
            transforms,
            sample_size=cfg.DATA.SAMPLE_SIZE,
            config=cfg,
            mode=mode,
        )
        batch_size = cfg.DATA_LOADER.TEST_BATCH_SIZE
        shuffle=False
    # print('[audio_visual_data_udiva.py] batch_size: ', batch_size, ' mode: ', mode)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,  # cfg.NUM_WORKS
        drop_last=cfg.DATA_LOADER.DROP_LAST,
        prefetch_factor=cfg.DATA_LOADER.PREFETCH_FACTOR,
    )
    return data_loader


def over_sampling(dataset, sampler_name, bimodal_option): 
    #****************** 以下是为了让数据集更均衡，使用 oversampling 过采样方法来让数据集更均衡 ******************
    # choose sampler
    if sampler_name == 'RandomOverSampler':
        print('[oversampling]- Using RandomOverSampler to oversampling')
        sampler = RandomOverSampler(random_state=0)
    elif sampler_name == 'SMOTE':
        print('[oversampling]- Using SMOTE to oversampling')
        sampler = SMOTE(random_state=42)
    elif sampler_name == 'BorderlineSMOTE':
        print('[oversampling]- Using BorderlineSMOTE to oversampling')
        sampler = BorderlineSMOTE(random_state=42)
    else:
        print('[oversampling]- Using RandomOverSampler to oversampling')
        sampler = RandomOverSampler(random_state=0)
    
    # 由于udiva 训练集所有的116个session中，认识的label有44个，不认识的label有72个(有一个坏数据，所以减去1为71个)，所以使用 oversampling 过采样方法来让数据集更均衡, 借助imblearn.over_sampling.RandomOverSampler
    # 查看均衡前的数据分布
    # print('[oversampling]- len(dataset)=', len(dataset), ', type(dataset)=', type(dataset), ', dataset[0].keys()=', dataset[0].keys(), ', type(dataset[0])=', type(dataset[0])) 
    # len(dataset)= 115 , type(dataset)= <class 'dpcv.data.datasets.audio_visual_data_udiva.AudioVisualLstmDataUdiva'> , dataset[0].keys()= dict_keys(['image', 'label']) , type(dataset[0])= <class 'dict'>
    # print('[oversampling]- image, label shape=', dataset[0]['image'].shape, dataset[0]['label'].shape) 
    # image shape= torch.Size([sample_size, channel:6, w:224, h:224]) label shape: torch.Size([2]) label的shape固定是[2]，因为是二分类问题，所以只有两个元素，分别是认识和不认识的概率
    
    if bimodal_option == 1 or bimodal_option == 2:
        #****************** 1. 构造 X_train, y_train 作为fit_resample的输入参数********************
        X_train, y_train, X_session, X_segment = [], [], [], []
        for i in range(len(dataset)):
            if bimodal_option == 1: # only image modality
                X_train.append(dataset[i]['image'])
            else: # only audio modality
                X_train.append(dataset[i]['audio'])
            y_train.append(dataset[i]['label'])
        # print('[oversampling]- len(X_train)=', len(X_train), ', len(y_train)=', len(y_train), ', X_train[0].shape=', X_train[0].shape, ', y_train[0].shape=', y_train[0].shape)
        # len(X_train)= 115 , len(y_train)= 115 , X_train[0].shape= torch.Size([sample_size, 6, 224, 224]) , y_train[0].shape= torch.Size([2])
        
        X_train = torch.stack(X_train, dim=0)
        y_train = torch.stack(y_train, dim=0)
        # print('[oversampling]- after stack: len(X_train)=', len(X_train), ', len(y_train)=', len(y_train), 'X_train.shape=', X_train.shape, ', y_train.shape=', y_train.shape, ', X_train[0].shape=', X_train[0].shape, ', y_train[0].shape=', y_train[0].shape)
        if bimodal_option == 1: # only image modality
            sample_size = X_train.shape[1]
        else: # only audio modality
            audio_last_shape = X_train.shape[-1]
        # print('[oversampling]- sample_size=', sample_size)
        # after stack: len(X_train)= 115 , len(y_train)= 115, X_train.shape= torch.Size([115, sample_size, 6, 224, 224]) , y_train.shape= torch.Size([115, 2]) , X_train[0].shape= torch.Size([4, 6, 224, 224]) , y_train[0].shape= torch.Size([2])
        
        # fit_resample的输入参数 第一个参数X_tarin需要是 (n_samples, n_features), 第二个参数y_train需要是 (n_samples,)，所以需要对X_train和y_train进行reshape
        X_train = X_train.reshape(X_train.shape[0], -1) # 将X_train转换为shape为[115, sample_size*6*224*224]的二维数组，方便后续的过采样
        y_train = y_train.argmax(dim=1) # 将y_train转换为shape为[115, ]的一维数组，方便后续的过采样
        # print('[oversampling]- after reshape: X_train.shape=', X_train.shape, ', y_train.shape=', y_train.shape, ', type(X_train)=', type(X_train), ', type(y_train)=', type(y_train))
        # after reshape: X_train.shape= torch.Size([115, 1204224]) , y_train.shape= torch.Size([115]) , type(X_train)= <class 'torch.Tensor'> , type(y_train)= <class 'torch.Tensor'>
        # print('[oversampling]- ******** Counter(y_train)=', sorted(Counter(y_train.numpy()).items())) # 使用Counter查看均衡前的label类别数据分布 # 统计y_train中各类label的数量
        # Counter(y_train)= [(0, 44), (1, 71)] 即认识的label有44个，不认识的label有71个
        
        #****************** 2. 核心：执行过采样 ******************
        # print('[oversampling]- sampler: ', sampler)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        # print('[oversampling]- after oversampling: X_resampled.shape=', X_resampled.shape, ', y_resampled.shape=', y_resampled.shape, ', type(X_resampled)=', type(X_resampled), ', type(y_resampled)=', type(y_resampled))
        # after oversampling: X_resampled.shape= (142, 1204224) , y_resampled.shape= (142,) , type(X_resampled)= <class 'numpy.ndarray'> , type(y_resampled)= <class 'numpy.ndarray'>
        print('[oversampling]- ******** Before oversampling: Counter(y_train)=', sorted(Counter(y_train.numpy()).items()), '; After  oversampling: Counter(y_resampled)=', sorted(Counter(y_resampled).items())) # 使用Counter查看均衡后的label类别数据分布
        # Counter(y_resampled)= [(0, 71), (1, 71)] 即认识的label有71个，不认识的label也有71个
        # print('[oversampling]- y_resampled=', y_resampled, ', type(y_resampled)=', type(y_resampled)) # type(y_resampled)= <class 'numpy.ndarray'>
        
        # 打印 X_resampled中的所有len(X_resampled)个元素，但是每个元素只打印前4个tensor数值, 且都把tensor中的元素保留前4位小数
        # a = X_train[:, :4]
        # b = X_resampled[:, :4]
        # b = torch.from_numpy(np.round(X_resampled[:, :4], decimals=4))
        # print('[oversampling]- diff X_train[:, :4]=\n', a, '\n X_resampled[:, :4]=\n', b, ', a.shape=', a.shape, ', b.shape=', b.shape)
        # 经过print可以看到，X_resampled比X_train多的部分，是通过复制X_train中的元素得到的，但并不只是复制同一个。例如增加了10个元素，这10个元素虽然可能有3个元素是一样的，但是也有7个元素是不一样的 即多的部分不是完全重复的，符合预期
        
        if bimodal_option == 1: # image
            X_resampled = torch.from_numpy(X_resampled.reshape(-1, sample_size, 6, 224, 224))
        else: # audio
            X_resampled = torch.from_numpy(X_resampled.reshape(-1, 2, 1, audio_last_shape))
        y_resampled = F.one_hot(torch.tensor(y_resampled), num_classes=2).float() # 将y_resampled恢复为维度为[num_samples, 2]的tensor
        
        # print('[oversampling]- X_resampled.shape=', X_resampled.shape, ', type(X_resampled)=', type(X_resampled))
        # print('[oversampling]- after one_hot: y_resampled.shape=', y_resampled.shape, ', type(y_resampled)=', type(y_resampled), ', y_resampled=', y_resampled)
        # after torch.tensor: y_resampled.shape= torch.Size([142, 2]) , type(y_resampled)= <class 'torch.Tensor'> , y_resampled= tensor([[1., 0.], ... , [0., 1.]])
        #****************** 3. 构造dataset 用于传给dataloader ******************
        dataset = TensorDataset(X_resampled, y_resampled) # refer: https://stackoverflow.com/questions/67683406/difference-between-dataset-and-tensordataset-in-pytorch # torch TensorDataset 可以接受一个或多个参数，每个参数都是一个张量（Tensor），并将它们打包成一个数据集（Dataset）。dataset = TensorDataset(tensor1, tensor2, tensor3, ...) 张量的第一个维度大小应该相同，以确保它们能够正确地打包成数据集。 tensors that have the same size of the first dimension.
        # dataset 是一个TensorDataset对象，一共有142个元素，每个元素是一个tuple，tuple的第一个元素是一个shape为[sample_size, 6, 224, 224]的image tensor，第二个元素是一个shape为[2]的label
    elif bimodal_option == 3: # bimodal_option=3, 表示使用image和audio数据
        #****************** 1. 构造 X_train, y_train 作为fit_resample的输入参数********************
        X_train_img, X_train_audio, y_train = [], [], []
        for i in range(len(dataset)):
            X_train_img.append(dataset[i]['image'])
            X_train_audio.append(dataset[i]['audio'])
            y_train.append(dataset[i]['label'])
        # print('1-[oversampling]- len(X_train_img)=', len(X_train_img), ', len(y_train)=', len(y_train), ', X_train_img[0].shape=', X_train_img[0].shape, ', y_train[0].shape=', y_train[0].shape)
        # print('1-[oversampling]- len(X_train_audio)=', len(X_train_audio), ', X_train_audio[0].shape=', X_train_audio[0].shape)
        # len(X_train_img)= 115 , len(y_train)= 115 , X_train_img[0].shape= torch.Size([sample_size, 6, 224, 224]) , y_train[0].shape= torch.Size([2])
        
        X_train_img = torch.stack(X_train_img, dim=0)
        X_train_audio = torch.stack(X_train_audio, dim=0)
        y_train = torch.stack(y_train, dim=0)
        # print('2-[oversampling]- after stack: len(X_train_img)=', len(X_train_img), ', len(y_train)=', len(y_train), 'X_train_img.shape=', X_train_img.shape, ', y_train.shape=', y_train.shape, ', X_train_img[0].shape=', X_train_img[0].shape, ', y_train[0].shape=', y_train[0].shape)
        # print('2-[oversampling]- after stack: len(X_train_audio)=', len(X_train_audio), 'X_train_audio.shape=', X_train_audio.shape, ', X_train_audio[0].shape=', X_train_audio[0].shape)
        img_sample_size = X_train_img.shape[1]
        audio_last_shape = X_train_audio.shape[-1]
        # after stack: len(X_train_img)= 115 , len(y_train)= 115, X_train_img.shape= torch.Size([115, sample_size, 6, 224, 224]) , y_train.shape= torch.Size([115, 2]) , X_train_img[0].shape= torch.Size([4, 6, 224, 224]) , y_train[0].shape= torch.Size([2])
        
        # fit_resample的输入参数 第一个参数X_tarin需要是 (n_samples, n_features), 第二个参数y_train需要是 (n_samples,)，所以需要对X_train和y_train进行reshape
        X_train_img = X_train_img.reshape(X_train_img.shape[0], -1) # 将X_train_img转换为shape为[115, sample_size*6*224*224]的二维数组，方便后续的过采样
        X_train_audio = X_train_audio.reshape(X_train_audio.shape[0], -1) # 将X_train_audio转换为shape为[115, 2*1*sample_size*16000]的二维数组，方便后续的过采样
        y_train = y_train.argmax(dim=1) # 将y_train转换为shape为[115, ]的一维数组，方便后续的过采样
        # print('3-[oversampling]- after reshape: X_train_img.shape=', X_train_img.shape, ', y_train.shape=', y_train.shape, ', type(X_train_img)=', type(X_train_img), ', type(y_train)=', type(y_train))
        # print('3-[oversampling]- after reshape: X_train_audio.shape=', X_train_audio.shape, ', y_train.shape=', y_train.shape, ', type(X_train_audio)=', type(X_train_audio))
        # after reshape: X_train_img.shape= torch.Size([115, 1204224]) , y_train.shape= torch.Size([115]) , type(X_train_img)= <class 'torch.Tensor'> , type(y_train)= <class 'torch.Tensor'>
        # print('4-[oversampling]- ******** Counter(y_train)=', sorted(Counter(y_train.numpy()).items())) # 使用Counter查看均衡前的label类别数据分布 # 统计y_train中各类label的数量
        # Counter(y_train)= [(0, 44), (1, 71)] 即认识的label有44个，不认识的label有71个
        
        #****************** 2. 核心：执行过采样 ******************
        X_img_resampled, y_resampled = sampler.fit_resample(X_train_img, y_train) 
        X_audio_resampled, y_resampled = sampler.fit_resample(X_train_audio, y_train)
        # print('5-[oversampling]- after oversampling: X_img_resampled.shape=', X_img_resampled.shape, ', X_audio_resampled.shape=', X_audio_resampled.shape, ', y_resampled.shape=', y_resampled.shape, ', type(X_img_resampled)=', type(X_img_resampled), ', type(y_resampled)=', type(y_resampled))
        # after oversampling: X_img_resampled.shape= (142, 1204224) , y_resampled.shape= (142,) , type(X_img_resampled)= <class 'numpy.ndarray'> , type(y_resampled)= <class 'numpy.ndarray'>
        print('[oversampling]- ******** Before oversampling: Counter(y_train)=', sorted(Counter(y_train.numpy()).items()), '; After  oversampling: Counter(y_resampled)=', sorted(Counter(y_resampled).items())) # 使用Counter查看均衡后的label类别数据分布
        # Counter(y_resampled)= [(0, 71), (1, 71)] 即认识的label有71个，不认识的label也有71个
        # print('7-[oversampling]- y_resampled=', y_resampled, ', type(y_resampled)=', type(y_resampled)) # type(y_resampled)= <class 'numpy.ndarray'>

        X_img_resampled = torch.from_numpy(X_img_resampled.reshape(-1, img_sample_size, 6, 224, 224))
        X_audio_resampled = torch.from_numpy(X_audio_resampled.reshape(-1, 2, 1, audio_last_shape))
        y_resampled = F.one_hot(torch.tensor(y_resampled), num_classes=2).float() # 将y_resampled恢复为维度为[num_samples, 2]的tensor
        
        # print('8-[oversampling]- after torch.from_numpy: X_img_resampled.shape=', X_img_resampled.shape, ', X_audio_resampled.shape=', X_audio_resampled.shape)
        # print('8-[oversampling]- after one_hot: y_resampled.shape=', y_resampled.shape, ', type(y_resampled)=', type(y_resampled), ', y_resampled=', y_resampled)
        # after torch.tensor: y_resampled.shape= torch.Size([142, 2]) , type(y_resampled)= <class 'torch.Tensor'> , y_resampled= tensor([[1., 0.], ... , [0., 1.]])
        
        #****************** 3. 构造dataset 用于传给dataloader ******************
        dataset = TensorDataset(X_img_resampled, X_audio_resampled, y_resampled) # refer: https://stackoverflow.com/questions/67683406/difference-between-dataset-and-tensordataset-in-pytorch # torch TensorDataset 可以接受一个或多个参数，每个参数都是一个张量（Tensor），并将它们打包成一个数据集（Dataset）。dataset = TensorDataset(tensor1, tensor2, tensor3, ...) 张量的第一个维度大小应该相同，以确保它们能够正确地打包成数据集。 tensors that have the same size of the first dimension.
        # dataset 是一个TensorDataset对象，一共有142个元素，每个元素是一个tuple，tuple的第一个元素是一个shape为[sample_size, 6, 224, 224]的image tensor，第二个元素是shape为[2, 1, sample_size*16000]的audio tensor，第3个元素是一个shape为[2]的label
    else:
        raise ValueError('bimodal_option must be 1, 2 or 3')
    return dataset



if __name__ == "__main__":
    # from tqdm import tqdm
    # args = ("../../../datasets", "ImageData/trainingData", "VoiceData/trainingData_50176", "annotation_training.pkl")
    trans = set_audio_visual_transform()
    # data_set = AudioVisualDataUdiva(*args, trans)
    # # print(len(data_set))
    # data = data_set[1]
    # print(data["image"].shape, data["audio"].shape, data["label"].shape)

    dataset = AudioVisualDataUdiva(
        "../../../datasets",
        ["image_data/training_data_01", "image_data/validation_data_01"],
        ["voice_data/trainingData", "voice_data/validationData"],
        ["annotation/annotation_training.pkl", "annotation/annotation_validation.pkl"],
        trans,
    )
    print(len(dataset))
    a = dataset[1]
    print(a)

"""
[oversampling]- diff X_train[:, :4]=
 tensor([[ 0.7591,  0.7762,  0.7762,  0.7933],
        [-2.0665, -2.0665, -2.0665, -2.0494],
        [ 1.1015,  1.1015,  1.0673,  1.0502],
        [-0.4739, -0.4739, -0.4568, -0.4568],
        [ 0.2967,  0.6734,  1.0502,  1.4269],
        [-1.1760, -1.1760, -1.1760, -1.1932],
        [-0.8849, -0.6452, -0.3883, -0.1486],
        [ 2.0605,  1.8550,  1.5982,  1.3070],
        [-1.6555, -1.7069, -1.6384, -1.5870],
        [ 0.6392,  0.6392,  0.6392,  0.6392],
        [-0.3369, -0.2342, -0.1486, -0.0629],
        [-1.8439, -1.8439, -1.8268, -1.8097],
        [-0.9020, -0.5938, -0.3027, -0.0116],
        [-1.1075, -1.0733, -1.0390, -1.0048],
        [-0.9363, -0.9363, -0.9363, -0.9020],
        [ 0.8447,  0.8789,  0.9132,  0.9474],
        [-0.9020, -0.7479, -0.5767, -0.4054],
        [ 0.7248,  0.7248,  0.7248,  0.7419],
        [ 1.1529,  1.2043,  1.2385,  1.2899],
        [-2.0665, -2.0665, -2.0665, -2.0665],
        [-1.3473, -1.3473, -1.3473, -1.3302]])
 X_resampled[:, :4]=
 tensor([[ 0.7591,  0.7762,  0.7762,  0.7933],
        [-2.0665, -2.0665, -2.0665, -2.0494],
        [ 1.1015,  1.1015,  1.0673,  1.0502],
        [-0.4739, -0.4739, -0.4568, -0.4568],
        [ 0.2967,  0.6734,  1.0502,  1.4269],
        [-1.1760, -1.1760, -1.1760, -1.1932],
        [-0.8849, -0.6452, -0.3883, -0.1486],
        [ 2.0605,  1.8550,  1.5982,  1.3070],
        [-1.6555, -1.7069, -1.6384, -1.5870],
        [ 0.6392,  0.6392,  0.6392,  0.6392],
        [-0.3369, -0.2342, -0.1486, -0.0629],
        [-1.8439, -1.8439, -1.8268, -1.8097],
        [-0.9020, -0.5938, -0.3027, -0.0116],
        [-1.1075, -1.0733, -1.0390, -1.0048],
        [-0.9363, -0.9363, -0.9363, -0.9020],
        [ 0.8447,  0.8789,  0.9132,  0.9474],
        [-0.9020, -0.7479, -0.5767, -0.4054],
        [ 0.7248,  0.7248,  0.7248,  0.7419],
        [ 1.1529,  1.2043,  1.2385,  1.2899],
        [-2.0665, -2.0665, -2.0665, -2.0665],
        [-1.3473, -1.3473, -1.3473, -1.3302],
        [-0.3256, -0.2978, -0.2793, -0.2515],
        [-1.2792, -1.1195, -0.9483, -0.7829],
        [-1.8194, -1.8047, -1.7925, -1.7656],
        [-0.9029, -0.7468, -0.5796, -0.4116],
        [-0.2529, -0.1530, -0.0703,  0.0134],
        [ 0.5721,  0.6092,  0.6339,  0.6806],
        [-0.9331, -0.9184, -0.9026, -0.8557]]) , a.shape= torch.Size([21, 4]) , b.shape= torch.Size([28, 4])

"""