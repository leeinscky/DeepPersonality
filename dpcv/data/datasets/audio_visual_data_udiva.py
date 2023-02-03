import os
from matplotlib import test
import torch
import glob
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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

class AudioVisualDataUdiva(VideoDataUdiva):

    def __init__(self, data_root, img_dir, audio_dir, label_file, transform=None, sample_size=100):
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- class AudioVisualDataUdiva(VideoDataUdiva) 开始执行 __init__')
        super().__init__(data_root, img_dir, label_file, audio_dir)
        self.transform = transform
        self.sample_size = sample_size
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- class AudioVisualDataUdiva(VideoDataUdiva) 结束执行 __init__')

    def __getitem__(self, idx): # idx means the index of video in the video directory
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py] - class AudioVisualDataUdiva(VideoDataUdiva) 开始执行 __getitem__ , idx = ', idx) # idx 和 get_ocean_label(self, index) 里的index含义一样，表示video目录里的第几个video样本
        label = self.get_ocean_label(idx)  # label是True或者False, 代表关系Known是True或者False
        
        # img = self.get_image_data(idx) # img是一个PIL.Image.Image对象, 代表该video的一帧图像, 例如：PIL.Image.Image object, mode=RGB, size=224x224, 代表该video的一帧图像的大小是224x224, 且是RGB三通道的, 也就是说该video的一帧图像是224x224x3的
        fc1_img, fc2_img = self.get_image_data(idx)
        
        # wav = self.get_wave_data(idx) # wav是一个numpy.ndarray对象, 代表该video的一帧音频, 例如：array([[[ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ]]], dtype=float32), 代表该video的一帧音频是50176维的, 也就是说该video的一帧音频是50176x1x1的, 且是float32类型的, 且是三维的
        fc1_wav, fc2_wav  = self.get_wave_data(idx) # fc1_wav是一个numpy.ndarray对象, 代表该video的一帧音频, 例如：array([[[ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ]]], dtype=float32), 代表该video的一帧音频是50176维的, 也就是说该video的一帧音频是50176x1x1的, 且是float32类型的, 且是三维的

        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ - transform之前 type(img)=', type(img), 'img=', img) # type(img)= <class 'PIL.Image.Image'> img= <PIL.Image.Image image mode=RGB size=256x256 at 0x7FC8AA03EB20>
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
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ - torch.as_tensor之前 type(wav)=', type(wav), 'wav=', wav) # type(wav)= <class 'numpy.ndarray'> wav= [[[-1.0260781e-03 -2.3528279e-03 -2.2199661e-03 ...  9.4422154e-05 2.8776360e-04 -7.4460535e-05]]]
        # wav原始逻辑
        # wav = torch.as_tensor(wav, dtype=img.dtype) # torch.as_tensor - 将输入转换为张量, 并返回一个新的张量, 与输入共享内存, 但是不同的是, 如果输入是一个张量, 则返回的张量与输入不共享内存, 例如：tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        # wav修改后逻辑
        fc1_wav = torch.as_tensor(fc1_wav, dtype=img.dtype) # shape=(1,1,50176)
        fc2_wav = torch.as_tensor(fc1_wav, dtype=img.dtype) # shape=(1,1,50176)
        # 将两个tensor拼接在一起 concatenate the fc1_wav and fc2_wav, 拼接后的维度为(2,1,50176)，即将两个(1,1,50176)的tensor拼接在一起
        wav = torch.cat((fc1_wav, fc2_wav), 0) # shape=(2,1,50176) 
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ - torch.as_tensor之后 type(wav)=', type(wav), 'wav=', wav) # type(wav)= <class 'torch.Tensor'> wav= tensor([[[-1.0261e-03, -2.3528e-03, -2.2200e-03,  ...,  9.4422e-05, 2.8776e-04, -7.4461e-05]]])

        label = torch.as_tensor(label, dtype=img.dtype)
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ - torch.as_tensor之后 type(label)=', type(label), 'label=', label, ' label.shape=', label.shape) # type(label)= <class 'torch.Tensor'> label= tensor([0.5111, 0.4563, 0.4019, 0.3626, 0.4167])
        

        # 返回的sample原始逻辑（也适用于将2个视频的帧的tensor拼接后的逻辑）
        sample = {"image": img, "audio": wav, "label": label} # 非udiva的shape: img.shape()=torch.Size([3, 224, 224]) wav.shape()=torch.Size([1, 1, 50176]) label.shape()=torch.Size([5])  # udiva的shape: img.shape()= torch.Size([6, 224, 224]) wav.shape()= torch.Size([1, 2, 50176]) label.shape()= torch.Size([1])
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py] - class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ 函数返回的结果中 , img.shape()=', img.shape, 'wav.shape()=', wav.shape, 'label.shape()=', label.shape) # 
        # 因为__getitem__ 需要传入参数 idx，所以返回的sample也是一个视频对应的img，wav，label，至于为什么只有1帧image, 因为get_image_data函数只返回了1帧image，而没有返回多帧image
        
        # 返回的sample修改后逻辑
        # sample = {
        #     "fc1_image": fc1_img,
        #     "fc1_audio": fc1_wav,
        #     "fc2_image": fc2_img,
        #     "fc2_audio": fc2_wav,
        #     "label": label
        # }
        # print('[ deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py ] - class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ 函数返回的结果中 , fc1_img.shape() = ', fc1_img.shape, 'fc1_wav.shape() = ', fc1_wav.shape, 'fc2_img.shape() = ', fc2_img.shape, 'fc2_wav.shape() = ', fc2_wav.shape, 'label.shape() = ', label.shape) 
        
        return sample # sample是一个dict对象, 例如：{'image': tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],

    def get_image_data(self, idx):
        # print('[ deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py ] - get_image_data 函数开始执行')
        # get image data, return PIL.Image.Image object, for example: PIL.Image.Image object, mode=RGB, size=224x224, means the image size is 224x224, and is RGB three channels
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) 开始执行 get_image_data')
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
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data fc1_img_dir_path:', fc1_img_dir_path, "fc2_img_dir_path:", fc2_img_dir_path)
        # 打印结果: get_image_data fc1_img_dir_path: datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128/FC1_A     fc2_img_dir_path: datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128/FC2_A

        # get fc1 image
        fc1_img_paths = glob.glob(fc1_img_dir_path + "/*.jpg") # fc1_img_paths是FC1_A目录下所有的jpg图像文件路径的集合。 例如：train session id=055128, len(fc1_img_paths): 7228
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data fc1_img_paths:', fc1_img_paths, 'len(fc1_img_paths):', len(fc1_img_paths))
        #从所有的frame中按照，取出sample_size个frame，即随机取出sample_size个图片. 例如：一共有9293帧图片 随机取出sample_size = 100个图片. 那么公差=9293/100=92.93，即每隔92.93个图片取一个图片
        simple_fc1_frames = np.linspace(0, len(fc1_img_paths), self.sample_size, endpoint=False, dtype=np.int16) #从所有的frame中   np.linspace 返回等差数列，endpoint=False表示不包含最后一个数 例如：np.linspace(0, 10, 5, endpoint=False) 结果为：array([0., 2., 4., 6., 8.])，即不包含10，只包含0-8，共5个数，间隔为2，即等差数列
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data simple_fc1_frames:', simple_fc1_frames) # self.sample_size = 100
        selected_fc1 = random.choice(simple_fc1_frames)
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data selected_fc1:', selected_fc1)
        # show the selected fc1 image
        # show_selected_fc1 = mpimg.imread(fc1_img_paths[selected_fc1])
        # plt.imshow(show_selected_fc1) # show the image
        # plt.show()
        try:
            fc1_img = Image.open(fc1_img_paths[selected_fc1]).convert("RGB") # PIL.Image.open() 打开图片，返回一个Image对象，Image对象有很多方法，如：Image.show()，Image.save()，Image.convert()等，Image.convert()用于转换图片模式，如：RGB，L等，为了方便后续处理，这里转换为RGB模式，即3通道
        except:
            # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]exception: fc1_img_paths:', fc1_img_paths)
            return None
        
        # get fc2 image
        fc2_img_paths = glob.glob(fc2_img_dir_path + "/*.jpg")
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data fc2_img_paths:', fc2_img_paths)
        simple_fc2_frames = np.linspace(0, len(fc2_img_paths), self.sample_size, endpoint=False, dtype=np.int16) 
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data simple_fc2_frames:', simple_fc2_frames) # self.sample_size = 100
        # selected_fc2 = random.choice(simple_fc2_frames)
        selected_fc2 = selected_fc1 # 保证fc1和fc2的图片对应于同一个时刻
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data selected_fc2:', selected_fc2)
        # show the selected fc2 image
        # show_selected_fc2 = mpimg.imread(fc2_img_paths[selected_fc2])
        # plt.imshow(show_selected_fc2) # show the image
        # plt.show()
        try:
            fc2_img = Image.open(fc2_img_paths[selected_fc2]).convert("RGB")
        except:
            # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]exception: fc2_img_paths:', fc2_img_paths)
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
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data 函数开始执行')
        
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
                # print('[Ddeeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data函数 fc1_wav_path:', fc1_wav_path)
            # judge file is a file and start with FC2 and end with .wav.npy
            if os.path.isfile(os.path.join(wav_dir_path, file)) and file.startswith('FC2') and file.endswith('.wav.npy'):
                fc2_wav_path = os.path.join(wav_dir_path, file)
                # print('[Ddeeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data函数 fc2_wav_path:', fc2_wav_path)
        
        # process fc1 wave data
        fc1_wav_ft = np.load(fc1_wav_path) # fc1_wav_ft.shape: (1, 1, 4626530) len(fc1_wav_ft):  1
        # print('[Ddeeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data函数 fc1_wav_ft:', fc1_wav_ft, 'fc1_wav_ft.shape:', fc1_wav_ft.shape, '  len(fc1_wav_ft): ', len(fc1_wav_ft))
        try:
            n = np.random.randint(0, len(fc1_wav_ft) - 50176) # np.random.randint 生成一个随机整数，范围是[0, len(fc1_wav_ft) - 50176)
        except:
            # 看日志发现程序在这里执行了，说明len(fc1_wav_ft) - 50176 < 0
            n = 0
        fc1_wav_tmp = fc1_wav_ft[..., n: n + 50176] # 日志：fc1_wav_tmp.shape: (1, 1, 50176)  n: n + 50176 表示从n开始，取50176个采样点，每个采样点是1个声道，每个声道是1个采样点。
        # print('[Ddeeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data函数 fc1_wav_tmp:', fc1_wav_tmp, 'fc1_wav_tmp.shape:', fc1_wav_tmp.shape)
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

        return fc1_wav_tmp, fc2_wav_tmp
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


class AudioVisualLstmDataUdiva(VideoDataUdiva): # 基于AudioVisualDataUdiva 增加针对LSTM的数据处理
    def __init__(self, data_root, img_dir, audio_dir, label_file, transform=None, sample_size=48):
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- class AudioVisualDataUdiva(VideoDataUdiva) 开始执行 __init__')
        super().__init__(data_root, img_dir, label_file, audio_dir)
        self.transform = transform
        self.sample_size = sample_size # 表示从一个视频中采样sample_size个连续的帧图片
        self.frame_idx = 0
        wandb.config.sample_size = sample_size
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- class AudioVisualDataUdiva(VideoDataUdiva) 结束执行 __init__')

    def __getitem__(self, idx): # idx means the index of session in the directory
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py] - class AudioVisualDataUdiva(VideoDataUdiva) 开始执行 __getitem__ , idx = ', idx) # idx 和 get_ocean_label(self, index) 里的index含义一样，表示video目录里的第几个video样本
        label = self.get_ocean_label(idx)  # label是True或者False, 代表关系Known是True或者False
        fc1_img_tensor_list, fc2_img_tensor_list = self.get_image_data(idx)
        fc1_wav, fc2_wav  = self.get_wave_data(idx) # fc1_wav是一个numpy.ndarray对象, 代表该video的一帧音频, 例如：代表该video的一帧音频是50176维的, 也就是说该video的一帧音频是50176x1x1的, 且是float32类型的, 且是三维的
        img_tensor_list = []
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py] - class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ , len(fc1_img_tensor_list) = ', len(fc1_img_tensor_list), 'len(fc2_img_tensor_list) = ', len(fc2_img_tensor_list))
        for i in range(len(fc1_img_tensor_list)):
            fc1_img_tensor = fc1_img_tensor_list[i]
            fc2_img_tensor = fc2_img_tensor_list[i]
            img = torch.cat((fc1_img_tensor, fc2_img_tensor), 0) # concatenate the fc1_img_tensor and fc2_img_tensor
            # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py] - class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ , img.shape = ', img.shape) # img.shape =  torch.Size([6, 224, 224])
            img_tensor_list.append(img)
        img = torch.stack(img_tensor_list) # 将img_tensor_list中的tensor拼接在一起, 拼接后的维度为(16,6,224,224)，即将16个6x224x224的tensor拼接在一起
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py] - class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ , len(img_tensor_list) = ', len(img_tensor_list), ', img.shape = ', img.shape) # len(img_tensor_list) =  16 , img.shape =  torch.Size([16, 6, 224, 224])

        fc1_wav = torch.as_tensor(fc1_wav, dtype=img.dtype) # shape=(1,1,50176)
        fc2_wav = torch.as_tensor(fc1_wav, dtype=img.dtype) # shape=(1,1,50176)
        # 将两个tensor拼接在一起 concatenate the fc1_wav and fc2_wav, 拼接后的维度为(2,1,50176)，即将两个(1,1,50176)的tensor拼接在一起
        wav = torch.cat((fc1_wav, fc2_wav), 0) # shape=(2,1,50176) 
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ - torch.as_tensor之后 wav.shape=', wav.shape, ', fc1_wav.shape=', fc1_wav.shape, ', fc2_wav.shape=', fc2_wav.shape) # wav.shape= torch.Size([2, 1, 256000]) fc1_wav.shape= torch.Size([1, 1, 256000]) fc2_wav.shape= torch.Size([1, 1, 256000])

        label = torch.as_tensor(label, dtype=img.dtype)
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ - torch.as_tensor之后 type(label)=', type(label), 'label=', label, ' label.shape=', label.shape) # type(label)= <class 'torch.Tensor'> label= tensor([0.5111, 0.4563, 0.4019, 0.3626, 0.4167])

        # 返回的sample原始逻辑（也适用于将2个视频的帧的tensor拼接后的逻辑）
        sample = {"image": img, "audio": wav, "label": label} # 非udiva的shape: img.shape()=torch.Size([3, 224, 224]) wav.shape()=torch.Size([1, 1, 50176]) label.shape()=torch.Size([5])  # udiva的shape: img.shape()= torch.Size([6, 224, 224]) wav.shape()= torch.Size([1, 2, 50176]) label.shape()= torch.Size([1])
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py] - class AudioVisualDataUdiva(VideoDataUdiva) __getitem__ 函数返回的结果中 , len(img_tensor_list)=', len(img_tensor_list), 'wav.shape()=', wav.shape, 'label.shape()=', label.shape) # 
        # 因为__getitem__ 需要传入参数 idx，所以返回的sample也是一个session对应的img，wav，label
        
        return sample # sample是一个dict对象, 例如：{'image': tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],

    def get_image_data(self, idx):
        # print('[ deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py ] - get_image_data 函数开始执行')
        # get image data, return PIL.Image.Image object, for example: PIL.Image.Image object, mode=RGB, size=224x224, means the image size is 224x224, and is RGB three channels
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-class AudioVisualDataUdiva(VideoDataUdiva) 开始执行 get_image_data')
        img_dir_path = self.img_dir_ls[idx]
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- img_dir_path=', img_dir_path)
        # img_dir_path 是session的路径，例如 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128'
        # 对session路径下的FC1_A 文件夹和FC2_A文件夹分别进行提取帧
        fc1_img_dir_path, fc2_img_dir_path = '', ''
        for file in os.listdir(img_dir_path):
            # print('file:', file, 'type:', type(file))
            # judge file is a directory and start with FC1 and not end with .mp4
            if os.path.isdir(os.path.join(img_dir_path, file)) and file.startswith("FC1") and not file.endswith(".mp4"):
                fc1_img_dir_path = os.path.join(img_dir_path, file)
            # judge file is a directory and start with FC2 and not end with .mp4
            if os.path.isdir(os.path.join(img_dir_path, file)) and file.startswith("FC2") and not file.endswith(".mp4"):
                fc2_img_dir_path = os.path.join(img_dir_path, file)
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data idx:', idx, 'fc1_img_dir_path:', fc1_img_dir_path, "fc2_img_dir_path:", fc2_img_dir_path)
        # 打印结果: get_image_data fc1_img_dir_path: datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128/FC1_A     fc2_img_dir_path: datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128/FC2_A

        ########### get fc1 image - start ###########
        fc1_img_paths = glob.glob(fc1_img_dir_path + "/*.jpg") # fc1_img_paths是FC1_A目录下所有的jpg图像文件路径的集合。 例如：train session id=055128, len(fc1_img_paths): 7228
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data fc1_img_paths:', fc1_img_paths, 'len(fc1_img_paths):', len(fc1_img_paths))
        # 将fc1_img_paths中所有的图片按照后缀数字升序排序，然后随机取出连续的sample_size个图片帧
        fc1_img_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data, after sort, fc1_img_paths:', fc1_img_paths, 'len(fc1_img_paths):', len(fc1_img_paths))
        # 在0，len(fc1_img_paths)-sample_size 之间，生成一个随机数frame_idx，表示索引为frame_idx的图片帧
        self.frame_idx = random.randint(0, (len(fc1_img_paths) - self.sample_size)) # 如果一共16帧,索引值候选范围:从index=0到15, 共16个, sample_size=4, 那么frame_index=随机数，且取值范围是[0,12], 12=16-4
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data, self.frame_idx:', self.frame_idx, ' img_dir_path=', img_dir_path)
        # 在 fc1_img_paths 取出从frame_idx开始的sample_size个图片帧
        sample_fc1_frames = fc1_img_paths[self.frame_idx:self.frame_idx + self.sample_size] # 从所有的frame中按照self.frame_idx开始，取出sample_size个frame，即随机取出sample_size个图片，包含self.frame_idx，不包含self.frame_idx+sample_size，左闭右开
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data, sample_fc1_frames:', sample_fc1_frames, 'len(sample_fc1_frames):', len(sample_fc1_frames))
        # 遍历sample_fc1_frames里的每个图片帧，将其转换为RGB模式
        fc1_img_tensor_list = []
        for i, fc1_frame_path in enumerate(sample_fc1_frames):
            # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data, enumerate(sample_fc1_frames), fc1_frame_path:', fc1_frame_path, 'i:', i)
            try:
                fc1_img = Image.open(fc1_frame_path).convert("RGB") # PIL.Image.open() 打开图片，返回一个Image对象，Image对象有很多方法，如：Image.show()，Image.save()，Image.convert()等，Image.convert()用于转换图片模式，如：RGB，L等，为了方便后续处理，这里转换为RGB模式，即3通道
            except Exception as e:
                # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]exception:', e, 'fc1_frame_path:', fc1_frame_path)
                return None
            # 将图片转换为tensor
            if self.transform:
                fc1_img_tensor = self.transform(fc1_img)
            # 将图片tensor添加到fc1_imgs中
            fc1_img_tensor_list.append(fc1_img_tensor)
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data, fc1_img_tensor_list:', fc1_img_tensor_list, 'len(fc1_img_tensor_list):', len(fc1_img_tensor_list))
        ########### get fc1 image - done ###########
        
        ########### get fc2 image - start###########
        fc2_img_paths = glob.glob(fc2_img_dir_path + "/*.jpg") # fc2_img_paths是FC2_A目录下所有的jpg图像文件路径的集合。 例如：train session id=055128, len(fc2_img_paths): 7228
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data fc2_img_paths:', fc2_img_paths, 'len(fc2_img_paths):', len(fc2_img_paths))
        # 将fc2_img_paths中所有的图片按照后缀数字升序排序，然后随机取出连续的sample_size个图片帧
        fc2_img_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data, after sort, fc2_img_paths:', fc2_img_paths, 'len(fc2_img_paths):', len(fc2_img_paths))
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data, self.frame_idx:', self.frame_idx)
        # 在 fc2_img_paths 取出从self.frame_idx开始的sample_size个图片帧
        sample_fc2_frames = fc2_img_paths[self.frame_idx:self.frame_idx + self.sample_size] # 从所有的frame中按照self.frame_idx开始，取出sample_size个frame，即随机取出sample_size个图片
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data, sample_fc2_frames:', sample_fc2_frames, 'len(sample_fc2_frames):', len(sample_fc2_frames))
        # 遍历sample_fc2_frames里的每个图片帧，将其转换为RGB模式
        fc2_img_tensor_list = []
        for i, fc2_frame_path in enumerate(sample_fc2_frames):
            # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data, enumerate(sample_fc2_frames), fc2_frame_path:', fc2_frame_path, 'i:', i)
            try:
                fc2_img = Image.open(fc2_frame_path).convert("RGB") # PIL.Image.open() 打开图片，返回一个Image对象，Image对象有很多方法，如：Image.show()，Image.save()，Image.convert()等，Image.convert()用于转换图片模式，如：RGB，L等，为了方便后续处理，这里转换为RGB模式，即3通道
            except Exception as e:
                # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]exception:', e, 'fc2_frame_path:', fc2_frame_path)
                return None
            # 将图片转换为tensor
            if self.transform:
                fc2_img_tensor = self.transform(fc2_img)
            # 将图片tensor添加到fc2_imgs中
            fc2_img_tensor_list.append(fc2_img_tensor)
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- get_image_data, fc2_img_tensor_list:', fc2_img_tensor_list, 'len(fc2_img_tensor_list):', len(fc2_img_tensor_list))
        ########### get fc2 image - done ###########
        
        return fc1_img_tensor_list, fc2_img_tensor_list

    def get_wave_data(self, idx):
        # get wave data, return numpy.ndarray object, for example: numpy.ndarray object, shape=(1, 1, 128), means the wave data is 1x1x128, 音频数据是1x1x128，1代表1个声道，1代表1个采样点，128代表128个采样点，也就是说音频数据是128个采样点，每个采样点是1个声道，每个声道是1个采样点。
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data 函数开始执行')
        
        img_dir_path = self.img_dir_ls[idx] # img_dir_path 是训练集图像帧session的路径，例如 'datasets/udiva_tiny/train/recordings/animals_recordings_train_img/055128'
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data, img_dir_path:', img_dir_path)
        session_id = os.path.basename(img_dir_path) # session_id 是训练集图像帧session的名称，例如 '055128'
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data, session_id:', session_id)
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data, self.audio_dir:', self.audio_dir)
        
        # 原始的生成wav_dir_path的方式
        # wav_dir_path = os.path.join(self.data_root, self.audio_dir, session_id) # wav_dir_path 是训练集音频帧session的路径，例如 'datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/055128'
        
        # 修改后的生成wav_dir_path的方式
        # 构造wav所在的路径，需要和img的路径保持一致，为同一task的同一个session，即如果 img_dir_path: datasets/udiva_tiny/train/recordings/animals_recordings_train_img/059134，那么 wav_dir_path: datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/059134
        # 将 img_dir_path 路径里的 _img 替换为 _wav，例如 datasets/udiva_tiny/train/recordings/animals_recordings_train_img/059134 替换为 datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/059134
        wav_dir_path = img_dir_path.replace('_img', '_wav')
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data, img_dir_path:', img_dir_path, 'session_id:', session_id, 'wav_dir_path:', wav_dir_path)
        
        fc1_wav_path, fc2_wav_path = '', ''
        for file in os.listdir(wav_dir_path):
            # print('file:', file, 'type:', type(file)) # datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/128129/FC1_A.wav.npy 
            # datasets/udiva_tiny/train/recordings/animals_recordings_train_wav/055128/FC1_A.wav.npy
            # judge file is a file and start with FC1 and end with .wav.npy
            if os.path.isfile(os.path.join(wav_dir_path, file)) and file.startswith('FC1') and file.endswith('.wav.npy'):
                fc1_wav_path = os.path.join(wav_dir_path, file)
                # print('[Ddeeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data函数 fc1_wav_path:', fc1_wav_path)
            # judge file is a file and start with FC2 and end with .wav.npy
            if os.path.isfile(os.path.join(wav_dir_path, file)) and file.startswith('FC2') and file.endswith('.wav.npy'):
                fc2_wav_path = os.path.join(wav_dir_path, file)
                # print('[Ddeeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data函数 fc2_wav_path:', fc2_wav_path)
        
        ########### process fc1 wave data ###########
        fc1_wav_ft = np.load(fc1_wav_path) # fc1_wav_ft.shape: (1, 1, 4626530) len(fc1_wav_ft):  1
        # print('[Ddeeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data函数 fc1_wav_ft:', fc1_wav_ft, 'fc1_wav_ft.shape:', fc1_wav_ft.shape, '  len(fc1_wav_ft): ', len(fc1_wav_ft))
        # 举例：animals_recordings_test_wav/008105/FC1_A.wav   时长:6分20秒=380秒    其对应的fc1_wav_ft.shape: (1, 1, 6073598)   6073598/380=15983.1526316，即每秒采样15983.1526316个采样点，为什么不完全等于16000呢？通过运行soxi -D FC1_A.wav，得到其精确的时长=379.599819秒，因此 379.599819*16000=6073597.104，四舍五入为6073598，能匹配！！
        # 举例：animals_recordings_val_wav/001081/FC1_A.wav    时长:10分07秒=607秒   其对应的fc1_wav_ft.shape: (1, 1, 9707799)   9707799/607=15993.0790774，即每秒采样15993.0790774个采样点, 精确时间: 606.737415秒, 因此 606.737415*16000=9707798.64，四舍五入为9707799，能匹配！！
        # 举例：animals_recordings_val_wav/001080/FC1_A.wav    时长:8分59秒=539秒    其对应的fc1_wav_ft.shape: (1, 1, 8621477)   8621477/539=15995.3191095，即每秒采样15995.3191095个采样点
        # 举例：animals_recordings_train_wav/055125/FC1_A.wav  时长:6分25秒=385秒    其对应的fc1_wav_ft.shape: (1, 1, 6161648)   6161648/385=16004.2805195，即每秒采样16004.2805195个采样点
        # 举例：animals_recordings_train_wav/058110/FC1_A.wav  时长:7分07秒=427秒    其对应的fc1_wav_ft.shape: (1, 1, 6824438)   6824438/427=15982.2903981，即每秒采样15982.2903981个采样点
        # 结合 leenote/video_to_wav/raw_audio_process.py 里的librosa_extract函数，wav_ft = librosa.load(wav_file_path, 16000)[0][None, None, :]，即当wav转为npy时，采样率为16000，所以每秒采样16000个点，所以每个npy文件的第三个维度值就是时长*16000
        
        # 从音频中的第self.frame_idx秒开始，取时长为self.sample_size秒的音频片段, 即从第self.frame_idx*16000个采样点开始，取连续的self.sample_size*16000个采样点
        start_point = self.frame_idx * 16000
        end_point = start_point + self.sample_size * 16000
        if end_point > fc1_wav_ft.shape[-1]: # 如果end_point > fc1_wav_ft.shape[-1]，则说明从采样的那一秒往后加sample_size秒会超过音频的最后一秒，因此只需要取到音频的最后一秒（最后一个采样点）即可
            end_point = fc1_wav_ft.shape[-1]
        # print('[dpcv/data/datasets/audio_visual_data_udiva.py] fc1 start_point:', start_point, 'end_point:', end_point)
        fc1_wav_tmp = fc1_wav_ft[..., start_point: end_point] # fc1_wav_tmp.shape: 

        # print('[Ddeeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data函数 fc1_wav_tmp.shape:', fc1_wav_tmp.shape)
        if fc1_wav_tmp.shape[-1] < self.sample_size * 16000: # 如果采样点数小于self.sample_size * 16000，就用0填充剩余的采样点
            fc1_wav_fill = np.zeros((1, 1, self.sample_size * 16000))
            fc1_wav_fill[..., :fc1_wav_tmp.shape[-1]] = fc1_wav_tmp # 将fc1_wav_tmp的采样点填充到fc1_wav_fill的前fc1_wav_tmp.shape[-1]个采样点
            fc1_wav_tmp = fc1_wav_fill # 赋值给fc1_wav_tmp
        
        ########### process fc2 wave data ###########
        fc2_wav_ft = np.load(fc2_wav_path)
        
        start_point = self.frame_idx * 16000
        end_point = start_point + self.sample_size * 16000
        if end_point > fc2_wav_ft.shape[-1]: 
            end_point = fc2_wav_ft.shape[-1]
        # print('[dpcv/data/datasets/audio_visual_data_udiva.py] fc2 start_point:', start_point, 'end_point:', end_point)
        fc2_wav_tmp = fc2_wav_ft[..., start_point: end_point]
        
        if fc2_wav_tmp.shape[-1] < self.sample_size * 16000:
            fc2_wav_fill = np.zeros((1, 1, self.sample_size * 16000))
            fc2_wav_fill[..., :fc2_wav_tmp.shape[-1]] = fc2_wav_tmp
            fc2_wav_tmp = fc2_wav_fill

        # print('[Ddeeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]-get_wave_data函数 fc1_wav_tmp.shape:', fc1_wav_tmp.shape, 'fc2_wav_tmp.shape:', fc2_wav_tmp.shape) # fc1_wav_tmp.shape: (1, 1, 256000) fc2_wav_tmp.shape: (1, 1, 256000)
        return fc1_wav_tmp, fc2_wav_tmp


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
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- bimodal_resnet_data_loader_udiva 开始进行训练集的dataloader初始化...')
        dataset = AudioVisualDataUdiva(
            cfg.DATA.ROOT,
            cfg.DATA.TRAIN_IMG_DATA, # 'TRAIN_IMG_DATA': 'ChaLearn2016_tiny/train_data',
            cfg.DATA.TRAIN_AUD_DATA, # 'TRAIN_AUD_DATA': 'ChaLearn2016_tiny/voice_data/voice_librosa/train_data',
            cfg.DATA.TRAIN_LABEL_DATA, # 'TRAIN_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_training.pkl',
            transforms
        )
        batch_size = cfg.DATA_LOADER.TRAIN_BATCH_SIZE
    elif mode == "valid":
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- bimodal_resnet_data_loader_udiva 开始进行验证集的dataloader初始化...')
        dataset = AudioVisualDataUdiva(
            cfg.DATA.ROOT,
            cfg.DATA.VALID_IMG_DATA, #'VALID_IMG_DATA': 'ChaLearn2016_tiny/valid_data',
            cfg.DATA.VALID_AUD_DATA, # 'VALID_AUD_DATA': 'ChaLearn2016_tiny/voice_data/voice_librosa/valid_data',
            cfg.DATA.VALID_LABEL_DATA, # 'VALID_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_validation.pkl',
            transforms
        )
        batch_size = cfg.DATA_LOADER.VALID_BATCH_SIZE
    elif mode == "full_test":
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- bimodal_resnet_data_loader_udiva 开始进行测试集的dataloader初始化...')
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
def bimodal_resnet_lstm_data_loader_udiva(cfg, mode): # # 基于AudioVisualDataUdiva 增加针对LSTM的数据处理
    assert (mode in ["train", "valid", "test", "full_test"]), " 'mode' only supports 'train' 'valid' 'test' "
    transforms = build_transform_spatial(cfg) # TRANSFORM: "standard_frame_transform"
    if mode == "train":
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- bimodal_resnet_lstm_data_loader_udiva 开始进行训练集的dataloader初始化...')
        
        # 如果 cfg.DATA.SESSION 字符串里包含了 'ANIMALS' 关键词，那么就在list里添加cfg.DATA.ANIMALS_TRAIN_IMG_DATA, 否则就不添加
        # 如果 cfg.DATA.SESSION 字符串里包含了 'GHOST' 关键词，那么就在list里添加cfg.DATA.GHOST_TRAIN_IMG_DATA, 否则就不添加
        # 如果 cfg.DATA.SESSION 字符串里包含了 'LEGO' 关键词，那么就在list里添加cfg.DATA.LEGO_TRAIN_IMG_DATA, 否则就不添加
        # 如果 cfg.DATA.SESSION 字符串里包含了 'TALK' 关键词，那么就在list里添加cfg.DATA.TALK_TRAIN_IMG_DATA, 否则就不添加
        train_img_data = [
            cfg.DATA.ANIMALS_TRAIN_IMG_DATA if 'ANIMALS' in cfg.DATA.SESSION else None,
            cfg.DATA.GHOST_TRAIN_IMG_DATA if 'GHOST' in cfg.DATA.SESSION else None,
            cfg.DATA.LEGO_TRAIN_IMG_DATA if 'LEGO' in cfg.DATA.SESSION else None,
            cfg.DATA.TALK_TRAIN_IMG_DATA if 'TALK' in cfg.DATA.SESSION else None,
        ]
        train_aud_data = [
            cfg.DATA.ANIMALS_TRAIN_AUD_DATA if 'ANIMALS' in cfg.DATA.SESSION else None,
            cfg.DATA.GHOST_TRAIN_AUD_DATA if 'GHOST' in cfg.DATA.SESSION else None,
            cfg.DATA.LEGO_TRAIN_AUD_DATA if 'LEGO' in cfg.DATA.SESSION else None,
            cfg.DATA.TALK_TRAIN_AUD_DATA if 'TALK' in cfg.DATA.SESSION else None,
        ]
        # 去掉list里的None元素
        train_img_data = [x for x in train_img_data if x is not None]
        train_aud_data = [x for x in train_aud_data if x is not None]
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- train_img_data = ', train_img_data, ', train_aud_data = ', train_aud_data, ', len(train_img_data)=', len(train_img_data), ' len(train_aud_data)=', len(train_aud_data))
        dataset = AudioVisualLstmDataUdiva(
            cfg.DATA.ROOT,
            train_img_data, 
            train_aud_data, 
            cfg.DATA.TRAIN_LABEL_DATA,
            transforms
        )
        batch_size = cfg.DATA_LOADER.TRAIN_BATCH_SIZE
    elif mode == "valid":
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- bimodal_resnet_lstm_data_loader_udiva 开始进行验证集的dataloader初始化...')
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
            transforms
        )
        batch_size = cfg.DATA_LOADER.VALID_BATCH_SIZE
    elif mode == "full_test":
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data_udiva.py]- bimodal_resnet_lstm_data_loader_udiva 开始进行测试集的dataloader初始化...')
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
            transforms
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
