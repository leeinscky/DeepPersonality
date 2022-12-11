import os
import torch
import glob
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import random
import pickle
import numpy as np
from dpcv.data.datasets.bi_modal_data import VideoData
from dpcv.data.transforms.transform import set_audio_visual_transform
from dpcv.data.transforms.build import build_transform_spatial
from .build import DATA_LOADER_REGISTRY
from random import shuffle


class AudioVisualData(VideoData):

    def __init__(self, data_root, img_dir, audio_dir, label_file, transform=None, sample_size=100):
        print('[deeppersonality/dpcv/data/datasets/audio_visual_data.py]- class AudioVisualData(VideoData) 开始执行 __init__')
        super().__init__(data_root, img_dir, label_file, audio_dir)
        self.transform = transform
        self.sample_size = sample_size
        print('[deeppersonality/dpcv/data/datasets/audio_visual_data.py]- class AudioVisualData(VideoData) 结束执行 __init__')

    def __getitem__(self, idx):
        print('[deeppersonality/dpcv/data/datasets/audio_visual_data.py] - class AudioVisualData(VideoData) 开始执行 __getitem__ , idx = ', idx) # idx 和 get_ocean_label(self, index) 里的index含义一样，表示video目录里的第几个video样本
        label = self.get_ocean_label(idx)  # label是一个list，长度为5，每个元素是一个0-1之间的浮点数, 代表该video的5个personality维度的分数, 例如：[0.5, 0.5, 0.5, 0.5, 0.5], 代表该video的5个personality维度的分数都是0.5, 也就是说该video的personality维度的分数都是中性的
        img = self.get_image_data(idx) # img是一个PIL.Image.Image对象, 代表该video的一帧图像, 例如：PIL.Image.Image object, mode=RGB, size=224x224, 代表该video的一帧图像的大小是224x224, 且是RGB三通道的, 也就是说该video的一帧图像是224x224x3的
        wav = self.get_wave_data(idx) # wav是一个numpy.ndarray对象, 代表该video的一帧音频, 例如：array([[[ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ]]], dtype=float32), 代表该video的一帧音频是50176维的, 也就是说该video的一帧音频是50176x1x1的, 且是float32类型的, 且是三维的

        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data.py]-class AudioVisualData(VideoData) __getitem__ - transform之前 type(img)=', type(img), 'img=', img) # type(img)= <class 'PIL.Image.Image'> img= <PIL.Image.Image image mode=RGB size=256x256 at 0x7FC8AA03EB20>
        if self.transform: # self.transform是一个Compose对象, 代表对img和wav的一系列变换, 对于bimodal_resnet_data_loader， transforms = build_transform_spatial(cfg)，'TRANSFORM': 'standard_frame_transform', 参考 dpcv/data/transforms/transform.py 里的def standard_frame_transform()
            img = self.transform(img) # img是一个torch.Tensor对象, 代表该video的一帧图像, 例如：tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
    
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data.py]-class AudioVisualData(VideoData) __getitem__ - torch.as_tensor之前 type(label)=', type(label), 'label=', label) # type(label)= <class 'list'> label= [0.5111111111111111, 0.4563106796116505, 0.4018691588785047, 0.3626373626373627, 0.4166666666666667]
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data.py]-class AudioVisualData(VideoData) __getitem__ - torch.as_tensor之前 type(wav)=', type(wav), 'wav=', wav) # type(wav)= <class 'numpy.ndarray'> wav= [[[-1.0260781e-03 -2.3528279e-03 -2.2199661e-03 ...  9.4422154e-05 2.8776360e-04 -7.4460535e-05]]]
        wav = torch.as_tensor(wav, dtype=img.dtype) # torch.as_tensor - 将输入转换为张量, 并返回一个新的张量, 与输入共享内存, 但是不同的是, 如果输入是一个张量, 则返回的张量与输入不共享内存, 例如：tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        label = torch.as_tensor(label, dtype=img.dtype)
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data.py]-class AudioVisualData(VideoData) __getitem__ - torch.as_tensor之后 type(label)=', type(label), 'label=', label) # type(label)= <class 'torch.Tensor'> label= tensor([0.5111, 0.4563, 0.4019, 0.3626, 0.4167])
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data.py]-class AudioVisualData(VideoData) __getitem__ - torch.as_tensor之后 type(wav)=', type(wav), 'wav=', wav) # type(wav)= <class 'torch.Tensor'> wav= tensor([[[-1.0261e-03, -2.3528e-03, -2.2200e-03,  ...,  9.4422e-05, 2.8776e-04, -7.4461e-05]]])

        sample = {"image": img, "audio": wav, "label": label} # img.shape() =  torch.Size([3, 224, 224]) wav.shape() =  torch.Size([1, 1, 50176]) label.shape() =  torch.Size([5])
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data.py] - class AudioVisualData(VideoData) __getitem__ 函数返回的结果中 , img.shape() = ', img.shape, 'wav.shape() = ', wav.shape, 'label.shape() = ', label.shape) # img.shape() =  torch.Size([3, 224, 224]) wav.shape() =  torch.Size([1, 1, 50176]) label.shape() =  torch.Size([5])
        # 因为__getitem__ 需要传入参数 idx，所以返回的sample也是一个视频对应的img，wav，label，至于为什么只有1帧image, 因为get_image_data函数只返回了1帧image，而没有返回多帧image
        return sample # sample是一个dict对象, 例如：{'image': tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],

    def get_image_data(self, idx):
        # get image data, return PIL.Image.Image object, for example: PIL.Image.Image object, mode=RGB, size=224x224, means the image size is 224x224, and is RGB three channels
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data.py]-class AudioVisualData(VideoData) 开始执行 get_image_data')
        img_dir_path = self.img_dir_ls[idx]
        img_paths = glob.glob(img_dir_path + "/*.jpg") # glob.glob()返回所有匹配的文件路径列表
        sample_frames = np.linspace(0, len(img_paths), self.sample_size, endpoint=False, dtype=np.int16) # np.linspace()返回在指定的间隔内返回均匀间隔的数字, np.int16是指定返回的数据类型是int16
        selected = random.choice(sample_frames) # random.choice()从序列中获取一个随机元素, 这里的序列是sample_frames, 也就是说从sample_frames中随机选取一个元素, 例如：sample_frames=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 那么selected就是0, 1, 2, 3, 4, 5, 6, 7, 8, 9中的一个
        # img_path = random.choice(img_paths)
        try:
            img = Image.open(img_paths[selected]).convert("RGB") # Image.open()打开一个图像文件, convert("RGB")将图像转换为RGB模式, 也就是说将图像转换为三通道, 例如：PIL.Image.Image object, mode=RGB, size=224x224, means the image size is 224x224, and is RGB three channels, 如果不转换为RGB模式, 那么就是单通道, 例如：PIL.Image.Image object, mode=L, size=224x224, means the image size is 224x224, and is L one channel, L means luminance, 亮度, 也就是灰度图, 也就是黑白图.
            return img
        except:
            print(img_paths)

    def get_wave_data(self, idx):
        # get wave data, return numpy.ndarray object, for example: numpy.ndarray object, shape=(1, 1, 128), means the wave data is 1x1x128, 音频数据是1x1x128，1代表1个声道，1代表1个采样点，128代表128个采样点，也就是说音频数据是128个采样点，每个采样点是1个声道，每个声道是1个采样点。
        # print('[deeppersonality/dpcv/data/datasets/audio_visual_data.py]-class AudioVisualData(VideoData) 开始执行 get_wave_data')
        img_dir_path = self.img_dir_ls[idx]
        # wav_path = img_dir_path.replace("image_data", "voice_data/voice_librosa") + ".wav.npy"
        video_name = os.path.basename(img_dir_path)
        wav_path = os.path.join(self.data_root, self.audio_dir, f"{video_name}.wav.npy")
        wav_ft = np.load(wav_path) # np.load()读取.npy文件, 返回的是一个numpy.ndarray对象, 例如：wav_ft.shape=(1, 128, 128), wav_ft.dtype=float32, wav_ft.max()=0.99999994, wav_ft.min()=-0.99999994, wav_ft.mean()=0.000101, wav_ft.std()=0.000101, 
        try:
            n = np.random.randint(0, len(wav_ft) - 50176)
        except:
            n = 0
        wav_tmp = wav_ft[..., n: n + 50176] 
        if wav_tmp.shape[-1] < 50176:
            wav_fill = np.zeros((1, 1, 50176))
            wav_fill[..., :wav_tmp.shape[-1]] = wav_tmp
            wav_tmp = wav_fill
        return wav_tmp


class ALLSampleAudioVisualData(AudioVisualData):

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


class ALLSampleAudioVisualData2(AudioVisualData):

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
        data_set = AudioVisualData(
            cfg.DATA_ROOT,  # "/home/ssd500/personality_data",
            cfg.TRAIN_IMG_DATA,  # "image_data/train_data",
            cfg.TRAIN_AUD_DATA,  # "voice_data/train_data",
            cfg.TRAIN_LABEL_DATA,  # "annotation/annotation_training.pkl",
            trans
        )
    elif mode == "valid":
        data_set = AudioVisualData(
            cfg.DATA_ROOT,  # "/home/ssd500/personality_data",
            cfg.VALID_IMG_DATA,  # "image_data/valid_data",
            cfg.VALID_AUD_DATA,  # "voice_data/valid_data",
            cfg.VALID_LABEL_DATA,  # annotation/annotation_validation.pkl",
            trans
        )
    elif mode == "trainval":
        data_set = AudioVisualData(
            cfg.DATA_ROOT,  # "../datasets",
            cfg.TRAINVAL_IMG_DATA,  # ["image_data/training_data_01", "image_data/validation_data_01"],
            cfg.TRANIVAL_AUD_DATA,  # ["voice_data/trainingData", "voice_data/validationData"],
            cfg.TRAINVAL_LABEL_DATA,  # ["annotation/annotation_training.pkl", "annotation/annotation_validation.pkl"],
            trans,
        )
    elif mode == "test":
        data_set = AudioVisualData(
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


@DATA_LOADER_REGISTRY.register()
def bimodal_resnet_data_loader(cfg, mode):
    assert (mode in ["train", "valid", "test", "full_test"]), " 'mode' only supports 'train' 'valid' 'test' "
    transforms = build_transform_spatial(cfg)
    if mode == "train":
        dataset = AudioVisualData(
            cfg.DATA.ROOT,
            cfg.DATA.TRAIN_IMG_DATA, # 'TRAIN_IMG_DATA': 'ChaLearn2016_tiny/train_data',
            cfg.DATA.TRAIN_AUD_DATA, # 'TRAIN_AUD_DATA': 'ChaLearn2016_tiny/voice_data/voice_librosa/train_data',
            cfg.DATA.TRAIN_LABEL_DATA, # 'TRAIN_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_training.pkl',
            transforms
        )
        batch_size = cfg.DATA_LOADER.TRAIN_BATCH_SIZE
    elif mode == "valid":
        dataset = AudioVisualData(
            cfg.DATA.ROOT,
            cfg.DATA.VALID_IMG_DATA, #'VALID_IMG_DATA': 'ChaLearn2016_tiny/valid_data',
            cfg.DATA.VALID_AUD_DATA, # 'VALID_AUD_DATA': 'ChaLearn2016_tiny/voice_data/voice_librosa/valid_data',
            cfg.DATA.VALID_LABEL_DATA, # 'VALID_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_validation.pkl',
            transforms
        )
        batch_size = cfg.DATA_LOADER.VALID_BATCH_SIZE
    elif mode == "full_test":
        return ALLSampleAudioVisualData(
            cfg.DATA.ROOT,
            cfg.DATA.TEST_IMG_DATA, # 'TEST_IMG_DATA': 'ChaLearn2016_tiny/test_data',
            cfg.DATA.TEST_AUD_DATA, # 'TEST_AUD_DATA': 'ChaLearn2016_tiny/voice_data/voice_librosa/test_data',
            cfg.DATA.TEST_LABEL_DATA, # 'TEST_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_test.pkl',
            transforms
        )
    else:
        dataset = AudioVisualData(
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


if __name__ == "__main__":
    # from tqdm import tqdm
    # args = ("../../../datasets", "ImageData/trainingData", "VoiceData/trainingData_50176", "annotation_training.pkl")
    trans = set_audio_visual_transform()
    # data_set = AudioVisualData(*args, trans)
    # # print(len(data_set))
    # data = data_set[1]
    # print(data["image"].shape, data["audio"].shape, data["label"].shape)

    dataset = AudioVisualData(
        "../../../datasets",
        ["image_data/training_data_01", "image_data/validation_data_01"],
        ["voice_data/trainingData", "voice_data/validationData"],
        ["annotation/annotation_training.pkl", "annotation/annotation_validation.pkl"],
        trans,
    )
    print(len(dataset))
    a = dataset[1]
    print(a)
