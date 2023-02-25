"""
transform operation for different networks
for vgg face mean = (131.0912, 103.8827, 91.4953) no std
"""
from .build import TRANSFORM_REGISTRY


def set_transform_op():
    import torchvision.transforms as transforms
    norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    norm_std = [0.229, 0.224, 0.225]
    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return transforms


@TRANSFORM_REGISTRY.register()
def standard_frame_transform():
    import torchvision.transforms as transforms # transfroms是一个类, 里面有很多函数, 比如Resize, CenterCrop, ToTensor, Normalize, Compose等, 这些函数都是用来对图片进行处理的, 例如Resize就是用来调整图片的大小的, CenterCrop就是用来裁剪图片的, ToTensor就是用来将图片转换成tensor的, Normalize就是用来对图片进行归一化的, Compose就是用来将多个函数组合成一个函数的, 这样就可以一次性对图片进行多种处理了
    # 例如这里就是先调整图片的大小, 然后再裁剪图片, 然后再将图片转换成tensor, 最后再对图片进行归一化
    transforms = transforms.Compose([
        transforms.Resize(256), # 调整图片的大小, 这里是调整成256*256, 也就是说图片的长和宽都是256, 如果只想调整图片的长或者宽, 可以使用transforms.Resize((256, None))或者transforms.Resize((None, 256)), 这样就只会调整图片的长或者宽, 不会调整图片的长和宽,
        transforms.RandomHorizontalFlip(0.5), # 随机水平翻转图片, 这里的0.5是指翻转的概率, 也就是说每张图片都有50%的概率会被翻转
        transforms.CenterCrop((224, 224)), # 裁剪图片, 这里是裁剪成224*224, 也就是说图片的长和宽都是224, 如果只想裁剪图片的长或者宽, 可以使用transforms.CenterCrop((224, None))或者transforms.CenterCrop((None, 224)), 这样就只会裁剪图片的长或者宽, 不会裁剪图片的长和宽
        transforms.ToTensor(), # 将图片转换成tensor, 这里的tensor是pytorch中的tensor, 也就是说图片的数据类型是torch.Tensor, 如果想要将图片转换成numpy中的array, 可以使用transforms.ToPILImage(), 这样就可以将图片转换成numpy中的array了, 但是这里要注意的是, 如果图片是灰度图, 那么转换成numpy中的array之后, 图片的数据类型是numpy.uint8, 如果图片是彩色图, 那么转换成numpy中的array之后, 图片的数据类型是numpy.float32
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 对图片进行归一化, 如果想要用自己的数据集的均值和方差对图片进行归一化, 可以使用transforms.Normalize(mean=[mean1, mean2, mean3], std=[std1, std2, std3]), 这里的mean1, mean2, mean3分别是图片的三个通道的均值, std1, std2, std3分别是图片的三个通道的方差
    ])
    return transforms

@TRANSFORM_REGISTRY.register()
def udiva_frame_transforms():
    import torchvision.transforms as transforms
    transforms = transforms.Compose([
        transforms.Resize(256), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # 随机更改图像颜色
        transforms.RandomRotation(degrees=10), # 设计角度旋转图像
        transforms.GaussianBlur(kernel_size=(3,3)), # 高斯模糊 kernel_size: 高斯核的大小(模糊半径), 模糊半径越大, 正态分布标准差越大, 图像就越模糊; sigma: 高斯核的标准差, sigma越大，模糊程度越大, 一般设置为0.1到2.0之间
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        AddGaussianNoise(mean=0., std=0.2), # 增加噪声, 0.2是标准差, 0.是均值, 这么做的目的是为了让模型更加鲁棒, 也就是说模型不会因为图片中有一些噪声而导致模型的效果变差
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.23)) # 随机擦除, scale: 遮挡区域的面积。如(a, b)，则会随机选择 (a, b) 中的一个遮挡比例 ; ratio: 遮挡区域长宽比。如(a, b)，则会随机选择 (a, b) 中的一个长宽比 例如: scale=(0.02, 0.33), ratio=(0.3, 3) 会随机选择一个遮挡比例为 0.02 到 0.33 之间的一个值, 然后随机选择一个长宽比为 0.3 到 3 之间的一个值, 然后根据这个长宽比和遮挡比例来确定遮挡区域的大小, 然后在图片中随机选择一个位置, ratio=(0.99, 0.99) 会用一个正方形来遮挡图片
    ])
    # print('Using udiva_frame_transforms augmentation method...')
    return transforms

@TRANSFORM_REGISTRY.register()
def face_image_transform():
    import torchvision.transforms as transforms
    norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    norm_std = [0.229, 0.224, 0.225]
    transforms = transforms.Compose([
        transforms.Resize(112),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return transforms


@TRANSFORM_REGISTRY.register()
def face_image_x2_transform():
    import torchvision.transforms as transforms
    norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    norm_std = [0.229, 0.224, 0.225]
    transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return transforms


@TRANSFORM_REGISTRY.register()
def crnet_frame_face_transform():
    import torchvision.transforms as transforms
    norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    norm_std = [0.229, 0.224, 0.225]

    frame_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(0.5),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    face_transforms = transforms.Compose([
        transforms.Resize(112),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return {"frame": frame_transforms, "face": face_transforms}


@TRANSFORM_REGISTRY.register()
def set_tpn_transform_op():
    import torchvision.transforms as transforms
    norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    norm_std = [0.229, 0.224, 0.225]
    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(0.5),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return transforms


@TRANSFORM_REGISTRY.register()
def set_vat_transform_op():
    import torchvision.transforms as transforms
    norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    norm_std = [0.229, 0.224, 0.225]
    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 112)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return transforms


def set_crnet_transform():
    import torchvision.transforms as transforms
    # norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
    # norm_std = [0.229, 0.224, 0.225]
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((112, 112))
        # transforms.Normalize(norm_mean, norm_std)
    ])
    return {"frame": transforms, "face": transforms}


def set_audio_visual_transform():
    import torchvision.transforms as transforms
    transforms = transforms.Compose([
        # transforms.RandomVerticalFlip(0.5),
        # transforms.RandomRotation(15),
        # transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
    ])
    return transforms


def set_per_transform():
    import torchvision.transforms as transforms
    transforms = transforms.Compose([
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
    ])
    return transforms

class AddGaussianNoise(object):
    # copy from https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/23
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
