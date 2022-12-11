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

