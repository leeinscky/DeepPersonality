import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

# 定义要加载和转换的图像文件
img_root_dir = '/home/zl525/code/DeepPersonality/datasets/udiva_full/train/recordings/talk_recordings_train_img/'
img_file = os.path.join(img_root_dir, '002003/FC2_T/frame_50.jpg')


# 定义转换
# transform = transforms.Compose([
#     transforms.Resize(256),   # 调整图像大小为 256 x 256, 保持长宽比不变，图像较小的一边填充为 256，较大的一边等比例缩放，多余部分裁剪。例如：原始图像为 200 x 100，调整后为 256 x 128
#     transforms.CenterCrop(224),   # 裁剪图像中心为 224 x 224
#     transforms.RandomHorizontalFlip(p=0.5),   # 随机水平翻转图像
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),   # 随机更改图像颜色
#     transforms.ToTensor(),   # 将图像转换为张量
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # 标准化张量
# ])

# standard_frame_transform, 同DeepPersonality代码库的逻辑 def standard_frame_transform(), DeepPersonality/dpcv/data/transforms/transform.py
standard_frame_transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.RandomHorizontalFlip(1), # 随机水平翻转图片, 这里的0.5是指翻转的概率, 也就是说每张图片都有50%的概率会被翻转
    transforms.CenterCrop((224, 224)), # 裁剪图片, 这里是裁剪成224*224,
    transforms.ToTensor(), # 将图片转换成tensor, 这里的tensor是pytorch中的tensor, 也就是说图片的数据类型是torch.Tensor, 如果想要将图片转换成numpy中的array, 可以使用transforms.ToPILImage(), 这样就可以将图片转换成numpy中的array了, 但是这里要注意的是, 如果图片是灰度图, 那么转换成numpy中的array之后, 图片的数据类型是numpy.uint8, 如果图片是彩色图, 那么转换成numpy中的array之后, 图片的数据类型是numpy.float32
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 对图片进行归一化, 如果想要用自己的数据集的均值和方差对图片进行归一化, 可以使用transforms.Normalize(mean=[mean1, mean2, mean3], std=[std1, std2, std3]), 这里的mean1, mean2, mean3分别是图片的三个通道的均值, std1, std2, std3分别是图片的三个通道的方差
])

# face_image_transform 同DeepPersonality代码库的逻辑
norm_mean = [0.485, 0.456, 0.406]  # statistics from imagenet dataset which contains about 120 million images
norm_std = [0.229, 0.224, 0.225]
face_image_transform = transforms.Compose([
    transforms.Resize(112),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 增加噪声处理，个人测试
noise_frame_transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.RandomHorizontalFlip(1), 
    transforms.CenterCrop((224, 224)), 
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载和转换图像
img = Image.open(img_file)
img_transformed = standard_frame_transform(img)
face_img_transformed = face_image_transform(img)
noise_img_transformed = noise_frame_transform(img)

# 验证图像像素是否服从正态分布，如果是对像素进行了归一化（Normalization），那么均值应该接近于0，标准差应该接近于1
# print(img_transformed.mean(), img_transformed.std()) # 验证代码里原始图像img 经过transform后，像素确实服从(0,1)的均值为0，标准差为1的正态分布
print('mean:', np.mean(img_transformed.numpy()), 'std:', np.std(img_transformed.numpy())) # 输出均值和标准差

row, col = 2, 2
# 设置画布大小为 10*10
plt.figure(figsize=(10, 10))
plt.subplot(row, col, 1)
plt.imshow(img)
plt.title('Original Image')
# plt.savefig('original_image.png')
print('***********Original Image Size: ', img.size, ', ratio: ', img.size[0]/img.size[1])


plt.subplot(row, col, 2)
img_transformed = img_transformed.permute(1, 2, 0) # 将通道数放在最后一维，是为了后面的可视化，因为 plt.imshow() 要求通道数在最后一维，而 pytorch 中通道数在第一维，所以要转换一下
img_transformed = img_transformed.numpy()
img_transformed = img_transformed * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406] # 反归一化，因为 pytorch 中的归一化是减去均值除以方差，所以反归一化就是乘以方差加上均值
plt.imshow(img_transformed)
plt.title('Standard Image')
# plt.savefig('standard.png')
print('***********Transformed Image Size: ', img_transformed.shape, ', ratio: ', img_transformed.shape[1]/img_transformed.shape[0])


plt.subplot(row, col, 3)
noise_image_transformed = noise_img_transformed.permute(1, 2, 0)
noise_image_transformed = noise_image_transformed.numpy()
noise_image_transformed = noise_image_transformed * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
plt.imshow(noise_image_transformed)
plt.title('Noise Image')
# plt.savefig('noise.png')
print('***********Noise Image Size: ', noise_image_transformed.shape, ', ratio: ', noise_image_transformed.shape[1]/noise_image_transformed.shape[0])


plt.subplot(row, col, 4)
face_image_transformed = face_img_transformed.permute(1, 2, 0)
face_image_transformed = face_image_transformed.numpy()
face_image_transformed = face_image_transformed * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
plt.imshow(face_image_transformed)
plt.title('Face Image')
plt.savefig('face.png')
print('***********Face Image Size: ', face_image_transformed.shape, ', ratio: ', face_image_transformed.shape[1]/face_image_transformed.shape[0])


plt.show()
