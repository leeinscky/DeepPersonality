# 处理UDIVA数据集，将视频转换为图片, 并从图片中提取人脸  参考：# https://github.com/kb22/Create-Face-Data-from-Images

import cv2
import os
from pathlib import Path
import numpy as np

def crop_to_square(img):
    h, w, _ = img.shape
    c_x, c_y = int(w / 2), int(h / 2)
    img = img[:, c_x - c_y: c_x + c_y]
    return img

def frame_extract(video_path, save_dir, resize=(456, 256), transform=None, frame_num=1):
    """
    Creating folder to save all frames from the video
    """
    try:
        cap = cv2.VideoCapture(video_path)
        file_name = Path(video_path).stem

        save_path = Path(save_dir).joinpath(file_name)
        # print('[frame_extract] after Path(save_dir).joinpath(file_name), save_path = ', save_path)
        os.makedirs(save_path, exist_ok=True) # exist_ok=True, if the directory exists, do nothing

        fps = cap.get(cv2.CAP_PROP_FPS) # 查看视频的帧率 CAP_PROP_FPS=5
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT) # 获取视频的总帧数，参考：https://blog.csdn.net/zhicai_liu/article/details/110518578
        print(f"video:{str(video_path)} length = {length}, fps = {fps}")
        # 打印结果：video:/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/023191/FC1_T.mp4 length = 8079.0, fps = 25.0
    except Exception as e:
        # print('[frame_extract] exception = ', e)
        return None
    
    # Running a loop to each frame and saving it in the created folder
    count = 0
    count_face = 0
    while cap.isOpened():
        if length == count:
            break
        ret, frame = cap.read()
        if frame is None:
            continue
        if ret:
            count += 1
            # 如果count是fps的倍数，就保存frame图片，即每隔1秒抽取一帧。 如果每秒抽n帧，那么就是count % (fps / n) == 0 举例：每秒抽取5帧，那么就是count % (fps/5) == 0, fps=25, 那么就是count % 5 == 0, 即每隔5个连续的帧抽取一帧，即每隔0.2秒抽取一帧
            if count % (fps / frame_num) == 0:
                if transform is not None:
                    frame = transform(frame)
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC) # Resizing it to w, h = resize to save the disk space and fit into the model
                # frame.shape =  (256, 256, 3) , type(frame)= <class 'numpy.ndarray'>
                
                # Saves image of the current frame to a jpg file
                name = f"{str(save_path)}/frame_{str(count)}.jpg"  # type(name)= <class 'str'>
                # print('name = ', name, ', type(name)=', type(name), ', frame.shape = ', frame.shape, ', type(frame)=', type(frame)) # name =  /home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/017150/FC2_T/frame_5.jpg , type(name)= <class 'str'> , frame.shape =  (256, 256, 3) , type(frame)= <class 'numpy.ndarray'>
                if os.path.exists(name):
                    continue
                # cv2.imwrite(name, frame) # 因为后续保存人脸图片，所以这里不保存frame图片
                
                ################# --------------> 重要逻辑：从图片中提取人脸， Reference：https://github.com/kb22/Create-Face-Data-from-Images
                image = frame
                (h, w) = image.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)) # 从图像中提取出人脸区域，将其转换为 300x300 的 blob 格式，然后输入到模型中进行人脸检测。四个参数分别是：图像，缩放比例，图像大小，均值。104.0 177.0 123.0 是在训练模型时所用的均值，需要在这里进行相同的处理。计算方式是：(R-104.0)/255.0，(G-177.0)/255.0，(B-123.0)/255.0，其中R、G、B分别是图像的RGB三个通道的值。这样做的目的是将图像的像素值归一化到[-1,1]之间，使得模型的输入更加稳定。

                model.setInput(blob)
                detections = model.forward()
                # print('detections.shape: ', detections.shape, ', detections=', detections[0, 0, 0]) # detections.shape:  (1, 1, 200, 7) detections= [0. 1. 0.9969796  0.49035808 0.20711678 0.66898656 0.4190783 ]

                # Identify each face
                for i in range(0, detections.shape[2]):
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # 从检测结果中提取出人脸区域的坐标，然后将其绘制到原图像上。0, 0, i, 3:7 表示第 i 个人脸的坐标，乘以 w 和 h 是为了将坐标从 blob 格式转换为原图像的坐标。
                    # print('box: ', box) # box: [94.79769897  50.82873535 137.63249207  99.23338318]
                    (startX, startY, endX, endY) = box.astype("int") # 人脸检测的结果是一个 1x1xNx7 的 blob，其中 N 表示检测到的人脸数目。blob 的第 3 维是一个长度为 7 的向量，包含了 [image_id, label, confidence, left, top, right, bottom]，其中 image_id 表示图像的 ID，label 表示类别，confidence 表示置信度，left、top、right、bottom 分别表示人脸区域的坐标。

                    confidence = detections[0, 0, i, 2]

                    # If confidence > 0.5, save it as a separate file
                    if (confidence > 0.5):
                        count_face += 1
                        frame = image[startY:endY, startX:endX] # 从原图像中提取出人脸区域
                        cv2.imwrite(name, frame)
                        # print('frame.shape: ', frame.shape) # frame.shape:  (48, 42, 3) or (54, 46, 3) # (h, w, c) 由于每个人脸的尺寸不一样，所以打印出来的尺寸也不一样
                ################# <-------------- 重要逻辑：从图片中提取人脸
                
                if count % 1000 == 0: # 如果count是xxx的倍数，打印
                    print(f"video:{str(video_path)} saved image {count}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'): # 如果按下q键，退出
                break
    print( "Video:" + str(video_path) + "Extracted " + str(count_face) + " faces from " + str(count) + " images")

def long_time_task(video, parent_dir, frame_num):
    # print(f"execute {video} ...")
    return frame_extract(video_path=video, save_dir=parent_dir, resize=(256, 256), transform=crop_to_square, frame_num=frame_num)

if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='extract image frames from videos')
    parser.add_argument('-v', '--video-dir', help="path to video directory", default=None, type=str)
    parser.add_argument("-o", "--output-dir", default=None, type=str, help="path to the extracted frames")
    parser.add_argument("-f", "--frame-num", default=1, type=str, help="number of frames to extract per second")
    args = parser.parse_args()

    # Read the model
    prototxt_path = 'face_detection_model_data/deploy.prototxt'
    caffemodel_path = 'face_detection_model_data/weights.caffemodel'
    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    p = Pool(8)
    path = Path(args.video_dir)
    # print('session path:',path)
    i = 0
    video_pts = list(path.rglob("*.mp4"))
    print('video_pts:', video_pts)
    frame_num = int(args.frame_num) # 每秒抽取的帧数, 默认为1即每秒抽取1帧, 如果为5, 则每秒抽取5帧
    for video in tqdm(video_pts):
        i += 1
        video_path = str(video)
        if args.output_dir is not None:
            saved_dir = args.output_dir
        else:
            saved_dir = Path(video).parent
        print('index:', i, ', video_path:', video_path)
        print('index:', i, ', saved_dir:', saved_dir)
        # print('index:', i, ', extract ', frame_num, ' frames per second')
        p.apply_async(long_time_task, args=(video_path, saved_dir, frame_num)) # 异步执行
        print('---------------------')
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    print(f"processed {i} videos")
