# 处理UDIVA数据集，将视频转换为图片

import cv2
import os
import zipfile
from pathlib import Path

'''
def process():
    data_dir = "./chalearn/test"
    video_list = os.listdir(data_dir)
    for video in tqdm(video_list):
        video_path = os.path.join(data_dir, video)
        frame_sample(video_path, "./ImageData/testData/")


def frame_sample(video, save_dir):
    """
    Creating folder to save all the 100 frames from the video
    """
    cap = cv2.VideoCapture(video)

    # file_name = (os.path.basename(video).split('.mp4'))[0]
    file_name = Path(video).stem
    try:
        # if not os.path.exists(save_dir + file_name):
        #     os.makedirs(save_dir + file_name)

        save_path = Path(save_dir).joinpath(file_name)
        if not save_path.exists():
            save_path.mkdir()
    except OSError:
        print('Error: Creating directory of data')

    # Setting the frame limit to 100
    # cap.set(cv2.CAP_PROP_FRAME_COUNT, 120)
    # print(cap.get(cv2.CAP_PROP_FPS))
    # print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 6 * 5))
    cap.set(cv2.CAP_PROP_FPS, 25)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 6 * 5)
    count = 0
    # Running a loop to each frame and saving it in the created folder
    while cap.isOpened():
        count += 1
        if length == count:
            break
        ret, frame = cap.read()
        if frame is None:
            continue
        # Resizing it to 256*256 to save the disk space and fit into the model
        frame = cv2.resize(frame, (456, 256), interpolation=cv2.INTER_CUBIC)
        # Saves image of the current frame in jpg file
        name = save_dir + str(file_name) + '/frame' + str(count) + '.jpg'
        cv2.imwrite(name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
'''

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
        # file_name = (os.path.basename(video).split('.mp4'))[0]
        file_name = Path(video_path).stem
        # print('[frame_extract] after Path(video_path).stem')
        # try:
        # if not os.path.exists(save_dir + file_name):
        #     os.makedirs(save_dir + file_name)

        save_path = Path(save_dir).joinpath(file_name)
        # print('[frame_extract] after Path(save_dir).joinpath(file_name), save_path = ', save_path)
        os.makedirs(save_path, exist_ok=True) # exist_ok=True, if the directory exists, do nothing
        # print('[frame_extract] after os.makedirs')
        # if not save_path.exists():
        #     save_path.mkdir()
        # except OSError:
        #     print('Error: Creating directory of data')

        fps = cap.get(cv2.CAP_PROP_FPS) # 查看视频的帧率 CAP_PROP_FPS=5
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT) # 获取视频的总帧数，参考：https://blog.csdn.net/zhicai_liu/article/details/110518578
        print(f"video:{str(video_path)} length = {length}, fps = {fps}")
        # 打印结果：video:/home/zl525/rds/hpc-work/datasets/udiva_full/train/recordings/talk_recordings_train_img/023191/FC1_T.mp4 length = 8079.0, fps = 25.0
    except Exception as e:
        # print('[frame_extract] exception = ', e)
        return None
    
    # Running a loop to each frame and saving it in the created folder
    count = 0
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
                # Resizing it to w, h = resize to save the disk space and fit into the model
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)
                # Saves image of the current frame to a jpg file
                name = f"{str(save_path)}/frame_{str(count)}.jpg"
                if os.path.exists(name):
                    continue
                cv2.imwrite(name, frame) # 保存图片
                if count % 1000 == 0: # 如果count是200的倍数，打印一下
                    print(f"video:{str(video_path)} saved image {count}")
            if cv2.waitKey(1) & 0xFF == ord('q'): # 如果按下q键，退出
                break


def long_time_task(video, parent_dir, frame_num):
    # print('start running long_time_task function')
    print(f"execute {video} ...")
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

    # def long_time_task(video, parent_dir):
    #     print(f"execute {video} ...")
    #     return frame_extract(video_path=video, save_dir=parent_dir, resize=(256, 256), transform=crop_to_square)

    p = Pool(8)
    # path = Path("/root/personality/datasets/chalearn2021/train/lego_train")
    path = Path(args.video_dir)
    print('session path:',path)
    i = 0
    video_pts = list(path.rglob("*.mp4"))
    print('video_pts:', video_pts)
    frame_num = int(args.frame_num) # 每秒抽取的帧数, 默认为1即每秒抽取1帧, 如果为5, 则每秒抽取5帧
    for video in tqdm(video_pts):
        print('video index: ', i)
        i += 1
        video_path = str(video)
        if args.output_dir is not None:
            saved_dir = args.output_dir
        else:
            saved_dir = Path(video).parent
        print('video_path:', video_path)
        print('saved_dir:', saved_dir)
        print('extract ', frame_num, ' frames per second')
        p.apply_async(long_time_task, args=(video_path, saved_dir, frame_num)) # 异步执行
        print('---------------------')
        # frame_extract(video_path=video_path, save_dir=saved_dir, resize=(256, 256), transform=crop_to_square)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    print(f"processed {i} videos")
