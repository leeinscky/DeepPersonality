import cv2
import os
import zipfile
from pathlib import Path


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

def crop_to_square(img):
    h, w, _ = img.shape
    c_x, c_y = int(w / 2), int(h / 2)
    img = img[:, c_x - c_y: c_x + c_y]
    return img


def frame_extract(video_path, save_dir, resize=(456, 256), transform=None):
    """
    Creating folder to save all frames from the video
    """
    try:
        # print('[frame_extract] start running frame_extract function')
        cap = cv2.VideoCapture(video_path)
        # print('[frame_extract] after cv2.VideoCapture')

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

        length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # print('[frame_extract] after cap.get(cv2.CAP_PROP_FRAME_COUNT)')
        count = 0
        # print('[frame_extract] before while loop')
        # Running a loop to each frame and saving it in the created folder
    # print exception 
    except Exception as e:
        # print('[frame_extract] exception = ', e)
        return None
     
    while cap.isOpened():
        # print('[frame_extract] in while loop')
        count += 1
        if length == count:
            break
        ret, frame = cap.read()
        if frame is None:
            continue
        if transform is not None:
            frame = transform(frame)

        # Resizing it to w, h = resize to save the disk space and fit into the model
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)
        # Saves image of the current frame to a jpg file
        name = f"{str(save_path)}/frame_{str(count)}.jpg"
        if os.path.exists(name):
            continue
        cv2.imwrite(name, frame)
        if count % 200 == 0:
            print(f"video:{str(video_path)} saved image {count}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def long_time_task(video, parent_dir):
    # print('start running long_time_task function')
    print(f"execute {video} ...")
    return frame_extract(video_path=video, save_dir=parent_dir, resize=(256, 256), transform=crop_to_square)

if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='extract image frames from videos')
    parser.add_argument('-v', '--video-dir', help="path to video directory", default=None, type=str)
    parser.add_argument("-o", "--output-dir", default=None, type=str, help="path to the extracted frames")
    args = parser.parse_args()

    # def long_time_task(video, parent_dir):
    #     print(f"execute {video} ...")
    #     return frame_extract(video_path=video, save_dir=parent_dir, resize=(256, 256), transform=crop_to_square)

    p = Pool(8)
    v_path = args.video_dir
    # path = Path("/root/personality/datasets/chalearn2021/train/lego_train")
    path = Path(v_path)
    print('path:', path)
    i = 0
    video_pts = list(path.rglob("*.mp4"))
    print('video_pts:', video_pts)
    for video in tqdm(video_pts):
        print('i:', i)
        i += 1
        video_path = str(video)
        if args.output_dir is not None:
            saved_dir = args.output_dir
        else:
            saved_dir = Path(video).parent
        print('video_path:', video_path)
        print('saved_dir:', saved_dir)
        print('before apply_async')
        p.apply_async(long_time_task, args=(video_path, saved_dir)) # 异步执行
        print('after apply_async')
        print('---------------------')
        # frame_extract(video_path=video_path, save_dir=saved_dir, resize=(256, 256), transform=crop_to_square)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    print(f"processed {i} videos")
