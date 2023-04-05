# Reference: Script for Video Face Extraction https://github.com/liaorongfan/DeepPersonality/blob/main/datasets/README.md

import os
import cv2
from tqdm import tqdm
import glob
from face_detector import FaceDetection
import time

class FaceImageExtractor:

    def __init__(self, data_root, level, detector_path):
        """
        args:
            data_root:(str) : where to save the processed video directory
            detector_path:(str) : the path to dlib face detector model
        """
        self.data_root = data_root
        self.video_name = None
        self.video_file = None
        self.video_path_name = None
        self.save_dir = None
        self.level = level
        self.frame_count = 0
        self.frame_count_new = 0
        self.fps = 0
        self.duration = 0
        self.step = 0
        self.max_frame_count = 5000 # max number of frames to extract from a video, to meet quota limit of login node

        self.face_detector = FaceDetection(
            os.path.join(detector_path, "shape_predictor_68_face_landmarks.dat")
        )

    def load_video(self, video_name):
        """extract face images form a video and the video name will be the name of a directory to save face images

        args:
            video_name:(str) name of video file end with ".mp4"
        """
        self.video_path_name = video_name.split("/")[-2] + "/" + video_name.split("/")[-1]
        
        if self.level in ["dir", "directory"]:
            # self.video_name = f"{os.path.basename(video_name)[:-4]}_face"
            self.video_name = f"{os.path.basename(video_name)[:-4]}"
        else:
            self.video_name = os.path.basename(video_name)[:-4]  # [:-4] to remove .mp4

        self.save_dir = os.path.join(self.data_root, self.video_name)
        # if not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)

        self.video_file = cv2.VideoCapture(video_name)

    def reduce_frame_rate(self, fps_new):
        """ extract face images with the sample rate of fps_new

        args:
            fps_new:(int) face images sample rate
        return:
            save detected face images to specified directory
        """
        ret = self.video_file.isOpened()
        if ret:
            self.frame_count = self.video_file.get(cv2.CAP_PROP_FRAME_COUNT)
            self.fps = int(self.video_file.get(cv2.CAP_PROP_FPS))
            self.duration = int(self.frame_count / self.fps)

            self.frame_count_new = self.duration * fps_new
            self.step = int(self.frame_count / self.frame_count_new)

            print(f"\n------------{self.video_path_name}----------------")
            print("frame_count original: " + str(self.frame_count))
            print("fps original: " + str(self.fps))
            print("duration: " + str(self.duration))
            print("- - - - - - - - -")
            print("frame_count new: " + str(self.frame_count_new))
            print("fps new: " + str(fps_new))
            print("duration: " + str(self.duration))
            print("step: " + str(self.step))
            print("-------------------------------------------------")

            if self.step == 0:
                self.step = 1
                print("WARNING: selected framerate is higher as original!")

            cnt = 0
            # for i in tqdm(range(0, int(self.frame_count), int(self.step))):
            for i in range(0, int(self.frame_count), int(self.step)):
                # if i % 1000 == 0:
                #     print('video:', self.video_path_name, ', processing frame:', i)
                cnt = cnt + 1
                
                # if cnt >= self.frame_count_new:
                if cnt > self.frame_count_new or cnt > self.max_frame_count: # move this line before the save logic, to avoid the one more frame generated when the video is processed again
                    break
                
                save_to = f"{self.save_dir}/face_{cnt}.jpg"
                if os.path.exists(save_to):
                    # print(f"{save_to} already exists...")
                    continue
                
                ret, frame = self.video_file.read()
                if ret:
                    if self.face_detector.find_face(frame):
                        frame_crop = self.face_detector.run(frame)
                        if not os.path.exists(self.save_dir):
                            os.makedirs(self.save_dir)
                        # save_to = f"{self.save_dir}/face_{cnt}.jpg"
                        # if os.path.exists(save_to):
                        #     print(f"{save_to} already exists...")
                        #     continue
                        if cnt % 100 == 0:
                            print(f'save face in {save_to}')
                        cv2.imwrite(save_to, frame_crop)
                    else:
                        if cnt == 1:
                            print(f"[Warning] No face detected in {self.video_path_name}!")

    def process_frames(self):
        ret = self.video_file.isOpened()
        if ret:
            frame_count = self.video_file.get(cv2.CAP_PROP_FRAME_COUNT)
            print(f"video:{self.video_name}, Num of frames={frame_count}, FPS={self.video_file.get(cv2.CAP_PROP_FPS)}") # e.g. datasets/noxi_full/img/001/Expert_video.mp4: duration is 17mins 46s=17*60+46=1066s, fps=25, so num of frames should be around 26650, and the print result is Num of frames=26644.0
            cnt = 0
            for i in tqdm(range(int(frame_count))):
                cnt = cnt + 1
                ret, frame = self.video_file.read()
                if ret:
                    if self.face_detector.find_face(frame):
                        frame_crop = self.face_detector.run(frame)
                        save_to = f"{self.save_dir}/face_{cnt}.jpg"
                        if os.path.exists(save_to):
                            # print("image_exist...")
                            continue

                        cv2.imwrite(save_to, frame_crop)

                if cnt >= frame_count:
                    break

    def play_video(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.video_file.read()
            if ret:
                cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        self.video_file.release()
        cv2.destroyAllWindows()


def run_on_videos(
        video_dir,
        data_root,
        level=None,
        detector_path="/home/zl525/code/DeepPersonality/leenote/video_to_img/video_to_face/pre_trained_weights/",
):
    image_extractor = FaceImageExtractor(
        data_root=data_root,
        level=level,
        detector_path=detector_path,
    )
    # dirs = [name for name in os.listdir(data_root) if "face" in name]
    # if len(dirs) == 2:
    #     return
    input_video_ls = glob.glob(f"{video_dir}/*.mp4")
    # input_video = "/home/rongfan/11-personality_traits/apa_paper/FaceDBGenerator_V2/Facedetector/_QXI4n_FRN4.003.mp4"
    for input_video in input_video_ls:
        print(f"processing {input_video} ...")
        image_extractor.load_video(input_video)
        image_extractor.reduce_frame_rate(5)
        # image_extractor.process_frames()


if __name__ == "__main__":
    import argparse
    import multiprocessing
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(description="detect and extract face images from videos")
    parser.add_argument("-v", "--video-path", help="path to video directory", default=None, type=str)
    parser.add_argument("-o", "--output-dir", help="path to save processed videos", default=None, type=str)
    parser.add_argument("-l", "--level", default="video", type=str,
                        help="datasets should be one of [video, directory/dir]",)
    args = parser.parse_args()

    if args.level == "video":
        video_dir = args.video_path
        data_root = args.output_dir
        run_on_videos(video_dir=video_dir, data_root=data_root)
    elif args.level in ["directory", "dir"]:
        video_root = args.video_path
        # video_root = "/home/rongfan/05-personality_traits/DeepPersonality/datasets/chalearn2021/valid/talk_valid"
        data_root_ls = os.listdir(video_root)
        data_root_ls_pt = [os.path.join(video_root, d) for d in data_root_ls]
        
        # 关于process 个数的设置，参考：https://stackoverflow.com/a/20039972 
        print('cpu_count:', multiprocessing.cpu_count(), ', os.cpu_count():', os.cpu_count()) # login-icelake.hpc.cam.ac.uk: cpu_count: 76 , os.cpu_count(): 76
        # num_process = multiprocessing.cpu_count() // 2
        # num_process = 2
        # num_process = multiprocessing.cpu_count() - 1
        num_process = 29
        print('num_process:', num_process) # login-icelake.hpc.cam.ac.uk: num_process: 38
        
        p = Pool(num_process) # 理论上这里process的数量可以设置为multiprocessing.cpu_count() - 1, 即76-1=75（机器: login-icelake.hpc.cam.ac.uk）
        for d in data_root_ls_pt:
            print('process ', d, '...')
            time.sleep(1)
            p.apply_async(run_on_videos, args=(d, d, args.level))
            # run_on_videos(
            #     video_dir=d,
            #     data_root=d,
            # )
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')
