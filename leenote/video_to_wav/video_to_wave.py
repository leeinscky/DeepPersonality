import glob
import subprocess
import os
from tqdm import tqdm
import zipfile
import librosa
from pathlib import Path
import numpy as np

""" 
class ChaLearn16AudioExtract:
    @staticmethod
    def video2wave_train(zipfile_dir):
        # Running a loop through all the zipped training file to extract all .wav audio files
        for i in range(1, 76):
            if i < 10:
                zipfilename = 'training80_0' + str(i) + '.zip'
            else:
                zipfilename = 'training80_' + str(i) + '.zip'
            # Accessing the zipfile i
            archive = zipfile.ZipFile(zipfile_dir + zipfilename, 'r')
            zipfilename = zipfilename.split('.zip')[0]
            # archive.extractall('unzippedData/'+zipfilename)
            for file_name in archive.namelist():
                file_name = (file_name.split('.mp4'))[0]
                try:
                    if not os.path.exists('../../../datasets/VoiceData/trainingData/'):
                        os.makedirs('../../../datasets/VoiceData/trainingData/')
                except OSError:
                    print('Error: Creating directory of data')
                command = "ffmpeg -i unzippedData/{}/{}.mp4 -ab 320k -ac 2 -ar 44100 -vn VoiceData/trainingData/{}.wav"\
                    .format(zipfilename, file_name, file_name)
                subprocess.call(command, shell=True)

    @staticmethod
    def video2wave_val(zipfile_dir):
        for i in range(1, 26):
            if i < 10:
                zipfilename = 'validation80_0' + str(i) + '.zip'
            else:
                zipfilename = 'validation80_' + str(i) + '.zip'
            # Accessing the zipfile i
            archive = zipfile.ZipFile(zipfile_dir + zipfilename, 'r')
            zipfilename = zipfilename.split('.zip')[0]
            # archive.extractall('unzippedData/'+zipfilename)
            for file_name in archive.namelist():
                file_name = (file_name.split('.mp4'))[0]
                try:
                    if not os.path.exists('../../../datasets/VoiceData/validationData/'):
                        os.makedirs('../../../datasets/VoiceData/validationData/')
                except OSError:
                    print('Error: Creating directory of data')
                command = "ffmpeg -i unzippedData/{}/{}.mp4 -ab 320k -ac 2 -ar 44100 -vn VoiceData/validationData/{}.wav"\
                    .format(zipfilename, file_name, file_name)
                subprocess.call(command, shell=True)

    @staticmethod
    def video2wave_tst(data_dir):
        for video in os.listdir(data_dir):
            file_name = video.split(".mp4")[0]
            if not os.path.exists("VoiceData/testData"):
                os.makedirs("VoiceData/testData")
            command = f"ffmpeg -i {data_dir}/{video} -ab 320k -ac 2 -ar 44100 -vn VoiceData/testData/{file_name}.wav"
            subprocess.call(command, shell=True)


def chalearn21_audio_extract_ffmpeg(dir_path):
    path = Path(dir_path)
    mp4_ls = path.rglob("./*/*.mp4")
    for mp4 in mp4_ls:
        parent_dir, name = mp4.parent, mp4.stem
        cmd = f"ffmpeg -i {mp4} -ab 320k -ac 2 -ar 44100 -vn {parent_dir}/{name}.wav"
        subprocess.call(cmd, shell=True)

def chalearn21_audio_process(dir_path):
    wav_path_ls = glob.glob(f"{dir_path}/*/*.wav")
    for wav_path in tqdm(wav_path_ls):
        try:
            wav_ft = librosa.load(wav_path, 16000)[0][None, None, :]  # output_shape = (1, 1, 244832)
            wav_name = wav_path.replace(".wav", ".npy")
            np.save(wav_name, wav_ft)
        except Exception:
            print("error:", wav_path)

"""
 
def audio_extract_ffmpeg(dir_path, output_dir=None):
    path = Path(dir_path)
    format_str = "./*.mp4"
    mp4_ls = path.rglob(format_str)
    for mp4 in mp4_ls:
        name = mp4.stem
        if output_dir is None:
            parent_dir = mp4.parent,
        else:
            os.makedirs(output_dir, exist_ok=True)
            parent_dir = output_dir
        print(f'命令：ffmpeg -i {mp4} -ab 320k -ac 2 -ar 44100 -vn {parent_dir}/{name}.wav')
        # cmd = f"ffmpeg -i {mp4} -ab 320k -ac 2 -ar 44100 -vn {parent_dir}/{name}.wav"
        cmd = f"/home/zl525/tools/ffmpeg-git-20220910-amd64-static/ffmpeg -i {mp4} -ab 320k -ac 2 -ar 44100 -vn {parent_dir}/{name}.wav"
        subprocess.call(cmd, shell=True)
        
    """ 
        -ab 320k:
            ab是audio bitrate的缩写(Audio bit rate in kbit/s), 意思是指定音频的比特率为320k，比特率越高，音质越好，但是文件也越大，一般来说，音频的比特率在32k-320k之间就可以了，
        -ac 2:
            ac是audio channel的缩写, 意思是指定声道数为2，即立体声，如果是单声道的话，就是1，如果是4声道的话，就是4
        -ar 44100:
            ar是audio rate的缩写(Audio sampling rate in Hz), 意思是指定采样率为44100，即每秒采集44100个点，采样率越高，音质越好，但是文件也越大，一般来说，音频的采样率在44100-48000之间就可以了， 
            采样频率一般共分为11025Hz、22050Hz、24000Hz、44100Hz、48000Hz五个等级，11025Hz能达到AM调幅广播的声音品质，而22050Hz和24000HZ能达到FM调频广播的声音品质，44100Hz则是理论上的CD音质界限，48000Hz则更加精确一些。
        -vn:
            {parent_dir}/{name}.wav 意思是指定输出的文件名为{name}.wav, vn是video none的缩写，意思是指定不要输出视频，只要输出音频

    参考资料：
        https://www.zhihu.com/question/21896882、https://baike.baidu.com/item/%E9%9F%B3%E9%A2%91%E9%87%87%E6%A0%B7%E7%8E%87/9023551
    """ 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ffmpeg audio extraction")
    parser.add_argument("-v", "--video-dir", default=None, type=str, help="path to video directory")
    parser.add_argument("-o", "--output-dir", default=None, type=str, help="path to save processed videos")
    args = parser.parse_args()

    # dir_path = "/home/rongfan/05-personality_traits/DeepPersonality/datasets/chalearn2021/test/talk_test"
    # chalearn21_audio_extract_ffmpeg(dir_path)
    # chalearn21_audio_process(dir_path)
    audio_extract_ffmpeg(dir_path=args.video_dir, output_dir=args.output_dir)
