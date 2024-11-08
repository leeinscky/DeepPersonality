import os
import librosa
import opensmile
import numpy as np
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import glob


class RawAudioProcessor():

    def __init__(self, mode, aud_dir, save_to):
        self.mode = mode
        self.saved_file = save_to
        os.makedirs(save_to, exist_ok=True)
        if mode == "opensmile":
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        self.aud_file_ls = list(glob.glob(f"{aud_dir}/*.wav"))

    def __getitem__(self, idx):
        wav_file_path = self.aud_file_ls[idx]
        video_name = os.path.basename(wav_file_path)  # .replace(".wav", "")
        if self.mode == "librosa":
            self.librosa_extract(wav_file_path, video_name)
        elif self.mode == "logfbank":
            self.logfbank_extract(wav_file_path, video_name)
        elif self.mode == "opensmile":
            self.opensmile_extract(wav_file_path, video_name)

    def __len__(self):
        return len(self.aud_file_ls)

    def librosa_extract(self, wav_file_path, video_name):
        r"""Extract audio features using librosa
        librosa.load函数返回的是音频信号的样本值y以及采样率sampling rate。这里使用[0]选取了返回的音频信号y，并使用[None, None, :]对其进行了reshape操作。
        [None, None, :]将数据在第1个维度和第2个维度上各增加了一个维度，使得音频数据变为[1, 1, len(y)]的三维数组。这是为了与神经网络模型的输入要求相符。
        总体而言，这段代码将一个音频文件读取后进行了预处理并保存为.npy文件。
        """
        try:
            # sample rate 16000 Hz
            wav_ft = librosa.load(wav_file_path, 16000)[0][None, None, :]  # output_shape = (1, 1, 244832) 16000指的是采样率，即每秒钟采集的样本数=16000(16000 samples are recorded per second.) librosa.load函数可以读取音频文件 [None, None, :]表示增加两个维度，第一个维度为1，第二个维度为1，第三个维度为xxx
            # wav_ft = librosa.load(wav_path, 3279)[0][None, None, :]  # output_shape = (1, 1, 50176)

            np.save(f"{self.saved_file}/{video_name}.npy", wav_ft)
        except Exception:
            print("error:", wav_file_path)

    def logfbank_extract(self, wav_file_path, video_name):
        try:
            (rate, sig) = wav.read(wav_file_path)
            fbank_feat = logfbank(sig, rate)  # fbank_feat.shape = (3059,26)
            a = fbank_feat.flatten()
            single_vec_feat = a.reshape(1, -1)  # single_vec_feat.shape = (1,79534)
            np.save(f"{self.saved_file}/{video_name}.npy", single_vec_feat)
        except Exception:
            print("error:", wav_file_path)

    def opensmile_extract(self, wav_file_path, video_name):
        try:
            out = self.smile.process_file(wav_file_path)
            arr = np.array(out)
            np.save(f"{self.saved_file}/{video_name}.npy", arr)
        except Exception:
            print("error:", wav_file_path)

    @staticmethod
    def processed_files(save_to):
        processed_file = os.listdir(save_to)
        file_name = [item.replace(".npy", "") for item in processed_file]
        return file_name


def audio_process(mode, aud_dir, saved_dir):
    from tqdm import tqdm

    processor = RawAudioProcessor(mode, aud_dir, saved_dir)
    for idx in tqdm(range(len(processor))):
        processor[idx]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Raw audio processor")
    parser.add_argument("-m", "--mode", default="librosa", type=str, help="audio processing methods")
    parser.add_argument("-a", "--audio-dir", default=None, type=str, help="path to audio dir")
    parser.add_argument("-o", "--output-dir", default=None, type=str, help="path to save processed audio files")
    args = parser.parse_args()

    audio_process(args.mode, args.audio_dir, args.output_dir)


