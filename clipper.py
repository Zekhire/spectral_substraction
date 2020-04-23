import scipy.io.wavfile as wave
from matplotlib import pyplot as plt
import numpy as np




def cut_audio_clip(input_path, start, end, output_path):
    audio = wave.read(input_path)
    samples = np.array(audio[1], dtype=float)/(65536/2)
    Fs = audio[0]
    clip_audio = samples[start:start + end*Fs][:,0]
    # n = []
    # print(clip_audio)
    # for i in range(len(audio)):
    #     n.append(i)
    # plt.plot(n, audio)
    # plt.show()
    wave.write(output_path, Fs, clip_audio)


if __name__ == "__main__":
    #input_path = "./audio1_orig.wav"
    input_path = "./KJW_ŚR.wav"
    #output_path = "audio1_clip.wav"
    output_path = "./KJW_ŚR_clip.wav"
    number_of_samples = 5000
    start = 0
    end = 60
    cut_audio_clip(input_path, start, end, output_path)