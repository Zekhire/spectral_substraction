import scipy.io.wavfile as wave
import scipy.optimize
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def load_audios(input_path):
    audio = wave.read(input_path)
    samples = np.array(audio[1], dtype=float)
    Fs = audio[0]
    return samples, Fs


def save_audios(output_paths, z, Fs):
    
    for i in range(len(output_paths)):
        # print("MAX:", max(z[i]))
        wave.write(output_paths[i], Fs, np.array(z)[i])



if __name__ == "__main__":
    printing = True
    show = False
    save = False

    input_paths  = ["audio1_clip.wav"]
    noised_paths = ["noised.wav"]
    output_paths = ["filtered.wav"]

    


