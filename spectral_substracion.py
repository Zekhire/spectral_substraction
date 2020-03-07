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


def save_audios(output_paths, samples, Fs):
    wave.write(output_paths, Fs, samples)


def show_signal(samples):
    t = np.ones(len(samples))
    t = np.cumsum(t)
    t = t-1
    plt.plot(t, samples)
    plt.show()


def add_noise(samples, Fs, noise):
    noise = np.random.normal(0, noise, len(samples)+Fs)
    zeros = np.zeros(Fs)
    signal_noised = np.hstack((zeros, samples))
    signal_noised += noise
    # show_signal(signal_noised)
    return signal_noised



def power_spectral_density_estimation_of_the_noise(y, Fs, N):
    z = y[:Fs]
    frames = int(Fs/N)
    SZ = np.zeros(int(N/2))
    for i in range(frames):
        zi = z[i*N: (i+1)*N]
        Zi = np.fft.fft(zi, N)
        Ziabs = np.abs(Zi)
        SZi = np.power(Ziabs[:int(N/2)],2)/N
        SZ += SZi
    SZ = SZi/frames
    return SZ


def power_spectral_density_estimation_of_the_noisy_signal(Yi, N):
    Yiabs = np.abs(Yi)
    SYi = np.power(Yiabs[:int(N/2)],2)/N
    return SYi

def power_spectral_density_function_of_the_noiseless_signal(SZ, SYi):
    SXi = SYi - SZ
    SXi[SXi<0] = 0
    return SXi


def create_denoising_filter(SXi, SYi):
    Ail = np.power(np.divide(SXi, SYi), 0.5)
    Air = np.flip(Ail, 0)
    Ai = np.hstack((Air, Ail))
    return Ai

def evaluate_denoised_signal(Ai, Yi):
    Xi = Ai*Yi
    return Xi


if __name__ == "__main__":
    printing = True
    show = False
    save = False

    noise = 0.01
    N = 512

    input_path  = "audio1_clip.wav"
    noisy_path = "noisy.wav"
    output_path = "filtered.wav"

    x, Fs = load_audios(input_path)
    y = add_noise(x, Fs, noise)
    save_audios(noisy_path, y, Fs)
    SZ = power_spectral_density_estimation_of_the_noise(y, Fs, N)

    frames = len(y[Fs:])/N
    if int(frames) - frames < 0:
        frames = int(frames) + 1
    xe = np.zeros(len(y)-Fs)

    for i in range(int(frames)):
        print("frame:", i)
        if i==int(frames)-1:
            yi = y[Fs+i*N:]
            padding_size = N-len(yi)
            zeros = np.zeros(padding_size)
            yi = np.hstack((yi, zeros))
        else:
            yi = y[Fs+i*N: Fs+(i+1)*N]

        Yi = np.fft.fft(yi, N)                                                  # v
        SYi = power_spectral_density_estimation_of_the_noisy_signal(Yi, N)      
        SXi = power_spectral_density_function_of_the_noiseless_signal(SZ, SYi)
        Ai = create_denoising_filter(SXi, SYi)                                    # v
        Xi = evaluate_denoised_signal(Ai, Yi)                                    # v
        xei = np.fft.ifft(Xi, N).real                                           # v

        if i==int(frames)-1:
            xe[i*N: i*N+(N-padding_size)] = xei[:-padding_size]
        else:
            xe[i*N: (i+1)*N] = xei
    
    samples = x -xe

    show_signal(samples)

    save_audios(output_path, xe, Fs)




