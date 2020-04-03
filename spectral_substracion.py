import scipy.io.wavfile as wave
import matplotlib.pyplot as plt
import numpy as np


def load_audios(input_path):
    audio = wave.read(input_path)
    samples = np.array(audio[1], dtype=float)
    Fs = audio[0]
    return samples, Fs


def save_audios(output_path, samples, Fs):
    wave.write(output_path, Fs, samples)


def show_signal(samples):
    t = np.ones(len(samples))
    t = np.cumsum(t)
    t = t-1
    plt.plot(t, samples)
    plt.grid()
    plt.show()


def add_noise(samples, Fs, noise_level=0.01):
    noise = np.random.normal(0, noise_level, len(samples)+Fs)
    # print(min(noise), max(noise))
    # exit()
    zeros = np.zeros(Fs)
    signal_noised = np.hstack((zeros, samples))
    signal_noised += noise
    show_signal(signal_noised)
    return signal_noised


def get_frame(y, frames, N, i):
    padding_size = 0
    if i==int(frames)-1:
        yi = y[Fs+i*N:]
        padding_size = N-len(yi)
        zeros = np.zeros(padding_size)
        yi = np.hstack((yi, zeros))
    else:
        yi = y[Fs+i*N: Fs+(i+1)*N]
    return yi, padding_size


def power_spectral_density_estimation_of_the_noise(y, Fs, N):
    z = y[:Fs]
    frames = int(Fs/N)
    SZ = np.zeros(int(N/2))
    for i in range(frames):
        zi = z[i*N: (i+1)*N]
        Zi = np.fft.fft(zi, N)
        # print(Zi[:3])
        Ziabs = np.abs(Zi)
        # print(Ziabs[:3])
        SZi = np.power(Ziabs[:int(N/2)], 2)/N
        # print(SZi[:3])
        # exit()
        SZ += SZi
    SZ = SZ/frames                                 # V
    # show_signal(SZ)
    return SZ


def power_spectral_density_estimation_of_the_noisy_signal(Yi, N):
    Yiabs = np.abs(Yi)
    SYi = np.power(Yiabs[:int(N/2)], 2)/N
    # show_signal(SYi)
    return SYi


def power_spectral_density_function_of_the_noiseless_signal(SYi, SZ):
    SXi = SYi - SZ
    SXi[SXi<0] = 0
    return SXi


def create_denoising_filter(SXi, SYi):
    Ail = np.sqrt(np.divide(SXi, SYi))
    Air = np.flip(Ail, 0)
    Ai = np.hstack((Ail, Air))
    # show_signal(Ai)
    return Ai


def evaluate_denoised_signal(Ai, Yi):       # V
    Xi = Ai*Yi
    # print(Ai[:5])
    # print(Yi[:5])
    # print(Xi[:5])
    # exit()

    return Xi


def power_spectral_substraction(y, Fs, N=512):
    SZ = power_spectral_density_estimation_of_the_noise(y, Fs, N)
    # N1 = N
    frames = len(y[Fs:])/N
    if int(frames) - frames < 0:
        frames = int(frames) + 1

    xe = np.zeros(len(y)-Fs)

    for i in range(int(frames)):
        print("frame:", i)
        yi, padding_size = get_frame(y, frames, N, i)
        Yi = np.fft.fft(yi, N)                                                      # v
        SYi = power_spectral_density_estimation_of_the_noisy_signal(Yi, N)      
        SXi = power_spectral_density_function_of_the_noiseless_signal(SYi, SZ)
        Ai = create_denoising_filter(SXi, SYi)                                      # v
        Xi = evaluate_denoised_signal(Ai, Yi)                                       # v
        xei = np.fft.ifft(Xi, N).real                                               # v
        if i == int(frames)-1:
            xe[i*N: i*N+(N-padding_size)] = xei[:-padding_size]
        else:
            xe[i*N: (i+1)*N] = xei

    return xe


if __name__ == "__main__":
    # printing = True
    # show = False
    # save = False

    input_path  = "audio1_clip.wav"
    noisy_path = "noisy.wav"
    output_path = "filtered.wav"

    x, Fs = load_audios(input_path)
    y = add_noise(x, Fs)
    save_audios(noisy_path, y, Fs)


    y, Fs = load_audios(noisy_path)
    xe = power_spectral_substraction(y, Fs, 1024)
    save_audios(output_path, xe, Fs)
