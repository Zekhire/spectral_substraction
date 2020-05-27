import scipy.io.wavfile as wave
import matplotlib.pyplot as plt
import numpy as np


def load_audios(input_path):
    audio = wave.read(input_path)
    samples = np.array(audio[1], dtype=np.float32)
    Fs = audio[0]
    return samples, Fs

def scaling_down(x):
    x_scaled = x/np.max(abs(x))
    return x_scaled

def save_audios(output_path, samples, Fs, scaling=True):
    if scaling:
        samples_scaled = scaling_down(samples)
    else:
        samples_scaled = samples
    wave.write(output_path, Fs, samples_scaled)

def show_signal(samples):
    t = np.ones(len(samples))
    t = np.cumsum(t)
    t = t-1
    plt.plot(t, samples)
    plt.grid()
    plt.show()


def add_noise(samples, clear_noise_length, noise_level=0.1):
    print(samples)
    noise = np.array(np.random.normal(0, noise_level, len(samples)+clear_noise_length), dtype=type(samples[0]))
    # noise = np.random.randint(0, noise_level, len(samples)+clear_noise_length, np.int16)
    zeros = np.zeros(clear_noise_length)
    signal_noised = np.hstack((zeros, samples))
    signal_noised += noise
    print(signal_noised[clear_noise_length:])
    return signal_noised

def add_noise_multichannel(samples, clear_noise_length, noise_level=0.1):
    h, w = np.transpose(samples).shape
    signal_noised = np.zeros((h, w+clear_noise_length))
    for i in range(samples.ndim):
        signal_noisedi = add_noise(samples[:, i], clear_noise_length, noise_level)
        signal_noised[i, :] = signal_noisedi
    signal_noised = np.transpose(signal_noised)
    return signal_noised

def add_noise_foyer(samples, clear_noise_length, noise_level=0.1):
    if samples.ndim > 1:
        signal_noised = add_noise_multichannel(samples, clear_noise_length, noise_level)
    else:
        signal_noised = add_noise(samples, clear_noise_length, noise_level)
    return signal_noised


def get_frame(y, clear_noise_end, frames, N, i):
    padding_size = 0
    if i==int(frames)-1:
        yi = y[clear_noise_end+i*N:]
        padding_size = N-len(yi)
        zeros = np.zeros(padding_size)
        yi = np.hstack((yi, zeros))
    else:
        yi = y[clear_noise_end+i*N: clear_noise_end+(i+1)*N]
    return yi, padding_size


def power_spectral_density_estimation_of_the_noise(y, clear_noise_end, N):
    z = y[:clear_noise_end]
    frames = int(clear_noise_end/N)
    SZ = np.zeros(int(N/2)+1)
    for i in range(frames):
        zi = z[i*N: (i+1)*N]
        Zi = np.fft.fft(zi, N)
        Ziabs = np.abs(Zi)
        SZi = np.power(Ziabs[:int(N/2)+1], 2)/N
        SZ += SZi
    SZ = SZ/frames                                 # V
    # show_signal(SZ)
    return SZ


def power_spectral_density_estimation_of_the_noisy_signal(Yi, N):
    Yiabs = np.abs(Yi)
    SYi = np.power(Yiabs[:int(N/2)+1], 2)/N
    # show_signal(SYi)
    return SYi


def power_spectral_density_function_of_the_noiseless_signal(SYi, SZ):
    SXi = SYi - SZ
    SXi[SXi<0] = 0
    return SXi


def create_denoising_filter(SXi, SYi, eps=1e-6):
    if eps is not None:
        SYi[SYi<eps] = eps
    Ail = np.sqrt(np.divide(SXi, SYi))
    Air = np.flip(Ail[1:-1], 0)
    Ai = np.hstack((Ail, Air))
    return Ai


def evaluate_denoised_signal(Ai, Yi):       # V
    Xi = Ai*Yi
    return Xi


def power_spectral_substraction(y, clear_noise_end, N=512, general=True, alfa=1.5, beta=2):
    SZ = power_spectral_density_estimation_of_the_noise(y, clear_noise_end, N)

    frames = len(y[clear_noise_end:])/N
    if int(frames) - frames < 0:
        frames = int(frames) + 1

    xe = np.zeros(len(y)-clear_noise_end)

    for i in range(int(frames)):
        # print("frame:", i)
        yi, padding_size = get_frame(y, clear_noise_end, frames, N, i)
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



def generalized_spectral_density_estimation_of_the_noise(y, clear_noise_end, N):
    z = y[:clear_noise_end]
    frames = int(clear_noise_end/N)
    Z = np.zeros(N)
    for i in range(frames):
        zi = z[i*N: (i+1)*N]
        Zi = np.fft.fft(zi, N)
        Ziabs = np.abs(Zi)
        Z += Ziabs
    Z = Z/frames                                 # V
    # show_signal(SZ)
    return Z


def get_frame_overlap(y, clear_noise_end, frames, N, i, overlap):
    padding_size = 0
    if clear_noise_end+i*(N-overlap)+N > len(y):
        print("wszedłem")
        yi = y[clear_noise_end+i*(N-overlap):]
        padding_size = N-len(yi)
        zeros = np.zeros(padding_size)
        yi = np.hstack((yi, zeros))
    else:
        yi = y[clear_noise_end+i*(N-overlap): clear_noise_end+i*(N-overlap)+N]
    return yi, padding_size


def get_number_of_frames(Nx, N, overlap):
    frames = int(np.ceil(Nx/(N-overlap)))
    return frames

def window(x, N):
    t = np.arange(0, N)
    xwi = x * (0.5)*(1-np.cos(  np.pi*t/((N-1)/2)  ))
    return xwi

def generalized_spectral_substraction(Yi, Zi, alfa, beta):
    denominator = np.power(abs(Yi), beta)
    nominator = denominator - alfa*np.power(abs(Zi), beta)
    nominator[nominator<0] = 0
    Xi = np.power(nominator/denominator, 1/beta)*Yi
    return Xi

def generalized_spectral_substraction_foyer(y, clear_noise_end, N=513, overlap=257, alfa=1.5, beta=2):
    Zi = generalized_spectral_density_estimation_of_the_noise(y, clear_noise_end, N)

    Nx = len(y)-clear_noise_end
    frames = get_number_of_frames(Nx, N, overlap)

    xe = np.zeros(Nx)

    for i in range(int(frames)):
        print("frame:", i, frames, N, Nx, (N-overlap)*frames)

        yi, padding_size = get_frame_overlap(y, clear_noise_end, frames, N, i, overlap)
        Yi = np.fft.fft(yi, N)                                                              # v
        Xi = generalized_spectral_substraction(Yi, Zi, alfa, beta)
        xei = np.fft.ifft(Xi, N).real                                                       # v
        xwi = window(xei, N)
        if clear_noise_end+i*(N-overlap)+N > len(y):
            xe[i*(N-overlap): i*(N-overlap)+(N-padding_size)] += xwi[:-padding_size]
        else:
            # print(xe[i*(N-overlap): i*(N-overlap)+2*N].shape, xwi.shape)
            xe[i*(N-overlap): i*(N-overlap)+N] += xwi

    return xe


def spectral_substraction_foyer(y, clear_noise_end, N=512, general=True, overlap=257, alfa=2, beta=2):
    if y.ndim > 1:
        # several channels
        h, w = np.transpose(y).shape
        xe = np.zeros((h, w-clear_noise_end))
        for i in range(y.ndim):
            if general:
                xei = generalized_spectral_substraction_foyer(y[:, i], clear_noise_end, N, overlap, alfa, beta)
            else:
                xei = power_spectral_substraction(y[:, i], clear_noise_end, N)
            xe[i,:] = xei
        xe = np.transpose(xe)
    else:
        # one channel
        if general:
            xe = generalized_spectral_substraction_foyer(y, clear_noise_end, N, overlap, alfa, beta)
        else:
            xe = power_spectral_substraction(y, clear_noise_end, N)
    return xe




if __name__ == "__main__":
    # printing = True
    # show = False
    # save = False


    input_path  = "KJW_ŚR_stereo.wav"
    noisy_path = "noisy.wav"
    output_path = "filtered.wav"

    N = 513
    general = True
    overlap = int((N+1)/2)
    alfa = 6
    beta = 2.5

    x, Fs = load_audios(input_path)
    clear_noise_length = Fs

    y = add_noise_foyer(x, clear_noise_length)
    save_audios(noisy_path, y, Fs)

    y, Fs = load_audios(noisy_path)
    xe = spectral_substraction_foyer(y, clear_noise_length, N, general, overlap, alfa, beta)
    save_audios(output_path, xe, Fs)
