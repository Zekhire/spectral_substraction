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
    noise = np.array(np.random.normal(0, noise_level, len(samples)+clear_noise_length), dtype=type(samples[0]))
    zeros = np.zeros(clear_noise_length)
    signal_noised = np.hstack((zeros, samples))
    signal_noised += noise
    return signal_noised


def add_noise_multichannel(samples, clear_noise_length, noise_level=0.1):
    h, w = np.transpose(samples).shape
    signal_noised = np.zeros((h, w+clear_noise_length))
    for i in range(samples.ndim):
        signal_noisedi = add_noise(samples[:, i], clear_noise_length, noise_level)
        signal_noised[i, :] = signal_noisedi
    signal_noised = np.transpose(signal_noised)
    return signal_noised


def add_noise_foyer(samples, clear_noise_length, noise_level=0.005):
    if samples.ndim > 1:
        signal_noised = add_noise_multichannel(samples, clear_noise_length, noise_level)
    else:
        signal_noised = add_noise(samples, clear_noise_length, noise_level)
    return signal_noised


def get_number_of_frames(Nx, N, overlap):
    frames = int(np.ceil(Nx/(N-overlap)))
    return frames


def get_frame(y, clear_noise_end, frames, N, i, overlap):
    padding_size = 0
    if clear_noise_end+i*(N-overlap)+N > len(y):
        yi = y[clear_noise_end+i*(N-overlap):]
        padding_size = N-len(yi)
        zeros = np.zeros(padding_size)
        yi = np.hstack((yi, zeros))
    else:
        yi = y[clear_noise_end+i*(N-overlap): clear_noise_end+i*(N-overlap)+N]
    return yi, padding_size


def generalized_spectral_density_estimation_of_the_noise(y, clear_noise_end, N, general, beta):
    z = y[:clear_noise_end]
    frames = int(clear_noise_end/N)

    if general:
        Z = np.zeros(N)
    else:
        Z = np.zeros(int(N/2)+1)

    for i in range(frames):
        zi = z[i*N: (i+1)*N]
        Zi = np.fft.fft(zi, N)
        Ziabs = np.abs(Zi)
        if general:
            Z += np.power(Ziabs, beta)
        else:
            Z += np.power(Ziabs[:int(N/2)+1], 2)/N

    Z = Z/frames                                 # V
    # show_signal(SZ)
    return Z


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


def window(x, N):
    t = np.arange(0, N)
    xwi = x * (0.5)*(1-np.cos(  np.pi*t/((N-1)/2)  ))
    return xwi


def spectral_substraction(Yi, Zi, alfa, beta):
    denominator = np.power(abs(Yi), beta)
    nominator = denominator - alfa*abs(Zi)
    nominator[nominator<0] = 0
    Xi = np.power(nominator/denominator, 1/beta)*Yi
    return Xi


def insert_frame(xe, xei, clear_noise_end, N, overlap, frames, i, padding_size):
    if i*(N-overlap)+N > len(xe):
        xe[i*(N-overlap): i*(N-overlap)+(N-padding_size)] += xei[:-padding_size]
    else:
        xe[i*(N-overlap): i*(N-overlap)+N] += xei
    return xe


def generalized_spectral_substraction(y, clear_noise_end, N=512, general=True, overlap=257, alfa=1.5, beta=2):
    Zi = generalized_spectral_density_estimation_of_the_noise(y, clear_noise_end, N, general, beta)
    if general is False:
        SZ = Zi

    Nx = len(y)-clear_noise_end
    frames = get_number_of_frames(Nx, N, overlap)
    xe = np.zeros(Nx)

    for i in range(int(frames)):
        print("frame:", i)
        yi, padding_size = get_frame(y, clear_noise_end, frames, N, i, overlap)
        Yi = np.fft.fft(yi, N)                                                      # v
        
        if general:
            Xi = spectral_substraction(Yi, Zi, alfa, beta)
            xei = np.fft.ifft(Xi, N).real                                           # v
            xwi = window(xei, N)
            xei = xwi
        else:
            SYi = power_spectral_density_estimation_of_the_noisy_signal(Yi, N)      
            SXi = power_spectral_density_function_of_the_noiseless_signal(SYi, SZ)
            Ai = create_denoising_filter(SXi, SYi)                                      # v
            Xi = evaluate_denoised_signal(Ai, Yi)                                       # v
            xei = np.fft.ifft(Xi, N).real                                               # v

        xe = insert_frame(xe, xei, clear_noise_end, N, overlap, frames, i, padding_size)
    return xe


def generalized_spectral_substraction_foyer(y, clear_noise_end, N=512, general=True, overlap=257, alfa=2, beta=2):
    if general is False:
        overlap = 0
        alfa = 0
        beta = 0

    if y.ndim > 1:
        # several channels
        h, w = np.transpose(y).shape
        xe = np.zeros((h, w-clear_noise_end))
        for i in range(y.ndim):
            xei = generalized_spectral_substraction(y[:, i], clear_noise_end, N, general, overlap, alfa, beta)
            xe[i,:] = xei
        xe = np.transpose(xe)
    else:
        # one channel
        xe = generalized_spectral_substraction(y, clear_noise_end, N, general, overlap, alfa, beta)
    return xe




if __name__ == "__main__":
    # Audio files paths
    input_path  = "LOTR.wav"
    noisy_path = "noisy.wav"
    output_path = "filtered.wav"

    N = 513                     # length of windows (odd for general method, even for non general)
    general = True              # choose if general or not general method should be used
    overlap = int((N+1)/2)      # overlap length (for general only)
    alfa = 8                    # parameter (for general only)
    beta = 2                    # parameter (for general only)


    # Add noise (You can comment following lines so that
    # this script will only denoise noisy audio)
    x, Fs = load_audios(input_path)
    clear_noise_length = Fs
    y = add_noise_foyer(x, clear_noise_length)
    save_audios(noisy_path, y, Fs)

    # Denoising with spectral_substraction algorithm
    y, Fs = load_audios(noisy_path)
    xe = generalized_spectral_substraction_foyer(y, clear_noise_length, N, general, overlap, alfa, beta)
    save_audios(output_path, xe, Fs)
