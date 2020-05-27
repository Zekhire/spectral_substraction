import numpy as np


x = np.array([1,2,3,4,5])
print(np.fft.fft(x, 5))
print(np.fft.fft(x, 8))


for i in range(5):
    print(i)