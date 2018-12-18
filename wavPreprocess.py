# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np

import imageio
from skimage import util


filename = "./data/79/79(3).wav"
rate, audio = wavfile.read(filename)
print(rate)
audio = np.mean(audio, axis=1)
#print(len(audio))

audio = np.trim_zeros(audio)
N = audio.shape[0]
L = N / rate
#print(L)

t = np.linspace(0, L, len(audio))
plt.figure(figsize=(12,4))

plt.plot(t, audio)

M = 1024
filename = filename.replace(".wav", "")

freqs, times, Sx = signal.spectrogram(audio, fs=rate, window='hanning',
                                      nperseg=1024, noverlap=M - 100,
                                      detrend=False, scaling='spectrum')
print(Sx.shape)

f, ax = plt.subplots(figsize=(12, 4))
ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='viridis')
ax.set_ylabel('Frequency [kHz]')
ax.set_xlabel('Time [s]');
plt.savefig(filename + '.png')
