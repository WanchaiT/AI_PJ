import numpy as npf
import librosa

def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    print("****", mfcc)

    pad_width = max_pad_len - mfcc.shape[1]
    print("****" , mfcc.shape[1])
    print("****", pad_width)
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc
