import numpy as np
import os
from mfcc import wav2mfcc
from keras.utils import to_categorical
DATA_PATH = "./data"

# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)

    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

def save_data_to_array(path=DATA_PATH, max_pad_len=130):
    labels, _ ,_ = get_labels(path)
    print("label = " ,labels)




    for wavfile in os.listdir(path ):
        print(wavfile)

    for label in labels:
        print()
        print("--lable = ",label)
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + '/' + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        #wavfiles = [path + '/' + label for wavfile in os.listdir(path + '/' + label)]
        print("////" , wavfiles)
        for wavfile in wavfiles:
            print("--wav = ",wavfile)
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)


get_labels()
save_data_to_array()
