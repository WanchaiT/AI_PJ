import numpy as np
from get_labels import get_labels
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    print("eieiei")
    labels, indices, _ = get_labels("./data")

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    u=0
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))
        u+=1
        print("eqwfweqd" ,label)

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)
