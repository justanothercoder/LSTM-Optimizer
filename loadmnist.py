import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

def load_data(test_size=0.2):
    mnist = fetch_mldata('MNIST original')
    X_train, X_val, y_train, y_val = train_test_split(mnist.data, mnist.target, test_size=test_size)

    X_train = X_train.astype(np.float32) / 255.
    X_val = X_val.astype(np.float32) / 255.

    return X_train, y_train, X_val, y_val
