import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# load training data from numpy files
X_train = np.load("resized_data.npy")
Y_train = np.load("labels.npy")

# convert string labels to one hot encoding.
# Code from: https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# How to One Hot Encode Sequence Data in Python by Jason Brownlee on July 12, 2017 in Long Short-Term Memory Networks
values = array(Y_train)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y_train = onehot_encoder.fit_transform(integer_encoded)

# reshape X data
X_train = X_train.transpose().reshape(-1, 50*50)
