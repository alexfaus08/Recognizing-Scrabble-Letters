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
from keras.models import model_from_json
import os

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

# set the random seed
np.random.seed(3520)

# Set/get relevant parameters
num_inputs = 2500  # from X_train.shape
num_outputs = 27  # 26 letters plus skip
batch_size = 100
learning_rate = .02
epochs = 100

# 2500
model = Sequential()
# Hidden Layers
model.add(Dense(units=500, activation='sigmoid', input_dim=num_inputs))
model.add(Dense(units=500, activation='sigmoid', input_dim=num_inputs))
model.add(Dense(units=500, activation='sigmoid', input_dim=num_inputs))
# Output Layer
model.add(Dense(units=num_outputs, activation='sigmoid'))
# Set learning rate
sgd = keras.optimizers.SGD(lr=learning_rate)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
# Print Summary
model.summary()

history = model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2)

score = model.evaluate(X_train, Y_train, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: {0} %".format(score[1] * 100))

# code taken from: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# Save and Load Your Keras Deep Learning Models
# by Jason Brownlee on June 13, 2016 in Deep Learning

# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")