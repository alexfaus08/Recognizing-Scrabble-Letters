import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

# load training data from numpy files
X_train = np.load("resized_data.npy")
Y_train = np.load("labels.npy")
# all the indices of the skipped data
skip = np.where(Y_train == 'skip')
Y_train = np.delete(Y_train, skip, axis=0)

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

# delete data that is labeled as skip
X_train = np.delete(X_train, skip, axis=0)
X_train = np.true_divide(X_train, 255)

# set the random seed
np.random.seed(3520)

# Set/get relevant parameters
num_inputs = 2500  # from X_train.shape
num_outputs = 26  # 26 letters plus skip
batch_size = 500
learning_rate = .01
epochs = 1000

# 2500
model = Sequential()
# Hidden Layers
model.add(Dense(units=500, activation='relu', input_dim=num_inputs))
model.add(Dense(units=500, activation='relu', input_dim=num_inputs))
# Output Layer
model.add(Dense(units=num_outputs, activation='softmax'))
# Set learning rate
sgd = keras.optimizers.SGD(lr=learning_rate)
model.compile(loss='mean_squared_error', optimizer="adam", metrics=['accuracy'])
# Print Summary
model.summary()

history = model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2)

score = model.evaluate(X_train, Y_train, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: {0} %".format(score[1] * 100))

# run training data through built NN
pred = model.predict(X_train)

Y_train = np.argmax(Y_train, axis=1)
pred = np.argmax(pred, axis=1)

cm = confusion_matrix(Y_train, pred)

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
          's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


# Code taken from https://gist.github.com/zachguo/10296432
# Author Zach Guo user @zachguo on GitHub
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


print_cm(cm, labels)

plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()