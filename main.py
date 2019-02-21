import numpy as np


from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from keras import losses
import matrix_print
from keras.models import load_model
import convert_arrays

X = np.load("data.npy")
Y = np.load("labels.npy")
X, Y = convert_arrays.convert_arrays(X, Y)
X_train = X[:2000]
Y_train = Y[:2000]

X_test = X[2000:]
Y_test = Y[2000:]
# set the random seed
np.random.seed(3520)

# Set/get relevant parameters
num_inputs = 2500  # from X_train.shape
num_outputs = 26  # 26 letters plus skip
batch_size = 500
# learning_rate = .01
epochs = 400

# 2500
model = Sequential()
# Hidden Layers
model.add(Dense(units=500, activation='relu', input_dim=num_inputs))
model.add(Dense(units=500, activation='relu', input_dim=num_inputs))
# Output Layer
model.add(Dense(units=num_outputs, activation='softmax'))
# Set learning rate
# sgd = keras.optimizers.SGD(lr=learning_rate)
model.compile(loss=losses.mean_squared_error, optimizer="adam", metrics=['accuracy'])
# Print Summary
model.summary()

history = model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: {0} %".format(score[1] * 100))

model.save('my_model_tweaked.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model_tweaked.h5')

# run training data through built NN
pred = model.predict(X_test)

Y_train = np.argmax(Y_train, axis=1)
pred = np.argmax(pred, axis=1)

# cm = confusion_matrix(Y_test, pred)

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


# print_cm(cm, labels)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
# matrix_print.print_mat(cm)