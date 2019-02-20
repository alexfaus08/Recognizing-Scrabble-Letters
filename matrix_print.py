import matplotlib.pyplot as plt
import numpy
from numpy import ravel, reshape, swapaxes
import scipy.io
from sklearn import svm
from sklearn.metrics import confusion_matrix
from random import sample

def print_mat(cm):
    fig = plt.figure()
    plt.matshow(cm)
    plt.title('Recognizing Scrabble Letters')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig('confusion_matrix' + '.jpg')