import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import pandas
import numpy as np
from skimage import io
import time


# loads testing data

def extract_label(label):
    if (label[0] == '{'):
        # it is a letter
        # the 17th position of the str is the letter
        return label[17]
    else:
        # it is a skip
        return "skip"

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

fields = ["label"]

df = pandas.read_csv(file_path, skipinitialspace=True, usecols=fields)

print(df.label)

labels = df.label

Y = []

# for label in labels:
#     Y.append(extract_label(label))

# print(len(Y))



timestr = time.strftime("%Y-%m-%d %H:%M%:%S")


def open_image(url):
    image = io.imread(url)
    return image

fields = ["data"]

df = pandas.read_csv(file_path, skipinitialspace=True, usecols=fields)
print(df.keys())
print(df.data)

urls = df.data

X = []
i = 0

# for url in urls:
#     X.append(open_image(url))
#     print(i)
#     i += 1
#
i = 0
for u in range(0, len(urls)):
    print(i)
    X.append(open_image(urls[u]))
    Y.append(extract_label(labels[u]))
    i+=1



X = np.asarray(X)
Y = np.asarray(Y)
skipped = np.where(Y == "skip")
X = np.delete(X, skipped)
Y = np.delete(Y, skipped)



np.save("labels", Y)

np.save("data", X)





