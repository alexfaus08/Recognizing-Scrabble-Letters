from skimage import io
import cv2
import pandas
import numpy as np
import time

timestr = time.strftime("%Y-%m-%d %H:%M%:%S")


def open_image (url):
    image = io.imread(url)
    return image

fields = ["labeled_data", "Label"]

df = pandas.read_csv('letters.csv', skipinitialspace=True, usecols=fields)
print(df.keys())
print(df.labeled_data)

urls = df.labeled_data

X = []
i = 0
try:
    for url in urls:
        X.append(open_image(url))
        print(i)
        i += 1
except:
    X = np.asarray(X)

    np.save("data " + timestr, X)

X = np.asarray(X)

np.save("data " + timestr, X)

