from skimage import io
import cv2
import pandas
import numpy as np

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
for url in urls:
    X.append(open_image(url))
    print(i)
    i += 1

X = np.asarray(X)

np.save("data", X)

