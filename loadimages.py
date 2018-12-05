from skimage import io
import pandas
import numpy as np
import time

timestr = time.strftime("%Y-%m-%d %H:%M%:%S")


def open_image (url):
    image = io.imread(url)
    return image

fields = ["labeled_data"]

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

np.save("data " + timestr, X)

