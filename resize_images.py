import numpy as np
import cv2

HEIGHT = 50
WIDTH = 50
def resize_image(image):
    # resize and make images grayscale
    newimg = cv2.resize(image, (HEIGHT, WIDTH))
    newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)
    return newimg


def resize_array(data):
    resized = []
    for image in data:
        resized.append(resize_image(image))

    return np.asarray(resized)

X = np.load("testing_data.npy")
resized = []
i = 0

for image in X:
    resized.append(resize_image(image))

np.asarray(resized)
np.save("resized_test_data", resized)






