import numpy as np
import cv2

HEIGHT = 50
WIDTH = 50
def resize_image(image):
    # resize and make images grayscale
    newimg = cv2.resize(image, (HEIGHT, WIDTH))
    newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)
    return newimg


X = np.load("data 2018-12-03 19:17:22.npy")
resized = []
i = 0

for image in X:
    resized.append(resize_image(image))

np.asarray(resized)
np.save("resized_data", resized)






