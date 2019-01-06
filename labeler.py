import glob
import cv2
import numpy as np
files = glob.glob("images/*.png")
x = cv2.imread(files[0])

X = []
i = 0
for file in files:
    img = cv2.imread(file)
    cv2.imshow(str(i), img)
    cv2.waitKey(0)
    label = input()
    i += 1
    X.append(label)
    print(X)
