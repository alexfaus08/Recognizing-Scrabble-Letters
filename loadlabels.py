import pandas
import numpy as np


def extract_label(label):
    if (label[0] == '{'):
        # it is a letter
        # the 17th position of the str is the letter
        return label[17]
    else:
        # it is a skip
        return 'skip'


fields = ["Label"]

df = pandas.read_csv('letters.csv', skipinitialspace=True, usecols=fields)

print(df.Label)

labels = df.Label

Y = []

for label in labels:
    Y.append(extract_label(label))

print(len(Y))

Y = np.asarray(Y)
print(Y)
np.save("labels", Y)
