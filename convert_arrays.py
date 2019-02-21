import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import resize_images

def convert_arrays(data, labels):
    values = np.array(labels)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    del_these = np.where(integer_encoded == 0)
    integer_encoded = np.delete(integer_encoded, del_these)
    data = np.delete(data, del_these)
    data = resize_images.resize_array(data)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    labels = onehot_encoder.fit_transform(integer_encoded)
    # reshape X data
    data = data.transpose().reshape(-1, 50 * 50)
    data = data.transpose().reshape(-1, 50 * 50)
    data = np.true_divide(data, 255)
    return data, labels
