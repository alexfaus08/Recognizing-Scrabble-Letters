# Recognizing Scrabble Letters
After labeling images of Scrabble letters on [LabelBox](app.labelbox.com), a [Keras](https://keras.io/) model is fed these images and labels. Then, predictions of future letters can be made. 

## Getting Started
You will need:
* [Python 3.6](https://www.python.org/downloads/release/python-360/)
* [Keras](https://keras.io/) and [Tensorflow](tensorflow.org)
* [NumPy](http://www.numpy.org/)
* [OpenCV-python](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_table_of_contents_setup/py_table_of_contents_setup.html#py-table-of-content-setup)
* [pandas](https://pandas.pydata.org/)
* [scikit-image](http://scikit-image.org/docs/dev/api/skimage.html)
* [sklearn](https://scikit-learn.org/)
* [matplotlib.pyplot](https://matplotlib.org/api/pyplot_api.html)

## File List
* `letters.csv` Downloaded labeled images of Scrabble letters from [LabelBox](app.labelbox.com) 
* `loadimages.py` Converts data (images) from `letters.csv` to a numpy array
* `loadlabels.py` Converts data (labels) from `letters.csv` to a numpy array
* `resize_images.py` Takes the numpy array from `loadimages.py` and resizes each element to 100 by 100 and makes them grayscale. 
* `main.py` creates a [Keras](https://keras.io/) model for recognizing scrabble images

## Author
* **Alexander Faus**
