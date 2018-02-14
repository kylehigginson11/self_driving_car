import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt2
from sklearn.model_selection import train_test_split
import cv2
import glob
import numpy as np

training_data = glob.glob('training_data/*.npz')
# loop through all collected image files

image_array = np.zeros((1, 38400))
label_array = np.zeros((1, 4), 'float')

for npz_file in training_data:
    with np.load(npz_file) as data:
        train_temp = data['train']
        train_labels_temp = data['train_labels']
    image_array = np.vstack((image_array, train_temp))
    label_array = np.vstack((label_array, train_labels_temp))

image_data_x = image_array[1:, :]
label_data_y = label_array[1:, :]

x_train, x_test, y_train, y_test = train_test_split(image_data_x, label_data_y, test_size=0.3)

model = cv2.ml.ANN_MLP_load('neural_networks/neural_network.xml')

diff = []
ratio = []
_, pred = model.predict(x_train)

## The line / model
plt2.scatter(y_train, pred)
plt2.xlabel("True Values")
plt2.ylabel("Predictions")
plt2.show()
