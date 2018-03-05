import cv2
import numpy as np
import glob
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
from datetime import datetime

# Configure logger
logging.basicConfig(filename='/var/log/driverless_car/driverless_car_training.log', level=logging.DEBUG,
                    format="%(asctime)s:%(levelname)s:%(message)s")

OUTPUT_LAYER_SIZE = 3
IMAGE_PIXELS = 38400  # 320 * (240/2)


class TrainMLP:

    def __init__(self, net_name):

        self.net_name = net_name
        logging.info("Training MLP ...")
        self.start_time = datetime.now()
        logging.info('Loading training data...')
        # load training data
        self.image_array = np.zeros((1, IMAGE_PIXELS))
        self.label_array = np.zeros((1, OUTPUT_LAYER_SIZE), 'float')
        self.training_data = glob.glob('training_data/*.npz')
        self.load_training_data()

    def load_training_data(self):
        try:
            # loop through all collected image files
            for npz_file in self.training_data:
                with np.load(npz_file) as data:
                    train_temp = data['train']
                    train_labels_temp = data['train_labels']
                self.image_array = np.vstack((self.image_array, train_temp))
                self.label_array = np.vstack((self.label_array, train_labels_temp))

            image_data_x = self.image_array[1:, :]
            label_data_y = self.label_array[1:, :]
            logging.info('Image array shape: ' + str(image_data_x.shape))
            logging.info('Label array shape: ' + str(label_data_y.shape))

            end_time = datetime.now()
            load_image_time = end_time - self.start_time
            logging.info('Loading image duration:' + str(load_image_time.seconds) + 'seconds')
            self.create_mlp(image_data_x, label_data_y)
        except FileNotFoundError:
            logging.error("Cant find training data!")
            sys.exit()

    def create_mlp(self, images, labels):
        # train test split, can use random_state=42 to set a seed here, splits data so can calculate accuracy
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3)

        # set start time
        train_start_time = datetime.now()

        # create artificial neural network
        ann_mlp = cv2.ml.ANN_MLP_create()
        ann_mlp.setLayerSizes(np.int32([IMAGE_PIXELS, 32, OUTPUT_LAYER_SIZE]))
        ann_mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)
        ann_mlp.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        ann_mlp.setBackpropMomentumScale(0.0)
        # set weight between each neuron
        ann_mlp.setBackpropWeightScale(0.001)
        ann_mlp.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001))

        logging.info('Training MLP ...')
        train_ann = ann_mlp.train(np.float32(x_train), cv2.ml.ROW_SAMPLE, np.float32(y_train))

        # set end time
        train_end_time = datetime.now()
        train_time = (train_end_time - train_start_time)
        logging.info('Training duration: ' + str(train_time.seconds) + 'seconds')
        self.save_mlp(x_train, x_test, y_train, y_test, ann_mlp)

    def save_mlp(self, x_train, x_test, y_train, y_test, ann_mlp):

        # calculate train accuracy
        _train, train_prediction = ann_mlp.predict(x_train)

        train_rate = accuracy_score(y_train.argmax(-1), train_prediction.argmax(axis=-1))
        logging.info('Train accuracy: {}'.format(train_rate))

        # calculate test accuracy
        _test, test_prediction = ann_mlp.predict(x_test)

        test_rate = accuracy_score(y_test.argmax(-1), test_prediction.argmax(axis=-1))
        logging.info('Test accuracy: {}'.format(test_rate))

        # save model
        ann_mlp.save('neural_networks/' + self.net_name + '_neural_network.xml')
        logging.info("Model Saved")


if __name__ == '__main__':
    try:
        TrainMLP(sys.argv[1])
    except IndexError:
        logging.error("No File Name Specified")
        sys.stdout.write("No File Name Specified")
