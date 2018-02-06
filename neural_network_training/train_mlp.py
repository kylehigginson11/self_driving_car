import cv2
import numpy as np
import glob
import sys
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

# Configure logger
logging.basicConfig(filename='/var/log/driverless_car/driverless_car.log', level=logging.DEBUG,
                    format="%(asctime)s:%(levelname)s:%(message)s")


class TrainMLP:

    def __init__(self):
        logging.info("Training MLP ...")
        self.start_time = datetime.now()
        logging.info('Loading training data...')
        # load training data
        self.image_array = np.zeros((1, 38400))
        self.label_array = np.zeros((1, 4), 'float')
        self.training_data = glob.glob('training_data/*.npz')
        self.load_training_data()

    def load_training_data(self):
        # if no data, exit
        if not self.training_data:
            logging.error("Cant find training data!")
            sys.exit()

        # loop through all collected image files
        for npz_file in self.training_data:
            with np.load(npz_file) as data:
                train_temp = data['train']
                train_labels_temp = data['train_labels']
            self.image_array = np.vstack((self.image_array, train_temp))
            self.label_array = np.vstack((self.label_array, train_labels_temp))

        image_data_x = self.image_array[1:, :]
        label_data_y = self.label_array[1:, :]
        logging.info('Image array shape: ', image_data_x.shape)
        logging.info('Label array shape: ', label_data_y.shape)

        end_time = datetime.now()
        load_image_time = end_time - self.start_time
        logging.info('Loading image duration:' + str(load_image_time.seconds) + 'seconds')
        self.create_mlp(image_data_x, label_data_y)

    def create_mlp(self, images, labels):
        # train test split, can use random_state=42 to set a seed here
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3)

        # set start time
        train_start_time = datetime.now()

        # create MLP
        layer_sizes = np.int32([38400, 32, 4])
        ann_mlp = cv2.ml.ANN_MLP_create()
        ann_mlp.setLayerSizes(layer_sizes)
        ann_mlp.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        ann_mlp.setBackpropMomentumScale(0.0)
        ann_mlp.setBackpropWeightScale(0.001)
        ann_mlp.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001))
        ann_mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

        logging.info('Training MLP ...')
        train_ann = ann_mlp.train(np.float32(x_train), cv2.ml.ROW_SAMPLE, np.float32(y_test))

        # set end time
        train_end_time = datetime.now()
        train_time = (train_end_time - train_start_time)
        logging.info('Training duration: ' + str(train_time.seconds) + 'seconds')
        self.save_mlp(x_train, x_test, y_train, y_test, ann_mlp)

    def save_mlp(self, x_train, x_test, y_train, y_test, ann_mlp):

        # calculate train accuracy
        label_1, confidence_1 = ann_mlp.predict(x_train)
        prediction = np.argmax(confidence_1, axis=-1)
        true_labels = np.argmax(y_train, axis=-1)

        train_rate = np.mean(prediction == true_labels)
        logging.info('Train accuracy: ', "{0:.2f}%".format(train_rate * 100))

        # calculate test accuracy
        label_2, confidence_2 = ann_mlp.predict(x_test)
        prediction_1 = np.argmax(confidence_2, axis=-1)
        true_labels_1 = np.argmax(y_test, axis=-1)

        test_rate = np.mean(prediction_1 == true_labels_1)
        logging.info('Test accuracy: ', "{0:.2f}%".format(test_rate * 100))

        # save model
        ann_mlp.save('neural_networks/neural_network.xml')
        logging.info("Model Saved")
        logging.info("MLP Training Complete!")


if __name__ == '__main__':
    TrainMLP()
