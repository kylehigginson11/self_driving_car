import cv2
import numpy as np
import glob
import sys
from sklearn.model_selection import train_test_split
import logging

# Configure logger
logging.basicConfig(filename='/var/log/driverless_car/driverless_car.log', level=logging.DEBUG,
                    format="%(asctime)s:%(levelname)s:%(message)s")


class TrainMLP:
    def __init__(self):
        logging.info("Training MLP ...")
        self.start_time = cv2.getTickCount()
        logging.info('Loading training data...')
        # load training data
        self.image_array = np.zeros((1, 38400))
        self.label_array = np.zeros((1, 4), 'float')
        self.training_data = glob.glob('training_data/*.npz')
        self.load_training_data()

    def load_training_data(self):
        # if no data, exit
        if not self.training_data:
            logging.error("No training data in directory, exit")
            sys.exit()

        for single_npz in self.training_data:
            with np.load(single_npz) as data:
                train_temp = data['train']
                train_labels_temp = data['train_labels']
            self.image_array = np.vstack((self.image_array, train_temp))
            self.label_array = np.vstack((self.label_array, train_labels_temp))

        image_data_x = self.image_array[1:, :]
        label_data_y = self.label_array[1:, :]
        logging.info('Image array shape: ', image_data_x.shape)
        logging.info('Label array shape: ', label_data_y.shape)

        end_time = cv2.getTickCount()
        time0 = (end_time - self.start_time) / cv2.getTickFrequency()
        logging.info('Loading image duration:', time0)
        self.create_mlp(image_data_x, label_data_y)

    def create_mlp(self, images, labels):
        # train test split, 7:3
        train, test, train_labels, test_labels = train_test_split(images, labels, test_size=0.3)

        # set start time
        start_time = cv2.getTickCount()

        # create MLP
        layer_sizes = np.int32([38400, 32, 4])
        ann = cv2.ml.ANN_MLP_create()
        ann.setLayerSizes(layer_sizes)
        ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        ann.setBackpropMomentumScale(0.0)
        ann.setBackpropWeightScale(0.001)
        ann.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001))
        ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

        logging.info('Training MLP ...')
        train_ann = ann.train(np.float32(train), cv2.ml.ROW_SAMPLE, np.float32(train_labels))

        # set end time
        end_time = cv2.getTickCount()
        train_time = (end_time - start_time) / cv2.getTickFrequency()
        logging.info('Training duration:', train_time)
        self.train_data(train, test, train_labels, test_labels, ann)

    def train_data(self, train, test, train_labels, test_labels, ann):

        # train data
        train_prediction_1, train_prediction_2 = ann.predict(train)
        prediction_0 = train_prediction_2.argmax(-1)
        true_labels_0 = train_labels.argmax(-1)

        train_rate = np.mean(prediction_0 == true_labels_0)
        logging.info('Train accuracy: ', "{0:.2f}%".format(train_rate * 100))

        # test data
        test_prediction_1, test_prediction_2 = ann.predict(test)
        prediction_1 = test_prediction_2.argmax(-1)
        true_labels_1 = test_labels.argmax(-1)

        test_rate = np.mean(prediction_1 == true_labels_1)
        logging.info('Test accuracy: ', "{0:.2f}%".format(test_rate * 100))

        # save model
        ann.save('mlp_xml/mlp.xml')
        logging.info("Model Saved")
        logging.info("MLP Training Complete!")


if __name__ == '__main__':
    TrainMLP()
