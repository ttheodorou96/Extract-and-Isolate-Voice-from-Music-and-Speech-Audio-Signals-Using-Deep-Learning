import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle
from sklearn.model_selection import train_test_split
from numpy import newaxis


def load_images(path):
    images_array = []
    for img in tqdm(os.listdir(path)):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            images_array.append(img_array)
        except Exception as e:
            Exception

    return images_array


def create_feature_label_pairs(path_x, path_y):
    X = []
    y = []
    for X_train, y_train in zip(tqdm(os.listdir(path_x)), tqdm(os.listdir(path_y))):
        try:
            with open(os.path.join(path_x, X_train), 'rb') as f:
                pickle_item_X = pickle.load(f)
            with open(os.path.join(path_y, y_train), 'rb') as f:
                pickle_item_y = pickle.load(f)
            for training_input, label in zip(pickle_item_X, pickle_item_y):
                X.append(training_input)
                y.append(label)
        except Exception as e:
            print(e)
            Exception

    return X, y


class DataLoader:
    """
        Create input - output labels from the selected training set and modes
    """

    def __init__(self, dataset_path, categories, model_path, model_details):
        self.dataset_path = dataset_path
        self.categories = categories
        self.model_path = model_path
        self.model_details = model_details

    def create_training_data(self):
        """
        Method that adds into the training_data list images of the the non-vocal, vocal pairs (0,1) and returns it.

        Returns
        -------
        training_data: list
            List of numpy arrays that each represent a spectrogram image of the non-vocal/vocal pairs (1,0).
        """
        training_data = []
        print(self.categories[:2])
        for category in self.categories[:2]:

            path = os.path.join(self.dataset_path, category)
            class_num = self.categories.index(category)  # get the classification  (0 or a 1). 0=non-vocals 1=vocals

            for set in tqdm(os.listdir(path)):  # iterate over each sett per vocals and non-vocals
                try:
                    with open(os.path.join(path, set), 'rb') as f:
                        pickle_item = pickle.load(f)
                    for item in pickle_item:
                        training_data.append([item, class_num])
                except Exception as e:  # in the interest in keeping the output clean...
                    Exception
        return training_data

    def split_data(self):
        """
        Method that creates a training/valid/test split and saves the sets into pickle files.

        Returns
        -------
        None
        """
        training_data = self.create_training_data()
        random.shuffle(training_data)

        X = []
        y = []

        for features, label in training_data:
            X.append(features)
            y.append(label)

        X = np.array(X)
        y = np.array(y)
        X = X[:, :, :, newaxis]

        X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)

        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

        model_path_test_data = os.path.join(self.model_path,
                                            self.model_details.get('name') + '_'
                                            + str(self.model_details.get('epochs')) + '_'
                                            + str(self.model_details.get('batch_size')) + '_'
                                            + str(int((self.model_details.get('frame_size') / 2) + 1)) + 'x'
                                            + str(self.model_details.get('img_size').get('width'))
                                            )
        if not os.path.isdir(model_path_test_data):
            os.mkdir(model_path_test_data)

        pickle_out = open(self.model_path + "\\X_train.pickle", "wb")
        pickle.dump(X_train, pickle_out)
        pickle_out.close()

        pickle_out = open(self.model_path + "\\y_train.pickle", "wb")
        pickle.dump(y_train, pickle_out)
        pickle_out.close()

        pickle_out = open(model_path_test_data + "\\X_test.pickle", "wb")
        pickle.dump(X_test, pickle_out)
        pickle_out.close()

        pickle_out = open(model_path_test_data + "\\y_test.pickle", "wb")
        pickle.dump(y_test, pickle_out)
        pickle_out.close()

        pickle_out = open(self.model_path + "\\X_valid.pickle", "wb")
        pickle.dump(X_valid, pickle_out)
        pickle_out.close()

        pickle_out = open(self.model_path + "\\y_valid.pickle", "wb")
        pickle.dump(y_valid, pickle_out)
        pickle_out.close()
        print("saved pickle")

    def load_training_data(self, category_index_x, category_index_y):
        X_train_path = os.path.join(self.dataset_path, self.categories[category_index_x])
        y_train_path = os.path.join(self.dataset_path, self.categories[category_index_y])
        X_train, y_train = create_feature_label_pairs(X_train_path, y_train_path)
        X_train = np.array(X_train)
        X_train = X_train[:, :, :, np.newaxis]
        y_train = np.array(y_train)

        return X_train, y_train

