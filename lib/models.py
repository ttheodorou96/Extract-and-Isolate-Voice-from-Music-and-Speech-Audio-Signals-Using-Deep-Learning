import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split
import pickle
from lib.autoencoder import Autoencoder
from lib.cnn import Cnn
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


class Models:
    """
        Models represent the various architectures and modes of training by initiating user's input parameters
    """

    def __init__(self, height, width, model_path, model_name, epochs, batch_size, X_train=None, y_train=None):
        """

        Parameters
        ----------
        height: int
            Image's height of the input's spectrogram
        width: int
             Image's width of the input's spectrogram
        model_path: str
            Absolute path of the model directory
        model_name:
            Positional argument the user provided for the saved model, the method will add to the name some of the
            parameters
        epochs: int
            Number of iterations the model is trained
        batch_size: int
            Number of samples will be feed to the model in each epoch
        """
        self.height = height
        self.width = width
        self.model_path = model_path
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.X_train = X_train
        self.y_train = y_train

        self.X_valid = None
        self.y_valid = None

    def model_train(self, model, shuffle=False):
        """
            Trains the model
        """
        tensorboard = TensorBoard(log_dir="logs/{}".format(self.model_name))
        es, mc = self.early_stopping_callback()
        model.fit(self.X_train,
                  self.y_train,
                  validation_data=(self.X_valid, self.y_valid),
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  shuffle=shuffle,
                  callbacks=[tensorboard, es, mc]
                  )
        return

    def early_stopping_callback(self):
        """
            create and return early stopping and checkpoint callbacks
        """
        best_model_path = os.path.join(self.model_path, self.model_name)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
        mc = ModelCheckpoint(f'{best_model_path}_best_model.h5', monitor='val_loss', mode='min', verbose=1,
                             save_best_only=True)
        return es, mc

    def print_summary(self, model):
        """
            Prints summary of the model and the returns the model's output path.
        """
        model_path = os.path.join(self.model_path, self.model_name)
        model_summary = os.path.join(model_path + '_summary.txt')
        with open(model_summary, 'w') as f:
            with redirect_stdout(f):
                if isinstance(model, Cnn):
                    model.print_summary()
                else:
                    model.summary()
        return model_path

    def create_training_valid_splits(self, split=0.8):
        """
            Splits training data into training/validation splits
        """
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train,
                                                                                  self.y_train,
                                                                                  train_size=split)
        return

    def convolutional_auto_encoder(self):
        """
            A simple convolutional auto encoder trained with spectrogram excerpts of the mixture as the encoded input
            and the ideal vocals as the decoded output. A Xavier's initializer is also added to further improve
            the network's performance. This class can be used for transfer learning or for weight transfer by name
            since it has a different architecture from the CNN model.
        """
        # X_train is excerpts of mixture spectrogram while y_train is excerpts of the ideal vocal spectrogram
        _, _, X_train, X_valid = train_test_split(self.X_train, self.y_train, train_size=0.8)

        autoencoder = Autoencoder(
            input_shape=(self.height, self.width, 1),
            conv_filters=(8, 16, 32, 8),
            conv_kernels=(3, 3, 3, 3),
            conv_strides=(1, 1, 1, 1),
            latent_space_dim=2
        )
        autoencoder.summary()

        autoencoder.compile(0.0005)
        tensorboard = TensorBoard(log_dir="logs/{}".format(self.model_name))
        # Train the model
        autoencoder.train(X_train, X_valid, X_train, X_valid, self.batch_size, self.epochs, tensorboard, True)
        model_path = self.print_summary(autoencoder)
        # save model
        autoencoder.save(model_path)
        return

    def vad_model(self):
        """
            Method that loads the pickle files/sets and creates/saves the Convolutional Neural Network model with
            the corresponding layers and parameters.
        """
        # To load training/test/valid dataset
        pickle_in = open(self.model_path + "\\X_train.pickle", "rb")
        self.X_train = pickle.load(pickle_in)

        pickle_in = open(self.model_path + "\\y_train.pickle", "rb")
        self.y_train = pickle.load(pickle_in)

        pickle_in = open(self.model_path + "\\X_valid.pickle", "rb")
        self.X_valid = pickle.load(pickle_in)

        pickle_in = open(self.model_path + "\\y_valid.pickle", "rb")
        self.y_valid = pickle.load(pickle_in)

        # VAD MODEL
        model = tf.keras.Sequential()
        # Layer 1
        model.add(layers.Conv2D(16, (3, 3), padding='same', input_shape=(self.height, self.width, 1)))

        model.add(layers.LeakyReLU())
        # Layer 2
        model.add(layers.Conv2D(16, (3, 3), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.MaxPooling2D(pool_size=(3, 3)))
        model.add(layers.Dropout(0.25))
        # Layer 3
        model.add(layers.Conv2D(16, (3, 3), padding='same'))
        model.add(layers.LeakyReLU())
        # Layer 4
        model.add(layers.Conv2D(16, (3, 3), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.MaxPooling2D(pool_size=(3, 3)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(64))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))

        # sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'],
                      )

        self.model_train(model, True)
        model_path = self.print_summary(model)
        model.save(model_path)

    def cnn_separator(self):
        """
            CNN_separator architecture, input shape (img_height, img_width, 1) -> output shape (1, img_height).
            Default Values:
                img_height = 2049
                img_width = 9
        """
        self.create_training_valid_splits(0.8)
        model = tf.keras.Sequential()
        # Layer 1
        model.add(layers.Conv2D(32, (12, 3), padding='same', input_shape=(self.height, self.width, 1)))
        model.add(layers.LeakyReLU())
        # Layer 2
        model.add(layers.Conv2D(16, (12, 3), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.MaxPooling2D(pool_size=(12, 1)))
        model.add(layers.Dropout(0.25))
        # Layer 3
        model.add(layers.Conv2D(64, (12, 3), padding='same'))
        model.add(layers.LeakyReLU())
        # Layer 4
        model.add(layers.Conv2D(32, (12, 3), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.MaxPooling2D(pool_size=(12, 1)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2049, activation='sigmoid'))
        print(model.summary())

        loss = keras.losses.BinaryCrossentropy()
        # adam_optimizer = Adam(learning_rate=0.00005)
        adam_optimizer = Adam()
        model.compile(loss=loss, optimizer=adam_optimizer)
        self.model_train(model, True)
        model_path = self.print_summary(model)
        model.save(model_path)
        model_json = model.to_json()
        with open(f"{model_path}.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(model_path + '.h5')
        return

    def initiate_svs_as_autoencoder(self):
        """
            Calls the Cnn Class to build the corresponding architecture, then trains the model with the selected
            training data. If using preprocess tool training data are in the auto_encoder folder and
            saves the model in the same directory
        """
        self.create_training_valid_splits(0.8)

        cnn = Cnn(
            input_shape=(self.height, self.width, 1),
            conv_filters=(32, 16, 64, 16),
            conv_kernels=((12, 3), (12, 3), (12, 3), (12, 3)),
            pool_size=((12, 1), (12, 1)),
            dense_neuron=(2048, 512, 18441),
            initial_train=True
        )
        tensorboard = TensorBoard(log_dir="logs/{}".format(self.model_name))
        loss = BinaryCrossentropy()
        optimizer = Adam(learning_rate=0.00001)
        cnn.compile(loss, optimizer)
        cnn.train(self.X_train, self.X_valid, self.X_train, self.X_valid, self.batch_size, self.epochs, tensorboard,
                  True)
        model_path = self.print_summary(cnn)
        cnn.save(model_path)

    def deep_cnn_separator(self):
        """
        method that loads the pickle files/sets and creates/saves the Convolutional Neural Network model with
        the corresponding layers and parameters.

        Returns
        -------
        None
        """
        # Split training/valid dataset
        self.create_training_valid_splits(0.8)
        # Load weights from previously trained model
        json_file = open('auto_encoder/ae_model/initialize_cnn_deep_as_autoencoder.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        # initiate model
        loaded_model = keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights('auto_encoder/ae_model/initialize_cnn_deep_as_autoencoder.h5', by_name=True)
        loss = keras.losses.BinaryCrossentropy()
        adam_optimizer = Adam(learning_rate=0.001)
        loaded_model.compile(loss=loss, optimizer=adam_optimizer)
        self.model_train(loaded_model, True)
        model_path = self.print_summary(loaded_model)
        loaded_model.save(model_path)
        model_json = loaded_model.to_json()
        with open(f"{model_path}.json", "w") as json_file:
            json_file.write(model_json)
        loaded_model.save_weights(model_path + '.h5')
