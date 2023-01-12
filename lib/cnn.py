import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


class Cnn:
    """
        Cnn represents a Deep Convolutional Neural network architecture.
    """

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 pool_size,
                 dense_neuron,
                 initial_train):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.pool_size = pool_size
        self.dense_neuron = dense_neuron
        self.initial_train = initial_train

        self._num_conv_layers = len(conv_filters)
        self._num_dense_layers = len(dense_neuron)
        self.model = keras.Sequential(name='CNN_deep_separator')

        self.build_cnn()
        self._print()

    def conv_input_layer(self, layer_number, layer_index):
        """
            Add conv input layers
        """
        self.model.add(layers.Conv2D(self.conv_filters[layer_index],
                                     self.conv_kernels[layer_index],
                                     padding="same",
                                     input_shape=self.input_shape,
                                     kernel_initializer='glorot_normal' if self.initial_train else None,
                                     name=f"conv_layer_{layer_number}"))
        return

    def _add_conv_layer(self, layer_index):
        """
            Add conv layers
        """
        layer_number = layer_index + 1
        if layer_number == 1:
            self.conv_input_layer(layer_number, layer_index)
        else:
            self.model.add(layers.Conv2D(self.conv_filters[layer_index],
                                         self.conv_kernels[layer_index],
                                         padding='same',
                                         kernel_initializer='glorot_normal' if self.initial_train else None,
                                         name=f'conv_layer_{layer_number}'))
            self.model.add(layers.BatchNormalization(name=f'batch_norm_{layer_number}'))
        self.model.add(layers.LeakyReLU(name=f'leaky_relu_{layer_number}'))
        if layer_number % 2 == 0:
            self.model.add(layers.MaxPooling2D(pool_size=self.pool_size[int(layer_index/2)],
                                               name=f'max_pooling_{layer_number}'))
        return

    def _add_conv_layers(self):
        """
            Add conv blocks
        """
        for layer_index in range(self._num_conv_layers):
            self._add_conv_layer(layer_index)
        return

    def _add_dense_layer(self, layer_index):
        """
            Add dense layers
        """
        layer_number = layer_index + 1
        self.model.add(layers.Dense(self.dense_neuron[layer_index], name=f'dense_layer_{layer_number}'))
        self.model.add(layers.LeakyReLU(name=f'hidden_leaky_relu_{layer_number}'))
        if layer_number != self._num_dense_layers:
            self.model.add(layers.Dropout(0.5, name=f'dropout_layer_{layer_number}'))
        else:
            self.model.add(layers.Activation('sigmoid', name=f'output_layer'))
        return

    def _add_dense_layers(self):
        """
            Add dense layer blocks
        """
        for layer_index in range(self._num_dense_layers):
            self._add_dense_layer(layer_index)
        return

    def build_cnn(self):
        """
            Build the proposed Convolutional Neural Network
        """
        self._add_conv_layers()
        self.model.add(layers.Flatten())
        self._add_dense_layers()
        self.model.add(layers.Reshape(self.input_shape[:-1]))

        return self.model

    def compile(self, loss, optimizer, metric=None):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metric)

    def train(self, X_train, X_valid, y_train, y_valid, batch_size, num_epochs, tensorboard, shuffle=False):
        self.model.fit(X_train,
                       y_train,
                       validation_data=(X_valid, y_valid),
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=shuffle,
                       callbacks=[tensorboard])

    def save(self, save_folder="."):
        if self.initial_train:
            model_json = self.model.to_json()
            with open(f"{save_folder}.json", "w") as json_file:
                json_file.write(model_json)
            self.model.save_weights(save_folder+'.h5')
        else:
            self.model.save(save_folder+'.h5')

    def load_initial_weights(self, model_path):
        self.model = self.model.load_weights(model_path)
        return self.model

    def print_summary(self):
        return self.model.summary()

    def _print(self):
        print(self.model.summary())
