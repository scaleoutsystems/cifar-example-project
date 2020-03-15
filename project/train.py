from __future__ import print_function

import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from read_data import read_training_data
from scaleout.runtime.runtimeclient import RuntimeClient


def train(model_id=None):
    print("-- RUNNING TRAINING --")
    """ Respond to a training request:
        Download the global model and train on the new local data. """

    client = RuntimeClient()

    # Fetch global model for project
    print("inited client")
    global_model_candidate = client.get_global_model()
    print("got the global model")

    batch_size = 32
    num_classes = 10
    epochs = 1

    # Input image dimensions
    img_rows, img_cols = 28, 28

    # The data, split between train and test sets
    (x_train, y_train, classes) = read_training_data(sample_fraction=0.01)

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_train /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)

    global_model_candidate.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    # Submit the candidate model
    client.send_model(global_model_candidate)

    print("-- TRAINING COMPLETED --")


if __name__ == '__main__':
    train()
