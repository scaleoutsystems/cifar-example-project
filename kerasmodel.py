import pickle
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn import metrics
import json

import keras.backend as K

import psutil
model_folder = 'models'
validation_folder = 'validations'

learning_rate = 0.001
decay=0


def construct_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=[32,32,3]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(learning_rate=learning_rate, decay=decay)
    model.compile(loss='categorical_crossentropy',
                       optimizer=opt,
                       metrics=['accuracy'])

    return model


# def construct_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), padding='same',
#                  input_shape=[32,32,3]))
#     model.add(Activation('relu'))
#     model.add(Conv2D(32, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(64, (3, 3), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#     model.add(Dense(50))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(10))
#     model.add(Activation('softmax'))
#
#     opt = keras.optimizers.Adam(learning_rate=learning_rate, decay=decay)
#     model.compile(loss='categorical_crossentropy',
#                        optimizer=opt,
#                        metrics=['accuracy'])
#
#     return model

class ANN_Model:

    def __init__(self, graph=None):

        if graph is not None:
            self.model = graph
        else:
            self.model = construct_model()

        self.training_calls = 0
        self.training_history = []

    def fit(self, x_data, y_data, x_val=None, y_val=None, batch_size=32, epochs=10, data_augmentation=True, validation_split=0.1, verbose=2,
            ):

        if x_val is None:
            new_ind = np.arange(len(x_data))
            np.random.shuffle(new_ind)
            x_data = x_data[new_ind]
            y_data = y_data[new_ind]
            split_index = int(len(x_data)*(1-validation_split))
            x_train = x_data[:split_index]
            y_train = y_data[:split_index]
            x_val = x_data[split_index:]
            y_val = y_data[split_index:]
        else:
            x_train = x_data
            y_train = y_data



        print("x_train shape: ", x_train.shape, ", y_train shape: ", y_train.shape)


        h = []
        if not data_augmentation:
            print('Not using data augmentation.')
            h = self.model.fit(x_train,y_train, validation_data=(x_val,y_val), batch_size=batch_size, epochs=epochs,
                               verbose=verbose) #callbacks=[mcp_save, es])

        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                zca_epsilon=1e-06,  # epsilon for ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                # randomly shift images horizontally (fraction of total width)
                width_shift_range=0.1,
                # randomly shift images vertically (fraction of total height)
                height_shift_range=0.1,
                shear_range=0.,  # set range for random shear
                zoom_range=0.,  # set range for random zoom
                channel_shift_range=0.,  # set range for random channel shifts
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                cval=0.,  # value used for fill_mode = "constant"
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,  # randomly flip images
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                validation_split=0.1)

                # Compute quantities required for feature-wise normalization
                # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            h = self.model.fit_generator(datagen.flow(x_train, y_train,
                                                      batch_size=batch_size),
                                epochs=epochs,
                                verbose=verbose,
                                validation_data=(x_val, y_val),
                                workers=4)

        self.training_calls += 1
        self.training_history += [h]
        return h

    def predict(self, x_data):
        return self.model.predict(x_data)

    def validate(self, x_data, y_data):

        y_pred = np.argmax(self.predict(x_data),1)
        y_data = np.argmax(y_data,1)

        validation_data = metrics.classification_report(y_data,y_pred,labels=np.unique(y_pred))

        return validation_data

    def set_weights(self, weights):

        self.model.set_weights(weights=weights)

    def get_weights(self):

        return self.model.get_weights()













