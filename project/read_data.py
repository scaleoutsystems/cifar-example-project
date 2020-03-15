""" We simulate the situation that we have a small amount of new data coming in. 
Every time we call read_data  it return a randomized, small fraction of the digits dataset. """
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split


def load_digits_biased(remove_digits):
    import os
    import pickle
    if os.path.isfile("../data/x_train.py"):
        x_train = pickle.load(open("x_train.p", "rb"))
        y_train = pickle.load(open("y_train.p", "rb"))
    else:
        from keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        pickle.dump(x_train, open("x_train.p", "wb"))
        pickle.dump(y_train, open("y_train.p", "wb"))
        pickle.dump(x_test, open("x_test.p", "wb"))
        pickle.dump(y_test, open("y_test.p", "wb"))


    X = x_train
    y = y_train

    if remove_digits:
        delete_rows = []
        for digit, amount in remove_digits.items():
            idx = numpy.where(y == digit)[0]
            delete_rows += list(idx[0:int(len(idx) * amount)])

        X = numpy.delete(X, delete_rows, 0)
        y = numpy.delete(y, delete_rows, 0)

    return {'data': X, 'target': y}


def read_training_data(remove_digits=None, sample_fraction=1.0):
    dataset = load_digits_biased(remove_digits)
    x = dataset['data']
    y = dataset['target']
    if sample_fraction < 1.0:
        foo, x, bar, y = train_test_split(x, y, test_size=sample_fraction)
    classes = range(10)
    return (x, y, classes)


def read_test_data():
    test_data = numpy.array(pd.read_csv("../data/test.csv"))
    X_validate = test_data[:, 1::]
    y_validate = test_data[:, 0]

    classes = range(10)
    return (X_validate, y_validate, classes)
