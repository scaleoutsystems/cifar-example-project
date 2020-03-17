import sys
import numpy
from minio import Minio
from sklearn.metrics import classification_report
import keras
import pickle
import numpy as np
import json


mk = json.load(open('miniokeys.json'))

minioClient = Minio(endpoint=mk['ClientUrl'],
               access_key=mk['MinioAccessKey'],
               secret_key=mk['MinioSecretKey'],
               secure=False
               )

def read_data(client, datasort, split_validation=False, validation_split=0.1):
    # x_data = pickle.loads(client.get_object('dataset', 'c10/x_' + datasort).read())
    # y_data = pickle.loads(client.get_object('dataset', 'c10/y_' + datasort).read())

    data = pickle.loads(client.get_object('dataset', 'cifardata.p').read())
    x_data = data['x_' + datasort]
    y_data = data['y_' + datasort]

    num_classes = 10
    # Convert class vectors to binary class matrices.
    y_data = keras.utils.to_categorical(y_data, num_classes)
    x_data = x_data.astype('float32')
    x_data /= 255

    if split_validation:

        split = int(len(x_data) * validation_split)
        x_train = x_data[split:]
        y_train = y_data[split:]
        x_val = x_data[:split]
        y_val = y_data[:split]

        return x_train, y_train, x_val, y_val

    else:
        return x_data, y_data

def read_test_data():


    X_validate, y_validate = read_data(minioClient, datasort='test', split_validation=False, validation_split=0.1)

    classes = range(10)

    return X_validate, y_validate, classes


def generate_report(model_uid):
    import pickle

    data = minioClient.get_object('models', model_uid)
    model = pickle.loads(data.read())

    # img_rows, img_cols = 28, 28
    x_test, y_test, classes = read_test_data()
    # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    # x_test = x_test.astype('float32')
    # x_test /= 255
    y_predict = model.predict_classes(x_test)
    y_test = np.argmax(y_test,1)
    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    report = classification_report(y_test, y_predict, target_names=cifar_classes)

    return report


if __name__ == '__main__':
    model_uid = str(sys.argv[1])

    report = generate_report(model_uid)

    print(report)