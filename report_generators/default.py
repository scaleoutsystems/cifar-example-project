import sys
import numpy
from minio import Minio
from sklearn.metrics import classification_report

minioClient = Minio('minio.generators.platform.demo.scaleout.se',
                    access_key='',
                    secret_key='',
                    secure=False)


def read_test_data():
    import pandas

    data = minioClient.get_object('dataset', 'test.csv')
    with open('media/datasets/test.csv', 'wb') as file_data:
        for d in data.stream(32 * 1024):
            file_data.write(d)

    test_data = numpy.array(pandas.read_csv("media/datasets/test.csv"))

    X_validate = test_data[:, 1::]
    y_validate = test_data[:, 0]
    classes = range(10)

    return X_validate, y_validate, classes


def generate_report(model_uid):
    import pickle

    data = minioClient.get_object('models', model_uid)
    model = pickle.loads(data.read())

    img_rows, img_cols = 28, 28
    x_test, y_test, classes = read_test_data()
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_predict = model.predict_classes(x_test)
    report = classification_report(y_test, y_predict)

    return report


if __name__ == '__main__':
    model_uid = str(sys.argv[1])

    report = generate_report(model_uid)

    print(report)
