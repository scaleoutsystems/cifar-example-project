from io import BytesIO
from PIL import Image
import numpy as np
import keras
import pickle

def read_data(client, datasort, split_validation=False, validation_split=0.1):


    #x_data = pickle.loads(client.get_object('dataset', 'c10/x_' + datasort).read())
    #y_data = pickle.loads(client.get_object('dataset', 'c10/y_' + datasort).read())

    data = pickle.loads(client.get_object('dataset', 'cifardata.p').read())
    x_data = data['x_' + datasort]
    y_data = data['y_' + datasort]

    
    
    num_classes = 10
    # Convert class vectors to binary class matrices.
    y_data = keras.utils.to_categorical(y_data, num_classes)
    x_data = x_data.astype('float32')
    x_data /= 255


    if split_validation:

        split = int(len(x_data)*validation_split)
        x_train = x_data[split:]
        y_train = y_data[split:]
        x_val = x_data[:split]
        y_val = y_data[:split]

        return x_train, y_train, x_val, y_val

    else:
        return x_data, y_data
