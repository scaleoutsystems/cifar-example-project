from kerasmodel import ANN_Model
import sys
from read_cifardata import read_data
from read_model import load_model, save_model
from minio import Minio
import json
import io
import pickle
import requests
import uuid

import os

if __name__ == '__main__':

    load_model_id = str(sys.argv[1])
    save_model_id = str(sys.argv[2])

    if len(sys.argv) > 3:
        epochs = int(sys.argv[3])

    print("arguments: ", sys.argv)

    if load_model_id == 'None':
        load_model_id = None

    if save_model_id == 'None':
        save_model_id = None

    mk = json.load(open('miniokeys3.json'))

    client = Minio(endpoint=mk['ClientUrl'],
                   access_key=mk['MinioAccessKey'],
                   secret_key=mk['MinioSecretKey'],
                   secure=False
                   )

    x_train, y_train, x_val, y_val = read_data(client, datasort='train', split_validation=True, validation_split=0.1)

    if load_model_id is not None:
        model = load_model(client, load_model_id)
    else:
        model = ANN_Model()

    model.fit(x_data=x_train, y_data=y_train, x_val=x_val, y_val=y_val, batch_size=32, epochs=epochs, data_augmentation=True,
              validation_split=0.1, verbose=1)

    model_uid = str(uuid.uuid4())

    #if save_model_id is not None:

    save_model(client,model.model, model_uid)


    url = 'https://platform.demo.scaleout.se/api/models/'

    model_name = 'afternoon'
    model_description = 'keras sequential model'
    model_url = os.path.join('https://minio.test.platform.demo.scaleout.se/minio/models',)
    project_id = '8'
    myobj = {
        'uid': model_uid,
        'name': model_name,
        'description': model_description,
        'url': model_url,
        'project': project_id,
        }

    x = requests.post(url, data = myobj)

    print(x.status_code)