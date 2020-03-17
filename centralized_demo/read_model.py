import pickle
import io

def save_weights(client, model, name):

    if not client.bucket_exists(bucket_name='models'):
        client.make_bucket(bucket_name='models')

    weights = model.get_weights()
    obj = pickle.dumps(weights)
    client.put_object('models', name, io.BytesIO(obj), len(obj))

def load_weights(client, model, name):

    objects = [m.object_name for m in client.list_objects('models')]
    if name in objects:
        obj = client.get_object(bucket_name='models', object_name=name)
        weights = pickle.loads(obj.read())
        model.load_weights(weights=weights)

def save_model(client, model, name):

    if not client.bucket_exists(bucket_name='models'):
        client.make_bucket(bucket_name='models')

    obj = pickle.dumps(model)
    client.put_object('models', name, io.BytesIO(obj), len(obj))

def load_model(client, name):

    #if name in objects:
    try:
        #objects = [m.object_name for m in client.list_objects('models')]
        obj = client.get_object(bucket_name='models', object_name=name)
        model = pickle.loads(obj.read())

    except:
        print("Couldn't find model: ", name)
        model = None

    return model





