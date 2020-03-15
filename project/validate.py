import sys
import keras
from read_data import read_training_data
from scaleout.runtime.runtimeclient import RuntimeClient


def validate(model_id=None):
    print("-- RUNNING VALIDATION! --")
    """ Respond to a validation request:
        Download a candidate model and score it on own training data. """

    client = RuntimeClient()
    if model_id == None:
        print("failed to run validation, no model id!")
        return

    # Fetch candidate model to validate
    candidate_model = client.get_model(model_id)

    try:
        x_test, y_test, classes = read_training_data(sample_fraction=0.01)
        # Input image dimensions
        img_rows, img_cols = 28, 28
        num_classes = 10
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = keras.utils.to_categorical(y_test, num_classes)
        model_score = candidate_model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', model_score[0])
        print('Test accuracy:', model_score[1])
    except Exception as e:
        print ("failed to run scoring {}".format(e))
        exit(-1)

    evaluation = {"model_id": model_id, "score": model_score[1]}
    client.send_evaluation(evaluation)

    print("-- VALIDATION COMPLETED --")
    return model_score

if __name__ == '__main__':

    model_id = str(sys.argv[1])
    score = validate(model_id)
    print("Model id: {0} Test loss: {1} Test accuracy: {2}".format(model_id, score[0],score[1]))
