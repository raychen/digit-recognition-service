from flask import Flask
from flask import request, Response
from flask_json import FlaskJSON, json_response

from io import BytesIO
from base64 import b64decode
import binascii
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

from keras import models
import tensorflow as tf
import numpy as np

ERROR_MESSAGE_IMAGE_MISSING_OR_EMPTY = "field \"image\" is missing or has empty content"
ERROR_MESSAGE_INVALID_IMAGE_DATA = "invalid image data"
ERROR_MESSAGE_JSON_EXPECTED = "only content-type: application/json is accepted"

model = models.load_model('mnist.h5')
graph = tf.get_default_graph()

IMG_ROWS = 28
IMG_COLS = 28

app = Flask(__name__)
FlaskJSON(app)


def predict(image):
    """
    predict label with the given image i.e. digits from [0-9]
    :param image: the image as a numpy ndarray
    :return: predicted label to the given image
    """
    #TODO: seperate pre-processing and prediction
    # using a Model class?

    gray = rgb2gray(image)[:, :, np.newaxis]
    resized_image = resize(gray, (IMG_ROWS, IMG_COLS), mode='reflect')

    # keras model only accept array of input, so the image needs to be put into an array
    x = resized_image[np.newaxis, :, :, :]

    # seems Keras has a unclosed bug when dealing with threads
    # https://github.com/fchollet/keras/issues/2397
    with graph.as_default():
        label = model.predict_classes(x, batch_size=1, verbose=0)

    return label[0]


@app.route('/')
def hello_world():
    introduction = """
    <h2>usage:</h2>
        <ul>
            <li>Request /recognize</li>
            <li>content-type: application/json</li>
            <li>body: {'image': [base64 encoded image binary]}</li>
        </ul>
    """
    return Response(introduction)


@app.route('/recognize', methods=['POST'])
def recognize():
    """
    process incoming request, decode the image and turn it into numpy ndarray
    :return:
    """
    json_request = request.get_json()
    if json_request is None:
        return json_response(400, description=ERROR_MESSAGE_JSON_EXPECTED)

    encoded_image = json_request.get('image', None)
    if encoded_image is None or len(encoded_image) == 0:
        return json_response(400, description=ERROR_MESSAGE_IMAGE_MISSING_OR_EMPTY)

    # Once the image data is corrupted in transfer,
    # there might be several possible exceptions in this simple process pipeline
    try:
        raw_image = b64decode(encoded_image)
        image = imread(BytesIO(raw_image))
    except (OSError, binascii.Error, ValueError):
        return json_response(400, description=ERROR_MESSAGE_INVALID_IMAGE_DATA)

    prediction = predict(image)
    return json_response(200, label=str(prediction))

if __name__ == '__main__':
    app.run(debug=True)
