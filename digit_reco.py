from flask import Flask
from flask import request
from flask_json import FlaskJSON, json_response

from io import BytesIO
from base64 import b64decode
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

from keras import models
import tensorflow as tf
import numpy as np

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
    resized = resize(gray, (IMG_ROWS, IMG_COLS), mode='reflect')

    # keras model will only accept array of input
    x = resized[np.newaxis, :, :, :]

    # seems Keras has a unclosed bug when dealing with threads
    # https://github.com/fchollet/keras/issues/2397
    with graph.as_default():
        label = model.predict_classes(x, batch_size=1, verbose=0)

    return label[0]


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/recognize', methods=['POST'])
def recognize():
    """
    process incoming request, decode the image and turn it into numpy ndarray
    :return:
    """
    json_request = request.get_json()

    #TODO check if flask will raise proper error message when 'image' is absent
    encoded_image = json_request['image']

    raw_image = b64decode(encoded_image)
    image = imread(BytesIO(raw_image))

    prediction = predict(image)
    return json_response(200, label=str(prediction))

if __name__ == '__main__':
    app.run(debug=True)
