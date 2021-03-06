from flask import Flask
from flask import request, Response
from flask_json import FlaskJSON, json_response

from io import BytesIO
from base64 import b64decode
import binascii
from skimage.io import imread
from models import load_model


ERROR_MESSAGE_IMAGE_MISSING_OR_EMPTY = "field \"image\" is missing or has empty content"
ERROR_MESSAGE_INVALID_IMAGE_DATA = "invalid image data"
ERROR_MESSAGE_JSON_EXPECTED = "only content-type: application/json is accepted"

#TODO it would be better to pass in model name and file path through configuration file
model = load_model('mnist.h5', 'MNISTKeras')
app = Flask(__name__)
FlaskJSON(app)


@app.route('/')
def intro():
    introduction = """
    <h2>usage:</h2>
        <ul>
            <li>endpoint: <b>/recognize</b></li>
            <li>method: <b>POST</b></li>
            <li>content-type: <b>application/json</b></li>
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

    prediction = model.predict(image)
    return json_response(200, label=str(prediction))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
