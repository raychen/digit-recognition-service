from keras.models import load_model as load_keras_model
import tensorflow as tf
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from os.path import abspath, dirname, join


def load_model(model_name, model_type):
    file_path = join(dirname(abspath(__file__)), 'model_files', model_name)
    if model_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_type](file_path)
    else:
        raise ValueError('unsupported model type')


class MNISTKeras(object):

    MNIST_IMG_SIZE = (28, 28)

    def __init__(self, file_path):
        self._model = load_keras_model(file_path)
        self._graph = tf.get_default_graph()

    @classmethod
    def pre_process(cls, image, resize_to):
        """
        :param image: the image as a numpy ndarray
        :param resize_to: the shape of processed image
        :return: image array ready to feed into model
        """
        gray = rgb2gray(image)[:, :, np.newaxis]
        resized_image = resize(gray, resize_to, mode='reflect')
        # keras model only accept array of input, so the image needs to be put into an array
        x = resized_image[np.newaxis, :, :, :]
        return x

    def predict(self, image, resize_to=MNIST_IMG_SIZE):
        """
        predict label with the given image i.e. digits from [0-9]
        :param image: the image as a numpy ndarray
        :return: predicted label to the given image
        """
        x = self.pre_process(image, resize_to)

        # seems Keras has a unclosed bug when dealing with threads
        # https://github.com/fchollet/keras/issues/2397
        with self._graph.as_default():
            label = self._model.predict_classes(x, batch_size=1, verbose=0)

        return label[0]


MODEL_REGISTRY = {
    'MNISTKeras': MNISTKeras
}