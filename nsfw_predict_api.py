#-*-coding:utf-8-*-

import os
import sys

import numpy as np
from PIL import Image
import tensorflow as tf
import io
import logging
logger = logging.getLogger(__name__)


_MODEL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/models/1547856517')

_IMAGE_SIZE = 64
_BATCH_SIZE = 128

_LABEL_MAP = {0:'drawings', 1:'hentai', 2:'neutral', 3:'porn', 4:'sexy'}

def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    return img

def load_image( image_byte ) :
    img_data = io.BytesIO(image_byte)
    img = Image.open(img_data)
    img = img.resize((_IMAGE_SIZE, _IMAGE_SIZE))
    img.load()
    data = np.asarray( img, dtype=np.float32 )
    data = standardize(data)
    return data

sess = tf.Session()
graph = tf.get_default_graph()
tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], _MODEL_DIR)
inputs = graph.get_tensor_by_name("input_tensor:0")
probabilities_op = graph.get_tensor_by_name('softmax_tensor:0')
class_index_op = graph.get_tensor_by_name('ArgMax:0')
def predict(image_byte):
    logger.debug("predict start")
    image_data = load_image(image_byte)
    logger.debug("predict load_image ok")
    feed_dict = {inputs: [image_data] * _BATCH_SIZE}
    logger.debug("predict feed_dict ok")
    probabilities, class_index = sess.run([probabilities_op, class_index_op],
                                              feed_dict=feed_dict)
    logger.debug("predict tensorflow ok")
    probabilities_dict = {_LABEL_MAP.get(i): l for i, l in enumerate(probabilities[0])}
    pre_label = _LABEL_MAP.get(class_index[0])
    result = {"class": pre_label, "probability": probabilities_dict}
    logger.debug("all ok")
    return result


