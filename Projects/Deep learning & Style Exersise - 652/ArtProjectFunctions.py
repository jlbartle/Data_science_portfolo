import numpy as np
from keras import backend as K
from keras.preprocessing.image import load_img, save_img, img_to_array
import tensorflow as tf


# from keras import optimizers
# from keras.applications.vgg19 import VGG19
# vgg19_weights = '../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
# vgg19 = VGG19(include_top = False, weights=vgg19_weights)

# converts a image, and its size into a img array a keras object in the form of a numpy aray
def preprocess_image(image_path, img_nrows, img_ncols):
    from keras.applications import vgg19
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def get_content_loss(base_content, target):
    return K.sum(K.square(target - base_content))


# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(input_tensor):
    assert K.ndim(input_tensor) == 3
    # if K.image_data_format() == 'channels_first':
    #    features = K.batch_flatten(input_tensor)
    # else:
    #    features = K.batch_flatten(K.permute_dimensions(input_tensor,(2,0,1)))
    # gram = K.dot(features, K.transpose(features))
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram  # /tf.cast(n, tf.float32)


def get_style_loss(style, combination, img_nrows, img_ncols):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C))  # /(4.0 * (channels ** 2) * (size ** 2))

def deprocess_image(x, img_nrows, img_ncols):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def save_img(imgx, filename):
    from PIL import Image
    rescaled = (255.0 / imgx.max() * (imgx - imgx.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(filename + '.png')


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values