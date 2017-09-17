import random
import os
import random
import glob
import numpy as np
from PIL import Image
import cv2
import scipy.misc
import tensorflow as tf
from tensorflow.contrib import learn

def save_gen(generated_images, n_ex = 36, epoch = 0, iter = 0):
    for i in range(generated_images.shape[0]):
        cv2.imwrite('/root/code/Video_Generation/gen_images/image_' + str(epoch) + '_' + str(iter) + '_' + str(i) + '.jpg', generated_images[i, :, :, :])
def resize(image, width=None, height=None, inter=cv2.INTER_LANCZOS4):
    '''resize the image such that its aspect ration is preserved. Only height or width is required as the
    other parameter is calculated using the aspect raito of the given image.
    input parameters
      image_name:- str. Image name along with the full path
      width:- integer. The width of the output image
      height:- integer. If widht is provided then height is not needed. The height of the output image.
    output
      returns the path + resized image name (str) which is resized to the image widht or height given in the input.
    '''

    # read the image
    #image = cv2.imread(image_name)

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image_name

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image_name
    #ext = image_name.split(".")[-1]
    #if ext == "png" or ext == "PNG":
    #    resized_img_name = image_name.split(".")[0] + "_resized.png"
    #else:
    #    resized_img_name = image_name.split(".")[0] + "_resized.jpg"
    #cv2.imwrite(resized_img_name, resized)
    return resized


# making square image
def make_square_image(img, size=None):
    '''
    Resize the image to a desired height or width. Once this is done pad the shorter side such that both the height and the width are equal.
    Only height or width is required as the other parameter will be the same as we are constructing square images.
    input parameters
      image_name:- str. Image name along with the full path
      width:- integer. The width of the output image
      height:- integer. If widht is provided then height is not needed. The height of the output image.
    output
      returns the path + squared image name:- str.
    '''

    if size == None:
        return "Please return a valid size"
    else:
        #ext = image_name.split(".")[-1]
        #img = cv2.imread(image_name)
        h, w = img.shape[:2]
        if h >= w:
            resized_img = resize(img, height=size)
            #resized_img = cv2.imread(resized_img_name)
            r_h, r_w = resized_img.shape[:2]
            padding_size = size - r_w
            if img.ndim==3:
                square_image = np.pad(resized_img, ((
                0, 0), (padding_size / 2, padding_size / 2 + padding_size % 2), (0, 0)), "constant")
            else:
                square_image = np.pad(resized_img, ((
                0, 0), (padding_size / 2, padding_size / 2 + padding_size % 2)), "constant")
        else:
            resized_img = resize(img, width=size)
            #resized_img = cv2.imread(resized_img_name)
            r_h, r_w = resized_img.shape[:2]
            padding_size = size - r_h
            if img.ndim==3:
                square_image = np.pad(resized_img, ((
                padding_size / 2, padding_size / 2 + padding_size % 2), (0, 0), (0, 0)), "constant")
            else:
                square_image = np.pad(resized_img, ((
                padding_size / 2, padding_size / 2 + padding_size % 2), (0, 0)), "constant")

        #if ext == "png" or ext == "PNG":
        #    square_img_name = image_name.split(".")[0] + "_square_img." + ext
        #else:
        #    square_img_name = image_name.split(".")[0] + "_square_img.jpg"

        #cv2.imwrite(square_img_name, square_image)
        return square_image

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def get_real_images(files, isone = False):
    file_path = '/root/code/img_align_celeba/'
    x = np.zeros((len(files), 64, 64, 3))
    counter = 0
    for file in files:
        img = cv2.imread(file_path + file)
        img = img/127.5
        img = img - 1
        x[counter, :, :, :] = make_square_image(img, 64)
        counter = counter + 1
    return x



def make_squared_image(generated_images):
  N = len(generated_images)
  black_image = np.zeros(generated_images[0].shape, dtype=np.int32)
  w = int(np.minimum(10, np.sqrt(N)))
  h = int(np.ceil(N / w))

  one_row_image = generated_images[0]
  for j in range(1, w):
    one_row_image = np.concatenate((one_row_image, generated_images[j]), axis=1)

  image = one_row_image
  for i in range(1, h):
    one_row_image = generated_images[i*w]
    for j in range(1, w):
      try:
        one_row_image = np.concatenate((one_row_image, generated_images[i*w + j]), axis=1)
      except:
        one_row_image = np.concatenate((one_row_image, black_image), axis=1)
    image = np.concatenate((image, one_row_image), axis=0)

  return image

def l1_loss(tensor, weight=1.0, scope=None):
  with tf.name_scope(scope, 'L1Loss', [tensor]):
    weight = tf.convert_to_tensor(weight,
                                  dtype=tensor.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.multiply(weight, tf.reduce_sum(tf.abs(tensor)), name='value')
    return loss


def conv2d(input_, output_dim,name,k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                  initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.Variable(tf.random_normal([shape[1], output_size], dtype = tf.float32, stddev = stddev))
    bias = tf.Variable(tf.constant(bias_start, shape = [output_size]))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))

    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv
