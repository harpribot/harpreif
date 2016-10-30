import tensorflow as tf


def weight_variable(shape):
    """
    This creates the weight placeholder for the DQN layers
    :param shape: shape of the weight variable
    :return: The tensorflow placeholder for weights
    """
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    This creates the bias placeholder for the DQN layers
    :param shape: shape of the bias variable
    :return: The tensorflow placeholder for the bias
    """
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w, stride):
    """
    This creates the 2D convolution layer
    :param x: The input placeholder to the convolution layer
    :param w: The weight placeholder for the convolution layer
    :param stride: The stride of the convolution layer
    :return: The convolution layer Tensorflow placeholder
    """
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    """
    This create a maxpooling of 2x2 window with stride of 1 in each direction
    :param x: The input to the maxpooling layer
    :return: The maxpooling layer Tensorflow placeholder
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
