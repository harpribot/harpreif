def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv1d(x, W, stride):
    return tf.nn.conv1d(x, W, strides=stride, padding="SAME")


def max_pool_4x1(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding="SAME")
