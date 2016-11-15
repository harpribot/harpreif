import numpy as np


def sliding_window(image, step_size, window_size):
    """
    This creates an iterator for the sliding window in the row major format for a given image.
    :param image: The jigsaw image
    :param step_size: The stride of the sliding window
    :param window_size: The window sie of the sliding window
    :return: Iterator of the image patch for the next sliding window
    """
    # slide a window across the image
    for y in xrange(0, image.shape[0] - step_size, step_size):
        for x in xrange(0, image.shape[1] - step_size, step_size):
            # yield the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def gradient_discretizer(gradient, bins):
    """
    Digitizes the gradient to limit the state space, otherwise there are infinite possible gradient values
    :param gradient: The numpy array of gradient across provided orientations
    :param bins: The bins range in which the gradient is to be discretized into
    :return: The numpy array of bin indices in which each gradient falls into. Return size same as gradient size
    """
    return np.digitize(gradient, bins)


def performance_statistics(image_diff_list, reward_list):
    image_diff_list = np.array(image_diff_list)
    reward_list = np.array(reward_list)

    print 'The image matching performance - %f' % np.average(image_diff_list)
    print 'The average accumulated reward - %f' % np.average(reward_list)
