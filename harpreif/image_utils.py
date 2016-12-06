import numpy as np
import sys
from scipy.ndimage.interpolation import rotate


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
    sys.stderr.write('The image matching performance - ' + str(np.average(image_diff_list)) + '\n')
    sys.stderr.write('The average accumulated reward - ' + str(np.average(reward_list)) + '\n')


def subtract_image_mean(image, mean_file_location):
    """
    Subtract the imagenet mean from the image before further Deep Learning action
    :param image: The unprocessed image of shape 256 x 256
    :param mean_file_location: location of the Imagenet mean file
    :return: The mean subtracted image
    """
    imagenet_mean = np.load(mean_file_location)
    return image - np.mean(imagenet_mean, axis=0)/255.0


def supress_corners(jigsaw_piece, corner_fraction=1/8.0):
    """
    Supresses the corners of the jigsaw piece to remove any edge continuity that can be exploited as a shortcut
    :param jigsaw_piece: The original jigsaw piece
    :param corner_fraction: The fraction of the whole dimension that is to be supressed on each corner.
    :return: The edge blurred jigsaw piece
    """
    piece_size = jigsaw_piece.shape[0]
    corner_size = piece_size * corner_fraction
    # remove top corner
    jigsaw_piece[0:corner_size, :] = 0.
    # remove bottom corner
    jigsaw_piece[-corner_size:, :] = 0.
    # remove left corner
    jigsaw_piece[:, 0:corner_size] = 0.
    # remove right corner
    jigsaw_piece[:, -corner_size:] = 0.
    return jigsaw_piece


def jitter_image(jigsaw_piece):
    """
    Jitters the jigsaw piece randomly. The way this is done, is by randomly rotating the jigsaw piece about the center
    by a random angle between [-10 deg, 10 deg]. This will prevent edge continuity
    :param jigsaw_piece:The input jigsaw piece
    :return: The jittered version of the original jigsaw piece
    """
    rotation_angle = (np.random.random(1)[0] - 0.5) * 20
    jittered_image = rotate(jigsaw_piece, rotation_angle, reshape=False)

    assert jigsaw_piece.shape == jittered_image.shape, 'The jittered piece has changed size... Not allowed'
    return jittered_image
