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
