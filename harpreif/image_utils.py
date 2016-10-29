def sliding_window(image, stepSize, windowSize):
    """
    This creates an iterator for the sliding window in the row major format for a given image.
    :param image: The jigsaw image
    :param stepSize: The stride of the sliding window
    :param windowSize: The window sie of the sliding window
    :return: Iterator of the image patch for the next sliding window
    """
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
