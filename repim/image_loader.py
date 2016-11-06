from scipy import ndimage
import glob
from skimage.color import rgb2gray


class ImageLoader(object):
    def __init__(self, image_dir):
        """

        :param image_dir: The directory containing all the resized 256 x 256 images of train.
        """
        self.image_dir = image_dir
        self.image_list = None
        self.image_ptr = 0
        self.__index_images()
        self.image = None
        self.index2piece = dict()
        self.index2histogram = dict()
        self.tile_locations = None
        self.tries = 0
        self.image_name = None

    def __index_images(self):
        """
        Indexes all the images in the train needed for training.
        :return: None
        """
        self.image_list = [x for x in glob.glob(self.image_dir + '/' + '*.jpg')]

    def load_next_image(self):
        """
        Loads next image from train index for training.
        :return: True if the next image is present, else False
        """
        if len(self.image_list) == self.image_ptr:
            return False
        print 'Loaded New Image'
        self.image = ndimage.imread(self.image_list[self.image_ptr])
        self.image_name = self.image_list[self.image_ptr]

        is_color = self.__check_color()
        if is_color:
            self.image = rgb2gray(self.image)

        assert self.image.shape == (256, 256), 'Image not 256 x 256'
        self.image_ptr += 1

        return True

    def __check_color(self):
        """
        Checks if the input image is color image or not
        :return: True, if the image is color, False if the image is grayscale
        """
        if self.image.shape == (256, 256, 3):
            return True
        elif self.image.shape == (256, 256):
            return False
        else:
            raise TypeError('The image is not of standard dimension')

    def get_image(self):
        """
        Get the train image that the RL algo is currently training one
        :return: self.image
        """
        return self.image_name, self.image
