import sys
from scipy import ndimage
import glob
import image_slicer
import numpy as np
from skimage.color import rgb2gray
from random import shuffle


class ImageNet(object):
    def __init__(self, image_dir, grid_dim, num_images=None):
        """

        :param image_dir: The directory containing all the resized 256 x 256 images of train.
        :param grid_dim: The number of horizontal and vertical cuts required to form the jigsaw piece
        :param num_images: Total number of images to be used for training/ validation / testing. If None, use all.
        """
        self.image_dir = image_dir
        self.grid_dim = grid_dim
        self.num_images = num_images
        self.image_list = None
        self.image_ptr = 0
        self.__index_images()
        self.image = None
        self.index2piece = dict()
        self.index2histogram = dict()
        self.tile_locations = None
        self.tries = 1

    def __index_images(self):
        """
        Indexes all the images in the train needed for training. Keeps only the number of images specified by user
        via self.num_images. Discards rest.
        :return: None
        """
        self.image_list = [x for x in glob.glob(self.image_dir + '/' + '*.jpg')]
        shuffle(self.image_list)
        if self.num_images is not None:
            self.image_list = self.image_list[:self.num_images]

    def load_next_image(self):
        """
        Loads next image from train index for training.
        :return: True if the next image is present, else False
        """
        if len(self.image_list) == self.image_ptr:
            return False
        sys.stderr.write('Loaded Image #' + str(self.image_ptr) + ' ...\n')
        self.image = ndimage.imread(self.image_list[self.image_ptr])
        is_color = self.__check_color()
        if is_color:
            self.image = rgb2gray(self.image)

        assert self.image.shape == (256, 256), 'Image not 256 x 256'
        self.__break_into_jigzaw_pieces()
        self.image_ptr += 1
        self.tries = 1

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

    def __break_into_jigzaw_pieces(self):
        """
        Break the image into jigsaw pieces
        :return: None
        """
        image_loc = self.image_list[self.image_ptr]
        self.tiles = image_slicer.slice(image_loc, self.grid_dim ** 2, save=False)

    def get_puzzle_pieces(self):
        """
        returns the puzzle pieces, as well as their true locations in row major numbering format, as a dictionary,
        where the key, is row_major puzzle_piece_id and the value is the piece image itself
        :return: The dictionary of piece_id => piece_image
        """
        result = dict()
        for piece_id, piece in enumerate(self.tiles):
            piece_image = np.array(piece.image)
            result[piece_id] = rgb2gray(piece_image)

        return result

    def get_tries_per_image(self):
        """
        Returns the number of episodes that have been trained on a particular image.
        :return: The number of episodes used for training a image
        """
        return self.tries

    def increment_tries(self):
        """
        Increment the episode count after the current episode termination
        :return: None
        """
        self.tries += 1

    def get_image(self):
        """
        Get the train image that the RL algo is currently training one
        :return: self.image
        """
        return self.image
