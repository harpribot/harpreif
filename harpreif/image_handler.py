from scipy import ndimage
import glob
import image_slicer
from skimage import io


class ImageNet(object):
    def __init__(self, image_dir, grid_dim, gradient_directions):
        self.image_dir = image_dir
        self.grid_dim = grid_dim
        self.gradient_directions = gradient_directions
        self.image_list = None
        self.image_ptr = 0
        self.__index_images()
        self.image = None
        self.index2piece = dict()
        self.index2histogram = dict()
        self.tile_locations = None
        self.tries = 0

    def __index_images(self):
        self.image_list = [self.image_dir + '/' + x for x in glob.glob('*.JPEG')]

    def load_next_image(self):
        self.image = ndimage.imread(self.image_list[self.image_ptr])
        self.image_ptr += 1
        self.tries = 0

    def break_into_jigzaw_pieces(self):
        image_loc = self.image_list[self.image_ptr]
        tile = image_slicer.slice(image_loc, self.grid_dim ** 2)
        self.tile_locations = self.get_tile_locations(tile)

    def get_puzzle_pieces(self):
        # we will return the puzzle pieces, as well as their true locations in row major numbering format
        result = dict()
        for piece_id, image_loc in enumerate(self.tile_locations):
            result[piece_id] = io.imread(image_loc)

        return result

    def get_state_size(self):
        return 4 * self.gradient_directions * (self.grid_dim ** 2)

    def get_tries_per_image(self):
        return self.tries

    def increment_tries(self):
        self.tries += 1

    def get_image(self):
        return self.image

    def get_tile_locations(self, tile):
        tile_str = [x[1].strip() for x in str(tile)[1:-1].split('-')]
        tile_loc = [self.image_dir + '/' + x for x in tile_str]
        return tile_loc
