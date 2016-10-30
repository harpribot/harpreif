import numpy as np
from skimage.feature import hog
from image_utils import sliding_window

POSTIVE_REWARD = 1
NEGATIVE_REWARD = -0.1
DELAY_REWARD = -0.05
STEPS_MAX = 10


class Environment(object):
    def __init__(self, original_image, initial_gamestate, grid_dim,
                 puzzle_pieces, image_dim, window, stride, num_channels):
        """

        :param original_image: The true output expected. It is used to give reward
        :param initial_gamestate: The start state for each episode. It is all zeros.
        :param grid_dim: The number of horizontal and vertical splits each, required to form the puzzle pieces
        :param puzzle_pieces: The dictionary of puzzle piece image as value for the puzzle_piece id as key
        :param image_dim: The dimension (row/col) of the original image. The image must be a square image.
        :param window: The window dimension for HOG based state space construction
        :param stride: The stride of the sliding window for HOG
        :param num_channels: The number of channels of the state space (= number of gradients given by HOG)
        """
        self.original_image = original_image
        self.jigsaw_image = np.zeros([image_dim, image_dim])
        self.initial_gamestate = initial_gamestate
        self.gamestate = initial_gamestate
        self.grid_dim = grid_dim
        self.puzzle_pieces = puzzle_pieces
        self.image_dim = image_dim
        self.state_height = self.gamestate.shape[0]
        self.state_width = self.state_height
        self.window = tuple(window)
        self.num_gradients = num_channels
        self.stride = stride
        self.action = None
        self.placed_location_for_id = dict()
        self.id_for_placed_location = dict()
        self.terminal = False
        self.jigsaw_split = np.split(np.array(range(self.image_dim)), self.grid_dim)
        self.steps = 0

    def __update_placed_pieces(self, image_id, place_id):
        """
        Updates the jigsaw board
        :param image_id: The image_id of the piece that needs to be placed
        :param place_id: The place_id of the board location where the image is to be placed
        :return: None
        """
        if image_id in self.placed_location_for_id:
            if self.placed_location_for_id[image_id] != place_id:
                self.remove_piece(image_id, self.placed_location_for_id[image_id])

        if place_id in self.id_for_placed_location:
            if self.id_for_placed_location[place_id] != image_id:
                self.remove_piece(self.id_for_placed_location[place_id], place_id)
                self.place_piece(image_id, place_id)
        else:
            self.place_piece(image_id, place_id)

    def __update_state(self):
        """
        Updates the state space (self.gamestate) after the suggested action is taken
        :return: None
        """
        image_id, place_id = self.decode_action()
        self.__update_placed_pieces(image_id, place_id)
        self.__render_gamestate()

    def __render_gamestate(self):
        """
        Renders the new gamestate based on the changed board condition using HOG gradients over sliding window
        :return: None
        """
        slides = sliding_window(self.jigsaw_image, self.stride,self.window)

        hog_gradients = []
        for slide in slides:
            window_image = slide[2]

            gradient = np.array(hog(window_image,
                         orientations=self.num_gradients,
                         pixels_per_cell=self.window,
                         cells_per_block=(1, 1), visualise=False))

            assert gradient.size == self.num_gradients, "Gradient size not equal to desired size"
            hog_gradients.extend(gradient)

        hog_gradients = np.array(hog_gradients)

        hog_gradients = hog_gradients.reshape((self.state_height, self.state_width, self.num_gradients))

        assert self.gamestate.shape == hog_gradients.shape, "The state dimension is trying to be altered"
        self.gamestate = hog_gradients

    def remove_piece(self, image_id, place_id):
        """
        Remove the piece from the jigsaw board
        :param image_id: The id of the image to be removed
        :param place_id: The place from where the image is to be removed from the jigsaw board
        :return: None
        """
        self.placed_location_for_id.pop(image_id)
        self.id_for_placed_location.pop(place_id)
        placing_range = self.__get_placing_range(place_id)
        self.update_jigsaw(image_id, placing_range, removal=True)

    def place_piece(self, image_id, place_id):
        """
        Add the piece to the jigsaw board
        :param image_id: The id of the image to be added to the board
        :param place_id: The place on the board where the image is to be placed.
        :return: None
        """
        self.placed_location_for_id[image_id] = place_id
        self.id_for_placed_location[place_id] = image_id
        placing_range = self.__get_placing_range(place_id)
        self.update_jigsaw(image_id, placing_range, removal=False)

    def set_action(self, action):
        """
        Set the action that the agent suggests. This action is transmitted to the environment.
        The environment then updates the next state of the agent.
        :param action: The action to be transmitted to the environment
        :return: None
        """
        self.action = action
        self.__update_state()

    def __get_reward(self):
        """
        For the given action, transmitted to the environment by the agent, the environment rewards the agent.
        :return: Reward given by the environment to the agent for the action taken
        """
        # get the reward based on the afterstate
        if self.terminal:
            if np.all(self.jigsaw_image == self.original_image):
                return POSTIVE_REWARD
            else:
                return NEGATIVE_REWARD
        else:
            return DELAY_REWARD

    def __get_next_state(self):
        """
        Returns the next state of the agent
        :return: Next state of the agent
        """
        return self.gamestate

    def __is_terminal(self):
        """
        Checks if the terminal state has been reached. Terminal state is reached, when all the pieces are
        placed in the board.
        :return: None
        """
        # check if self.gamestate is terminal
        if len(self.placed_location_for_id) == len(self.puzzle_pieces):
            self.terminal = True
        print len(self.placed_location_for_id), len(self.puzzle_pieces)

    def get_state_reward_pair(self):
        """
        Return the (s,r, terminality) --> state, reward pair by the environment to the agent in response to the action
        taken by the agent
        :return: (state, reward, terminality) triple
        """
        self.steps += 1
        if self.steps >= STEPS_MAX:
            self.terminal = True
        else:
            self.__is_terminal()

        reward = self.__get_reward()
        next_state = self.__get_next_state()
        return reward, next_state, self.terminal

    def decode_action(self):
        """
        Decoded the action, and returns which piece is to be placed where
        :return: (image_id, place_id)
        """
        image_id = int(self.action / (self.grid_dim * self.grid_dim))
        place_id = self.action % (self.grid_dim * self.grid_dim)

        return image_id, place_id

    def __get_placing_range(self, place_id):
        """
        Returns the placing range where the new image of the puzzle piece is to be placed.
        :param place_id: The id of the place on the jigsaw grid where the piece is to be placed
        :return: (row_start, row_end, col_start, col_end)
        """
        place_row = int(place_id / self.grid_dim)
        place_col = place_id % self.grid_dim

        # get the range for placing the jigsaw piece in the jigsaw image
        row_range = self.jigsaw_split[place_row]
        col_range = self.jigsaw_split[place_col]

        return row_range[0], row_range[-1] + 1, col_range[0], col_range[-1] + 1

    def update_jigsaw(self, image_id, placing_range, removal):
        """
        Update the jigsaw board
        :param image_id: The id of the puzzle piece to be placed/ removed
        :param placing_range: The placing range where the image of puzzle piece should be placed /removed
        :param removal: puzzle piece is to be placed or removed. True, means remove.
        :return: None
        """
        x_s, x_e, y_s, y_e = placing_range
        if removal:
            self.jigsaw_image[x_s:x_e, y_s:y_e] = np.zeros([x_e-x_s, y_e-y_s])
        else:
            self.jigsaw_image[x_s:x_e, y_s:y_e] = self.puzzle_pieces[image_id]

        assert self.jigsaw_image.shape == (256,256), "Trying to alter the image size"

    def get_state(self):
        """
        return the current gamestate
        :return: self.gamestate
        """
        return self.gamestate

    def update_puzzle_pieces(self, puzzle_pieces):
        """
        Update the puzzle pieces for a new image
        :param puzzle_pieces: The dictionary for image_id as key, and puzzle piece image as value
        :return: None
        """
        self.puzzle_pieces = puzzle_pieces

    def update_original_image(self, original_image):
        """
        Update the new image
        :param original_image: The original image of the image under consideration for learning
        :return: None
        """
        self.original_image = original_image

    def reset(self):
        """
        Reset the environment, when a terminal state is reached.
        :return: None
        """
        self.jigsaw_image = np.zeros([self.image_dim, self.image_dim])
        self.gamestate = self.initial_gamestate
        self.placed_location_for_id = dict()
        self.id_for_placed_location = dict()
        self.terminal = False
        self.steps = 0
