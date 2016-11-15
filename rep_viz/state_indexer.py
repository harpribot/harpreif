import tensorflow as tf
import numpy as np
from harpreif.network_utils import weight_variable, bias_variable, conv2d, max_pool_2x2
from image_loader import ImageLoader
from harpreif.image_utils import sliding_window, gradient_discretizer
from skimage.feature import hog
import cPickle as pickle

WINDOW_SIZE = [8, 8]
SLIDING_STRIDE = WINDOW_SIZE[0]/2
IMAGE_HEIGHT = IMAGE_WIDTH = 256
NUM_BINS = 16


class Image2Feature(object):
    def __init__(self, image_dir, checkpoint_dir, num_actions, num_gradients):
        """

        :param image_dir: The test directory for images
        :param checkpoint_dir: The checkpoint containing the best learnt model weights and biases
        :param num_actions: Number of actions that the agent can take
        :param num_gradients: Number of gradients to be used for each window
        """
        self.image_dir = image_dir
        self.bins = np.array([x / float(NUM_BINS) for x in range(0, NUM_BINS, 1)])
        self.sess = None
        self.checkpoint_dir = checkpoint_dir
        self.num_actions = num_actions
        self.num_gradients = num_gradients
        self.input_channels = num_gradients
        self.input_height = len(range(0, IMAGE_HEIGHT - SLIDING_STRIDE, SLIDING_STRIDE))
        self.input_width = self.input_height
        self.imagenet = None
        self.feature_dict = dict()
        self.state_height = self.input_height
        self.state_width = self.state_height
        self.save_transform = True
        self.im2f_loc = None
        self.feature_size = None

    def __load_model(self):
        """
        Loads the model and populates it with checkpoint weights
        :return: None
        """
        print 'Initializing Session...'
        self.sess = tf.InteractiveSession()
        print 'Creating Network...'
        self.__create_network()
        print 'Populating Network with learned weights and biases...'
        self.__populate_network()

    def __load_images(self):
        """
        Loads the images into imagenet from which it can be queried
        :return: None
        """
        self.imagenet = ImageLoader(self.image_dir)

    def image2feature(self, save_transform=False, im2f_loc= None):
        """
        Transforms image to feature vector obtained from penultimate layer of dqn
        :param save_transform: True, if we wish to save the image 2 feature transform map
        :param im2f_loc: If save_transform == True, then this contains the location where to save the map
        :return: The image 2 feature map
        """
        print 'Loading The images...'
        self.__load_images()

        print 'Loading the model...'
        self.__load_model()

        print 'Obtaining deep features of images...'
        self.__get_image_features()

        print 'Returning the map for image to features...'
        self.save_transform = save_transform
        self.im2f_loc = im2f_loc
        return self.__save_and_get_features()

    def __initialize_weights_and_biases(self):
        """
        Creates the layers to be used by the DQN network, and initializes their weights and biases
        :return: None
        """
        self.W_conv1 = weight_variable([4, 4, self.input_channels, 16])
        self.b_conv1 = bias_variable([16])

        self.W_conv2 = weight_variable([4, 4, 16, 32])
        self.b_conv2 = bias_variable([32])

        self.W_conv3 = weight_variable([2, 2, 32, 32])
        self.b_conv3 = bias_variable([32])

        self.W_fc1 = weight_variable([2048, 1024])
        self.b_fc1 = bias_variable([1024])

        self.W_fc2 = weight_variable([1024, 512])
        self.b_fc2 = bias_variable([512])

        self.W_fc3 = weight_variable([512, self.num_actions])
        self.b_fc3 = bias_variable([self.num_actions])

    def __form_input_layer(self):
        """
        Forms the input layer of gradient images of dimension (self.input_height, self.input_width, self.input_channels)
        Here,
        input_channels = number of gradient directions considered by the HOG filter.
        input_height = number of windows of window_size = WINDOW_SIZE, and stride = SLIDING_STRIDE possible along row
        input_width <= input_height as all images are square images of size (INPUT_HEIGHT x INPUT_WIDTH)
        :return: None
        """
        self.s = tf.placeholder("float", [None, self.input_height, self.input_width, self.input_channels])

    def __form_hidden_layers(self):
        """
        Forms the convolution layers with non-linear recurrent units, followed by fully connected units
        :return: None
        """
        self.__form_convolution_layers()
        self.__form_fully_connected_layers()

    def __form_convolution_layers(self):
        """
        Forms 3 convolution layers, with a max-pooling layer after first convolution layer.
        :return: None
        """
        self.h_conv1 = tf.nn.relu(conv2d(self.s, self.W_conv1, 2) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2, 2) + self.b_conv2)

        self.h_conv3 = tf.nn.relu(conv2d(self.h_conv2, self.W_conv3, 1) + self.b_conv3)

        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 2048])

    def __form_fully_connected_layers(self):
        """
        Forms 2 fully connected layers
        :return: None
        """
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)

        self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)

    def __form_output_layer(self):
        """
        Forms The output layer (linear in this case) - The value represents the value function for the action
        :return: None
        """
        self.readout = tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3

    def __create_network(self):
        """
        Creates the entire DQN network
        :return: None
        """
        self.__initialize_weights_and_biases()
        self.__form_input_layer()
        self.__form_hidden_layers()
        self.__form_output_layer()

    def __populate_network(self):
        """
        Popilates the network with checkpoint weights and biases
        :return: None
        """
        saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir + "saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            raise ValueError("No checkpoint found. You must have saved weights")

    def __get_input_for_model(self, image):
        """
        Renders the new gamestate based on the changed board condition using HOG gradients over sliding window
        :return: None
        """
        slides = sliding_window(image, SLIDING_STRIDE, WINDOW_SIZE)

        hog_gradients = []
        for slide in slides:
            window_image = slide[2]

            gradient = np.array(hog(window_image,
                                    orientations=self.num_gradients,
                                    pixels_per_cell= WINDOW_SIZE,
                                    cells_per_block=(1, 1), visualise=False))

            assert gradient.size == self.num_gradients, "Gradient size not equal to desired size"
            gradient = gradient_discretizer(gradient, self.bins)
            hog_gradients.extend(gradient)

        hog_gradients = np.array(hog_gradients)

        hog_gradients = hog_gradients.reshape((self.state_height, self.state_width, self.num_gradients))

        assert hog_gradients.shape == (self.input_height, self.input_width, self.input_channels), \
            "The state dimension is trying to be altered"

        state = hog_gradients
        return state

    def __get_image_features(self):
        """
        Get the feature vector for all the images
        :return: None
        """
        while True:
            is_present = self.imagenet.load_next_image()

            if is_present:
                image_nm, image = self.imagenet.get_image()
                image_state = self.__get_input_for_model(image)
                im_feat = self.sess.run(self.h_fc2, feed_dict={self.s: [image_state]})
                self.feature_dict[image_nm] = im_feat
                self.feature_size == im_feat.size
            else:
                break

    def __save_and_get_features(self):
        """
        Saves and returns the feature map and feature size
        :return: (feature_map, feature_dimension)
        """
        if self.save_transform:
            pickle.dump(self.feature_dict, open(self.im2f_loc + "image2feature.p", "wb"))
        return self.feature_dict, self.feature_size
