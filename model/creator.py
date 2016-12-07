import tensorflow as tf
from harpreif.network_utils import weight_variable, bias_variable, conv2d, max_pool_2x2, debug_printer
from harpreif.myconstants import *


class Creator(object):
    def __init__(self, input_channels, num_actions, input_height, input_width):
        """
        Creates the DQN network for the Reinforcement Learning
        :param input_channels: The number of channels in the input state
        :param num_actions: Number of actions to be predicted by DQN, equals the output vector size of DQN
        :param input_height: The height of the input state
        :param input_width: The width of the input state
        """
        self.input_channels = input_channels
        self.num_actions = num_actions
        self.input_height = input_height
        self.input_width = input_width

    def _initialize_weights_and_biases(self):
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

    def _form_input_layer(self):
        """
        Forms the input layer of gradient images of dimension (self.input_height, self.input_width, self.input_channels)
        Here,
        input_channels = number of gradient directions considered by the HOG filter.
        input_height = number of windows of window_size = WINDOW_SIZE, and stride = SLIDING_STRIDE possible along row
        input_width <= input_height as all images are square images of size (INPUT_HEIGHT x INPUT_WIDTH)
        :return: None
        """
        self.s = tf.placeholder("float", [None, self.input_height, self.input_width, self.input_channels])
        self.s = debug_printer(self.s, "Input: ")

    def _form_hidden_layers(self):
        """
        Forms the convolution layers with non-linear recurrent units, followed by fully connected units
        :return: None
        """
        self._form_convolution_layers()
        self._form_fully_connected_layers()

    def _form_convolution_layers(self):
        """
        Forms 3 convolution layers, with a max-pooling layer after first convolution layer and Leaky RELU for
        each convolution layer
        :return: None
        """
        h_conv1_activation = conv2d(self.s, self.W_conv1, 2) + self.b_conv1
        self.h_conv1 = tf.maximum(ALPHA * h_conv1_activation, h_conv1_activation)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        h_conv2_activation = conv2d(self.h_pool1, self.W_conv2, 2) + self.b_conv2
        self.h_conv2 = tf.maximum(ALPHA * h_conv2_activation, h_conv2_activation)

        h_conv3_activation = conv2d(self.h_conv2, self.W_conv3, 1) + self.b_conv3
        self.h_conv3 = tf.maximum(ALPHA * h_conv3_activation, h_conv3_activation)
        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 2048])

    def _form_fully_connected_layers(self):
        """
        Forms 2 fully connected layers
        :return: None
        """
        h_fc1_activation = tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1
        self.h_fc1 = tf.maximum(ALPHA * h_fc1_activation, h_fc1_activation)

        h_fc2_activation = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        self.h_fc2 = tf.maximum(ALPHA * h_fc2_activation, h_fc2_activation)

    def _form_output_layer(self):
        """
        Forms The output layer (linear in this case) - The value represents the value function for the action
        :return: None
        """
        self.readout = tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3

    def _define_loss(self):
        """
        Defines the loss function for the model
        :return: None
        """
        self.action = tf.placeholder("float", [None, self.num_actions])
        self.label = tf.placeholder("float", [None])

        self.readout_action = tf.reduce_sum(tf.mul(self.readout, self.action))
        self.cost = tf.reduce_mean(tf.square(self.label - self.readout_action))
        self.cost = debug_printer(self.cost, "Cost(MINIMIZE): ")

    def _form_trainer(self):
        """
        Defines the trainer for the model. It is a Stochastic Gradient Descent using Adadelta optimizer
        :return: None
        """
        self.learning_rate_tf = tf.placeholder(tf.float32, shape=[])
        self.train_step = tf.train.AdamOptimizer(self.learning_rate_tf).minimize(self.cost)
