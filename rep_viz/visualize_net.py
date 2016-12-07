import sys
import tensorflow as tf
from model.creator import Creator
from harpreif.myconstants import *
import numpy as np
import matplotlib.pyplot as plt


class Net(Creator):
    def __init__(self, num_actions, num_gradients, checkpoint_dir, state_type):
        """
        Recreates the Net for visualization
        :param num_actions: Number of actions of agent (= output layer vector size of DQN)
        :param num_gradients: Number of gradients to be used if state_type = HOG
        :param checkpoint_dir: The checkpoint location to be used to pupulate weights
        :param state_type: The state type, if equals 'HOG' then mirrored HOG is used for state construction
                    if 'image' then currently formed jigsaw image is used for state construction
        """
        self.num_actions = num_actions
        if state_type == 'image':
            self.input_channels = 1
        elif state_type == 'hog':
            self.input_channels = num_gradients
        else:
            raise ValueError

        self.input_height = len(range(0, IMAGE_HEIGHT - SLIDING_STRIDE, SLIDING_STRIDE))
        self.input_width = self.input_height
        self.checkpoint_dir = checkpoint_dir
        Creator.__init__(self, self.input_channels, self.num_actions, self.input_height, self.input_width)
        self.sess = tf.InteractiveSession()
        sys.stderr.write('Creating Network...\n')
        self.__create_network()
        self.saver = tf.train.Saver()
        self.__populate_network()

    def __create_network(self):
        """
        Creates a network for visualization
        :return: None
        """
        self._initialize_weights_and_biases()
        self._form_input_layer()
        self._form_hidden_layers()
        self._form_output_layer()

    def __populate_network(self):
        """
        Populate the network with checkpoint weights
        :return: None
        """
        self.sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir + "saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            sys.stderr.write("Successfully loaded: " + checkpoint.model_checkpoint_path + '\n')
        else:
            sys.stderr.write("Could not find old checkpoint weights\n")

    def display_weights(self):
        """
        Display the weights filter for visualization
        :return: None
        """
        print 'Displaying Weights...'
        W_conv1, W_conv2, W_conv3, W_fc1, W_fc2, W_fc3 = \
            self.sess.run([self.W_conv1, self.W_conv2, self.W_conv3,
                           self.W_fc1, self.W_fc2, self.W_fc3])

        print 'Convolution-1:\n {}\nConvolution-2:\n {}\nConvolution-3:\n {}\n' \
              'FC-1:\n {}\nFC-2:\n {}\nFC-3:\n {}\n'.format(
                W_conv1, W_conv2, W_conv3, W_fc1, W_fc2, W_fc3)

        print 'Testing boundedness of weights from bottom to top (Max, Min)'
        print np.max(W_conv1), np.min(W_conv1)
        print np.max(W_conv2), np.min(W_conv2)
        print np.max(W_conv3), np.min(W_conv3)
        print np.max(W_fc1), np.min(W_fc1)
        print np.max(W_fc2), np.min(W_fc2)
        print np.max(W_fc3), np.min(W_fc3)

        print 'Plotting the filters for convolution networks'
        Net.__plot(W_conv1, title='Convolution Layer 1')
        Net.__plot(W_conv2, title='Convolution Layer 2')
        Net.__plot(W_conv3, title='Convolution Layer 3')
        plt.show()

    def display_biases(self):
        """
        Display the biases for visualization
        :return: None
        """
        print 'Displaying Biases...'
        b_conv1, b_conv2, b_conv3, b_fc1, b_fc2, b_fc3 = \
            self.sess.run([self.b_conv1, self.b_conv2, self.b_conv3,
                           self.b_fc1, self.b_fc2, self.b_fc3])

        print 'Convolution-1:\n {}\nConvolution-2:\n {}\nConvolution-3:\n {}\n' \
              'FC-1:\n {}\nFC-2:\n {}\nFC-3:\n {}\n'.format(
                b_conv1, b_conv2, b_conv3, b_fc1, b_fc2, b_fc3)

        print 'Testing boundedness of biases from bottom to top (Max, Min)'
        print np.max(b_conv1), np.min(b_conv1)
        print np.max(b_conv2), np.min(b_conv2)
        print np.max(b_conv3), np.min(b_conv3)
        print np.max(b_fc1), np.min(b_fc1)
        print np.max(b_fc2), np.min(b_fc2)
        print np.max(b_fc3), np.min(b_fc3)

    @staticmethod
    def __plot(weights, title):
        """
        Utility function for visualization of filters, (might require changes based on number of filters)
        :param weights: The weights of the layer to be plotted
        :param title: Title of the plot
        :return: None
        """
        plt.figure()
        weights = np.array(weights)
        plt.title(title)
        print weights.shape

        for in_ in range(8):
            for out_ in range(8):
                plt.subplot(8, 8, in_ * 8 + out_ + 1)
                plt.imshow(weights[:, :, in_, out_], cmap='gray')
                plt.axis('off')
