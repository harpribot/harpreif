import tensorflow as tf
from network_utils import weight_variable, bias_variable, conv2d, max_pool_2x2
from collections import deque
from image_handler import ImageNet
from environment import Environment
from image_utils import performance_statistics
import random
import numpy as np

GAME = 'jigsaw'
LEARNING_RATE = 1e-5
INITIAL_EPSILON = 0.3
FINAL_EPSILON = 0.05
OBSERVE = 1000
REPLAY_MEMORY = 500
BATCH_SIZE = 32
GAMMA = 0.99
WINDOW_SIZE = [8, 8]
SLIDING_STRIDE = WINDOW_SIZE[0]/2
IMAGE_HEIGHT = IMAGE_WIDTH = 256
TRIES_PER_IMAGE = 10
ALPHA = 0.01


class Agent(object):
    def __init__(self, num_actions, grid_dim, num_gradients):
        """

        :param num_actions: Number of actions possible for the agent - The encoding is such that,
                            if i_th piece is to be placed in j_th location (of grid_dim ** 2 = N possible)
                            locations, then the corresponding action index = i * N + j
        :param grid_dim: Number of horizontal (equalling vertical breaks) on the original image to form pieces
        :param num_gradients: Number of bins for HOG (Histogram of Oriented Gradients) for each patch in a
                            sliding window across the jigsaw image (the image that is already been constructed)
        """
        self.grid_dim = grid_dim
        self.num_gradients = num_gradients
        self.num_actions = num_actions
        self.input_height = len(range(0, IMAGE_HEIGHT - SLIDING_STRIDE, SLIDING_STRIDE))
        self.input_width = self.input_height
        self.input_channels = self.num_gradients
        self.sess = None
        self.train_dir = None
        self.val_dir = None
        self.test_dir = None
        self.checkpoint_dir = None

    def play_game(self, train_dir, val_dir, checkpoint_dir):
        """
        Initiates gameplay using DQN based reinforcement learning
        :param train_dir: The directory containing the training images
        :param val_dir: The directory containing the testing images
        :param checkpoint_dir: The directory where checkpoints are to be stored.
        :return: None
        """
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.checkpoint_dir = checkpoint_dir
        print 'Initializing Session...'
        self.sess = tf.InteractiveSession()
        print 'Creating Network...'
        self.__create_network()
        print 'Training Network...'
        self.__train_network()

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
	self.s = Agent.debug_printer(self.s, "Input: ")
	
    def __form_hidden_layers(self):
        """
        Forms the convolution layers with non-linear recurrent units, followed by fully connected units
        :return: None
        """
        self.__form_convolution_layers()
        self.__form_fully_connected_layers()
    
    @staticmethod	
    def debug_printer(variable, variable_str):
	return tf.Print(variable, [variable], variable_str)

    def __form_convolution_layers(self):
        """
        Forms 3 convolution layers, with a max-pooling layer after first convolution layer.
        :return: None
        """
	#self.W_conv1 = Agent.debug_printer(self.W_conv1, "Conv1 weight: ")
	#self.b_conv1 = Agent.debug_printer(self.b_conv1, "Conv1 bias: ")
        h_conv1_activation = conv2d(self.s, self.W_conv1, 2) + self.b_conv1
	#h_conv1_activation = Agent.debug_printer(h_conv1_activation, "Conv1 activation: ")
        # self.h_conv1 = tf.nn.relu(conv2d(self.s, self.W_conv1, 2) + self.b_conv1)
        self.h_conv1 = tf.maximum(ALPHA * h_conv1_activation, h_conv1_activation)
        self.h_pool1 = max_pool_2x2(self.h_conv1)
	#self.h_conv1 = Agent.debug_printer(self.h_conv1, "Conv1 output: ")
	#self.h_pool1 = Agent.debug_printer(self.h_pool1, "Pool1 output: ")

        h_conv2_activation = conv2d(self.h_pool1, self.W_conv2, 2) + self.b_conv2
        #h_conv2_activation = Agent.debug_printer(h_conv2_activation, "Conv2 activation: ")
	# self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2, 2) + self.b_conv2)
        self.h_conv2 = tf.maximum(ALPHA * h_conv2_activation, h_conv2_activation)
	#self.h_conv2 = Agent.debug_printer(self.h_conv2, "Conv2 output: ")

        h_conv3_activation = conv2d(self.h_conv2, self.W_conv3, 1) + self.b_conv3
	#h_conv3_activation = Agent.debug_printer(h_conv3_activation, "Conv3 activation: ")
        # self.h_conv3 = tf.nn.relu(conv2d(self.h_conv2, self.W_conv3, 1) + self.b_conv3)
        self.h_conv3 = tf.maximum(ALPHA * h_conv3_activation, h_conv3_activation)
	#self.h_conv3 = Agent.debug_printer(self.h_conv3, "Conv3 output: ")

        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 2048])
	#self.h_conv3_flat = Agent.debug_printer(self.h_conv3_flat, "Conv3 flattened output: ")

    def __form_fully_connected_layers(self):
        """
        Forms 2 fully connected layers
        :return: None
        """
	#self.W_fc1 = Agent.debug_printer(self.W_fc1, "FC1 weight: ")
        #self.b_fc1 = Agent.debug_printer(self.b_fc1, "FC1 bias: ")
        h_fc1_activation = tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1
	#h_fc1_activation = Agent.debug_printer(h_fc1_activation, "Fully 1 activation: ")
        # self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)
        self.h_fc1 = tf.maximum(ALPHA * h_fc1_activation, h_fc1_activation)
	#self.h_fc1 = Agent.debug_printer(self.h_fc1, "Fully 1 output: ")
        
	#self.W_fc2 = Agent.debug_printer(self.W_fc2, "FC2 weight: ")
        #self.b_fc2 = Agent.debug_printer(self.b_fc2, "FC2 bias: ")
	h_fc2_activation = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        #h_fc2_activation = Agent.debug_printer(h_fc2_activation, "Fully 2 activation: ")
	# self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)
        self.h_fc2 = tf.maximum(ALPHA * h_fc2_activation, h_fc2_activation)
	#self.h_fc2 = Agent.debug_printer(self.h_fc2, "Fully 2 output: ")	

    def __form_output_layer(self):
        """
        Forms The output layer (linear in this case) - The value represents the value function for the action
        :return: None
        """
	self.W_fc3 = Agent.debug_printer(self.W_fc3, "FC3 weight: ")
        self.b_fc3 = Agent.debug_printer(self.b_fc3, "FC3 bias: ")
        self.readout = tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3
	self.readout = Agent.debug_printer(self.readout, "Output Layer output: ")

    def __create_network(self):
        """
        Creates the entire DQN network
        :return: None
        """
        self.__initialize_weights_and_biases()
        self.__form_input_layer()
        self.__form_hidden_layers()
        self.__form_output_layer()

    def __train_network(self):
        """
        Trains the DQN network
        :return: None
        """
        # define the cost function
        self.action = tf.placeholder("float", [None, self.num_actions])
        self.label = tf.placeholder("float", [None])

        self.readout_action = tf.reduce_sum(tf.mul(self.readout, self.action))
	self.cost = tf.reduce_mean(tf.square(self.label - self.readout_action))
	self.cost = Agent.debug_printer(self.cost, "Cost(MINIMIZE): ")
        train_step = tf.train.AdadeltaOptimizer(LEARNING_RATE).minimize(self.cost)

        # train
        # get the start state of the network
        imagenet = ImageNet(self.train_dir, self.grid_dim)
        imagenet.load_next_image()
        puzzle_pieces = imagenet.get_puzzle_pieces()
        original_image = imagenet.get_image()

        state = np.zeros([self.input_height, self.input_width, self.input_channels])

        # initialize the environment
        env = Environment(original_image, state, self.grid_dim, puzzle_pieces,
                          IMAGE_HEIGHT, WINDOW_SIZE, SLIDING_STRIDE,
                          self.input_channels)

        # saving and loading networks
        saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir + "saved_networks")

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print "Could not find old checkpoint weights"

        # start training
        epsilon = INITIAL_EPSILON
        t = 0

        # Initialize the replay memory
        replay_memory = deque()

        while True:
            # choose an action epsilon greedily
            readout_t = self.readout.eval(feed_dict={self.s: [state]})
	    #print readout_t
            a_t, action_index = self.__greedy_action(readout_t, epsilon)

            # update the environment with new action
            env.set_action(action_index)

            # get the reward and next state from the environment
            reward, state_new, terminal = env.get_state_reward_pair()

            # store the transition in replay memory
            replay_memory.append((state, a_t, reward, state_new, terminal))

            if len(replay_memory) > REPLAY_MEMORY:
                replay_memory.popleft()

            if t > OBSERVE:
                minibatch = random.sample(replay_memory, BATCH_SIZE)

                # get the batch variables
                s_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_new_batch = [d[3] for d in minibatch]

                y_batch = []
                readout_new_batch = self.readout.eval(feed_dict={self.s: s_new_batch})
                for i in range(0, len(minibatch)):
		    print i, len(minibatch)
                    terminal = minibatch[i][4]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_new_batch[i]))

                self.sess.run(train_step, feed_dict={self.label: y_batch, self.action: a_batch, self.s: s_batch})

            '''
            print y_batch, self.sess.run([self.readout_action],
                                         feed_dict={self.action: a_batch,
                                                    self.s: s_batch}), reward, action_index
            '''
            # if terminal state has reached, and number of tries per image has crossed threshold
            # then move to the new image
            if terminal:
                if imagenet.get_tries_per_image() == TRIES_PER_IMAGE:
                    image_present = imagenet.load_next_image()
                    if image_present:
                        puzzle_pieces = imagenet.get_puzzle_pieces()
                        original_image = imagenet.get_image()
                        env.update_puzzle_pieces(puzzle_pieces)
                        env.update_original_image(original_image)
                    else:
                        break
                else:
                    imagenet.increment_tries()

                env.reset()

            # update the old values
            state = env.get_state()
            t += 1

            # save progress every 10000 iterations
            if t % 10000 == 0:
                saver.save(self.sess, self.checkpoint_dir + 'saved_networks/' + GAME + '-dqn', global_step=t)
                # test the network on the validation data
                self.test_network()
            if t % 100 == 0:
                print t
            if t % 2000 == 0:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 100

    def test_network(self):
        """
        Network Testing is done on the validation data, and the testing data is to see if the representation
        is learnt efficiently. Testing data is not meant to check RL accuracy. The validation data is meant for that.
        Testing data is meant to check if the algorithm learnt feature representations of objects.
        :return: None
        """
        imagenet = ImageNet(self.val_dir, self.grid_dim)
        imagenet.load_next_image()
        puzzle_pieces = imagenet.get_puzzle_pieces()
        original_image = imagenet.get_image()
        state = np.zeros([self.input_height, self.input_width, self.input_channels])
        env = Environment(original_image, state, self.grid_dim, puzzle_pieces,
                          IMAGE_HEIGHT, WINDOW_SIZE, SLIDING_STRIDE,
                          self.input_channels)

        reward_list = []
        image_diff_list = []
        episode_reward = 0.0

        while True:
            # choose an action epsilon greedily
            readout_t = self.readout.eval(feed_dict={self.s: [state]})
            a_t, action_index = self.__greedy_action(readout_t, epsilon=0.0)

            env.set_action(action_index)
            reward, state_new, terminal = env.get_state_reward_pair()

            episode_reward = reward + GAMMA * episode_reward

            # if terminal state has reached, then move to the next image
            if terminal:
                image_diff_list.append(env.get_normalized_image_diff())
                reward_list.append(episode_reward)

                image_present = imagenet.load_next_image()

                if image_present:
                    puzzle_pieces = imagenet.get_puzzle_pieces()
                    original_image = imagenet.get_image()
                    env.update_puzzle_pieces(puzzle_pieces)
                    env.update_original_image(original_image)
                    env.reset()
                    episode_reward = 0.0
                else:
                    break

            # update the old values
            state = env.get_state()

        # display the reward and image matching performance statistics
        performance_statistics(image_diff_list, reward_list)

    def __greedy_action(self, value_function, epsilon):
        a_t = np.zeros([self.num_actions])

        if random.random() < epsilon:
            action_index = random.randrange(self.num_actions)
        else:
            action_index = np.argmax(value_function)

        a_t[action_index] = 1

        return a_t, action_index
