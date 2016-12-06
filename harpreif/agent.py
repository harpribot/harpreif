import tensorflow as tf
from collections import deque
from image_handler import ImageNet
from environment import Environment
from image_utils import performance_statistics
import random
import numpy as np
from myconstants import *
from model.creator import Creator
import sys
import cPickle as pickle


class Agent(Creator):
    def __init__(self, num_actions, grid_dim, num_gradients, state_type, mean_removal, jitter, mean_file):
        """

        :param num_actions: Number of actions possible for the agent - The encoding is such that,
                            if i_th piece is to be placed in j_th location (of grid_dim ** 2 = N possible)
                            locations, then the corresponding action index = i * N + j
        :param grid_dim: Number of horizontal (equalling vertical breaks) on the original image to form pieces
        :param num_gradients: Number of bins for HOG (Histogram of Oriented Gradients) for each patch in a
                            sliding window across the jigsaw image (the image that is already been constructed)
        :param state_type: 'hog' -> state is windowed HOG filter ,
                           'image' -> state is just the partially solved jigsaw image
        :param mean_removal: True if the imagenet mean is to be removed from the input image, else False.
                            Default = True
        :param mean_file: The location of the imagenet mean file
        :param jitter: True when the jigsaw piece is to be jittered (corner supression, + random rotation),
                        Dafault = False
        """
        self.state_type = state_type
        self.mean_removal = mean_removal
        self.jitter = jitter
        self.grid_dim = grid_dim
        self.num_gradients = num_gradients
        self.num_actions = num_actions
        self.mean_file = mean_file
        self.input_height = len(range(0, IMAGE_HEIGHT - SLIDING_STRIDE, SLIDING_STRIDE))
        self.input_width = self.input_height
        if self.state_type == 'hog':
            self.input_channels = self.num_gradients
        elif self.state_type == 'image':
            self.input_channels = 1
        else:
            raise ValueError('State type not recognized, enter hog or image')

        self.sess = None
        self.train_dir = None
        self.val_dir = None
        self.test_dir = None
        self.checkpoint_dir = None
        self.image_handled = 0
        Creator.__init__(self, self.input_channels, self.num_actions, self.input_height, self.input_width)

    def play_game(self, train_dir, val_dir, checkpoint_dir, reward_type):
        """
        Initiates gameplay using DQN based reinforcement learning
        :param train_dir: The directory containing the training images
        :param val_dir: The directory containing the testing images
        :param checkpoint_dir: The directory where checkpoints are to be stored.
        :param reward_type: The reward type: 0 -> normalized_image_diff  + matching ,
                                             1 -> matching,
                                             2 -> matching + placing
        :return: None
        """
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.checkpoint_dir = checkpoint_dir
        sys.stderr.write('Initializing Session...\n')
        self.sess = tf.InteractiveSession()
        sys.stderr.write('Creating Network...\n')
        self.__create_network()
        sys.stderr.write('Training Network...\n')
        self.__train_network(reward_type)

    def __create_network(self):
        """
        Creates the entire DQN network
        :return: None
        """
        self._initialize_weights_and_biases()
        self._form_input_layer()
        self._form_hidden_layers()
        self._form_output_layer()
        self._define_loss()
        self._form_trainer()

    def __get_image_loader(self, num_images=None):
        """
        Loads the image iterator
        :return: image iterator
        """
        return ImageNet(self.train_dir, self.grid_dim, self.mean_removal, self.jitter, self.mean_file, num_images)

    def __get_initial_state(self):
        """
        Returns the starting state for the RL agent
        :return: starting state
        """
        return np.zeros([self.input_height, self.input_width, self.input_channels])

    def __model_loader(self):
        """
        Loads the model from the checkpoint
        :return: None
        """
        self.saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir + "saved_networks")

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            sys.stderr.write("Successfully loaded: " + checkpoint.model_checkpoint_path + '\n')
        else:
            sys.stderr.write("Could not find old checkpoint weights\n")

    def __play_one_move(self, state, env, reward_type, epsilon, time=None):
        """
        Plays for one step, i.e. agent places one piece on the board.
        :param state:
        :param env:
        :param epsilon:
        :param time:
        :return:
        """
        readout_t = self.readout.eval(feed_dict={self.s: [state]})
        a_t, action_index = self.__greedy_action(readout_t, epsilon, time)
        # update the environment with new action
        env.set_action(action_index)
        # get the reward and next state from the environment
        reward, state_new, terminal = env.get_state_reward_pair(reward_type)
        sys.stderr.write(str(reward) + '\n')

        return state_new, a_t, reward, terminal

    def __train_minibatch(self, replay_memory, learning_rate):
        """
        Trains the model for a minibatch
        :param replay_memory:
        :param learning_rate:
        :return: None
        """
        minibatch = random.sample(replay_memory, BATCH_SIZE)

        # get the batch variables
        s_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        s_new_batch = [d[3] for d in minibatch]

        y_batch = []
        readout_new_batch = self.readout.eval(feed_dict={self.s: s_new_batch})

        for i in range(0, len(minibatch)):
            terminal = minibatch[i][4]
            # if terminal, only equals reward
            if terminal:
                y_batch.append(r_batch[i])
            else:
                y_batch.append(r_batch[i] + GAMMA * np.max(readout_new_batch[i]))

        self.sess.run(self.train_step, feed_dict={self.label: y_batch,
                                                  self.action: a_batch,
                                                  self.s: s_batch,
                                                  self.learning_rate_tf: learning_rate
                                                  })

    def __train_network(self, reward_type):
        """
        Trains the DQN network
        :return: None
        """
        imagenet = self.__get_image_loader()
        imagenet.load_next_image()
        state = self.__get_initial_state()
        # initialize the environment
        env = Environment(imagenet.get_image(), state, self.grid_dim, imagenet.get_puzzle_pieces(),
                          IMAGE_HEIGHT, WINDOW_SIZE, SLIDING_STRIDE,
                          self.input_channels, self.state_type)
        # saving and loading networks
        self.__model_loader()
        # initialize parameters and replay memory
        epsilon = INITIAL_EPSILON
        learning_rate = LEARNING_RATE
        replay_memory = deque()
        # Start Training
        t = 0
        episode_reward = 0.
        episode_reward_list = []
        while True:
            state_new, a_t, reward, terminal = self.__play_one_move(state, env, reward_type, epsilon, t)
            episode_reward = reward + GAMMA * episode_reward
            # store the transition in replay memory
            replay_memory.append((state, a_t, reward, state_new, terminal))
            # if the replay memory is full remove the leftmost entry
            if len(replay_memory) > REPLAY_MEMORY:
                replay_memory.popleft()
            # if observation is completed then start training mini-batches
            if t > OBSERVE:
                self.__train_minibatch(replay_memory, learning_rate)
            # if terminal state has reached, and number of tries per image has crossed threshold
            # then move to the new image
            if terminal:
                if imagenet.get_tries_per_image() == TRIES_PER_IMAGE:
                    self.image_handled += 1
                    # update learning rate after handling number of images = NUMBER_OF_IMAGES_FOR_DECAY
                    if self.image_handled % NUMBER_OF_IMAGES_FOR_DECAY == 0:
                        learning_rate /= LEARNING_DECAY
                    # test the network on the validation data after training on certain number of images
                    # if self.image_handled % NUM_IMAGES_PER_VALIDATION == 1:
                    #    self.test_network(reward_type)
                    episode_reward_list.append(episode_reward)
                    image_present = imagenet.load_next_image()
                    if image_present:
                        episode_reward = 0.
                        env.update_puzzle_pieces(imagenet.get_puzzle_pieces())
                        env.update_original_image(imagenet.get_image())
                    else:
                        break
                else:
                    episode_reward = 0.
                    imagenet.increment_tries()
            # update the old values
            state = env.get_state()
            t += 1
            # save progress every 10000 iterations
            if t % ITERATIONS_PER_CHECKPOINT == 1:
                pickle.dump(episode_reward_list, open(self.checkpoint_dir +
                                                      'saved_networks/' + 'episode_reward-' + str(t) + '.p', 'wb'))
                self.saver.save(self.sess, self.checkpoint_dir + 'saved_networks/' + GAME + '-dqn', global_step=t)
            if t % 500 == 0:
                sys.stderr.write(str(t) + '\n')
            # increase greediness every 4000 iterations
            if t % ITERATIONS_PER_EPSILON_DECAY == 0:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 100

    def test_network(self, reward_type):
        """
        Network Testing is done on the validation data, and the testing data is to see if the representation
        is learnt efficiently. Testing data is not meant to check RL accuracy. The validation data is meant for that.
        Testing data is meant to check if the algorithm learnt feature representations of objects.
        :return: None
        """
        imagenet = self.__get_image_loader(NUM_VALIDATION_IMAGES)
        imagenet.load_next_image()

        state = self.__get_initial_state()
        # initialize the environment
        env = Environment(imagenet.get_image(), state, self.grid_dim, imagenet.get_puzzle_pieces(),
                          IMAGE_HEIGHT, WINDOW_SIZE, SLIDING_STRIDE,
                          self.input_channels, self.state_type)
        reward_list = []
        image_diff_list = []
        episode_reward = 0.0
        while True:
            state_new, a_t, reward, terminal = self.__play_one_move(state, env, reward_type, epsilon=0.0)
            episode_reward = reward + GAMMA * episode_reward
            # if terminal state has reached, then move to the next image
            if terminal:
                image_diff_list.append(env.get_normalized_image_diff())
                reward_list.append(episode_reward)

                image_present = imagenet.load_next_image()

                if image_present:
                    env.update_puzzle_pieces(imagenet.get_puzzle_pieces())
                    env.update_original_image(imagenet.get_image())
                    episode_reward = 0.0
                else:
                    break
            # update the old values
            state = env.get_state()
        # display the reward and image matching performance statistics
        performance_statistics(image_diff_list, reward_list)

    def __greedy_action(self, value_function, epsilon, t=None):
        """
        Returns an epsilon greedy action
        :param value_function:
        :param epsilon:
        :param t:
        :return:
        """
        a_t = np.zeros([self.num_actions])

        if t is not None:
            epsilon = OBSERVE_EPSILON if t < OBSERVE else epsilon

        if random.random() < epsilon:
            action_index = random.randrange(self.num_actions)
        else:
            action_index = np.argmax(value_function)

        a_t[action_index] = 1

        return a_t, action_index
