# list of all constants initialized for the harpreif module

GAME = 'jigsaw'                      # name of the game
LEARNING_RATE = 1e-4                 # learning rate initial
LEARNING_DECAY = 2                   # learning is halved every 100 images.
NUMBER_OF_IMAGES_FOR_DECAY = 150     # number of images before halving the learning rate
INITIAL_EPSILON = 0.3                # the starting epsilon of the learning phase in training
FINAL_EPSILON = 0.05                 # the final epsilon of the learning phase in training
OBSERVE_EPSILON = 0.9                # the random action selection during observe phase. kept high to get good dataset
OBSERVE = 5000                       # the number of iterations we use to observe
REPLAY_MEMORY = 5000                 # the number of elements to keep in the replay memory
BATCH_SIZE = 256                     # the batch size for training
GAMMA = 0.99                         # discount factor for reinforcement learning
WINDOW_SIZE = [8, 8]                 # window dimension over which hog is computed
SLIDING_STRIDE = WINDOW_SIZE[0]/2    # sliding stride for the window used for HOG
IMAGE_HEIGHT = IMAGE_WIDTH = 256     # the dimension of a true image
TRIES_PER_IMAGE = 1                  # Number of tries to do per image
ALPHA = 0.01                         # Leaky RELU parameter - prevents dyeing neurons
DELAY_REWARD = -0.05                 # the reward given for each action that leads to non-terminal state
REPLACING_PENALTY = -0.1             # Penalty when removing an already placed piece
STEPS_MAX = 75                       # The maximum allowed step for each episode
NUM_BINS = 16                        # Number of bins in which the histogram is discretised
NUM_VALIDATION_IMAGES = 100          # Number of images to be used for validation testing
NUM_IMAGES_PER_VALIDATION = 500      # Number of images to train upon before validating
ITERATIONS_PER_CHECKPOINT = 10000    # Number of iterations per checkpoint operation
ITERATIONS_PER_EPSILON_DECAY = 5000  # Number of iterations before increasing greediness
