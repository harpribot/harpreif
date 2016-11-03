import sys
from harpreif.agent import Agent

# python train.py './train' './val' './' 2 8
# condorify_gpu_email python train.py '/scratch/cluster/harshal/RLProj/train' '/scratch/cluster/harshal/RLProj/val' '/scratch/cluster/harshal/RLProj/checkpoint/' 2 8
args = sys.argv

train_dir = args[1]
val_dir = args[2]

checkpoint_dir = args[3]

grid_dim = int(args[4])

num_actions = grid_dim ** 4
num_gradients = int(args[5])

agent = Agent(num_actions, grid_dim, num_gradients)
agent.play_game(train_dir, val_dir, checkpoint_dir)
