import argparse
from harpreif.agent import Agent

'''
SAMPLE RUN INSTRUCTIONS

Local Machine
python train.py --train_images './train'
                --val_images './val'
                --checkpoint_dir './'
                --grid_dim 8
                --num_gradients 8

Maverick -->
python train.py --train_images /work/03713/harshal1/maverick/RLProj/train
                --val_images '/work/03713/harshal1/maverick/RLProj/val'
                --checkpoint_dir '/work/03713/harshal1/maverick/RLProj/checkpoint/'
                --grid_dim 8
                --num_gradients 8
'''

# define the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_images', type=str)
parser.add_argument('--val_images', type=str)
parser.add_argument('--checkpoint_dir', type=str)
parser.add_argument('--grid_dim', type=int, default=8)
parser.add_argument('--num_gradients', type=int, default=8)
opt = parser.parse_args()
num_actions = opt.grid_dim ** 4

# create and train the agent
agent = Agent(num_actions, opt.grid_dim, opt.num_gradients)
agent.play_game(opt.train_images, opt.val_images, opt.checkpoint_dir)
