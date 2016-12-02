from rep_viz.visualize_net import Net
import argparse

'''
SAMPLE RUN INSTRUCTIONS
python vizualize_network.py --saved_checkpoint './checkpoint/' --grid_dim 4 --num_gradients 8
'''

parser = argparse.ArgumentParser()
parser.add_argument('--saved_checkpoint', type=str)
parser.add_argument('--grid_dim', type=int, default=8)
parser.add_argument('--num_gradients', type=int, default=8)
opt = parser.parse_args()

num_actions = opt.grid_dim ** 4

net = Net(num_actions, opt.num_gradients, opt.saved_checkpoint)
net.display_weights()

# net.display_biases()

