from harpreif.agent import Agent

image_dir = './imagenet'

grid_dim = 2

num_actions = grid_dim ** 4
num_gradients = 8

agent = Agent(num_actions, grid_dim, num_gradients)
agent.play_game(image_dir)


