import sys
#from rep_viz.representation import NearestNeighbour
from rep_viz.state_indexer import  Image2Feature

#python test.py '/work/03713/harshal1/maverick/RLProj/test' '/work/03713/harshal1/maverick/RLProj/checkpoint/' 2 8
args = sys.argv

test_dir = args[1]
checkpoint_dir = args[2]

grid_dim = int(args[3])

num_actions = grid_dim ** 4
num_gradients = int(args[4])

im2f = Image2Feature(test_dir, checkpoint_dir, num_actions, num_gradients)

image2feature_map, feat_sz = im2f.image2feature()

#nb_obj = NearestNeighbour(image2feature_map, feat_sz)
#nb_obj.save_nearest_neighbors()






