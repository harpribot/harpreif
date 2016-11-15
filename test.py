import sys
from rep_viz.state_indexer import Image2Feature
'''
#python test.py '/work/03713/harshal1/maverick/RLProj/test' '/work/03713/harshal1/maverick/RLProj/checkpoint/' 4 8
args = sys.argv

test_dir = args[1]
checkpoint_dir = args[2]

grid_dim = int(args[3])

num_actions = grid_dim ** 4
num_gradients = int(args[4])

im2f = Image2Feature(test_dir, checkpoint_dir, num_actions, num_gradients)

image2feature_map, feat_sz = im2f.image2feature(save_transform=True, im2f_loc= './')
'''
from rep_viz.representation import NearestNeighbour
import cPickle as pickle

image2feature_map = pickle.load(open('image2feature.p', 'rb'))
feat_sz = None
for key, value in image2feature_map.iteritems():
    image2feature_map[key] = value[0]
    if feat_sz is None:
        feat_sz = value[0].size

num_neighbors = 5
out_file = './neighbors.txt'
nb_obj = NearestNeighbour(image2feature_map, feat_sz)
nb_obj.save_nearest_neighbors(num_neighbors, out_file)







