from rep_viz.representation import NearestNeighbour
import cPickle as pickle
import argparse

'''
SAMPLE RUN INSTRUCTIONS

python test_nb_finder.py --image_feat_pickle './image2feature.p' --nb_out_file './neighbors.txt' --num_neighbors 100
'''

parser = argparse.ArgumentParser()
parser.add_argument('image_feat_pickle', type=str, default='./image2feature.p')
parser.add_argument('nb_out_file', type=str, default='./neighbors.txt')
parser.add_argument('num_neighbors', type=int, default=10)
opt = parser.parse_args()


image2feature_map = pickle.load(open(opt.image_feat_pickle, 'rb'))
feat_sz = None

for key, value in image2feature_map.iteritems():
    image2feature_map[key] = value[0]
    if feat_sz is None:
        feat_sz = value[0].size

nb_obj = NearestNeighbour(image2feature_map, feat_sz)
result_list = nb_obj.save_nearest_neighbors(opt.num_neighbors, opt.nb_out_file)

