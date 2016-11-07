import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class NearestNeighbour(object):
    def __init__(self, image2feature_dict, feature_size):
        self.image2feature_dict = image2feature_dict
        self.numimages = len(self.image2feature_dict)
        self.feature_size = feature_size
        self.im2index = dict()
        self.index2im = dict()
        self.__vectorify()
        self.__construct_cs_matrix()

    def __vectorify(self):
        self.X = np.zeros([self.numimages, self.feature_size])
        counter = 0
        for key, value in self.image2feature_dict.iteritems():
            self.im2index[key] = counter
            self.index2im[counter] = key
            self.X[counter] = value

    def __construct_cs_matrix(self):
        self.similarity_mat = cosine_similarity(self.X, self.X, dense_output=True)

    def save_nearest_neighbors(self, num_neighbors, out_file):
        result_list = []
        for key, value in self.im2index.iteritems():
            neighbor_list = [key]
            similarity_scores = self.similarity_mat[value]
            ind = np.argpartition(similarity_scores, -num_neighbors)[-num_neighbors:]
            ind = ind[np.argsort(similarity_scores[ind])]
            neighbors = [self.index2im[x] for x in ind]
            neighbor_list.extend(neighbors)

            result_list.append(neighbor_list)

        with open(out_file, 'wb') as out:
            for item in result_list:
                print>>out, item








