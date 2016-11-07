import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class NearestNeighbour(object):
    def __init__(self, image2feature_dict, feature_size):
        """
        Gets the nearest neighbour using cosine similarity metric
        :param image2feature_dict: The key, value pair of image and its corresponding feature
        :param feature_size: The dimension of each feature of each image
        """
        self.image2feature_dict = image2feature_dict
        self.numimages = len(self.image2feature_dict)
        self.feature_size = feature_size
        self.im2index = dict()
        self.index2im = dict()
        self.__vectorify()
        self.__construct_cs_matrix()

    def __vectorify(self):
        """
        Converts the map to a vector of dimension (num_images x feature_dim) and saves it as self.X
        :return: None
        """
        self.X = np.zeros([self.numimages, self.feature_size])
        counter = 0
        for key, value in self.image2feature_dict.iteritems():
            self.im2index[key] = counter
            self.index2im[counter] = key
            self.X[counter] = value

    def __construct_cs_matrix(self):
        """
        Creates as cosine similarity matrix between self.X and itself.
        :return: None
        """
        self.similarity_mat = cosine_similarity(self.X, self.X, dense_output=True)

    def save_nearest_neighbors(self, num_neighbors, out_file):
        """

        :param num_neighbors: Number of neighbors to be retrieved
        :param out_file: The output file where the neighbours of each image is to be stored
        :return: None
        """
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








