import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class NearestNeighbour(object):
    def __init__(self, image2feature_dict):
        self.image2feature_dict = image2feature_dict

    def __construct_cs_matrix(self):
        pass

