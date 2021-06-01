import sklearn.neighbors as skln
import numpy as np

class KD_Tree:
    def __init__(self, data_arr, lsize, vi):
        self.tree_info = None
        self.tree = self.build_tree(data_arr, lsize, vi)
        return self.tree

    def build_tree(self, data_arr, lsize, vi):
        self.tree = skln.BallTree(data_arr,
                                  leaf_size=lsize,
                                  metric=skln.DistanceMetric.get_metric('mahalanobis', VI=vi)
                                  )
        self.tree_info = self.tree.__getstate__()
        """
        self.tree_info = (
                        self.data_arr,
                        self.idx_array_arr,
                        self.node_data_arr,
                        self.node_bounds_arr,
                        int(self.leaf_size),
                        int(self.n_levels),
                        int(self.n_nodes),
                        int(self.n_trims),
                        int(self.n_leaves),
                        int(self.n_splits),
                        int(self.n_calls),
                        self.dist_metric,
                        self.sample_weight
                        )
        """
        return self.tree

    def insert(self, new_seq):
        data_arr = self.tree_info[0]
        idx_array_arr = self.tree_info[1]