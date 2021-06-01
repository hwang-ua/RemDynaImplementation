import numpy as np
import random
import sys

class KMeans:
    def __init__(self, k, dim_sample, dim_state):
        self.k = k
        self.dim_sample = dim_sample
        self.dim_state = dim_state

    def kernel_dist(self, obj_a, obj_b):
        diff = obj_a - obj_b
        diff = diff[self.ssr_i]
        #return 1.0 - a_indicator * np.exp(-0.5*(diff*self.covmat_inv).T.dot(diff))
        return 1.0 - (x[self.act_i] == y[self.act_i])*np.exp(-0.5*diff.T.dot(self.covmat_inv).dot(diff))

class Blocks:
    def __init__(self, num_block, len_seq, prot_array, rerun_freq=10):
        self.count = 0
        self.num_block = num_block
        self.len_seq = len_seq
        self.prot_array = prot_array
        self.rerun_freq = rerun_freq

        self.centers = np.zeros((num_block, len_seq))
        return

    def clustering(self):

        return