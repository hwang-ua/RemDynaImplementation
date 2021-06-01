import numpy as np
import time
# from utils.max_heap import HeapNode
from utils.max_heap import MaxHeap

class Node:
    def __init__(self, val, ind):
        self.val = val
        self.ind = ind
        self.rc = None
        self.lc = None
        return

    def add_rc(self, rc):
        self.rc = rc

    def add_lc(self, lc):
        self.lc = lc

    def get_val(self):
        return self.val

    def get_ind(self):
        return self.ind

    def get_lc(self):
        return self.lc

    def get_rc(self):
        return self.rc

    def is_leaf(self):
        return (self.rc is None) and (self.lc is None)

class KD_Tree:
    def __init__(self, dim, n=10000):
        self.dim = dim
        # self.indices = np.array([0,1,3,4])
        self.root = None
        self.n = n
        self.b = 0
        self.data_array = np.zeros((self.n, dim))
        self.sum = np.zeros(dim)
        self.cov_first = np.zeros((dim, dim))
        self.covmat = np.zeros((dim, dim))
        self.covmat_inv = np.zeros((dim, dim))
        self.nearest_dist = np.inf

        self.running_time = {"insert":0.0,
                             "rebuild":0.0,
                             "knn":0.0}
        return

    def insert(self, val, ind):
        start = time.time()
        if self.b < self.n:
            self.data_array[self.b] = val
        else:
            np.concatenate((self.data_array, np.array([val])), axis = 0)
        self.b += 1

        self.sum += val
        mu = self.sum / float(self.b)
        self.cov_first = ((self.b - 1.0) * self.cov_first + np.outer(val, val)) / float(self.b)
        self.covmat = self.cov_first - np.outer(mu, mu)
        self.covmat_inv = np.linalg.inv(self.covmat + np.eye(len(val)) * 0.0001)

        node = Node(val, ind)

        if self.root is None:
            self.root = node
            return

        previous = None
        rorl = None
        current = self.root
        done = False
        i = 0
        while not done:
            if current is None:
                if rorl == 'l':
                    previous.add_lc(node)
                elif rorl == 'r':
                    previous.add_rc(node)
                done = True
            elif val[i % self.dim] < current.get_val()[i % self.dim]:
                previous = current
                rorl = 'l'
                current = current.get_lc()
            elif val[i % self.dim] >= current.get_val()[i % self.dim]:
                previous = current
                rorl = 'r'
                current = current.get_rc()
            i += 1
        self.running_time["insert"] += time.time() - start
        return

    def rebuild(self, data = None):
        start = time.time()
        self.root = None
        temp_data = np.copy(self.data_array[:self.b]) if data is None else data
        temp_inds = np.array([i for i in range(len(temp_data))])

        self.b = len(temp_data)
        self.sum = np.sum(temp_data, axis=0)
        if self.sum is not None:
            mu = self.sum / float(self.b)
            self.covmat = np.cov(temp_data.T)
            self.cov_first = self.covmat + np.outer(mu, mu)
            self.covmat_inv = np.linalg.inv(self.covmat + np.eye(len(temp_data[0])) * 0.0000001)

        count = 0
        self.root, ld, rd, li, ri = self._find_med(count, temp_data, temp_inds)
        parent = self.root
        self._rebuild_lc(parent, count + 1, ld, li)
        self._rebuild_rc(parent, count + 1, rd, ri)

        self.running_time["rebuild"] += time.time() - start
        return self.root

    def get_tree(self):
        return self.root

    def get_data(self):
        return self.data_array

    def get_size(self):
        return self.b

    def check_time(self):
        res = self.running_time.copy()
        res["tsize"] = self.b
        for key in self.running_time.keys():
            self.running_time[key] = 0.0
        return res

    def k_nearest(self, k, target, covmat_inv=None):
        covmat_inv = self.covmat_inv if covmat_inv is None else covmat_inv
        start = time.time()
        self.nearest_pt = MaxHeap()
        self.nearest_dist = np.inf
        res = self._rec_k_nearest(0, self.root, k, target, covmat_inv).get_heap()
        self.running_time["knn"] += time.time() - start
        return res

    def _rec_k_nearest(self, level, current, k, target, covmat_inv):

        if current == None:
            return self.nearest_pt

        if self.nearest_pt.get_size() < k:
            dist = self._kernel_pt_dist(current, target, covmat_inv)
            self.nearest_pt.insert(current.get_val(), current.get_ind(), dist)
            self.nearest_dist = self.nearest_pt.get_max_dist()
            d = level % self.dim
            tvalue = target[d]
            rvalue = current.get_val()[d]
            if tvalue < rvalue:
                self.nearest_pt = self._rec_k_nearest(level + 1, current.get_lc(), k, target, covmat_inv)
                self.nearest_pt = self._rec_k_nearest(level + 1, current.get_rc(), k, target, covmat_inv)
            else:
                self.nearest_pt = self._rec_k_nearest(level + 1, current.get_rc(), k, target, covmat_inv)
                self.nearest_pt = self._rec_k_nearest(level + 1, current.get_lc(), k, target, covmat_inv)
            return self.nearest_pt

        if self._kernel_space_dist(current, target, level, covmat_inv) > self.nearest_dist:
            return self.nearest_pt

        rdist = self._kernel_pt_dist(current, target, covmat_inv)
        if rdist < self.nearest_dist:
            self.nearest_pt.insert(current.get_val(), current.get_ind(), rdist)
            self.nearest_pt.del_max()
            self.nearest_dist = self.nearest_pt.get_max_dist()

        d = level % self.dim
        tvalue = target[d]
        rvalue = current.get_val()[d]
        if tvalue < rvalue:
            self.nearest_pt = self._rec_k_nearest(level + 1, current.get_lc(), k, target, covmat_inv)
            self.nearest_pt = self._rec_k_nearest(level + 1, current.get_rc(), k, target, covmat_inv)
        else:
            self.nearest_pt = self._rec_k_nearest(level + 1, current.get_rc(), k, target, covmat_inv)
            self.nearest_pt = self._rec_k_nearest(level + 1, current.get_lc(), k, target, covmat_inv)
        return self.nearest_pt

    def _kernel_space_dist(self, current, pt, level, covmat_inv):
        d = level % self.dim
        diff = current.get_val()[d] - pt[d]
        return 1.0 - np.exp(-1 * diff*covmat_inv[d,d]*diff)

    def _kernel_pt_dist(self, current, pt, covmat_inv):
        diff = current.get_val() - pt
        return 1.0 - np.exp(-1 * diff.dot(covmat_inv).dot(diff.T))

    def visualization(self, node, pref=''):
        print(pref + str(node.get_val())+"("+str(node.get_ind())+")")
        if node.get_lc() is not None:
            self.visualization(node.get_lc(), ' '*len(pref)+'l---')
        if node.get_rc() is not None:
            self.visualization(node.get_rc(), ' '*len(pref)+'r---')
        return

    def _rebuild_lc(self, parent, count, data, inds):
        if len(data) == 1:
            parent.add_lc(Node(data[0], inds[0]))
            return
        elif len(data) == 0:
            return
        else:
            node, ld, rd, li, ri = self._find_med(count, data, inds)
            parent.add_lc(node)
            self._rebuild_lc(node, count + 1, ld, li)
            self._rebuild_rc(node, count + 1, rd, ri)
            return

    def _rebuild_rc(self, parent, count, data, inds):
        if len(data) == 1:
            parent.add_rc(Node(data[0], inds[0]))
            return
        elif len(data) == 0:
            return
        else:
            node, ld, rd, li, ri = self._find_med(count, data, inds)
            parent.add_rc(node)
            self._rebuild_lc(node, count + 1, ld, li)
            self._rebuild_rc(node, count + 1, rd, ri)
            return

    def _find_med(self, count, data, inds):
        comp = data[:, count % self.dim]
        order = np.argsort(comp, kind='mergesort')
        med_i = order[len(order) // 2]
        node = Node(data[med_i], inds[med_i])

        left_data = data[order[:len(order) // 2]]
        right_data = data[order[len(order) // 2 + 1:]]
        left_ind = inds[order[:len(order) // 2]]
        right_ind = inds[order[len(order) // 2 + 1:]]
        return node, np.array(left_data), np.array(right_data), np.array(left_ind), np.array(right_ind)

