import numpy as np
import random
import time
import sklearn.neighbors as skln
# import sys
# sys.path.append("../")
# import utils.scikit_learn.sklearn.neighbors as skln


def _k_near_neighbor(self, X, seq, lsize, vi, ksize):
    start = time.time()
    tree = skln.BallTree(X, leaf_size=lsize,
                         metric=skln.DistanceMetric.get_metric('mahalanobis', VI=vi))
    # tree = skln.KDTree(X, leaf_size=lsize)
    cons = time.time()
    dist, index = tree.query([seq], k=ksize)
    search = time.time()
    self.running_time["cons"] += cons - start
    self.running_time["search"] += search - cons
    return dist[0], index[0]


# X = np.array(np.random.random(2000).reshape((1000,2)))
# X = np.array([[51, 75], [70, 70], [55, 1], [60, 80], [10, 30], [35, 90], [25, 40], [1, 10], [50, 50]])
X = np.array([[6],[2],[8],[4],[5]])

X2 = np.array(np.random.random(20).reshape((10,2)))

dim = X.shape[1]
lsize = len(X)//2
ksize = 2
v = np.cov(X.T)
vi = np.linalg.inv(v + np.eye(dim)*0.0001)

seq = np.array([0.5, 0.5])

tree = skln.BallTree(X, leaf_size=lsize, metric=skln.DistanceMetric.get_metric('mahalanobis', VI=vi))
print(tree.node_data)
print(np.asarray(tree.node_data))
tree_info = list(tree.__getstate__())
print("\n**__getstate__\n")
for i in range(len(tree_info)):
    print(i, tree_info[i])
print("\n**array in order:", tree_info[0][tree_info[1]])

print("\n**before chage", tree.query([seq], k=ksize))

vi2 = np.linalg.inv(np.cov(X2.T) + np.eye(dim)*0.0001)
tree2 = skln.BallTree(X2, leaf_size=len(X2)//2, metric=skln.DistanceMetric.get_metric('mahalanobis', VI=vi2))
print("\n**right answer", tree2.query([seq], k=ksize))
print("            ", tree2.__getstate__())

tree_info[0] = tree2.__getstate__()[0]
tree.__setstate__(tuple(tree_info))
print(tree.__getstate__())
print("\n**after chage", tree.query([seq], k=ksize))

#
# print(np.asarray(tree.data))
# print(np.asarray(tree.idx_array))
# tree.data = memoryview(X2)
# print(np.asarray(tree.data))
# print(tree.get_arrays())
