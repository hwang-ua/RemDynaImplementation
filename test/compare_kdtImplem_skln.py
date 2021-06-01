import sys
sys.path.append('../')
import numpy as np
from utils.kd_tree import *
import sklearn.neighbors as skln
import time
import matplotlib.pyplot as plt

def test1():
    print("\n\n-----------test1------------ compare ball tree and implementation")
    skl_tree = skln.BallTree(X, leaf_size=lsize,
                         metric=skln.DistanceMetric.get_metric('mahalanobis', VI=vi))
    kdt_tree = KD_Tree(dim)
    kdt_tree.rebuild(X)

    for seq in targets:
        skl_dist, skl_index = skl_tree.query([seq], k=ksize)
        kdt_seq, kdt_index, kdt_dist = kdt_tree.k_nearest(ksize, seq)

        print("\n****\nseq =", seq)
        print("skl res =", skl_dist, skl_index)
        print("kdt res =", kdt_dist, kdt_index)
        print("indexed seq =", X[skl_index])
        print("kdt indexed =", kdt_seq)


def test2():
    print("\n\n-----------test2------------ compare kd tree and implementation")

    skl_tree = skln.KDTree(X, leaf_size=lsize)
    kdt_tree = KD_Tree(dim)
    kdt_tree.rebuild(X)

    for seq in targets:
        skl_dist, skl_index = skl_tree.query([seq], k=ksize)
        kdt_seq, kdt_index, kdt_dist = kdt_tree.k_nearest(ksize, seq)

        print("\n****\nseq =", seq)
        print("skl res =", skl_dist, skl_index)
        print("kdt res =", kdt_dist, kdt_index)
        print("indexed seq =", X[skl_index])
        print("kdt indexed =", kdt_seq)


def test3():
    print("\n\n-----------test3------------ test running time")
    t1 = time.time()
    skl_kd_tree = skln.KDTree(X, leaf_size=lsize)
    print("skl  kd  tree  construction", time.time() - t1)

    t2 = time.time()
    skl_bl_tree = skln.BallTree(X, leaf_size=lsize,
                             metric=skln.DistanceMetric.get_metric('mahalanobis', VI=vi))
    print("skl ball tree  construction", time.time() - t2)

    t3 = time.time()
    kdt_tree = KD_Tree(dim)
    kdt_tree.rebuild(X)
    print("implementation construction", time.time() - t3)

    t4 = time.time()
    kdt_tree_ub = KD_Tree(dim)
    kdt_tree_ub.rebuild(X[:len(X)//2])
    print("implementation construction (half data)", time.time() - t4)
    t5 = time.time()
    for i in range(len(X)//2, len(X)):
        kdt_tree_ub.insert(X[i], i)
    print("implementation insert (half data)", time.time() - t5)


    run_time = [0.0] * 4
    for seq in targets:
        ts1 = time.time()
        skl_kd_dist, skl_kd_index = skl_kd_tree.query([seq], k=ksize)
        run_time[0] += time.time() - ts1

        ts2 = time.time()
        skl_bl_dist, skl_bl_index = skl_bl_tree.query([seq], k=ksize)
        run_time[1] += time.time() - ts2

        ts3 = time.time()
        kdt_seq, kdt_index, kdt_dist = kdt_tree.k_nearest(ksize, seq)
        run_time[2] += time.time() - ts3

        ts4 = time.time()
        kdt_seq, kdt_index, kdt_dist = kdt_tree_ub.k_nearest(ksize, seq)
        run_time[3] += time.time() - ts4

        print(run_time)

    print("skl  kd  tree  searching", run_time[0])
    print("skl ball tree  searching", run_time[1])
    print("implementation searching", run_time[2])
    print("unbalanced imp searching", run_time[3])

def test4():
    print("\n\n-----------test4------------ compare kd tree and implementation")

    # skl_kdtree = skln.KDTree(X, leaf_size=lsize)
    skl_balltree = skln.BallTree(X, leaf_size=lsize, metric=skln.DistanceMetric.get_metric('mahalanobis', VI=vi))
    kdt_tree = KD_Tree(dim)
    kdt_tree.rebuild(X)

    for i in range(len(targets)):
        seq = targets[i]
        # skl_kddist, skl_kdindex = skl_kdtree.query([seq], k=ksize)
        skl_balldist, skl_ballindex = skl_balltree.query([seq], k=ksize)
        kdt_seq, kdt_index, kdt_dist = kdt_tree.k_nearest(ksize, seq)

        print("\n****\nseq =", seq)
        # print("skl kd res =", skl_kddist, skl_kdindex)
        print("skl ball res =", skl_balldist, skl_ballindex)
        # print("kdt res =", kdt_dist, kdt_index)
        # print("kd indexed seq =", X[skl_kdindex])
        print("ball indexed seq =", X[skl_ballindex])
        print("kdt indexed =", kdt_seq)
        sys.stdout.flush()

        only_ball = []
        both = []
        for res in skl_ballindex[0]:
            if res not in kdt_index:
                only_ball.append(res)
            else:
                both.append(res)

        plt.figure(i)
        plt.plot(X[:, 0], X[:, 1], '.', color='lightgrey')
        # plt.plot(X[skl_kdindex[0]][:, 0], X[skl_kdindex[0]][:, 1], '.', color='orange')
        plt.plot(kdt_seq[:, 0], kdt_seq[:, 1], '.', color='red')
        plt.plot(X[both][:, 0], X[both][:, 1], '.', color='green')
        plt.plot(X[only_ball][:, 0], X[only_ball][:, 1], '.', color='blue')
        plt.plot(seq[0], seq[1], '.', color='black')

    plt.show(block=True)

def test5():
    print("\n\n-----------test5------------ compare kd tree and implementation")
    
    t1 = time.time()
    skl_bl_tree = skln.BallTree(X1, leaf_size=lsize,
                             metric=skln.DistanceMetric.get_metric('mahalanobis', VI=vi))
    print("skl ball tree  construction", time.time() - t1)

    t2 = time.time()
    kdt_tree_ub = KD_Tree(dim)
    kdt_tree_ub.rebuild(X1)
    print("implementation construction", time.time() - t2)
    
    t3 = time.time()
    for j in range(len(X1), len(X)):
        skl_bl_tree = skln.BallTree(X[:j], leaf_size=lsize, metric=skln.DistanceMetric.get_metric('mahalanobis', VI=vi))
        seq = targets[j%len(targets)]
        skl_bl_dist, skl_bl_index = skl_bl_tree.query([seq], k=ksize)
    print("sklearn ball t rebuild and search", time.time() - t3)

    t4 = time.time()
    for i in range(len(X1), len(X)):
        kdt_tree_ub.insert(X[i], i)
        seq = targets[i%len(targets)]
        kdt_tree_ub.k_nearest(ksize, seq)
    print("implementation insert and search", time.time() - t4)



dim = 5
# X = np.array([[51, 75], [70, 70], [55, 1], [60, 80], [10, 30], [35, 90], [25, 40], [1, 10], [50, 50]])
X = np.array(np.random.random(50000).reshape((10000,5)))
trunc = -5000
X1 = X[: trunc]
X2 = X[trunc:]
lsize = len(X)//2
ksize = 200
# targets = [[1, 1], [1, 100], [60, 80], [60, 81]]
targets = X[-5:]
# targets = np.array(np.random.random(2).reshape((1,2)))

v = np.cov(X.T)
vi = np.linalg.inv(v + np.eye(dim)*0.0001)

test5()