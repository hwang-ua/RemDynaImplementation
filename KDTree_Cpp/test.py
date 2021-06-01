import numpy as np

import sys
import sklearn.neighbors as skln

def test1():
    tree = kd.KDTree()
    tree.BuildTree(data)
    print("Tree built")
    print("Size", tree.GetTreeSize())
    tree.Insert(data[0])
    print("Tree insert")
    print("Size", tree.GetTreeSize())

    print("KNN:\n", tree.kNN(data[0, :-1]+3, 3, vi))

def test2():
    tree2 = kd.KDTree()
    tree2.BuildTree(data[:5])
    print("Tree2 built")
    print("Tree2 Size", tree2.GetTreeSize())

def test3():
    tree3 = kd.KDTree()
    tree3.BuildTree(data[:1])
    for i in range(1, len(data)):
        tree3.Insert(data[i])
    print("Tree3 Size", tree3.GetTreeSize())


def test4():
    data4 = np.array([[5],[6],[3]])
    idx4 = np.array([[0],[1],[2]])
    data4 = np.concatenate((data4, idx4), axis=1)
    tree4 = kd.KDTree()
    tree4.BuildTree(data4[:1])
    tree4.Insert(data4[1])
    tree4.Insert(data4[2])


def test5():
    print("\n\n-----------test5------------ compare kd tree and implementation")

    # skl_kdtree = skln.KDTree(X, leaf_size=lsize)
    skl_balltree = skln.BallTree(X, leaf_size=lsize, metric=skln.DistanceMetric.get_metric('mahalanobis', VI=vi))
    kdt_tree = kd.KDTree()
    kdt_tree.BuildTree(data)
    np.save("data", X)
    np.save("targets", targets)
    for i in range(len(targets)):
        seq = targets[i]
        # skl_kddist, skl_kdindex = skl_kdtree.query([seq], k=ksize)
        skl_balldist, skl_ballindex = skl_balltree.query([seq], k=ksize)
        res = kdt_tree.kNN(seq, ksize, vi)
        kdt_seq = res[:, :len(seq)]
        kdt_dist = res[:, len(seq)]
        kdt_index = res[:, -1].astype(int)

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

        np.save(str(seq)+"_sklearn", skl_ballindex)
        np.save(str(seq)+"_implem", kdt_index)

def plot_t5():
    import matplotlib.pylab as plt
    X = np.load("data.npy")
    targets = np.load("targets.npy")
    for i in range(len(targets)):
        seq = targets[i]
        skl_ballindex = np.load(str(seq) + "_sklearn.npy")
        kdt_index = np.load(str(seq) + "_implem.npy")
        kdt_seq = X[kdt_index]
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


# data = np.array([[51, 75], [70, 70], [55, 1], [60, 80], [10, 30], [35, 90], [25, 40], [1, 10], [50, 50]])
# data = np.array(np.random.random(50000).reshape((10000,5)))
#
# v = np.cov(data.T)
# vi = np.linalg.inv(v + np.eye(data.shape[1])*0.0001)
# print("cov inv", vi)

dim = 5
X = np.array(np.random.random(10000*dim).reshape((10000,dim)))
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

idx = np.array([[i] for i in range(len(X))])
data = np.concatenate((X, idx), axis=1)
print(data)


# import pieKDTree as kd
# test5()
plot_t5()