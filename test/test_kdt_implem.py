import sys
sys.path.append('../')
from utils.kd_tree import *

def test_kdt_1():
    print("------test 1------")
    tree = KD_Tree(2)
    root = tree.rebuild(data)
    tree.visualization(root)

def test_kdt_2():
    print("------test 2------")
    tree = KD_Tree(2)
    for d in range(len(data)):
        tree.insert(data[d], d)
    root = tree.get_tree()
    tree.visualization(root)

def test_kdt_3():
    print("------test 3------")
    tree = KD_Tree(2)
    for d in range(len(data)):
        tree.insert(data[d], d)
    np.random.shuffle(data)
    tree.rebuild(data)
    root = tree.get_tree()
    tree.visualization(root)

def test_kdt_4():
    print("------test 4------")
    tree = KD_Tree(2)
    root = tree.rebuild(data)
    print("cov:", tree.covmat_inv)
    tree.visualization(root)
    n, i, d = tree.k_nearest(2, [3,2])
    print("\n<1>  ", n, i, d, "--", np.where(data==n[0]))
    n, i, d = tree.k_nearest(2, [100, 100])
    print("\n<2>  ", n, i, d, "--", np.where(data==n[0]))
    n, i, d = tree.k_nearest(2, [60,81])
    print("\n<3>  ", n, i, d, "--", np.where(data==n[0]))

def test_kdt_5():
    print("------test 5------")
    tree = KD_Tree(2)
    for d in range(len(data)):
        tree.insert(data[d], d)
        print("---", data[d], d)
        root = tree.get_tree()
        tree.visualization(root)

    root = tree.get_tree()
    tree.visualization(root)
    print("\n<1>  ", tree.k_nearest(2, [3, 2]))
    print("\n<2>  ", tree.k_nearest(2, [100, 100]))
    print("\n<3>  ", tree.k_nearest(2, [60, 81]))


# data=np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
data = np.array([[51, 75], [70, 70], [55, 1], [60, 80], [10, 30], [35, 90], [25, 40], [1, 10], [50, 50]])

test_kdt_1()
test_kdt_2()
test_kdt_3()
test_kdt_4()
test_kdt_5()