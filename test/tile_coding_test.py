import sys
sys.path.append("../")
from utils.TileCoding import *
import numpy as np


def test1(dim, num_tiling, num_tile):
    tco = TileCoding(dim, num_tiling, num_tile)
    print(tco.len_tile, tco.tiling_dist)
    for i in range(num_tiling):
        print("\ntiling", i, ":", end=" ")
        for j in range(num_tile):
            print("{:8.4f} ({:4d})".format(tco.len_tile * j - tco.tiling_dist * i, j+i*num_tile), end=" ")
        print("{:8.4f}".format(tco.len_tile * num_tile - tco.tiling_dist * i))
    pt_list = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    pt_list = np.array(pt_list).reshape((-1, 1))

    for pt in np.array(pt_list):
        print("point: ", pt)
        print(tco.get_index(pt))


def test2(dim, num_tiling, num_tile):
    tco = TileCoding(dim, num_tiling, num_tile)
    #tco_bug = TileCoding_bug(dim, num_tiling, num_tile)
    print(tco.len_tile, tco.tiling_dist)
    for i in range(num_tiling):
        print("\ntiling", i, ":", end=" ")
        for j in range(num_tile):
            print(tco.len_tile * j + tco.tiling_dist * i, end=" ")
        print()
    pt_list = []
    xlist = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for i in range(len(xlist)):
        for j in range(len(xlist)):
            pt_list.append([xlist[i], xlist[j]])

    for pt in np.array(pt_list):
        print("point: ", pt)
        print(tco.get_index(pt))
        #print(tco_bug.get_index(pt))

def test3():
    num_tiling = 32
    num_tile = 4
    tco = TileCoding(1, 32, 4)
    print(tco.len_tile, tco.tiling_dist)
    for i in range(num_tiling):
        print("\ntiling", i, ":", end=" ")
        for j in range(num_tile):
            print("{:8.4f} ({:4d})".format(tco.len_tile * j - tco.tiling_dist * i, j+i*num_tile), end=" ")
        print("{:8.4f}".format(tco.len_tile * num_tile - tco.tiling_dist * i))
    x = [0.62581515]
    print(tco.get_index(x))


# test2(2, 1, 16)
# test2(2, 4, 16)
# test2(2, 10, 10)

test1(1, 1, 1)
test1(1, 2, 2)
test1(1, 4, 4)
test1(1, 10, 10)

test3()