import numpy as np
import scipy.spatial as sps
from utils.TileCoding import TileCoding
from sklearn import preprocessing


# get wall
def get_wall(env_param):
    walls = []
    for block in env_param["walls"]:
        wx_start, wx_len, wy_start, wy_len = block
        for wx in range(wx_start, wx_start + wx_len):
            for wy in range(wy_start, wy_start + wy_len):
                walls.append([wx, wy])
    print(walls)
    return walls

def plot_avg_training_set(data, len_f, len_state, size = 100):
    img = np.zeros((size, size, data.shape[1] - (len_state + len_f)))
    count = np.zeros((size, size, 1))
    for d in data:
        s, g = d[:len_state], d[len_state + len_f:]
        s = (s * size).astype(int)
        s[0] = size - 1 if s[0] == size else s[0]
        s[1] = 1 if s[1] == 0 else s[1]
        img[size - s[1], s[0]] = g
        count[size - s[1], s[0]] += 1
    img /= count
    return img

# find averaged ground truth
def average_training_set(data, len_f, len_state=0):
    if len_state != 0:
        new_set = {}
        count = {}
        state = {}
        for d in data:
            s, f, g = d[:len_state], d[len_state: len_state+len_f], d[len_state+len_f:]
            f = tuple(f)
            if f in new_set.keys():
                new_set[f] += g
                count[f] += 1
                state[f] += s
            else:
                new_set[f] = g
                count[f] = 1
                state[f] = s

        all_f = list(new_set.keys())
        training_set = np.zeros((len(all_f), data.shape[1]))
        for idx in range(len(all_f)):
            training_set[idx, : len_state] = state[all_f[idx]] / float(count[all_f[idx]])
            training_set[idx, len_state: len_state+len_f] = np.array(all_f[idx])
            training_set[idx, len_state+len_f: ] = new_set[all_f[idx]] / float(count[all_f[idx]])

    else:
        new_set = {}
        count = {}
        for d in data:
            f, g = d[:len_f], d[len_f:]
            f = tuple(f)
            if f in new_set.keys():
                new_set[f] += g
                count[f] += 1
            else:
                new_set[f] = g
                count[f] = 1
        all_f = list(new_set.keys())
        training_set = np.zeros((len(all_f), data.shape[1]))
        for idx in range(len(all_f)):
            training_set[idx, : len_f] = np.array(all_f[idx])
            training_set[idx, len_f: ] = new_set[all_f[idx]] / float(count[all_f[idx]])
    return training_set

def normalize_ground_truth(data, gamma):
    data = data * (1 - gamma) * 100
    xd, yd = data[:, : data.shape[1]//2], data[:, data.shape[1]//2: ]
    # xd = np.divide(xd, np.linalg.norm(xd, axis=1).reshape((-1, 1)))
    # yd = np.divide(yd, np.linalg.norm(yd, axis=1).reshape((-1, 1)))
    normd = np.concatenate((xd, yd), axis=1)
    # normd = np.divide(normd, np.linalg.norm(normd, axis=1).reshape((-1, 1)))
    return normd

# given one hot encoding feature, return xy position
def recover_oneHot(feature, size_x, size_y):
    one = np.where(feature == 1)[0][0]
    x = one % size_y
    y = one // size_y
    return x, y

def recover_oneHot_set(features, size_x, size_y):
    res = []
    for f in features:
        x, y = recover_oneHot(f, size_x, size_y)
        res.append([x, y])
    return np.array(res)


# given feature and representation, transfer feature to xy position
# and put representation on coresponding grid
def construct_grid(data, size_x, size_y):
    grid = np.zeros((size_x, size_y, len(data[0]) - size_x*size_y))
    count = np.zeros((size_x, size_y))
    for d in data:
        f = d[:size_x*size_y]
        x, y = recover_oneHot(f, size_x, size_y)
        # grid[x, y] += d[size_x*size_y: ]
        grid[x, y] = (grid[x, y] * count[x, y] + d[size_x*size_y: ])/ float(count[x, y] + 1)
        count[x, y] += 1

    return grid


# given representation on each grid, check distance
def check_distance_dgw(data, size_x, size_y, goal, check_rep = False):
    dist = np.zeros((size_x, size_y))
    for b in range(size_y):
        for a in range(size_x):
            # dist[a, b] = np.mean(np.abs(data[a, b] - data[int(goal[0]), int(goal[1])]))
            dist[a, b] = np.linalg.norm(data[a, b] - data[goal[0], goal[1]])
            # dist[a, b] = np.dot(data[a, b], data[goal[0], goal[1]])
            # dist[a, b] = sps.distance.cosine(data[a, b], data[goal[0], goal[1]])
            if check_rep:
                print("{:8.4f}".format(dist[a,b]), "({:8.4f})".format(data[a, b][0]), end=' ')
            else:
                print("{:8.4f}".format(dist[a, b]), end=' ')

        print()
    print()
    return dist

def check_distance_cgw(pts, rep, goal_rep, check_rep = False, need_avg = True):
    if need_avg:
        new_data = average_training_set(np.concatenate((pts, rep), axis=1), pts.shape[1])
        pts, rep = new_data[:, :2], new_data[:, 2:]
    dist = np.zeros((len(pts), 3))
    for idx in range(len(pts)):
        # d = np.mean(np.abs(rep[idx] - goal_rep))
        d = np.linalg.norm(rep[idx] - goal_rep)
        # d = np.exp(-1 * (np.sum((rep[idx] - goal_rep) ** 2 * 100)))
        # d = np.dot(rep[idx], goal_rep)
        dist[idx, :2] = pts[idx]
        dist[idx, 2] = d
        if check_rep:
            print(pts[idx], d, "output:", dist[idx])
    return dist

# single xy position to one hot encoding
def one_hot_feature(data, size_x, size_y):
    # One-hot encoding
    feature_ary = np.zeros((len(data), size_x * size_y))
    for i in range(len(data)):
        xi, yi = data[i]
        feature_ary[i][xi + yi * size_x] = 1
    return feature_ary

# def one_hot_feature(data, size_x, size_y):
#     # One-hot encoding
#     feature_ary = np.zeros((len(data), size_x * size_y))
#     for i in range(len(data)):
#         xi, yi = data[i]
#         feature_ary[i][xi + yi * size_x] = 1
#     return feature_ary


def preproc_dgw_data_oneHot(data, size_x, size_y, gamma):
    # One-hot encoding
    feature_ary = one_hot_feature(data, size_x, size_y)
    y_ary = np.zeros(feature_ary.shape)
    f_sum = np.zeros(size_x * size_y)
    for j in range(len(feature_ary)-1, -1, -1):
        f_sum = feature_ary[j] + gamma * f_sum
        y_ary[j] = np.copy(f_sum)
    return feature_ary, y_ary
    # return feature_ary[0].reshape((1, -1)), y_ary[0].reshape((1, -1))

# when there are multiple gammas
def preproc_dgw_data_oneHot_multigamma(data, size_x, size_y, gamma_list):
    all_y = np.zeros((len(data), 0))
    all_f = None
    for gamma in gamma_list:
        f, y = preproc_dgw_data_oneHot(data, size_x, size_y, gamma)
        all_f = f
        all_y = np.concatenate((all_y, y), axis=1)
    return all_f, all_y

# def preproc_cgw_data(data, gamma, tile_coding, tc=None, num_tile=0, num_tiling=0):
#     if tile_coding:
#         if tc is None:
#             # tc = TileCoding(2, num_tiling, num_tile)
#             tc = TileCoding(1, num_tiling, num_tile)
#
#         # y_ary = np.zeros((data.shape[0], num_tile**2 * num_tiling))
#         # f_ary = np.zeros((data.shape[0], num_tile**2 * num_tiling))
#         # f_sum = np.zeros(num_tile**2 * num_tiling)
#         y_ary = np.zeros((data.shape[0], num_tile * num_tiling * 2))
#         f_ary = np.zeros((data.shape[0], num_tile * num_tiling * 2))
#         f_sum = np.zeros(num_tile * num_tiling * 2)
#
#         for j in range(len(data)-1, -1, -1):
#             one_idx = tc.get_index(data[j])
#             f = np.zeros(num_tile**2 * num_tiling)
#             f[one_idx] = 1
#             f_sum *= gamma
#             f_sum[one_idx] += 1
#             y_ary[j] = np.copy(f_sum)
#             f_ary[j] = np.copy(f)
#         return data, f_ary, y_ary
#         # return data[0], tc.get_index(data[0]), y_ary[0]
#     else:
#         y_ary = np.zeros(data.shape)
#         f_sum = np.zeros(data.shape[1])
#         for j in range(len(data)-1, -1, -1):
#             # print("-----data", data)
#
#             f_sum = data[j] + gamma * f_sum
#             y_ary[j] = np.copy(f_sum)
#         return data, y_ary
#         # return data[0], y_ary[0]
def preproc_cgw_data(data, gamma, tile_coding, separate, num_tile=0, num_tiling=0):
    if tile_coding:

        if separate:
            tc = TileCoding(1, num_tiling, num_tile)
            len_f = num_tile * num_tiling * 2
        else:
            tc = TileCoding(2, num_tiling, num_tile)
            len_f = num_tile**2 * num_tiling

        # import utils.tiles3 as tc
        # tc_mem_size = len_f //2
        # iht = tc.IHT(tc_mem_size)

        y_ary = np.zeros((data.shape[0], len_f))
        f_ary = np.zeros((data.shape[0], len_f))
        f_sum = np.zeros(len_f)

        for j in range(len(data)-1, -1, -1):
            if separate:
                x_one_idx = tc.get_index(data[j, 0].reshape((-1, 1)))
                # x_one_idx = np.array(tc.tiles(iht, num_tiling, float(num_tile) * np.array(data[j, 0]).reshape((-1,1))))
                xf = np.zeros(num_tile * num_tiling)
                xf[x_one_idx] = 1
                y_one_idx = tc.get_index(data[j, 1].reshape((-1, 1)))
                # y_one_idx = np.array(tc.tiles(iht, num_tiling, float(num_tile) * np.array(data[j, 1]).reshape((-1,1))))
                yf = np.zeros(num_tile * num_tiling)
                yf[y_one_idx] = 1
                f = np.concatenate((xf, yf), axis=0)

            else:
                f = np.zeros(len_f)
                one_idx = tc.get_index(data[j])
                # one_idx = np.array(tc.tiles(iht, num_tiling, float(num_tile) * np.array(data[j])))
                f[one_idx] = 1

            y_ary[j] = np.copy(f_sum)
            f_ary[j] = np.copy(f)
            f_sum *= gamma
            f_sum += f

        return data, f_ary, y_ary
    else:
        y_ary = np.zeros(data.shape)
        f_sum = np.zeros(data.shape[1])
        for j in range(len(data)-1, -1, -1):
            # print("-----data", data)

            f_sum = data[j] + gamma * f_sum
            y_ary[j] = np.copy(f_sum)
        return data, data, y_ary # feature == raw observation
        # return data[0], y_ary[0]


def preproc_cgw_data_multigamma(data, gamma_list, tile_coding, separate, num_tile=0, num_tiling=0):
    all_y = np.zeros((len(data), 0))
    for gamma in gamma_list:
        d, f, y = preproc_cgw_data(data, gamma, tile_coding, separate, num_tile, num_tiling)
        all_y = np.concatenate((all_y, y), axis=1)
    return d, f, all_y

def preproc_graph_data(data, gamma, num_tile=None, size_x=None, size_y=None):
    if num_tile is None:
        feature_ary = np.array(data, dtype=int)
        y_ary = np.zeros(feature_ary.shape)
        f_sum = np.zeros(size_x * size_y)
        for j in range(len(feature_ary)-1, -1, -1):
            f_sum = feature_ary[j] + gamma * f_sum
            y_ary[j] = np.copy(f_sum)

    elif size_x is None and size_y is None:
        num_tiling = 1

        tc = TileCoding(2, num_tiling, num_tile)

        len_f = num_tile ** 2

        y_ary = np.zeros((data.shape[0], len_f))
        f_ary = np.zeros((data.shape[0], len_f))
        f_sum = np.zeros(len_f)

        for j in range(len(data)-1, -1, -1):
            one_idx = tc.get_index(data[j])
            f = np.zeros(num_tile ** 2 * num_tiling)
            f[one_idx] = 1

            f_sum *= gamma
            f_sum += f
            y_ary[j] = np.copy(f_sum)
            f_ary[j] = np.copy(f)
    return data, f_ary, y_ary

def preproc_graph_data_multigamma(data, gamma_list, num_tile=None, size_x=None, size_y=None):
    all_y = np.zeros((len(data), 0))
    all_f = None
    for gamma in gamma_list:
        d, f, y = preproc_graph_data(data, gamma, num_tile=num_tile, size_x=size_x, size_y=size_y)
        all_f = f
        all_y = np.concatenate((all_y, y), axis=1)
    return data, all_f, all_y

# if array a and b are same, return True, else False
def equal_array(a, b):
    # a = np.sort(a)
    # b = np.sort(b)
    if len(a) != len(b):
        return False
    else:
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
    return True