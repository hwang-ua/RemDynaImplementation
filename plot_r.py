import numpy as np
import os
import json
import matplotlib.pylab as plt
import matplotlib.patches as patches
from utils.distance_matrix_func import *
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pickle

# def plot_prototypes(folder="temp"):
#
#     global fig_count
#
#     jsonfile = "parameters/continuous_gridworld.json"
#     json_dat = open(jsonfile, 'r')
#     exp = json.load(json_dat)
#     json_dat.close()
#
#     goal_x = [0.71, 1]#exp["env_params"]["goal_x"]#
#     goal_y = [0.95, 1]#exp["env_params"]["goal_y"]#
#     if goal_x[0] > 1 or goal_y[0] > 1:
#         goal_x, goal_y = [0, 1], [0.555, 1]
#
#     for act in range(4):
#     # for act in range(1):
#         print(act)
#
#         fig = plt.figure(fig_count)
#         currentAxis = plt.gca()
#         currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
#         currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
#         with open(folder+str(act)+'s.pkl', 'rb') as f:
#         # with open(folder+'s.pkl', 'rb') as f:
#             values = np.array(pickle.load(f))
#         plt.scatter(values[:,0], values[:,1], color='red',s=2)
#         plt.xlim([0.0,1.0])
#         plt.ylim([0.0,1.0])
#         fig.savefig(folder + str(act)+'s.png', dpi=fig.dpi)
#         # fig.savefig(folder+'s.png', dpi=fig.dpi)
#         # print(np.sort(values.view('f8,f8'), order=['f0'], axis=0).view(np.float))
#
#         fig_count += 1
#         fig = plt.figure(fig_count)
#         currentAxis = plt.gca()
#         currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
#         currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
#         with open(folder+str(act)+'sdash.pkl', 'rb') as f:
#         # with open(folder+'sdash.pkl', 'rb') as f:
#             values = np.array(pickle.load(f))
#         plt.scatter(values[:,0], values[:,1], color='blue',s=2)
#         plt.xlim([0.0,1.0])
#         plt.ylim([0.0,1.0])
#         fig.savefig(folder + str(act)+'sdash.png', dpi=fig.dpi)
#         # fig.savefig(folder+'sdash.png', dpi=fig.dpi)
#
#         fig_count += 1
#
#         print(len(values))
#
#     plt.clf()
#     plt.cla()
#     plt.close()

def plot_prototypes(folder="temp"):

    global fig_count

    jsonfile = "parameters/continuous_gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    goal_x = [0.71, 1]#exp["env_params"]["goal_x"]#
    goal_y = [0.95, 1]#exp["env_params"]["goal_y"]#
    if goal_x[0] > 1 or goal_y[0] > 1:
        goal_x, goal_y = [0, 1], [0.555, 1]

    act=""
    fig = plt.figure(fig_count)
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    with open(folder+str(act)+'-forward-s.pkl', 'rb') as f:
    # with open(folder+'s.pkl', 'rb') as f:
        values = np.array(pickle.load(f))
    plt.scatter(values[:,0], values[:,1], color='red',s=2)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    fig.savefig(folder + str(act)+'-forward-s.png', dpi=fig.dpi)
    # fig.savefig(folder+'s.png', dpi=fig.dpi)
    # print(np.sort(values.view('f8,f8'), order=['f0'], axis=0).view(np.float))

    fig_count += 1
    fig = plt.figure(fig_count)
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    with open(folder+str(act)+'-forward-sdash.pkl', 'rb') as f:
    # with open(folder+'sdash.pkl', 'rb') as f:
        values = np.array(pickle.load(f))
    plt.scatter(values[:,0], values[:,1], color='blue',s=2)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    fig.savefig(folder + str(act)+'-forward-sdash.png', dpi=fig.dpi)
    # fig.savefig(folder+'sdash.png', dpi=fig.dpi)

    fig_count += 1


    fig = plt.figure(fig_count)
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    with open(folder+str(act)+'-reverse-s.pkl', 'rb') as f:
    # with open(folder+'s.pkl', 'rb') as f:
        values = np.array(pickle.load(f))
    plt.scatter(values[:,0], values[:,1], color='red',s=2)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    fig.savefig(folder + str(act)+'-reverse-s.png', dpi=fig.dpi)
    # fig.savefig(folder+'s.png', dpi=fig.dpi)
    # print(np.sort(values.view('f8,f8'), order=['f0'], axis=0).view(np.float))

    fig_count += 1
    fig = plt.figure(fig_count)
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    with open(folder+str(act)+'-reverse-sdash.pkl', 'rb') as f:
    # with open(folder+'sdash.pkl', 'rb') as f:
        values = np.array(pickle.load(f))
    plt.scatter(values[:,0], values[:,1], color='blue',s=2)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    fig.savefig(folder + str(act)+'-reverse-sdash.png', dpi=fig.dpi)
    # fig.savefig(folder+'sdash.png', dpi=fig.dpi)

    fig_count += 1

    print(len(values))

    plt.clf()
    plt.cla()
    plt.close()


def plot_reconstruction(folder="temp"):

    global fig_count

    jsonfile = "parameters/continuous_gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    goal_x = [0.71, 1]#exp["env_params"]["goal_x"]#
    goal_y = [0.95, 1]#exp["env_params"]["goal_y"]#
    if goal_x[0] > 1 or goal_y[0] > 1:
        goal_x, goal_y = [0, 1], [0.555, 1]

    # for act in range(4):
    for act in range(1):

        print(act)

        fig = plt.figure(fig_count)
        currentAxis = plt.gca()
        currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
        currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
        # with open(folder+str(act)+'s.pkl', 'rb') as f:
        with open(folder+'s.pkl', 'rb') as f:
            values = np.array(pickle.load(f))
        num_steps = values.shape[0]
        color_idx = np.linspace(0, 1, num_steps)
        for i in range(num_steps):
            plt.scatter(values[i,0], values[i,1], color=plt.cm.viridis(color_idx[i]),s=2)
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        # fig.savefig(folder + str(act)+'s.png', dpi=fig.dpi)
        fig.savefig(folder+'s.png', dpi=fig.dpi)

        fig_count += 1
        fig = plt.figure(fig_count)
        currentAxis = plt.gca()
        currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
        currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
        # with open(folder+str(act)+'srec.pkl', 'rb') as f:
        with open(folder+'srec.pkl', 'rb') as f:
            values = np.array(pickle.load(f))
        num_steps = values.shape[0]
        color_idx = np.linspace(0, 1, num_steps)
        for i in range(num_steps):
            plt.scatter(values[i,0], values[i,1], color=plt.cm.viridis(color_idx[i]),s=2)
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        # fig.savefig(folder + str(act)+'srec.png', dpi=fig.dpi)
        fig.savefig(folder+'srec.png', dpi=fig.dpi)

        fig_count += 1

        print(len(values))

    plt.clf()
    plt.cla()
    plt.close()

def plot_sampling(folder="temp"):

    global fig_count

    jsonfile = "parameters/continuous_gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    goal_x = [0.71, 1]#exp["env_params"]["goal_x"]#
    goal_y = [0.95, 1]#exp["env_params"]["goal_y"]#
    if goal_x[0] > 1 or goal_y[0] > 1:
        goal_x, goal_y = [0, 1], [0.555, 1]

    for act in range(4):
    # for act in range(1):

        print(act)

        fig = plt.figure(fig_count)
        currentAxis = plt.gca()
        currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
        currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
        with open(folder+str(act)+'splan.pkl', 'rb') as f:
        # with open(folder+'splan.pkl', 'rb') as f:
            values = np.array(pickle.load(f))
        num_steps = values.shape[0]
        color_idx = np.linspace(0, 1, num_steps)
        for i in range(num_steps):
            plt.scatter(values[i,0], values[i,1], color=plt.cm.viridis(color_idx[i]),s=2)
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        fig.savefig(folder + str(act)+'splan.png', dpi=fig.dpi)
        # fig.savefig(folder+'splan.png', dpi=fig.dpi)

        fig_count += 1
        fig = plt.figure(fig_count)
        currentAxis = plt.gca()
        currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
        currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
        with open(folder+str(act)+'sprev.pkl', 'rb') as f:
        # with open(folder+'sprev.pkl', 'rb') as f:
            values = np.array(pickle.load(f))
        num_steps = values.shape[0]
        color_idx = np.linspace(0, 1, num_steps)
        for i in range(num_steps):
            plt.scatter(values[i,0], values[i,1], color=plt.cm.viridis(color_idx[i]),s=2)
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        # plt.xlim([np.min(values[:,0]),np.max(values[:,0])])
        # plt.ylim([np.min(values[:,1]),np.max(values[:,1])])
        fig.savefig(folder + str(act)+'sprev.png', dpi=fig.dpi)
        # fig.savefig(folder+'sprev.png', dpi=fig.dpi)

        fig_count += 1

        print(len(values))

    plt.clf()
    plt.cla()
    plt.close()

def plot_sampling_forward(folder="temp"):

    global fig_count

    jsonfile = "parameters/continuous_gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    goal_x = [0.71, 1]#exp["env_params"]["goal_x"]#
    goal_y = [0.95, 1]#exp["env_params"]["goal_y"]#
    if goal_x[0] > 1 or goal_y[0] > 1:
        goal_x, goal_y = [0, 1], [0.555, 1]

    for act in range(4):
    # for act in range(1):

        print(act)

        fig = plt.figure(fig_count)
        currentAxis = plt.gca()
        currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
        currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
        with open(folder+str(act)+'splan.pkl', 'rb') as f:
        # with open(folder+'splan.pkl', 'rb') as f:
            values = np.array(pickle.load(f))
        num_steps = values.shape[0]
        color_idx = np.linspace(0, 1, num_steps)
        for i in range(num_steps):
            plt.scatter(values[i,0], values[i,1], color=plt.cm.viridis(color_idx[i]),s=2)
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        fig.savefig(folder + str(act)+'splan.png', dpi=fig.dpi)
        # fig.savefig(folder+'splan.png', dpi=fig.dpi)

        fig_count += 1
        fig = plt.figure(fig_count)
        currentAxis = plt.gca()
        currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
        currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
        with open(folder+str(act)+'snext.pkl', 'rb') as f:
        # with open(folder+'snext.pkl', 'rb') as f:
            values = np.array(pickle.load(f))
        num_steps = values.shape[0]
        color_idx = np.linspace(0, 1, num_steps)
        for i in range(num_steps):
            plt.scatter(values[i,0], values[i,1], color=plt.cm.viridis(color_idx[i]),s=2)
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        # plt.xlim([np.min(values[:,0]),np.max(values[:,0])])
        # plt.ylim([np.min(values[:,1]),np.max(values[:,1])])
        fig.savefig(folder + str(act)+'snext.png', dpi=fig.dpi)
        # fig.savefig(folder+'snext.png', dpi=fig.dpi)

        fig_count += 1

        print(len(values))

    plt.clf()
    plt.cla()
    plt.close()

def plot_knn_vis(folder="temp",folder2="temp"):

    global fig_count

    jsonfile = "parameters/continuous_gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    for state in range(5):
        for action in range(4):
            fig = plt.figure(fig_count)
            currentAxis = plt.gca()
            currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
            currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
            with open(folder + str(action) + 'sdash.pkl', 'rb') as f:
                values = np.array(pickle.load(f))
            plt.scatter(values[:, 0], values[:, 1], color='red', s=2)

            with open(folder2 + str(state) + str(action) + '.pkl', 'rb') as f:
                values2 = np.array(pickle.load(f))

            plt.scatter(values2[:-1,0], values2[:-1,1], color='blue')
            plt.scatter(values2[-1,0], values2[-1,1], color='green')

            plt.xlim([0.0,1.0])
            plt.ylim([0.0,1.0])

            fig.savefig(folder2+str(state)+str(action)+'.png', dpi=fig.dpi)

            fig_count += 1


# def plot_sampling_vis(folder="temp"):
#
#     global fig_count
#
#     jsonfile = "parameters/continuous_gridworld.json"
#     json_dat = open(jsonfile, 'r')
#     exp = json.load(json_dat)
#     json_dat.close()
#
#     goal_x = [0.71, 1]#exp["env_params"]["goal_x"]#
#     goal_y = [0.95, 1]#exp["env_params"]["goal_y"]#
#     if goal_x[0] > 1 or goal_y[0] > 1:
#         goal_x, goal_y = [0, 1], [0.555, 1]
#
#
#     files = [""]
#     color_idx = ['orange', 'green', 'red', 'purple']
#     # files =["world-forwardsampling","new_rem-forwardsampling-fcov","old_rem-forwardsampling-fcov","lap-forwardsampling-fcov"]
#     # files =["world-forwardsampling","new_rem-forwardsampling-mu-fcov","old_rem-forwardsampling-mu-fcov","lap-forwardsampling-mu-fcov"]
#     # color_idx = np.linspace(0, 1, len(files))
#     for state in range(13):
#
#         fileNum = 0
#         for file in files:
#             fig = plt.figure(fig_count)
#             currentAxis = plt.gca()
#             currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
#             currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
#             for action in range(4):
#
#                 with open(folder+file+"/"+str(state)+str(action)+'forward.pkl', 'rb') as f:
#                 # with open(folder+file+"/"+str(state)+str(action)+'backward.pkl', 'rb') as f:
#                     values = np.array(pickle.load(f))
#                 num_steps = values.shape[0]
#                 plt.scatter(values[:-1,0], values[:-1,1], s=2, color=color_idx[action])
#                 fileNum += 1
#             plt.scatter(values[-1, 0], values[-1, 1], s=2, color="black")
#             plt.xlim([0.0,1.0])
#             plt.ylim([0.0,1.0])
#
#             fig.savefig(folder+file+str(state)+'-forward.png', dpi=fig.dpi)
#             # fig.savefig(folder+file+str(state)+'-backward.png', dpi=fig.dpi)
#
#             fig_count += 1
#
#     plt.clf()
#     plt.cla()
#     plt.close()

def plot_sampling_vis(folder="temp"):

    # global fig_count
    fig_count = 0

    jsonfile = "parameters/continuous_gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    goal_x = [0.71, 1]#exp["env_params"]["goal_x"]#
    goal_y = [0.95, 1]#exp["env_params"]["goal_y"]#
    if goal_x[0] > 1 or goal_y[0] > 1:
        goal_x, goal_y = [0, 1], [0.555, 1]


    files = [""]#["world-forwardsampling","new_rem-forwardsampling","old_rem-forwardsampling","lap-forwardsampling","lapinput-forwardsampling"]
    # files =["world-forwardsampling","new_rem-forwardsampling-fcov","old_rem-forwardsampling-fcov","lap-forwardsampling-fcov"]
    # files =["world-forwardsampling","new_rem-forwardsampling-mu-fcov","old_rem-forwardsampling-mu-fcov","lap-forwardsampling-mu-fcov"]
    color_idx = ['orange', 'green', 'red', 'purple']
    for state_num in range(14):
        fileNum = 0
        for file in files:
            fig = plt.figure(state_num)
            currentAxis = plt.gca()
            # currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
            # currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
            currentAxis.add_patch(patches.Rectangle((0.0, 0.0), 1.0, 1.0, fill=None))

            for action in range(4):
                with open(folder+file+"/"+str(state_num)+str(action)+'-sample.pkl', 'rb') as f:
                    values = np.array(pickle.load(f))
                print(action, values[0])
                num_steps = values.shape[0]
                plt.scatter(values[:,0], values[:,1], c=color_idx[action], s=2)
                # plt.scatter(values[:,0], values[:,1],s=2)
                fileNum += 1

                plt.xlim([-0.1,1.1])
                plt.ylim([-0.1,1.1])

            with open(folder+file + "/" + str(state_num) + '-state.pkl', 'rb') as f:
                state = np.array(pickle.load(f))
                print(state)
                plt.scatter(state[0], state[1],s=2, color='black')

            fig.savefig(folder+file+str(state_num)+'-sample.png', dpi=fig.dpi)
            print(str(state_num)+'-sample.png')
            fig_count += 1
    plt.clf()
    plt.cla()
    plt.close()


def plot_sampling_vis_same(folder="temp"):

    global fig_count

    jsonfile = "parameters/continuous_gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    goal_x = [0.71, 1]#exp["env_params"]["goal_x"]#
    goal_y = [0.95, 1]#exp["env_params"]["goal_y"]#
    if goal_x[0] > 1 or goal_y[0] > 1:
        goal_x, goal_y = [0, 1], [0.555, 1]


    files = ["world-forwardsampling","lm-forwardsampling","flm-forwardsampling","new_rem-forwardsampling"]
    # files = ["world-forwardsampling","new_rem-forwardsampling","old_rem-forwardsampling","lap-forwardsampling"]
    # files =["world-forwardsampling","new_rem-forwardsampling-fcov","old_rem-forwardsampling-fcov","lap-forwardsampling-fcov"]
    # files =["world-forwardsampling","new_rem-forwardsampling-mu-fcov","old_rem-forwardsampling-mu-fcov","lap-forwardsampling-mu-fcov"]
    # color_idx = np.linspace(0, 1, len(files))
    colors=["r","b","y","m"]
    states=[[0.05,0.95],[0.5,0.2],[0.75,0.95],[0.5,0.5],[0.2,0.5]]
    for state in range(5):
        for action in range(4):

            fig = plt.figure(fig_count)
            currentAxis = plt.gca()
            currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
            currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))

            fileNum = 0
            for file in files:

                with open(folder+file+"/"+str(state)+str(action)+'.pkl', 'rb') as f:
                    values = np.array(pickle.load(f))
                num_steps = values.shape[0]
                # plt.scatter(values[:,0], values[:,1], color=plt.cm.viridis(color_idx[fileNum]),s=2)
                plt.scatter(values[:,0], values[:,1], color=colors[fileNum],s=2)
                fileNum += 1

            plt.scatter(states[state][0], states[state][1], color="g",s=20)

            plt.xlim([0.0,1.0])
            plt.ylim([0.0,1.0])

            fig.savefig(folder+str(state)+str(action)+'.png', dpi=fig.dpi)

            fig_count += 1

    plt.clf()
    plt.cla()
    plt.close()


def plot_samples(file="x",c='red',file_name="y"):

    global fig_count

    goal_x = [0.71, 1]
    goal_y = [0.95, 1]
    if goal_x[0] > 1 or goal_y[0] > 1:
        goal_x, goal_y = [0, 1], [0.555, 1]

    fig = plt.figure(fig_count)
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    with open(file, 'rb') as f:
        values = np.array(pickle.load(f))
    plt.scatter(values[:,0], values[:,1], color=c,s=2)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    fig.savefig(file_name, dpi=fig.dpi)

    plt.clf()
    plt.cla()
    plt.close()


fig_count = 0

# check_weight()
# check_reward()

# check_weight()

# plot_nn_result()
# save_figures()
#remove_data()
f = "temp/"
# plot_nn_dist_dgw(folder=f, save=False, clip_at=None, switch_y=True)
#plot_recovered_state_dgw(folder=f, save=True)
# check_ground_truth_dgw()
# param_sweep_plot_nn_dist()

# f = "temp/"
# plot_nn_dist_cgw(folder=f, save=False, min_clip_at=None, max_clip_at=None, normalize=False)
# plot_recovered_state_cgw(folder=f, save=False)
# check_ground_truth_cgw_xy()
# check_ground_truth_cgw_tc()

# combine_data()
# compare_result_old()
# compare_result_new()
# cgw_training_data()

# # rs_training_data()
# # river_swim_return()
f = "prototypes/rem-GCov-100p-randomwalk/local_linear_model/legalv_bw_mode10_trainingSetNormCov0.025_addProtLimit-1.5_kscale1.0/"
plot_prototypes(folder=f)
# plot_reconstruction(folder="temp/")
# plot_sampling(folder="sampling/rem-GCov-100p-randomwalk-flm/")
# plot_sampling_forward(folder="")
# plot_knn_vis(folder="prototypes/rem-GCov-100p-randomwalk-flm/",folder2="prototypes-knn/rem-GCov-100p-randomwalk-flm/")
plot_sampling_vis(folder=f)
# plot_sampling_vis_same(folder="sampling-vis/")
# plot_samples(file="lm-vis/world.pkl",c='red',file_name="lm-vis/world.png")
# plot_samples(file="lm-vis/sample.pkl",c='red',file_name="lm-vis/sample.png")
