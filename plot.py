import numpy as np
import pickle as pkl
import os
import json
# import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.distance_matrix_func import *
from sklearn.decomposition import PCA
from sklearn import preprocessing
import utils.tiles3 as tc


color_list = ["red", "blue", "orange", "green", "purple", "black", "deepskyblue", "teal", "yellowgreen", "brown", "chocolate"]
facecolor_list = ['mistyrose', 'skyblue', 'papayawhip', 'palegreen','violet', 'grey', 'skyblue', 'cyan', 'greenyellow', "lightcoral", "peachpuff"]
def exponential_smooth(data, beta):
    J = 0
    new_data = np.zeros(len(data))
    for idx in range(len(data)):
        J *= (1-beta)
        J += beta
        rate = beta / J
        if idx == 0:
            new_data[idx] = data[idx] * rate
        else:
            new_data[idx] = data[idx] * rate + new_data[idx - 1] * (1 - rate)
    return new_data

def cgw_training_data():
    # rec_list = [
        # ['REM_Dyna', 10, 2, 0.01, 0.999],
        # ['REM_Dyna', 10, 8, 0.01, 0.999],
        # ['REM_Dyna', 14, 2, 0.25, 0.0],
        # ['REM_Dyna', 14, 8, 0.125, 0.0],
        # ['REM_Dyna', 16, 2, 0.01, 0.999],
        # ['REM_Dyna', 16, 8, 0.001, 0.9],
        # ['REM_Dyna', 17, 2, 0.125, 0.0],
        # ['REM_Dyna', 17, 8, 0.5, 0.0],
        # ['REM_Dyna', 19, 2, 0.0625, 0.0],
        # ['REM_Dyna', 19, 8, 0.0625, 0.0],
        # ['REM_Dyna', 0, 2, 0.25, 0.0],
        # ['REM_Dyna', 0, 8, 0.25, 0.0],
        # ['Q_learning', 0, 0, 1.0, 0.0],
        # ['Q_learning', 2, 0, 0.001, 0.9],
        # ['Q_learning', 3, 0, 2.0, 0.0],
        # ['Q_learning', 4, 0, 1.0, 0.0],
        # ['random_ER', 0, 0, 0.125, 0.0],
        # ['random_ER', 2, 0, 0.001, 0.999],
        # ['random_ER', 3, 0, 0.5, 0.0],
        # ['random_ER', 4, 0, 0.5, 0.0],
        # ['REM_Dyna', 1, 2, 0.5, 0.0],
        # ['REM_Dyna', 1, 8, 0.5, 0.0],

        # ['REM_Dyna_dep', 0, 1, 0.5, 0.0]

    # ] # [ alg,     mode, knn,  lr, rms ]


    # rec_list = [
    #             ['random_ER', 0, 10, 0.1, 0.0, 0, 10],
    #             # ['random_ER', 0, 10, 0.4, 0.0, 0, 16],
    #
    #             # ['REM_Dyna', 17, 10, 0.2, 0.0, 4, 10],
    #             # ['REM_Dyna', 17, 25, 0.1, 0.0, 4, 10],
    #             # ['REM_Dyna', 17, 50, 0.4, 0.0, 4, 10],
    #             # ['REM_Dyna', 0, 10, 0.8, 0.0, 4, 10],
    #             # ['REM_Dyna', 0, 25, 0.8, 0.0, 4, 10],
    #             # ['REM_Dyna', 0, 50, 0.4, 0.0, 4, 10],
    #
    #             # ['REM_Dyna', 17, 10, 0.4, 0.0, 4, 16],
    #             # ['REM_Dyna', 17, 25, 0.4, 0.0, 4, 16],
    #             # ['REM_Dyna', 17, 50, 0.8, 0.0, 4, 16],
    #             # ['REM_Dyna', 0, 10, 0.4, 0.0, 4, 16],
    #             # ['REM_Dyna', 0, 25, 0.2, 0.0, 4, 16],
    #             # ['REM_Dyna', 0, 50, 0.1, 0.0, 4, 16],
    #
    #             ['REM_Dyna', 17, 10, 0.2, 0.0, 32, 10],
    #             ['REM_Dyna', 17, 25, 0.2, 0.0, 32, 10],
    #             ['REM_Dyna', 17, 50, 0.2, 0.0, 32, 10],
    #             ['REM_Dyna', 0, 10, 0.2, 0.0, 32, 10],
    #             ['REM_Dyna', 0, 25, 0.8, 0.0, 32, 10],
    #             ['REM_Dyna', 0, 50, 0.4, 0.0, 32, 10],
    #
    #             # ['REM_Dyna', 17, 10, 0.8, 0.0, 32, 16],
    #             # ['REM_Dyna', 17, 25, 0.4, 0.0, 32, 16],
    #             # ['REM_Dyna', 17, 50, 0.1, 0.0, 32, 16],
    #             # ['REM_Dyna', 0, 10, 0.2, 0.0, 32, 16],
    #             # ['REM_Dyna', 0, 25, 0.4, 0.0, 32, 16],
    #             # ['REM_Dyna', 0, 50, 0.1, 0.0, 32, 16],
    #             ]

    rec_list = [
                # ["random_ER", 0, 10, 0.4, 0.0, 4, 16, 0],

                # ["REM_Dyna", 17, 10, 0.1, 0.0, 4, 16, 0],
                # ["REM_Dyna", 17, 10, 0.2, 0.0, 4, 16, 0],
                # ["REM_Dyna", 17, 10, 0.4, 0.0, 4, 16, 0],
                # ["REM_Dyna", 17, 10, 0.8, 0.0, 4, 16, 0],
                #
                # ["REM_Dyna", 17, 25, 0.1, 0.0, 4, 16, 0],
                # ["REM_Dyna", 17, 25, 0.2, 0.0, 4, 16, 0],
                # ["REM_Dyna", 17, 25, 0.4, 0.0, 4, 16, 0],
                # ["REM_Dyna", 17, 25, 0.8, 0.0, 4, 16, 0],

                ["REM_Dyna", 17, 10, 0.1, 0.0, 4, 16, 1],
                ["REM_Dyna", 17, 10, 0.2, 0.0, 4, 16, 1],
                ["REM_Dyna", 17, 10, 0.4, 0.0, 4, 16, 1],
                ["REM_Dyna", 17, 10, 0.8, 0.0, 4, 16, 1],

                # ["REM_Dyna", 17, 25, 0.1, 0.0, 4, 16, 1],
                # ["REM_Dyna", 17, 25, 0.2, 0.0, 4, 16, 1],
                # ["REM_Dyna", 17, 25, 0.4, 0.0, 4, 16, 1],
                # ["REM_Dyna", 17, 25, 0.8, 0.0, 4, 16, 1],

                ]

    file_name_list = []
    for rec in rec_list:

        if rec[1] in [2, 10, 11, 16]:
            opt_mode = 1
        elif rec[1] in [0, 1, 3, 4, 12, 13, 14, 15, 17, 18, 19]:
            opt_mode = 4
        else:
            print("UNKNOWN MODE")
            exit(-1)

        name = "exp_result/"

        if rec[0] == "REM_Dyna" or rec[0] == "REM_Dyna_dep":

            if rec[5] == 32 and rec[1] == 17:
                limit = -25.0
            elif rec[5] ==4 or rec[1] == 0:
                limit = -1.0

            name = "exp_result_usable/offline_mode17_onPolicy+fixCov/REM_Dyna_mode" + str(rec[1]) + \
                   "_offline1"+ \
                   "_planning" + str(rec[2]) + \
                   "/always_add_prot_1/"
            name += "pri_pred_alpha" + str(rec[3]) + \
                    "_divAF" + str(rec[1]) + \
                    "_near8" + \
                    "_protLimit" + str(limit) + \
                    "_similarity0.0" + \
                    "_kscale1e-07" + \
                    "_fixCov0.001" + \
                    "_updateQ" + \
                    "_lambda0.0" + \
                    "_momentum0.0" + \
                    "_rms" + str(rec[4]) + \
                    "_optMode" + str(opt_mode) + \
                    "_30000x5.npy"


        elif rec[0] == "Q_learning":
            name += rec[0] + "_mode" + str(rec[1]) + "/" + \
                    "_alpha" + str(rec[3]) + \
                    "_divAF" + str(rec[1]) + \
                    "_updateQ" + \
                    "_lambda0.0" + \
                    "_momentum0.0" + \
                    "_rms" + str(rec[4]) + \
                    "_optMode" + str(opt_mode) + \
                    "_30000x5.npy"

        elif rec[0] == "random_ER":
            name = "exp_result_usable/test_rem/er_1x"+str(rec[6]) + "/"
            name += rec[0] + "_mode" + str(rec[1]) + "/" + \
                    "_alpha" + str(rec[3]) + \
                    "_divAF" + str(rec[1]) + \
                    "_updateQ" + \
                    "_lambda0.0" + \
                    "_momentum0.0" + \
                    "_rms" + str(rec[4]) + \
                    "_optMode" + str(opt_mode) + \
                    "_30000x5.npy"

        file_name_list.append(name)

    plt.figure(0)

    for i in range(len(file_name_list)):
        print("Loading", file_name_list[i])
        data = np.load(file_name_list[i])

        learning = []
        for l in data:
            start = 0
            while start < len(l):
                if l[start] != 0:
                    break
                else:
                    start += 1
            print("2nd ep starts from", start)
            learning.append(l[start: start + 9200])

        learning = np.array(learning)

        md = np.mean(learning, axis=0)
        std = np.abs(np.std(learning, axis=0)) / np.sqrt(len(learning))
        upper = md + std * 1
        lower = md - std * 1

        x = np.linspace(0, len(md), len(md))
        plt.plot(x, md,  label=str(rec_list[i]), color=color_list[i])
        # plt.errorbar(x, md, yerr=std)

        # plt.plot(x, upper, color = facecolor_list[i])
        # plt.plot(x, lower, color = facecolor_list[i])
        plt.fill_between(x, upper, lower, facecolor=facecolor_list[i], alpha=0.5)

        # for j in learning:
        #     plt.plot(j, color=color_list[i])
        # plt.plot(np.mean(learning, axis=0), '--', linewidth=1.5, label=str(rec_list[i]), color=color_list[i])

        # md = np.mean(data, axis=0)

        # top = 0
        # while top < len(md) - 1:
        #     if md[top] > md[top + 1]:
        #         break
        #     else:
        #         top += 1
        # plt.plot(md[:top + 1], label=str(rec_list[i]))

        # plt.plot(md[:10001], label=str(rec_list[i]))#, color=color_list[i])

    plt.legend()
    plt.show(block=True)

def combine_data():
    not_exist = []

    for offline in [1]:
        for num_near in [8]: #[1]:
            for num_planning in [10]:
                for mode in [17]:
                    if mode == 0:
                        limit_list = [-1000.0]#[-0.025]#[-0.1]#[-4.0]#
                    elif mode == 17:
                        limit_list = [-1000.0]
                    elif mode == 21:
                        limit_list = [0.0]

                    for limit in limit_list:

                        for kscale in [1.0]:

                            for similarity in [0]:
                                for sample_limit in [0.0]: #[0.0, 0.5, 0.75, 0.9, 0.95]:

                                    for add_prot in [1]:

                                        if mode in [0, 11, 12, 13, 18]:
                                            fix_cov_list = [0.025]
                                        elif mode in [0, 1, 2, 10, 14, 15, 16, 17, 19, 20]:
                                            fix_cov_list = [0.025]#[0.001]#[0.025]
                                        elif mode in [21]:
                                            fix_cov_list = [0.0]
                                        else:
                                            print("UNKNOWN MODE FOR FIX_COV")

                                        for fix_cov in fix_cov_list:

                                            # limit = -0.015 * (1.0 / fix_cov)

                                            if mode in [2, 10, 11, 16]:
                                                opt_mode_list = [1]
                                            elif mode in [0, 1, 12, 13, 14, 15, 17, 18, 19, 20, 21]:
                                                opt_mode_list = [4]
                                            else:
                                                print("UNKNOWN MODE FOR OPTIMIZER")

                                            for opt_mode in opt_mode_list:
                                                for alg in ["Q"]:
                                                    for lambda_ in [0.0]:

                                                        if opt_mode in [0]:
                                                            momentum_list = [0.9, 0.99]
                                                        elif opt_mode in [4, 1]:
                                                            momentum_list = [0.0]

                                                        for momentum in momentum_list:

                                                            if opt_mode in [4]:
                                                                rms_list = [0.0]
                                                            elif opt_mode in [0, 1]:
                                                                rms_list = [0.999]

                                                            for rms in rms_list:
                                                                if opt_mode in [4]:
                                                                    alpha_list = [0.004, 0.002, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 8e-06, 4e-06]#[0.03125, 0.0625, 0.125, 0.25, 0.3]#
                                                                elif opt_mode in [0, 1]:
                                                                    alpha_list = [0.01, 0.001, 0.0005]

                                                                for alpha in alpha_list:

                                                                    # for pri_thr in [0.0]:
                                                                    # for buffer_size in [1000]:
                                                                    for sync in [1]:

                                                                        estimate_max_step = 500000#30000 #
                                                                        new_data = np.zeros((0, estimate_max_step))

                                                                        pref = "exp_result/paper/illegalv_BW_ER/" \
                                                                               "REM_Dyna_mode" + str(mode) + "new"\
                                                                               "_offline" + str(offline) + \
                                                                               "_planning" + str(num_planning) + \
                                                                               "_priThrshd" + str(0.0) + \
                                                                                "_DQNc" + str(sync) + \
                                                                                "_buffer" + str(1000) + \
                                                                                "/always_add_prot_1/"
                                                                        pref += "/random_BufferOnly_alpha" + str(alpha) + \
                                                                                     "_divAF" + str(mode) + \
                                                                                     "_near" + str(num_near) + \
                                                                                     "_protLimit" + str(limit) + \
                                                                                     "_similarity" + str(similarity) + \
                                                                                     "_sampleLimit" + str(sample_limit) + \
                                                                                     "_kscale" + str(kscale) + \
                                                                                     "_fixCov" + str(fix_cov) + \
                                                                                     "_update" + str(alg) + \
                                                                                     "_lambda" + str(lambda_) + \
                                                                                     "_momentum" + str(momentum) + \
                                                                                     "_rms" + str(rms) + \
                                                                                     "_optMode" + str(opt_mode)

                                                                        # pref = "exp_result/test_llm/q_1x10/Q_learning_mode"+str(mode)+"/"
                                                                        # pref += "_alpha" + str(alpha) + \
                                                                        #         "_divAF" +str(mode) + \
                                                                        #         "_update" + str(alg) + \
                                                                        #         "_lambda" + str(lambda_) + \
                                                                        #         "_momentum" + str(momentum) + \
                                                                        #         "_rms" + str(rms) + \
                                                                        #         "_optMode" + str(opt_mode)

                                                                        # pref = "exp_result/random_ER_mode"+str(mode)+"_buffer200/"
                                                                        # pref += "_alpha" + str(alpha) + \
                                                                        #         "_divAF" +str(mode) + \
                                                                        #         "_update" + str(alg) + \
                                                                        #         "_lambda" + str(lambda_) + \
                                                                        #         "_momentum" + str(momentum) + \
                                                                        #         "_rms" + str(rms) + \
                                                                        #         "_optMode" + str(opt_mode)


                                                                        for run in range(0, 5):
                                                                            file = pref + "_run" + str(run) + ".npy"
                                                                            # print(file)
                                                                            if os.path.isfile(file):
                                                                                print(file)
                                                                                rem_record = np.load(file)
                                                                                print(len(rem_record))
                                                                                if len(rem_record < estimate_max_step):
                                                                                    print("length of record:", len(rem_record))
                                                                                    rem_record = np.concatenate((rem_record, np.zeros(estimate_max_step-len(rem_record))), axis=0)
                                                                                rem_record = rem_record.reshape((1, -1))
                                                                                new_data = np.concatenate((new_data, rem_record), axis=0)
                                                                            else:
                                                                                not_exist.append(file)
                                                                        if len(new_data) != 0:
                                                                            np.save(pref + "_"+str(len(new_data[0]))+"x" + str(len(new_data)), new_data)
                                                                            print("file saved:", pref)

                                                                        # step_data = []
                                                                        # for run in range(0, 5):
                                                                        #     file = pref + "_run" + str(run) + "_stepPerEp.npy"
                                                                        #     # print(file)
                                                                        #     if os.path.isfile(file):
                                                                        #         print(file)
                                                                        #         rem_record = np.load(file)
                                                                        #         step_data.append(rem_record)
                                                                        #     else:
                                                                        #         not_exist.append(file)
                                                                        # if len(step_data) != 0:
                                                                        #     with open(pref + "_stepPerEp_"+str(len(step_data))+".pkl", "wb") as f:
                                                                        #         pkl.dump(step_data, f)
                                                                        #     print("file saved:", pref)
    print("\nNot exist:")
    for f in not_exist:
        print(f)


def compare_result_old():
    agent_list = ["remDyna_noLearnNoProto_realCov"]#["qLearning", "priorityER", "randomER", "remDyna_noLearnNoProto_realCov"]
    agent_name = {"qLearning": "Q_learning",
                  "priorityER": "priority_ER",
                  "randomER": "random_ER",
                  "remDyna_noLearnNoProto_realCov": "REM_Dyna_pri_pred"}
    tile_list = ["32x4tile"]  # ["1x16tile", "4x16tile", "10x10tile", "32x4tile"]
    standard = {"1x16tile": 200,
                "4x16tile": 100,
                "10x10tile": 130,
                "32x4tile": 0,
                "32x2tile": 0}

    alpha_list = [0.001, 0.002, 0.004, 0.008, 0.0156, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    near_list = [8]
    limit_list = [0.25]
    x = 0
    for agent in agent_list:
        for tile in tile_list:
            plt.figure(x)
            x += 1

            for alpha in alpha_list:
                for near in near_list:
                    for limit in limit_list:
                        for run in range(10, 11):
                            file = "exp_result_cgw/" + agent + "/" + tile + "/ContinuousGridWorld_" + agent_name[
                                agent] + "_alpha" + \
                                   str(alpha) + "_divAB1_near" + str(near) + "_protLimit" + str(
                                limit) + "_10000x" + str(run) + ".npy"
                            print("looking for", file)
                            if os.path.isfile(file):
                                data = np.load(file)
                                print(file, "max =", np.mean(data[:, :], axis=0)[-1])

                                if agent == "remDyna_noLearnNoProto_realCov" and \
                                        np.mean(data[:, :], axis=0)[-1] >= standard[tile]:
                                    plt.plot(np.mean(data[:, :], axis=0),
                                             label="alpha=" + str(alpha) + " NN=" + str(near) + " thrsd=" + str(
                                                 limit) + " avg" + str(run))
                                elif agent == "remDyna_noLearnNoProto_realCov" and \
                                        np.mean(data[:, :], axis=0)[-1] < standard[tile]:
                                    plt.plot(np.mean(data[:, :], axis=0), label='')
                                else:
                                    plt.plot(np.mean(data[:, :], axis=0),
                                             label="alpha=" + str(alpha) + " avg" + str(run))

            plt.title(agent + " " + tile)
            plt.legend()
    plt.show(block=True)

def compare_result_new():
    num_run = 5
    fx = 0
    # color = ['b', 'p', 'r', 'g', 'orange']

    for offline in [1]:#[0]:#
        for num_near in [8]: #[1]:
            for num_planning in [10]:
                plt.figure(fx)
                fx += 1
                plt.xlim(0,100000)
                plt.ylim(0, 2700)

                for mode in [17]:
                    if mode == 0:
                        limit_list = [-1000.0]#[-0.1]#[-0.025]#[-4.0]#
                    elif mode == 17:
                        limit_list = [-1000.0]
                    elif mode == 21:
                        limit_list = [0.0]

                    for limit in limit_list:
                        for kscale in [1.0]:#[1e-05]:#[1.0]: #
                            for similarity in [0]:
                                for sample_limit in [0.0]:#[0.0, 0.5, 0.75]:#[0.0, 0.5, 0.75, 0.9, 0.95]:

                                    for add_prot in [1]:

                                        if mode in [0, 11, 12, 13, 18]:
                                            fix_cov_list = [0.025]#[0.025]#[0.001]#
                                        elif mode in [0, 1, 2, 10, 14, 15, 16, 17, 19, 20]:
                                            fix_cov_list = [0.025]#[0.001]#[0.025]
                                        elif mode in [21]:
                                            fix_cov_list = [0.0]
                                        else:
                                            print("UNKNOWN MODE FOR FIX_COV")

                                        for fix_cov in fix_cov_list:
                                            # for pri_thr in [0.0]:
                                            # for buffer_size in [1000]:
                                            for sync in [1]:


                                                if mode in [2, 10, 11, 16]:
                                                    opt_mode_list = [1]
                                                elif mode in [0, 1, 12, 13, 14, 15, 17, 18, 19, 20, 21]:
                                                    opt_mode_list = [4]
                                                else:
                                                    print("UNKNOWN MODE FOR OPTIMIZER")

                                                for opt_mode in opt_mode_list:
                                                    for alg in ["Q"]:
                                                        for lambda_ in [0.0]:

                                                            if opt_mode in [0]:
                                                                momentum_list = [0.9, 0.99]
                                                            elif opt_mode in [4, 1]:
                                                                momentum_list = [0.0]

                                                            for momentum in momentum_list:

                                                                if opt_mode in [4]:
                                                                    rms_list = [0.0]
                                                                elif opt_mode in [0, 1]:
                                                                    rms_list = [0.999]

                                                                for rms in rms_list:
                                                                    if opt_mode in [4]:
                                                                        alpha_list = [0.1, 0.01, 0.004, 0.002, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 8e-06, 4e-06]#[0.03125, 0.0625, 0.125, 0.25, 0.3]##[0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]#[0.03125, 0.0625, 0.125, 0.25, 0.3]#[0.00097656, 0.00195312, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
                                                                    elif opt_mode in [0, 1]:
                                                                        alpha_list = [0.01, 0.001, 0.0005]

                                                                    for lr_idx in range(len(alpha_list)):
                                                                        alpha = alpha_list[lr_idx]

                                                                        for n in range(num_run-5, num_run+1):
                                                                            # pref = "exp_result/nonLinearQ_baseline/minibatch1/" + \
                                                                            pref = "exp_result/noCons_BW_ER/" + \
                                                                                   "REM_Dyna_mode" + str(mode) +"noCons"+\
                                                                                   "_offline" + str(offline) + \
                                                                                   "_planning" + str(num_planning) + \
                                                                                   "_priThrshd0.0_DQNc" + str(sync) + \
                                                                                   "_buffer" + str(1000) + \
                                                                                   "/always_add_prot_1/"
                                                                            pref += "random_BufferOnly_alpha" + str(alpha) + \
                                                                                    "_divAF" + str(mode) + \
                                                                                    "_near" + str(num_near) + \
                                                                                    "_protLimit" + str(limit) + \
                                                                                    "_similarity" + str(similarity) + \
                                                                                    "_sampleLimit" + str(sample_limit) + \
                                                                                    "_kscale" + str(kscale) + \
                                                                                    "_fixCov" + str(fix_cov) + \
                                                                                    "_update" + str(alg) + \
                                                                                    "_lambda" + str(lambda_) + \
                                                                                    "_momentum" + str(momentum) + \
                                                                                    "_rms" + str(rms) + \
                                                                                    "_optMode" + str(opt_mode) + \
                                                                                    "_500000x"+str(n)+".npy"
                                                                                    # "_stepPerEp_" + str(n) + ".pkl"

                                                                            # # step per episode
                                                                            # pref = "exp_result/DQN_baseline/REM_Dyna_mode" + str(mode) + \
                                                                            #        "_offline" + str(offline) + \
                                                                            #        "_planning" + str(num_planning) + \
                                                                            #        "_priThrshd0.0_DQN_buffer" + str(buffer_size) + \
                                                                            #        "/always_add_prot_1/"
                                                                            # pref += "random_alpha" + str(alpha) + \
                                                                            #         "_divAF" + str(mode) + \
                                                                            #         "_near" + str(num_near) + \
                                                                            #         "_protLimit" + str(limit) + \
                                                                            #         "_similarity" + str(similarity) + \
                                                                            #         "_sampleLimit" + str(sample_limit) + \
                                                                            #         "_kscale" + str(kscale) + \
                                                                            #         "_fixCov" + str(fix_cov) + \
                                                                            #         "_update" + str(alg) + \
                                                                            #         "_lambda" + str(lambda_) + \
                                                                            #         "_momentum" + str(momentum) + \
                                                                            #         "_rms" + str(rms) + \
                                                                            #         "_optMode" + str(opt_mode) + \
                                                                            #         "_run"+str(n)+"_stepPerEp.npy"

                                                                            # pref = "exp_result/test_llm/q_1x10/Q_learning_mode" + str(mode) + "/"
                                                                            # pref += "_alpha" + str(alpha) + \
                                                                            #         "_divAF" + str(mode) + \
                                                                            #         "_update" + str(alg) + \
                                                                            #         "_lambda" + str(lambda_) + \
                                                                            #         "_momentum" + str(momentum) + \
                                                                            #         "_rms" + str(rms) + \
                                                                            #         "_optMode" + str(opt_mode) + \
                                                                            #         "_30000x"+str(n)+".npy"

                                                                            # pref = "exp_result/random_ER_mode" + str(mode) + "_buffer500/"
                                                                            # pref += "_alpha" + str(alpha) + \
                                                                            #         "_divAF" + str(mode) + \
                                                                            #         "_update" + str(alg) + \
                                                                            #         "_lambda" + str(lambda_) + \
                                                                            #         "_momentum" + str(momentum) + \
                                                                            #         "_rms" + str(rms) + \
                                                                            #         "_optMode" + str(opt_mode) + \
                                                                            #         "_30000x"+str(n)+".npy"

                                                                            f = pref
                                                                            if os.path.isfile(f):
                                                                                print("Loading", f)

                                                                                # accumulate reward
                                                                                d = np.load(f)
                                                                                learning = []
                                                                                for l in d:
                                                                                    start = 0
                                                                                    while start < len(l):
                                                                                        if l[start] != 0:
                                                                                            break
                                                                                        else:
                                                                                            start += 1
                                                                                    print("this run starts from", start)
                                                                                    # learning.append(l[start: start + 9200])
                                                                                    learning.append(l[start: start + 100000])
                                                                                learning = np.array(learning)

                                                                                # # # step per episode
                                                                                # # plt.plot(d, label="run "+str(n))
                                                                                # # plt.legend()
                                                                                # # step per episode find avg
                                                                                # with open(f, "rb") as r:
                                                                                #     d = pkl.load(r)
                                                                                # cut = 100000
                                                                                # for ep in d:
                                                                                #     length = len(ep)
                                                                                #     cut = length if length < cut else cut
                                                                                # ready_plot = np.zeros((len(d), cut))
                                                                                # for idx in range(len(d)):
                                                                                #     ready_plot[idx] = d[idx][:cut]
                                                                                # learning = ready_plot
                                                                                # md = np.mean(learning, axis=0)
                                                                                # if mode == 0:
                                                                                #     plt.plot(md, "--", label="raw - alpha="+str(alpha),
                                                                                #              color=color_list[lr_idx % len(color_list)]
                                                                                #              )
                                                                                # elif mode == 17:
                                                                                #     plt.plot(md, label="rep - alpha="+str(alpha),
                                                                                #              color=color_list[lr_idx % len(color_list)]
                                                                                #              )
                                                                                #
                                                                                # # sign = ":" if mode == 0 else "-."
                                                                                # # for i in range(len(learning)):
                                                                                # #     plt.plot(learning[i], sign, label="run "+str(i), color=color_list[i % len(color_list)])#, label=i, color=color_list[i % len(color_list)]
                                                                                # #     # if learning[i][-1] == 1:
                                                                                # #     #     print(i)

                                                                                md = np.mean(learning, axis=0)
                                                                                ste = np.abs(np.std(learning, axis=0)) / np.sqrt(len(learning))
                                                                                upper = md + ste * 1
                                                                                lower = md - ste * 1
                                                                                x = np.linspace(0, len(md), len(md))
                                                                                color_idx = np.where(np.array(alpha_list)==alpha)[0][0]
                                                                                if mode == 0:
                                                                                    plt.plot(x, md, "--", label="raw - alpha="+str(alpha),
                                                                                             # color=color_list[color_idx % len(color_list)]
                                                                                             )
                                                                                elif mode == 17:
                                                                                    plt.plot(x, md, label="rep - alpha="+str(alpha),
                                                                                             # color=color_list[color_idx % len(color_list)]
                                                                                             )
                                                                                # plt.errorbar(x, md, yerr=std)

                                                                                # plt.plot(x, upper, color=facecolor_list[color_idx % len(color_list)])
                                                                                # plt.plot(x, lower, color=facecolor_list[color_idx % len(color_list)])
                                                                                plt.fill_between(x, upper, lower,
                                                                                                 # facecolor=facecolor_list[color_idx % len(color_list)],
                                                                                                 alpha=0.3)

                                                                                # md = np.mean(d, axis=0)
                                                                                # plt.plot(md[:10001], label=f)

                                                                                # md = np.mean(d, axis=0)
                                                                                # top = 0
                                                                                # while top < len(md) - 1:
                                                                                #     if md[top] > md[top + 1]:
                                                                                #         break
                                                                                #     else:
                                                                                #         top += 1
                                                                                # plt.plot(md[:top+1], label=str([alpha, num_near, limit, similarity, fix_cov, rms, opt_mode]))
                                                                            else:
                                                                                print("Not found", f)
                                                                            plt.title(str(mode) + "_" + str(num_planning))
                                                                            plt.legend()


                                                # plt.title(str(mode) + "_" + str(alpha))
                                                # plt.legend()

                                                # rec = ["random_ER", 0, 32, 0.0625, 0.0, 4, 4, 0]
                                                # name = "exp_result_May/random_ER_mode0_32x4tiling/"
                                                # name += "_alpha" + str(rec[3]) + \
                                                #         "_divAF" + str(rec[1]) + \
                                                #         "_updateQ" + \
                                                #         "_lambda0.0" + \
                                                #         "_momentum0.0" + \
                                                #         "_rms" + str(rec[4]) + \
                                                #         "_optMode4" + \
                                                #         "_30000x5.npy"

                                                # buffer_size = 1000
                                                # if buffer_size in [250, 500, 1000]:
                                                #     er_lr = 0.125
                                                # else:
                                                #     er_lr = 0.0625
                                                # name = "exp_result/random_ER_mode0_buffer" + str(buffer_size) +"/"
                                                # name += "_alpha" + str(er_lr) + \
                                                #         "_divAF" + str(0) + \
                                                #         "_updateQ" + \
                                                #         "_lambda0.0" + \
                                                #         "_momentum0.0" + \
                                                #         "_rms" + str(0.0) + \
                                                #         "_optMode4" + \
                                                #         "_30000x5.npy"

                                                # if mode == 0:
                                                #     lr = 0.001
                                                #     lb = "raw-"
                                                # else:
                                                #     lr = 3e-05
                                                #     lb = "rep-"
                                                # name = "exp_result/sanity_check/er/REM_Dyna_mode"+str(mode)+"_offline1_planning"+str(num_planning)+"_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" \
                                                #        "random_BufferOnly_alpha"+str(lr)+"_divAF"+str(mode)+"_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.0_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_300000x5.npy"
                                                # d = np.load(name)
                                                # learning = []
                                                # for l in d:
                                                #     start = 0
                                                #     while start < len(l):
                                                #         if l[start] != 0:
                                                #             break
                                                #         else:
                                                #             start += 1
                                                #     print("this run starts from", start)
                                                #     # learning.append(l[start: start + 9200])
                                                #     learning.append(l[start: start + 100000])
                                                # learning = np.array(learning)
                                                # # for i in range(len(learning)):
                                                # #     plt.plot(learning[i], '--', label="run " + str(i), color=color_list[i % len(color_list)])  # , label=i, color=color_list[i % len(color_list)]
                                                # md = np.mean(learning, axis=0)
                                                # ste = np.abs(np.std(learning, axis=0)) / np.sqrt(len(learning))
                                                # upper = md + ste * 1
                                                # lower = md - ste * 1
                                                # x = np.linspace(0, len(md), len(md))
                                                # color_idx = np.where(np.array(alpha_list) == alpha)[0][0]
                                                # plt.plot(x, md, label=lb + "er best performance")
                                                # plt.fill_between(x, upper, lower, alpha=0.3)
                                                # plt.legend()

    plt.show()
    print("Shown")


def plot_nn_result():
    est = np.load("temp/test_estimation.npy")
    ftr = np.load("temp/test_feature.npy")
    exp = np.load("temp/test_expcted.npy")

    start = 0
    end = 20000
    plt.figure(0)
    fig, ax1 = plt.subplots()
    ax1.plot(est[start: end], '.', color="blue", label="prediction")
    ax1.plot(exp[start: end], '.', color="red", label="truth")
    ax2 = ax1.twinx()
    ax2.plot(ftr[start: end], '.', color="green", label="feature")
    # plt.legend()
    fig.tight_layout()
    plt.show(block=True)


def plot_nn_dist_cgw(folder="temp", save=True, min_clip_at=None, max_clip_at=None, normalize=True, wall=False, return_img=False):
    if not os.path.isfile(folder + "/test_points.npy"):
        print(folder, "doesn't exist")
        return
    else:
        print("Plotting:", folder)

    jsonfile = "parameters/continuous_gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    grid = 60
    # WW/BW
    goal_list = [[0.75, 0.95],
                 [0.6, 0.5],
                 [0.1, 0.9]]
    # # PW
    # goal_list = [[0.2, 0.5],
    #              [0.45, 0.75],
    #              [0.1, 0.2]]

    for goal in goal_list:
        goal_x = [goal[0], 1]#exp["env_params"]["goal_x"]#
        goal_y = [goal[1], 1]#exp["env_params"]["goal_y"]#
        if goal_x[0] > 1 or goal_y[0] > 1:
            goal_x, goal_y = [0, 1], [0.555, 1]
        gamma = exp["agent_params"]["nn_gamma"]
        num_tiling = 32 #exp["agent_params"]["num_tilings"]
        num_tile = 4 #exp["agent_params"]["num_tiles"]

        pts = np.load(folder + "/test_points.npy")
        found_goal = False
        pi = 0
        while not found_goal and pi < len(pts):
            p = pts[pi]
            if p[0] >= goal_x[0] and p[0] <= goal_x[1] and \
                    p[1] >= goal_y[0] and p[1] <= goal_y[1]:
                found_goal = True
                goal_idx = pi
                print("Goal", p)
            else:
                pi += 1

        print(pts.shape)
        rep = np.load(folder + "/test_representation.npy")

        # if normalize:
        #     for ind in range(len(rep)):
        #         rep[ind] = rep[ind] / np.linalg.norm(rep[ind])
        #         # rep[ind] = rep[ind] / np.max(np.abs(rep[ind]))

        rec = np.load(folder + "/test_reconstruction.npy") if os.path.isfile(folder + "/test_reconstruction.npy") else None
        loss = np.load(folder + "/rep_loss.npy") if os.path.isfile(folder + "/rep_loss.npy") else None

        print("Gamma =", gamma)

        # # check max and min similarity
        # maxsq, minsq = -10000, 10000
        # for r in rep:
        #     sq = np.dot(r,r)
        #     if sq > maxsq:
        #         maxsq = sq
        #     if sq < minsq:
        #         minsq = sq
        # print("max similarity =", maxsq)
        # print("min similarity =", minsq)
        print("input shape", pts.shape)
        print("output shape", rep.shape)

        # print("Range of Representation")
        # plt.figure()
        # count = np.zeros((rep.shape[1], 41))
        # for i in range(rep.shape[1]):
        #     print(np.min(rep[:, i]), np.max(rep[:, i]))
        #     for j in rep[:, i]:
        #         count[i, int((j+1)/2.0 * 40 // 1)] += 1
        # plt.imshow(count)
        # plt.colorbar()
        # plt.show()


        rep_dist = check_distance_cgw(pts, rep, rep[goal_idx], check_rep=False)
        if min_clip_at is not None:
            rep_dist[:, 2] = np.clip(rep_dist[:, 2], min_clip_at, np.max(rep_dist))
        if max_clip_at is not None:
            rep_dist[:, 2] = np.clip(rep_dist[:, 2], np.min(rep_dist), max_clip_at)

        if os.path.isfile(folder + "/test_reconstruction.npy"):
            if type(gamma) != list:
                rec_dist = check_distance_cgw(pts, rec, rec[goal_idx], check_rep=False)
            else:

                rec1 = rec[:, :num_tile * num_tiling * 2]
                rec2 = rec[:, num_tile * num_tiling * 2:]
                print("2 Successesor Features", rec.shape, rec1.shape, rec2.shape)
                rec_dist1 = check_distance_cgw(pts, rec1, rec1[goal_idx], check_rep=False, need_avg=False)
                rec_dist2 = check_distance_cgw(pts, rec2, rec2[goal_idx], check_rep=False, need_avg=False)

        print("Representation Distance")
        fig_count = 0
        fig = plt.figure(fig_count)
        plt.hexbin(rep_dist[:, 0], rep_dist[:, 1], C=rep_dist[:,2], gridsize=grid, cmap='autumn_r')
        plt.plot([goal_x[0]], [goal_y[0]], 'x', c='red')
        currentAxis = plt.gca()

        # # puddle world
        # plt.plot([0.45, 0.45], [0.8, 0.4], c="black")
        # plt.plot([0.45, 0.1], [0.75, 0.75], c="black")

        # block world
        if wall:
            currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=True, color='White'))
            currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=True, color='White'))
        else:
            currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
            currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))

        # plt.title("Representation")
        # cb = plt.colorbar()
        # cb.set_label('Distance')

        if return_img:
            return fig

        if save:
            slash_idx = list(folder).index("/")
            new_folder = folder[:slash_idx] + "_" + folder[slash_idx+1:]

        if save:
            # fig.savefig('../'+str(new_folder) +'_goal'+ str(p) + '_rep_clip' + str(min_clip_at)+"-"+str(max_clip_at) + '.png', dpi=fig.dpi)
            fig.savefig('../'+str(new_folder) +'_goal'+ str(p) + '_rep_clip' + str(min_clip_at)+"-"+str(max_clip_at) + '.pdf', dpi=fig.dpi)
            plt.clf()
            plt.cla()
            plt.close()
        else:
            plt.show()

        fig_count += 1


        print("Successesor feature Distance")
        if os.path.isfile(folder + "/test_reconstruction.npy"):
            if type(gamma) != list:
                fig = plt.figure(fig_count)
                plt.hexbin(rec_dist[:, 0], rec_dist[:, 1], C=rec_dist[:, 2], gridsize=grid, cmap='virdis')
                plt.plot([goal_x[0]], [goal_y[0]], 'x', c='red')
                currentAxis = plt.gca()
                currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
                currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
                plt.title("Reconstruction")
                cb = plt.colorbar()
                cb.set_label('Distance')
                if save:
                    fig.savefig('../'+str(new_folder) + '_rec.png', dpi=fig.dpi)
                    plt.clf()
                    plt.cla()
                    plt.close()
                else:
                    plt.show()
                fig_count += 1
            else:
                print("2 Figures")
                fig = plt.figure(fig_count)
                plt.hexbin(rec_dist1[:, 0], rec_dist1[:, 1], C=rec_dist1[:, 2], gridsize=grid, cmap='Blues')
                plt.plot([goal_x[0]], [goal_y[0]], 'x', c='red')
                currentAxis = plt.gca()
                currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
                currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
                plt.title("Reconstruction gamma=" + str(gamma[0]))
                cb = plt.colorbar()
                cb.set_label('Distance')
                if save:
                    fig.savefig('../'+str(new_folder) + '_rec1.png', dpi=fig.dpi)
                    plt.clf()
                    plt.cla()
                    plt.close()
                else:
                    plt.show()
                fig_count += 1

                fig = plt.figure(fig_count)
                plt.hexbin(rec_dist2[:, 0], rec_dist2[:, 1], C=rec_dist2[:, 2], gridsize=grid, cmap='Blues')
                plt.plot([goal_x[0]], [goal_y[0]], 'x', c='red')
                currentAxis = plt.gca()
                currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
                currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
                plt.title("Reconstruction gamma=" + str(gamma[1]))
                cb = plt.colorbar()
                cb.set_label('Distance')
                if save:
                    fig.savefig('../'+str(new_folder) + '_rec2.png', dpi=fig.dpi)
                    plt.clf()
                    plt.cla()
                    plt.close()
                else:
                    plt.show()
                fig_count += 1

            # w = np.load("last_layer_weight.npy")
            # w_inv = np.linalg.pinv(w)
            # p_rep = np.dot(rec, w_inv.T)
            # p_rep_dist = check_distance_cgw(pts, p_rep, p_rep[goal_idx], check_rep=False)
            # fig = plt.figure(fig_count)
            # plt.hexbin(p_rep_dist[:, 0], p_rep_dist[:, 1], C=p_rep_dist[:, 2], gridsize=grid, cmap='Blues')
            # currentAxis = plt.gca()
            # currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
            # currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
            # plt.title("Calculation")
            # cb = plt.colorbar()
            # cb.set_label('Distance')
            # if save:
            #     fig.savefig('../' + str(new_folder) + '_prep.png', dpi=fig.dpi)
            #     plt.clf()
            #     plt.cla()
            #     plt.close()
            # else:
            #     plt.show()

        # fig = plt.figure(fig_count)
        # plt.plot(loss)
        # plt.title("Loss")
        # if save:
        #     fig.savefig('../'+str(new_folder) + '_loss.png', dpi=fig.dpi)
        #     plt.clf()
        #     plt.cla()
        #     plt.close()
        # else:
        #     plt.show()


def plot_recovered_state_cgw(folder="temp", save=True):
    jsonfile = "parameters/continuous_gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()
    rcv_state = np.load(folder + "/recv_state.npy")
    pts = np.load(folder + "/test_points.npy")
    print(rcv_state)

    num_tiling = exp["agent_params"]["nn_num_tilings"]
    num_tile = exp["agent_params"]["nn_num_tiles"]
    length = 2 #num_tile * num_tiling * 2#num_tile ** 2 * num_tiling

    # assert length == rcv_state.shape[1] // 2
    # assert len(pts) == len(rcv_state)

    gt = rcv_state[:, :length]
    rcv = rcv_state[:, length:]
    comp = np.zeros(rcv_state.shape[0])
    samples = np.random.randint(0, high=len(gt), size=20)
    gt_sample = gt[samples]
    rcv_sample = rcv[samples]

    color_idx = np.linspace(0, 1, len(gt_sample))
    plt.figure()
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    for i in range(len(gt_sample)):
        plt.scatter(gt_sample[i, 0], gt_sample[i, 1], s=2, color=plt.cm.viridis(color_idx[i]))
    plt.figure()
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    for i in range(len(rcv_sample)):
        plt.scatter(rcv_sample[i, 0], rcv_sample[i, 1], s=2, color=plt.cm.viridis(color_idx[i]))
    plt.figure()
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    for i in range(len(rcv_sample)):
        plt.scatter(gt_sample[i, 0], gt_sample[i, 1], s=2, color=plt.cm.viridis(color_idx[i]))
        plt.scatter(rcv_sample[i, 0], rcv_sample[i, 1], s=2, color=plt.cm.viridis(color_idx[i]))
    plt.show()
    for i in range(len(rcv_state)):
        comp[i] = np.linalg.norm(gt[i] - rcv[i])
        
        # idx = rcv[i].argsort()[-num_tiling:]
        # f = np.zeros(num_tile ** 2 * num_tiling)
        # f[idx] = 1
        # if equal_array(gt[i], f):
        #     comp[i] = 0.0
        # else:
        #     comp[i] = 1.0
        #     print(gt[i].argsort()[-num_tiling:], idx)
        
    # for j in range(len(comp)):
    #     if comp[j] != 0.0 and comp[j] != 1.0:
    #         print(comp[j], pts[j])

    fig = plt.figure()
    plt.hexbin(pts[:, 0], pts[:, 1], comp, gridsize=500, cmap="Greens")
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    plt.colorbar()
    plt.title("Recovered states")
    if save:
        slash_idx = list(folder).index("/")
        new_folder = folder[:slash_idx] + "_" + folder[slash_idx+1:]
        fig.savefig('../'+str(new_folder) + '_state.png', dpi=fig.dpi)
        plt.clf()
        plt.cla()
        plt.close()
    else:
        plt.show()


def plot_recovered_state_dgw(folder="temp", save=False):
    jsonfile = "parameters/gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    size_x = exp["env_params"]["size_x"]
    size_y = exp["env_params"]["size_y"]

    rcv_state = np.load(folder + "/recv_state.npy")
    length = rcv_state.shape[1] // 2
    gt = rcv_state[:, :length]
    rcv = rcv_state[:, length:]
    comp = np.zeros((size_x, size_y))
    for i in range(len(rcv_state)):
        pts = recover_oneHot(gt[i], size_x, size_y)
        # comp[pts[0], pts[1]] = np.linalg.norm(gt[i] - rcv[i])

        f = np.zeros(length)
        f[np.where(rcv[i] == np.max(rcv[i]))[0][0]] = 1
        s = recover_oneHot(f, size_x, size_y)
        if equal_array(s, pts):
            comp[pts[0], pts[1]] = 0
        else:
            comp[pts[0], pts[1]] = 1.0
            print(s, pts)

    if not save:
        plt.figure(0)
        plt.imshow(comp.transpose(), cmap='Greens')
        plt.colorbar()
        plt.title("Recovered states")
        plt.show()
    else:
        slash_idx = list(folder).index("/")
        new_folder = folder[:slash_idx] + "_" + folder[slash_idx+1:]
        fig = plt.figure(0)
        plt.imshow(comp.transpose(), cmap='Greens')
        plt.colorbar()
        plt.title("Recovered states")
        fig.savefig('../'+str(new_folder) + '_state.png', dpi=fig.dpi)
        plt.clf()
        plt.cla()
        plt.close()


def plot_nn_dist_dgw(folder="temp_discrete", save=False, min_clip_at=None, max_clip_at=None, switch_y=False):
    jsonfile = "parameters/gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    size_x = exp["env_params"]["size_x"]
    size_y = exp["env_params"]["size_y"]

    print("Loading data from", folder)
    pts = np.load(folder + "/test_points.npy")
    rep = np.load(folder + "/test_representation.npy")
    if os.path.isfile(folder + "/test_reconstruction.npy"):
        rec = np.load(folder + "/test_reconstruction.npy")
        rec_list = [rec[:, : size_x * size_y], rec[:, size_x * size_y:], rec]
    else:
        rec_list = None
    loss = np.load(folder + "/loss.npy") if os.path.isfile(folder + "/loss.npy") else None

    goal = [exp["env_params"]["goal_x"], exp["env_params"]["goal_y"]]
    print("goal", goal)

    rep_copy = np.copy(rep)
    rep_copy = preprocessing.scale(rep_copy)
    rep_copy = normalize_ground_truth(rep_copy, 0.998)
    pca = PCA(n_components=1)
    rep_pca = pca.fit_transform(rep_copy)
    # print("************************")
    # print(pca.components_)
    # eigenv = pca.components_[:8]
    # rep_pca = np.dot(eigenv, rep.transpose()).transpose()
    # print(rep_pca.shape)

    feature = one_hot_feature(pts, size_x, size_y)
    rep_data = np.concatenate((feature, rep), axis=1)[:-1]
    if rec_list is not None:
        rec_data_list = []
        for rec in rec_list:
            rec_data_list.append(np.concatenate((feature, rec), axis=1)[:-1])
    else:
        rec_data_list = None

    representation = construct_grid(rep_data, size_x, size_y)
    print(representation.shape)
    print("============================")
    for i in representation[9, 0]:
        print("{:8.4f}".format(i), end=" ")
    print()
    for k in representation[9, 1]:
        print("{:8.4f}".format(k), end=" ")
    print()
    for j in representation[1, 6]:
        print("{:8.4f}".format(j), end=" ")
    print()
    for k in representation[0, 2]:
        print("{:8.4f}".format(k), end=" ")
    print()
    if rec_data_list is not None:
        reconstruction = []
        for rec_data in rec_data_list:
            reconstruction.append(construct_grid(rec_data, size_x, size_y))
    else:
        reconstruction = None

    # for xi in range(representation.shape[1]):
    #     for yi in range(representation.shape[0]):
    #         representation[yi, xi] /= np.linalg.norm(representation[yi, xi])

    print("Distance measured by representation: \n")
    rep_dist = check_distance_dgw(representation, size_x, size_y, goal)
    if min_clip_at is not None:
        rep_dist = np.clip(rep_dist, min_clip_at, np.max(rep_dist))
    if max_clip_at is not None:
        rep_dist = np.clip(rep_dist, np.min(rep_dist), max_clip_at)

    new_rep_dist = np.zeros((rep_dist.shape[0], rep_dist.shape[1]))
    if switch_y:
        for i in range(size_y):
            new_rep_dist[:, size_y-1 - i] = rep_dist[:, i]
        rep_dist = new_rep_dist

    pca_set = np.concatenate((feature, rep_pca), axis=1)
    temp = construct_grid(pca_set, size_x, size_y)
    print("After PCA")
    pca_dist = check_distance_dgw(temp, size_x, size_y, goal)

    # walls = get_wall(exp["env_params"])
    # for w in walls:
    #     rep_dist[w[0], w[1]] = 0

    print("\n")
    if reconstruction is not None:
        rec_dist_list = []
        for idx in range(len(reconstruction)):
            print("Distance measured by reconstruction:", idx, ": \n")
            rec_dist_list.append(check_distance_dgw(reconstruction[idx], size_x, size_y, goal))

    if not save:
        """
        if os.path.isfile(folder + "/test_reconstruction.npy"):

            fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
            axes[0, 0].imshow(rec_dist_list[0].transpose(), cmap='Blues')#, interpolation='nearest')
            axes[0, 0].set_title("Z1")
            axes[1, 0].imshow(rec_dist_list[1].transpose(), cmap='Blues')#, interpolation='nearest')
            axes[1, 0].set_title("Z2")
            axes[1, 1].imshow(rec_dist_list[2].transpose(), cmap='Blues')#, interpolation='nearest')
            axes[1, 1].set_title("[Z1,Z2]")

            axes[0, 1].imshow(rep_dist.transpose(), cmap='Blues', interpolation='nearest')
            axes[0, 1].set_title("Representation")


        if os.path.isfile(folder + "/loss.npy"):
            plt.figure(1)
            plt.plot(loss)
            plt.title("Loss")

        plt.show()

        """
        fig_count = 0

        # if os.path.isfile(folder + "/test_reconstruction.npy"):
        #     for rec_dist in rec_dist_list:
        #         plt.figure(fig_count)
        #         plt.imshow(rec_dist.transpose(), cmap='Blues', interpolation='nearest')
        #         plt.colorbar()
        #         plt.title("Reconstruction " + str(fig_count))
        #         fig_count += 1

        plt.figure(fig_count)
        plt.imshow(rep_dist.transpose(), cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.title("Representation")
        fig_count += 1

        plt.figure(fig_count)
        plt.imshow(pca_dist.transpose(), cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.title("representation_pca")
        fig_count += 1

        # if os.path.isfile(folder + "/rep_loss.npy"):
        #     plt.figure(fig_count)
        #     plt.plot(loss)
        #     plt.title("Loss")

        plt.show()

    else:
        slash_idx = list(folder).index("/")
        new_folder = folder[:slash_idx] + "_" + folder[slash_idx+1:]
        fig_count = 0
        # if reconstruction is not None:
        #     for rec_dist in rec_dist_list:
        #         fig = plt.figure(fig_count)
        #         plt.imshow(rec_dist.transpose(), cmap='Blues', interpolation='nearest')
        #         plt.title("Reconstruction " + str(fig_count))
        #         plt.colorbar()
        #         fig.savefig('../'+str(new_folder) + '_rec' + str(fig_count) + '.png', dpi=fig.dpi)
        #         plt.clf()
        #         plt.cla()
        #         plt.close()
        #         fig_count += 1

        fig = plt.figure(fig_count)
        plt.imshow(rep_dist.transpose(), cmap='Blues', interpolation='nearest')
        plt.title("Representation")
        plt.colorbar()
        fig.savefig('../'+str(new_folder) + '_rep_clip' + str(min_clip_at)+"-"+str(max_clip_at) + '.png', dpi=fig.dpi)
        plt.clf()
        plt.cla()
        plt.close()
        fig_count += 1
        
        if loss is not None:
            fig = plt.figure(fig_count)
            plt.plot(loss)
            plt.title("Loss")
            fig.savefig('../'+str(new_folder) + '_repLoss.png', dpi=fig.dpi)
            plt.clf()
            plt.cla()
            plt.close()


def param_sweep_plot_nn_dist():
    optmz_list = ['adam']
    lr_list = [0.001]
    num_epochs = [40, 80, 100, 150]
    batchsize_list = [128]
    beta_list = ['[1.0, 1.0, 0.5]', '[1.0, 1.0, 0.8]', '[1.0, 1.0, 0.9]', '[1.0, 1.0, 0.95]']
    d_list = [1, 0, -1]

    for opt in optmz_list:
        for lr in lr_list:
            for epoch in num_epochs:
                for size in batchsize_list:
                    for beta in beta_list:
                        for d in d_list:
                            folder = "temp/" + str(opt) + "_" + str(lr) + "_" + str(epoch) + "_" + str(
                                size) + "_" + str(beta) + "-" + str(float(d))
                            if os.path.exists(folder) and len(os.listdir(folder)) >= 5:
                                plot_nn_dist_dgw(folder=folder, save=True)
                                print("Success", folder)
                                # continue
                            else:
                                print("Not exist", folder)


def param_sweep_plot_nn_dist_lplcrl():
    numf_list = [16, 32]
    lr_list = [0.00005, 0.0001, 0.0005, 0.001]
    momentum_list = [0.0, 0.125]
    batchsize_list = [512]

    for f in numf_list:
        for lr in lr_list:
            for mmt in momentum_list:
                for bs in batchsize_list:
                    folder = "temp/" + str(f) + "_" + str(lr) + "_" + str(mmt) + "_" + str(bs)
                    if os.path.exists(folder) and len(os.listdir(folder)) >= 5:
                        plot_nn_dist_dgw(folder=folder, save=True)
                        print("Success", folder)
                        # continue
                    else:
                        print("Not exist", folder)


def check_ground_truth_dgw(normalize=True):
    jsonfile = "parameters/gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    file_name = exp["agent_params"]["nn_model_name"]
    file_name += "_discrete"

    size_x = exp["env_params"]["size_x"]
    size_y = exp["env_params"]["size_y"]
    goal_x = exp["env_params"]["goal_x"]
    goal_y = exp["env_params"]["goal_y"]
    gamma = exp["agent_params"]["nn_gamma"]

    training_set = np.load("random_data/fixed_env_suc_prob_1.0/dgw_training_set_noGoal_randomStart_0opt_[0.998, 0.8]gamma_1pts_x1_x500000.npy")

    training1 = training_set
    if normalize:
        training1[:, size_x*size_y:] = preprocessing.normalize(training1[:, size_x*size_y:])

    temp = construct_grid(training1, size_x, size_y)
    dist = check_distance_dgw(temp, size_x, size_y, [goal_x, goal_y])
    plt.figure()
    plt.imshow(dist.transpose(), cmap='Blues', interpolation='nearest')
    plt.colorbar()
    # plt.show()

    # print("Gamma =", gamma[1])
    # temp = construct_grid(training2, size_x, size_y)
    # dist = check_distance_dgw(temp, size_x, size_y, [goal_x, goal_y])
    # plt.figure()
    # plt.imshow(dist.transpose(), cmap='Blues', interpolation='nearest')
    plt.show()


def check_ground_truth_cgw_xy(normalize=True):
    jsonfile = "parameters/continuous_gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    # goal_x = exp["env_params"]["goal_x"][0]
    # goal_y = exp["env_params"]["goal_y"][0]
    gamma = exp["agent_params"]["nn_gamma"]
    num_tilings = 32#exp["agent_params"]["nn_num_tilings"]
    num_tiles = 4#exp["agent_params"]["nn_num_tiles"]
    gsize = 15

    # training_set = np.load("random_data/fixed_env_suc_prob_1.0/cgw_noGoal_separateTC"+
    #                        str(num_tilings)+"x"+str(num_tiles)+
    #                        "_training_set_randomStart_0opt_0.998gamma_1pts_x1_x100000.npy")
    training_set = np.load("random_data/fixed_env_suc_prob_1.0/pw_noGoal_raw_training_set_randomStart_0opt_[0.998, 0.8]gamma_1pts_x1_x100000.npy")
    print(list(training_set)[:10])

    len_f = 2#num_tiles*num_tilings*2

    training_set = average_training_set(training_set, len_f, len_state=2)
    print("Gamma =", gamma)

    gidx = 0
    while gidx < len(training_set):
        # if 0.7 <= training_set[gidx, 0] <= 0.75 and 0.8 <= training_set[gidx, 1] <= 1.0:
        if 0.95 <= training_set[gidx, 0] <= 1.0 and 0.95 <= training_set[gidx, 1] <= 1.0:
            break
        else:
            gidx += 1

    pca = PCA(n_components=4)
    gt = np.copy(training_set[:, 2 + len_f:])
    # gt = preprocessing.scale(gt)
    # gt = normalize_ground_truth(gt, 0.998)

    pts = np.copy(training_set[:, :2])
    rep_pca = pca.fit_transform(gt)
    # print(pts.shape, rep_pca.shape)
    # print(pca.singular_values_)
    # print(pca.explained_variance_)

    pca_dist = check_distance_cgw(pts, rep_pca, rep_pca[gidx], check_rep=False)
    # pca_dist = np.clip(pca_dist, 0, 500)
    goal_x, goal_y = training_set[gidx, 0], training_set[gidx, 1]
    plt.figure(0)
    plt.plot([goal_x], [goal_y], "x", color='red')
    plt.hexbin(pca_dist[:, 0], pca_dist[:, 1], C=pca_dist[:, 2], gridsize=gsize, cmap='Blues')
    plt.colorbar()
    plt.title("Distance given by pca")
    # plt.show()


    # if normalize:
    #     training_set[:, 2+len_f:] = normalize_ground_truth(training_set[:, 2+len_f:], [0.998, 0.8])
    goal_x, goal_y, goal_rep = training_set[gidx, 0], training_set[gidx, 1], training_set[gidx, 2+len_f:]

    dist = check_distance_cgw(training_set[:, :2], training_set[:, 2 + len_f:], goal_rep, check_rep=False)
    goal_x, goal_y = training_set[gidx, 0], training_set[gidx, 1]

    plt.figure(1)
    plt.plot([goal_x], [goal_y], "x", color='red')
    plt.hexbin(dist[:, 0], dist[:, 1], C=dist[:, 2], gridsize=gsize, cmap='Blues')
    plt.colorbar()
    plt.title("Distance given by successor features")
    plt.show()


    # size = 40
    # goal_x, goal_y = int(0.75*size), size - int(0.95*size)
    # training_set = plot_avg_training_set(training_set_copy, num_tiles**2*num_tilings, len_state=2, size=size)
    # # if normalize:
    # #     # training_set[:, 2+num_tilings*num_tiles*2:] = normalize_ground_truth(training_set[:, 2+num_tilings*num_tiles*2:], 0.998)
    # #     training_set[:, 2+num_tilings*num_tiles**2:] = normalize_ground_truth(training_set, 0.998)
    #
    # goal_rep = training_set[goal_y, goal_x]#training_set[gidx, 0], training_set[gidx, 1], training_set[gidx, 2+num_tilings*num_tiles*2:]
    # print("Goal = [", goal_x, goal_y,"]")er
    # print("size", training_set.shape)
    #
    # dist = np.zeros((size, size))
    # for y in range(size):
    #     for x in range(size):
    #         dist[y, x] = np.linalg.norm(training_set[y, x] - training_set[goal_y, goal_x])
    #
    # plt.figure(1)
    # plt.plot([goal_x], [goal_y], "x", color='red')
    # # plt.hexbin(dist[:, 0], dist[:, 1], C=dist[:, 2], gridsize=gsize, cmap='Blues')
    # plt.imshow(dist, cmap="Blues")
    # plt.colorbar()
    # plt.title("Distance given by successor features")
    # plt.show()





def check_ground_truth_cgw_tc():
    jsonfile = "parameters/continuous_gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    goal_x = exp["env_params"]["goal_x"][0]
    goal_y = exp["env_params"]["goal_y"][0]
    if goal_x > 1 or goal_y > 1:
        goal_x, goal_y = 0.45, 0

    gamma = exp["agent_params"]["nn_gamma"]
    num_tiling = exp["agent_params"]["num_tilings"]
    num_tile = exp["agent_params"]["num_tiles"]
    opt_prob = exp["agent_params"]["opt_prob"]
    num_point = 1
    num_ep = 1

    name = "random_data/cgw_noGoal_tc" + str(num_tiling) + "x" + str(num_tile) + "_training_set_randomStart_" + str(
        int(opt_prob * 100)) + "opt_" + str(gamma) + "gamma_" + str(num_point) + "pts_x" + str(num_ep) + ".npy"
    print("load from", name)
    training_set = np.load(name)
    print(training_set.shape)

    len_state = 2
    len_f = num_tile ** 2 * num_tiling

    training_set = average_training_set(training_set, len_f, len_state=2)
    state, f, gt = training_set[:, :len_state], training_set[:, len_state:len_state + len_f], training_set[:,
                                                                                              len_state + len_f:]
    if type(gamma) != list:
        gt = normalize_ground_truth(gt, gamma)
    else:
        gt1 = normalize_ground_truth(gt[:, :len_f], gamma[0])
        gt2 = normalize_ground_truth(gt[:, len_f:], gamma[1])
        gt = np.concatenate((gt1, gt2), axis=1)
    # gt = preprocessing.scale(gt)
    training_set = np.concatenate((f, gt), axis=1)
    print("Goal", state[-1])
    goal_rep = gt[-1]
    print(state.shape, gt.shape, goal_rep.shape)
    if type(gamma) != list:
        dist = check_distance_cgw(state, gt, goal_rep, check_rep=False)
    else:
        dist1 = check_distance_cgw(state, gt[:, :len_f], goal_rep[:len_f], check_rep=False)
        dist2 = check_distance_cgw(state, gt[:, len_f:], goal_rep[len_f:], check_rep=False)

    if type(gamma) != list:
        plt.figure(0)
        plt.hexbin(dist[:, 0], dist[:, 1], C=dist[:, 2], gridsize=10, cmap='Blues')
        cb = plt.colorbar()
        cb.set_label('Distance')
        plt.title("Distance given by successor features")
    else:
        plt.figure(0)
        plt.hexbin(dist1[:, 0], dist1[:, 1], C=dist1[:, 2], gridsize=10, cmap='Blues')
        cb = plt.colorbar()
        cb.set_label('Distance')
        plt.title("Distance given by successor features, gamma=" + str(gamma[0]))

        plt.figure(1)
        plt.hexbin(dist2[:, 0], dist2[:, 1], C=dist2[:, 2], gridsize=10, cmap='Blues')
        cb = plt.colorbar()
        cb.set_label('Distance')
        plt.title("Distance given by successor features, gamma=" + str(gamma[1]))

    pca = PCA(n_components=32)
    gt = pca.fit_transform(gt)
    training_set = np.concatenate((f, gt), axis=1)
    goal_rep = gt[-1]
    dist_rep = check_distance_cgw(state, gt, goal_rep, check_rep=False)

    plt.figure(2)
    plt.hexbin(dist_rep[:, 0], dist_rep[:, 1], C=dist_rep[:, 2], gridsize=10, cmap='Blues')
    cb = plt.colorbar()
    cb.set_label('Distance')
    plt.title("Distance given by PCA")

    plt.show()


def save_figures():
    gamma_list = ["0.998", "[0.998, 0.8]", "[0.998, 0.99]"]
    constraint_list = [0, 1]
    beta_list = [0.001, "1e-05", "1e-06"]
    delta_list = [1]
    lr_list = [0.0001]
    num_epochs = [100]
    scale_list = [0, 1]

    not_end = []

    for scale in scale_list:
        for gamma in gamma_list:
            for beta in beta_list:
                f1 = "temp_continuous/beta" + str(beta) + "_delta1.0_gamma" + str(gamma) + "_lr0.0001-0.0001_epoch200-200_scale" + str(scale)
                try:
                    plot_nn_dist_cgw(folder=f1, save=True, clip_at=None)
                    #plot_recovered_state_cgw(folder=f1, save=True)
                except:
                    not_end.append(f1)
            f1 = "temp_continuous/noConstraint_gamma" + str(gamma) + "_lr0.0001-0.0001_epoch200-00_scale" + str(scale)
            try:
                plot_nn_dist_cgw(folder=f1, save=True, clip_at=None)
                #plot_recovered_state_cgw(folder=f1, save=True)
            except:
                not_end.append(f1)
            """
            f2 = "temp_discrete/beta" + beta + "_delta1.0_lr0.0001-0.0001_epoch200-200_scale" + scale
            plot_nn_dist_dgw(folder=f2, save=True, clip_at=None)
            plot_recovered_state_dgw(folder=f2, save=True)
            """
            
    print(not_end)

def remove_data():
    betas = ['1e-07', '1e-06', '1e-05', '0.0001', '0.001']
    scales = ['0', '1']
    for beta in betas:
        for scale in scales:
            f1 = "temp_continuous/beta" + beta + "_delta1.0_lr0.0001-0.0001_epoch100-100_scale" + scale + "/"
            os.system("rm -rf " + f1 + "*.npy")
            f2 = "temp_discrete/beta" + beta + "_delta1.0_lr0.0001-0.0001_epoch100-100_scale" + scale + "/"
            os.system("rm -rf " + f2 + "*.npy")


def check_weight(save=True):
    for alg in ["Q"]:
        if alg == "Sarsa":
            lambda_ = [0.0, 0.9]
        else:
            lambda_ = [0.0]
        for l in lambda_:
            for alpha in [1e-4, 0.001, 0.01]:
                for opt_mode in [0, 1]:
                    if opt_mode in [1, 3]:
                        momentum = [0.0]
                    else:
                        momentum = [0.9, 0.99]
                    for m in momentum:
                        for rms in [0.9, 0.99, 0.999]:
                            filename = "exp_check_weight/random_ER/weight_log" + \
                                       "alg" + str(alg) + \
                                       "_lambda" + str(l) + \
                                       "_lr" + str(alpha) + \
                                       "_optMode" + str(opt_mode) + \
                                       "_beta" + str(m) + \
                                       "-" + str(rms)
                            try:
                                weight = np.load(filename + ".npy")

                                found = False
                                i = len(weight) - 1
                                while i > 0 and not found:
                                    if np.sum(weight[i]) != 0:
                                        num_ep = i + 1
                                        print(filename, "ep", num_ep)
                                        found = True
                                    else:
                                        i -= 1
                                if found:
                                    for a in range(4):
                                        fig = plt.figure(a)
                                        for w in range(32 * a, 32 * (a + 1)):
                                            plt.plot(weight[: num_ep, w])
                                        if save:
                                            print("saving")
                                            fig.savefig(filename + 'action' + str(a) + '.png', dpi=fig.dpi)
                                            fig.clf()
                                        if not save:
                                            plt.show()
                                else:
                                    print("Nothing saved", filename)

                            except:
                                print("Failed", filename)

def check_reward(save=True):

    fi = 0
    for agent in ["Q_learning", "random_ER"]:
        fig = plt.figure(fi)
        for learning_mode in [2]:

            for alg in ["Q"]:

                if alg == "Sarsa":
                    lambda_ = [0.0, 0.9]
                else:
                    lambda_ = [0.0]

                for l in lambda_:
                    for alpha in [0.001, 0.01, 0.1]:
                        for opt_mode in [0, 1]:

                            if opt_mode in [1, 3]:
                                momentum = [0.0]
                            else:
                                momentum = [0.9, 0.99]

                            for m in momentum:
                                for rms in [0.9, 0.99, 0.999]:

                                    if agent == "Q_learning":
                                        sample_list = []#["random"]
                                    elif agent == "random_ER":
                                        sample_list = ["random"]#["random", "priority"]

                                    for sample in sample_list:
                                        common_name = "alg" + str(alg) + \
                                                      "_lambda" + str(l) + \
                                                      "_lr" + str(alpha) + \
                                                      "_optMode" + str(opt_mode) + \
                                                      "_beta" + str(m) + \
                                                      "-" + str(rms)
                                        folder = "exp_check_weight/" + str(agent) + "_" + str(sample) + "/"
                                        filename = folder + common_name

                                        try:
                                            reward = np.load(filename + ".npy")
                                            if len(reward) >= 2 and reward[-1] > 12500:

                                                plt.plot(reward, label = agent+"_"+str(sample)+"_"+common_name)

                                            else:
                                                print("Nothing saved", filename)

                                        except:
                                            print("Failed", filename)

        if save:
            print("saving")
            fig.legend()
            try:
                fig.savefig(agent + '.png', dpi=fig.dpi)
                fig.clf()
            except:
                print("Nothing plot in figure")
        if not save:
            plt.show()

def visit_time():
    file_name = "random_data/fixed_env_suc_prob_1.0/cgw_noGoal_separateTC32x4" \
                "_training_set_randomStart_0opt_0.998gamma_1pts_x1_x50000.npy"
    seq = np.load(file_name)

    size = 40
    len_size = 1.0 / size
    grids = np.zeros((size, size))
    pts = seq[:, :2]
    for d in pts:
        x, y = int(d[0] // len_size), int(d[1] // len_size)
        grids[size-y-1, x] += 1

    grids = grids / grids[0, 30]
    # grids = np.clip(grids, 0, 20)

    print(list(grids))
    plt.figure()
    plt.imshow(grids, cmap="Blues")
    plt.colorbar()
    plt.show()

    return

def read_img_file(betas, deltas, epoch, legal, normalize=True):
    fig, axs = plt.subplots(ncols=len(deltas), nrows=len(betas), sharex=True, figsize=(500, 500))
    print(axs.shape)
    for i in range(len(betas)):
        for j in range(len(deltas)):
            # f = "temp_continuous_continuous_input[0.0, 1]_envSucProb1.0_legalv0_gamma0.998_epoch1000_nfeature32" + \
            #     "_beta" + str(betas[i]) + "_delta" + str(deltas[j]) + "_rep_clip0-None.png"
            # try:
            #     img = plt.imread(path+f)
            #     axs[i, j].imshow(img)
            #     axs[i, j].set_title("beta=" + str(betas[i]) + " delta=" + str(deltas[j]), fontsize=5)
            #     axs[i, j].axis('off')
            # except:
            #     axs[i, j].axis('off')
            #     print(f, "does not exist")
            folder = "temp_continuous/" + "continuous_input[0.0, 1]_envSucProb1.0_legalv"+str(legal)+"_gamma[0.998, 0.8]_epoch"+str(epoch)+"_nfeature32" + \
                "_beta" + str(betas[i]) + "_delta" + str(deltas[j])
            if not os.path.isfile(folder + "/test_points.npy"):
                print(folder, "doesn't exist")
                axs[i, j].axis('off')
            else:
                print("Plotting:", folder)

                jsonfile = "parameters/continuous_gridworld.json"
                json_dat = open(jsonfile, 'r')
                exp = json.load(json_dat)
                json_dat.close()

                grid = 60

                goal_x = [0.95, 1]  # exp["env_params"]["goal_x"]#
                goal_y = [0.95, 1]  # exp["env_params"]["goal_y"]#
                if goal_x[0] > 1 or goal_y[0] > 1:
                    goal_x, goal_y = [0, 1], [0.555, 1]
                gamma = exp["agent_params"]["nn_gamma"]
                num_tiling = 32  # exp["agent_params"]["num_tilings"]
                num_tile = 4  # exp["agent_params"]["num_tiles"]

                pts = np.load(folder + "/test_points.npy")
                found_goal = False
                pi = 0
                while not found_goal and pi < len(pts):
                    p = pts[pi]
                    if p[0] >= goal_x[0] and p[0] <= goal_x[1] and \
                            p[1] >= goal_y[0] and p[1] <= goal_y[1]:
                        found_goal = True
                        goal_idx = pi
                        print("Goal", p)
                    else:
                        pi += 1

                print(pts.shape)
                rep = np.load(folder + "/test_representation.npy")

                if normalize:
                    # rep = np.concatenate((rep, 10*np.ones((len(rep), 1))), axis=1)
                    for ind in range(len(rep)):
                        # rep[ind] = (rep[ind] / np.linalg.norm(rep[ind]) +1) /2.0
                        rep[ind] = rep[ind] / np.linalg.norm(rep[ind])

                print("Gamma =", gamma)

                print("input shape", pts.shape)
                print("output shape", rep.shape)
                rep_dist = check_distance_cgw(pts, rep, rep[goal_idx], check_rep=False)

                print("Representation Distance")

                im = axs[i, j].hexbin(rep_dist[:, 0], rep_dist[:, 1], C=rep_dist[:, 2], gridsize=grid, cmap='Blues')
                axs[i, j].plot([goal_x[0]], [goal_y[0]], 'x', c='red')
                # currentAxis = axs[i, j].gca()
                # axs[i, j].add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
                # axs[i, j].add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
                axs[i, j].set_title("beta=" + str(betas[i]) + " delta=" + str(deltas[j]), fontsize=5)

                fig.colorbar(im, ax=axs[i, j])
                # cb = axs[i, j].colorbar()
                # cb.set_label('Distance')

                axs[i, j].axis('off')

    plt.show()
    fig.savefig('param_sweep.pdf', bbox_inches='tight')

def plot_loss(folder, save=False):
    if not os.path.isfile(folder+"/rep_loss.npy"):
        print(folder+"/rep_loss.npy", "doesn't exist")
        return
    loss = np.load(folder+"/rep_loss.npy")
    fig, axs = plt.subplots(nrows=loss.shape[1], sharex=True)
    for i in range(loss.shape[1]):
        axs[i].plot(loss[:, i])

    if save:
        slash_idx = list(folder).index("/")
        new_folder = folder[:slash_idx] + "_" + folder[slash_idx + 1:]
        fig.savefig('../'+str(new_folder) + '_loss.png', dpi=fig.dpi)
        plt.clf()
        plt.cla()
        plt.close()
    else:
        plt.show()

def check_log(f):

    all_files = os.listdir(f)

    given = []
    predictions = []
    actions = []
    for record in all_files:
        if record[-4:] == ".out":
            print("processing:", record)
            name = f + record
            log = open(name, "r")
            walk = []
            plan = []
            for l in log:
                lst = [i.split("]") for i in l.strip("\n").split("[")]
                if lst[0][0] == "illegal prediction ":
                    # print(lst)
                    s_temp = lst[1][0].split(" ")
                    s = []
                    for i in s_temp:
                        if i != "":
                            s.append(i)
                    sx, sy = float(s[0]), float(s[1])
                    a = int(lst[1][1].strip(" ").strip("->"))
                    sp_temp = lst[2][0].split(" ")
                    sp = []
                    for i in sp_temp:
                        if i != "":
                            sp.append(i)
                    spx, spy = float(sp[0]), float(sp[1])
                    given.append([sx, sy])
                    actions.append(a)
                    predictions.append([spx, spy])
            log.close()

    given = np.array(given)
    predictions = np.array(predictions)
    # actions = np.array(actions)
    plt.figure()
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    plt.scatter(predictions[:, 0], predictions[:, 1], s=2)
    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)

    plt.figure()
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    plt.scatter(given[:, 0], given[:, 1], s=2, color="orange")
    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)
    plt.show()

def feature_construction(iht, state, action):
    state = np.clip(np.array(state), 0.0, 1.0)
    ind = np.array(tc.tiles(iht, 32, float(4) * np.array(state)))
    return ind

def pw_collect(pref, beta):
    num_run = 5
    num_ep = 200
    return_rec = np.zeros((num_run, num_ep))
    not_exist = []
    for run in range(num_run):
        accum_r = pref + "_run" + str(run) + ".npy"
        step_per_ep = pref + "_run" + str(run) + "_stepPerEp.npy"

        if os.path.isfile(accum_r):
            reward_record = np.load(accum_r)
            step_record = np.load(step_per_ep)
            # print(step_record)
            total_s = -1
            last_ep = 0
            r_ep = []
            for s_idx in range(len(step_record)):
                s = step_record[s_idx]
                total_s += s
                if last_ep != 0:
                    r_ep.append(reward_record[total_s] - reward_record[last_ep])
                    # return_rec[run, s_idx] = reward_record[total_s] - reward_record[last_ep]
                else:
                    r_ep.append(reward_record[total_s])
                    # return_rec[run, s_idx] = reward_record[total_s]
                last_ep = total_s
            r_ep = exponential_smooth(r_ep, beta=beta)
            length = len(r_ep)
            return_rec[run, :length] = r_ep
        else:
            not_exist.append(accum_r)

    cut = 0
    stop = False
    while cut < num_ep and not stop:
        k_ep = return_rec[:, cut]
        count = len(np.where(k_ep != 0)[0])
        if count >= num_run/2:
            cut += 1
        else:
            stop = True
    print("Not exist: \n", not_exist, "\n")
    return return_rec, cut

def pw_show_result():
    not_exist = []

    offline = 1
    for folder_label in ["rem", "llm"]:
        # limit = -0.015 * (1.0 / fix_cov)
        alg_list = ["random"]#, "random_pred", "pri_pred"]#["random_BufferOnly"]  #
        for alg in alg_list:

            fig = plt.figure(figsize=(8.0, 5.0))
            plt.ylim(-500, 10)
            plt.xlim(0, 200)

            for mode in [17]:

                for num_near in [8]: #[1]:
                    for num_planning in [10]:
                        if mode == 0:
                            if folder_label == "llm":
                                limit_list = [-0.12]  # [-0.025]#[-0.1]#[-4.0]#
                            elif folder_label == "rem":
                                limit_list = [-0.7]#[-0.025]#[-0.1]#[-4.0]#
                        elif mode == 17:
                            if folder_label == "llm":
                                limit_list = [-0.05]
                            elif folder_label == "rem":
                                limit_list = [-120.0]
                        elif mode == 21:
                            limit_list = [0.0]

                        for limit in limit_list:
                            if folder_label == "llm" or folder_label == "er":
                                kscale_list = [1.0]
                            elif folder_label == "rem":
                                kscale_list = [1e-07]
                            for kscale in kscale_list:#[1.0]:#[1e-07]: #

                                for similarity in [0]:
                                    for sample_limit in [0.0]: #[0.0, 0.5, 0.75, 0.9, 0.95]:

                                        for add_prot in [1]:

                                            if mode in [0, 11, 12, 13, 18]:
                                                if folder_label == "llm":
                                                    fix_cov_list = [0.025]#[0.001]#[0.025]
                                                elif folder_label == "rem":
                                                    fix_cov_list = [0.0]#[0.001]#[0.025]
                                                else:
                                                    fix_cov_list = [0.025]

                                            elif mode in [0, 1, 2, 10, 14, 15, 16, 17, 19, 20]:
                                                if folder_label == "llm":
                                                    fix_cov_list = [0.025]#[0.001]#[0.025]
                                                elif folder_label == "rem":
                                                    fix_cov_list = [0.0]#[0.001]#[0.025]
                                                else:
                                                    fix_cov_list = [0.025]

                                            elif mode in [21]:
                                                fix_cov_list = [0.0]
                                            else:
                                                print("UNKNOWN MODE FOR FIX_COV")

                                            for fix_cov in fix_cov_list:


                                                temp_limit = limit
                                                temp_kscale = kscale
                                                temp_cov = fix_cov
                                                if alg == "random_BufferOnly":
                                                    alg_label = "ER"
                                                    temp_limit = -1000.0
                                                    temp_kscale = 1.0
                                                    temp_cov = 0.025
                                                elif alg == "random":
                                                    alg_label = "random+forward"
                                                elif alg == "random_pred":
                                                    alg_label = "random+forward&backward"
                                                elif alg == "pri_pred":
                                                    alg_label = "priority+forward&backward"

                                                if mode in [2, 10, 11, 16]:
                                                    opt_mode_list = [1]
                                                elif mode in [0, 1, 12, 13, 14, 15, 17, 18, 19, 20, 21]:
                                                    opt_mode_list = [4]
                                                else:
                                                    print("UNKNOWN MODE FOR OPTIMIZER")

                                                for opt_mode in opt_mode_list:

                                                    for lambda_ in [0.0]:

                                                        if opt_mode in [0]:
                                                            momentum_list = [0.9, 0.99]
                                                        elif opt_mode in [4, 1]:
                                                            momentum_list = [0.0]

                                                        for momentum in momentum_list:

                                                            if opt_mode in [4]:
                                                                rms_list = [0.0]
                                                            elif opt_mode in [0, 1]:
                                                                rms_list = [0.999]

                                                            for rms in rms_list:

                                                                if opt_mode in [4]:
                                                                    alpha_list = [0.004, 0.002, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 4e-06, 8e-06]#[0.03125, 0.0625, 0.125, 0.25, 0.3]#
                                                                elif opt_mode in [0, 1]:
                                                                    alpha_list = [0.01, 0.001, 0.0005]

                                                                for alpha in alpha_list:

                                                                    # for pri_thr in [0.0]:
                                                                    # for buffer_size in [1000]:
                                                                    for sync in [1]:

                                                                        # estimate_max_step = 100000#30000 #300000
                                                                        new_data = []#np.zeros((0, estimate_max_step))
                                                                        path_name = "er" if alg == "random_BufferOnly" else folder_label
                                                                        pref = "exp_result/AE_PW_tp_"+str(path_name).upper()+"/"+\
                                                                               "REM_Dyna_mode" + str(mode) + "AE" + \
                                                                               "_offline" + str(offline) + \
                                                                               "_planning" + str(num_planning) + \
                                                                               "_priThrshd" + str(0.0) + \
                                                                                "_DQNc" + str(sync) + \
                                                                                "_buffer" + str(1000) + \
                                                                                "/always_add_prot_1/"
                                                                        pref += "/"+str(alg)+\
                                                                                "_alpha" + str(alpha) + \
                                                                                "_divAF" + str(mode) + \
                                                                                "_near" + str(num_near) + \
                                                                                "_protLimit" + str(temp_limit) + \
                                                                                "_similarity" + str(similarity) + \
                                                                                "_sampleLimit" + str(sample_limit) + \
                                                                                "_kscale" + str(temp_kscale) + \
                                                                                "_fixCov" + str(temp_cov) + \
                                                                                "_updateQ" + \
                                                                                "_lambda" + str(lambda_) + \
                                                                                "_momentum" + str(momentum) + \
                                                                                "_rms" + str(rms) + \
                                                                                "_optMode" + str(opt_mode)
                                                                        all_run, cut = pw_collect(pref, beta=0.1)
                                                                        all_run = all_run * 40
                                                                        mean = np.zeros(cut)
                                                                        upper = np.zeros(cut)
                                                                        lower = np.zeros(cut)
                                                                        for ep in range(cut):
                                                                            kth_ep = all_run[:, ep]
                                                                            non0idx = np.where(kth_ep != 0)[0]
                                                                            learning = kth_ep[non0idx]
                                                                            mean[ep] = np.mean(learning)
                                                                            ste = np.abs(
                                                                                np.std(learning, axis=0)) / np.sqrt(
                                                                                len(learning))
                                                                            upper[ep] = mean[ep] + ste
                                                                            lower[ep] = mean[ep] - ste
                                                                        x = np.linspace(0, len(mean), len(mean))
                                                                        color_idx = np.where(np.array(alpha_list) == alpha)[0][0]
                                                                        if mode == 0:
                                                                            plt.plot(x, mean, "--",
                                                                                     label="raw - alpha=" + str(alpha),
                                                                                     # color=color_list[color_idx % len(color_list)]
                                                                                     )
                                                                        elif mode == 17:
                                                                            plt.plot(x, mean,
                                                                                     label="rep - alpha=" + str(alpha),
                                                                                     # color=color_list[color_idx % len(color_list)]
                                                                                     )
                                                                        # plt.errorbar(x, md, yerr=std)
                                                                        # plt.plot(x, upper, color=facecolor_list[color_idx % len(color_list)])
                                                                        # plt.plot(x, lower, color=facecolor_list[color_idx % len(color_list)])
                                                                        plt.fill_between(x, upper, lower,
                                                                                         # facecolor=facecolor_list[color_idx % len(color_list)],
                                                                                         alpha=0.3)
                                                                        plt.title(str(alg_label)+" - "+str(folder_label)+"-"+str(num_planning))
                                                                        plt.legend()

                                                                        # for run in range(0, 5):
                                                                        #     accum_r = pref + "_run" + str(run) + ".npy"
                                                                        #     step_per_ep = pref + "_run" + str(run) + "_stepPerEp.npy"
                                                                        #     if os.path.isfile(accum_r):
                                                                        #         reward_record = np.load(accum_r)
                                                                        #         step_record = np.load(step_per_ep)
                                                                        #         # print(step_record)
                                                                        #         total_s = -1
                                                                        #         last_ep = 0
                                                                        #         r_ep = []
                                                                        #         for s in step_record:
                                                                        #             total_s += s
                                                                        #             if last_ep != 0:
                                                                        #                 r_ep.append(reward_record[total_s] - reward_record[last_ep])
                                                                        #             else:
                                                                        #                 r_ep.append(reward_record[total_s])
                                                                        #             last_ep = total_s
                                                                        #         # if last_ep != len(reward_record):
                                                                        #         #     r_ep.append(reward_record[-1] - reward_record[last_ep])
                                                                        #         #     print(r_ep)
                                                                        #         # print(len(reward_record))
                                                                        #         # print()
                                                                        #         # r_ep = np.concatenate((np.array(r_ep), np.zeros(estimate_max_step - len(r_ep))))
                                                                        #         # new_data = np.concatenate((new_data, r_ep), axis=0)
                                                                        #
                                                                        #         r_ep = exponential_smooth(r_ep, beta=0.1)
                                                                        #
                                                                        #         new_data.append(r_ep)
                                                                        #     else:
                                                                        #         not_exist.append(accum_r)
                                                                        #         # exit()
                                                                        #
                                                                        # if len(new_data) != 0:
                                                                        #     # with open(pref + "_rewardPerEp_"+str(len(new_data))+".pkl", "wb") as f:
                                                                        #     #     pkl.dump(new_data, f)
                                                                        #     # print("file saved:", pref)
                                                                        #
                                                                        #     d = new_data
                                                                        #     cut = 100000
                                                                        #     for ep in d:
                                                                        #         length = len(ep)
                                                                        #         cut = length if length < cut else cut
                                                                        #     ready_plot = np.zeros((len(d), cut))
                                                                        #     for idx in range(len(d)):
                                                                        #         ready_plot[idx] = d[idx][:cut]
                                                                        #     learning = ready_plot * 40                                              # if mode == 0 else ready_plot * 40
                                                                        #     md = np.mean(learning, axis=0)
                                                                        #     print("total runs", len(learning), "length", len(md))
                                                                        #     lr_idx = np.where(np.array(alpha_list) == alpha)[0][0]
                                                                        #     if mode == 0:
                                                                        #         plt.plot(md, "--", label="raw - alpha="+str(alpha),
                                                                        #                  color=color_list[lr_idx % len(color_list)]
                                                                        #                  )
                                                                        #     elif mode == 17:
                                                                        #         plt.plot(md, label="rep - alpha="+str(alpha),
                                                                        #                  color=color_list[lr_idx % len(color_list)]
                                                                        #                  )
                                                                        #
                                                                        #     ste = np.abs(
                                                                        #         np.std(learning, axis=0)) / np.sqrt(
                                                                        #         len(learning))
                                                                        #     upper = md + ste * 1
                                                                        #     lower = md - ste * 1
                                                                        #     x = np.linspace(0, len(md) - 1, len(md))
                                                                        #     color_idx = \
                                                                        #     np.where(np.array(alpha_list) == alpha)[0][
                                                                        #         0]
                                                                        #
                                                                        #     # if mode == 0:
                                                                        #     #     plt.plot(x, md, "--",
                                                                        #     #              label=alg_label + "_raw - alpha=" + str(
                                                                        #     #                  alpha),
                                                                        #     #              color=color_list[
                                                                        #     #                  color_idx % len(
                                                                        #     #                      color_list)]
                                                                        #     #              )
                                                                        #     # elif mode == 17:
                                                                        #     #     plt.plot(x, md,
                                                                        #     #              label=alg_label + "_rep - alpha=" + str(
                                                                        #     #                  alpha),
                                                                        #     #              color=color_list[
                                                                        #     #                  color_idx % len(
                                                                        #     #                      color_list)]
                                                                        #     #              )
                                                                        #
                                                                        #     # if alg == "random":
                                                                        #     #     plt.plot(x, md, "--",
                                                                        #     #              label=alg_label + "_" + path_name + "_alpha=" + str(
                                                                        #     #                  alpha),
                                                                        #     #              color=color_list[
                                                                        #     #                  color_idx % len(
                                                                        #     #                      color_list)]
                                                                        #     #              )
                                                                        #     # elif alg == "random_pred":
                                                                        #     #     plt.plot(x, md, "-.",
                                                                        #     #              label=alg_label + "_" + path_name + "_alpha=" + str(
                                                                        #     #                  alpha),
                                                                        #     #              color=color_list[
                                                                        #     #                  color_idx % len(
                                                                        #     #                      color_list)]
                                                                        #     #              )
                                                                        #     # elif alg == "pri_pred":
                                                                        #     #     plt.plot(x, md, "-",
                                                                        #     #              label=alg_label + "_" + path_name + "_alpha=" + str(
                                                                        #     #                  alpha),
                                                                        #     #              color=color_list[
                                                                        #     #                  color_idx % len(
                                                                        #     #                      color_list)]
                                                                        #     #              )
                                                                        #     # elif alg == "random_BufferOnly":
                                                                        #     #     plt.plot(x, md, ":",
                                                                        #     #              label=alg_label + "_" + path_name + "_alpha=" + str(
                                                                        #     #                  alpha),
                                                                        #     #              color=color_list[
                                                                        #     #                  color_idx % len(
                                                                        #     #                      color_list)]
                                                                        #     #              )
                                                                        #     # else:
                                                                        #     #     print("wrong label", alg_label)
                                                                        #     #     exit()
                                                                        #
                                                                        #     plt.fill_between(x, upper, lower,
                                                                        #                      facecolor=facecolor_list[
                                                                        #                          color_idx % len(
                                                                        #                              color_list)],
                                                                        #                      # facecolor=color_list[color_idx % len(color_list)],
                                                                        #                      alpha=0.3)
                                                                        #     plt.title(str(alg_label)+" - "+str(folder_label)+"-"+str(num_planning))
                                                                        #     plt.legend()
                                                                        #
                                                                        # else:
                                                                        #     print("Not found", pref)
            # fig.savefig("../" + str(alg) + "-" + path_name + '.png')

    plt.show()
    # print("\nNot exist:")
    # for f in not_exist:
    #     print(f)

def check_ground_truth_pw():
    # data = np.load("random_data/fixed_env_suc_prob_1.0/pw_noGoal_separateTC32x4_reward_training_set_randomStart_0opt_[0.998, 0.8]gamma_1pts_x1_x100000.npy")
    # reward = data[:, -1]
    # state = data[:, :2]
    # step = 0.005
    # r_sum = np.zeros((int(1.0 / step), int(1.0 / step)))
    # r_count = np.zeros((int(1.0 / step), int(1.0 / step)))
    # visit = np.zeros((int(1.0 / step), int(1.0 / step)))
    # for idx in range(len(data)):
    #     sx, sy = state[idx]
    #     r = reward[idx]
    #     px = np.clip(int(sx / step), 0, int(1.0/step) - 1)
    #     py = np.clip(int(1.0/step) - int(sy / step), 0, int(1.0/step) - 1)
    #     r_sum[px, py] += r
    #     r_count[px, py] += 1
    #     visit[px, py] = 1
    # r_sum /= r_count

    import environment.PuddleWorld as pw
    env = pw.puddleworld()
    # print(env._reward(0.45, 0.5, False))
    # exit()
    step = 0.01
    r_sum = np.zeros((int(1.0 / step), int(1.0 / step)))
    for px in range(int(1.0/step)):
        for py in range(int(1.0/step)):
            sx = px * step + step / 2.0
            sy = py * step + step / 2.0
            r = env._reward(sx, sy, bool((sx >= 0.95) and (sy >= 0.95)))
            r_sum[px, int(1.0/step) -1 - py] = r
            if r < -1:
                print(sx, sy, r)

    plt.figure()
    plt.imshow(r_sum.T, cmap='Blues')
    plt.colorbar()

    # plt.figure()
    # plt.imshow(visit.T, cmap='Blues')
    # plt.colorbar()
    plt.show()

def catcher_show_result():
    num_run = 5
    fx = 0
    # color = ['b', 'p', 'r', 'g', 'orange']
    not_exist = []
    for offline in [1]:#[0]:#
        for num_near in [8]: #[1]:
            for num_planning in [1]:

                for mode in [0, 17]:
                    plt.figure(fx)
                    fx += 1
                    plt.ylim(-1, 1)
                    if mode == 0:
                        limit_list = [-1000.0]#[-0.1]#[-0.025]#[-4.0]#
                    elif mode == 17:
                        limit_list = [-1000.0]#[-400.0]#[-20.0]#[-3.0]#
                    elif mode == 21:
                        limit_list = [0.0]

                    for limit in limit_list:
                        for kscale in [1.0]:#[1e-05]:#[1.0]: #
                            for similarity in [0]:
                                for sample_limit in [0.0]:#[0.0, 0.5, 0.75]:#[0.0, 0.5, 0.75, 0.9, 0.95]:

                                    for add_prot in [1]:

                                        if mode in [0, 11, 12, 13, 18]:
                                            fix_cov_list = [0.025]#[0.025]#[0.001]#
                                        elif mode in [0, 1, 2, 10, 14, 15, 16, 17, 19, 20]:
                                            fix_cov_list = [0.025]#[0.001]#[0.025]
                                        elif mode in [21]:
                                            fix_cov_list = [0.0]
                                        else:
                                            print("UNKNOWN MODE FOR FIX_COV")

                                        for fix_cov in fix_cov_list:
                                            # for pri_thr in [0.0]:
                                            # for buffer_size in [1000]:
                                            for sync in [1]:


                                                if mode in [2, 10, 11, 16]:
                                                    opt_mode_list = [1]
                                                elif mode in [0, 1, 12, 13, 14, 15, 17, 18, 19, 20, 21]:
                                                    opt_mode_list = [4]
                                                else:
                                                    print("UNKNOWN MODE FOR OPTIMIZER")

                                                for opt_mode in opt_mode_list:
                                                    for alg in ["Q"]:
                                                        for lambda_ in [0.0]:

                                                            if opt_mode in [0]:
                                                                momentum_list = [0.9, 0.99]
                                                            elif opt_mode in [4, 1]:
                                                                momentum_list = [0.0]

                                                            for momentum in momentum_list:

                                                                if opt_mode in [4]:
                                                                    rms_list = [0.0]
                                                                elif opt_mode in [0, 1]:
                                                                    rms_list = [0.999]

                                                                for rms in rms_list:
                                                                    if opt_mode in [4]:
                                                                        alpha_list = [0.004, 0.002, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 8e-06, 4e-06]#[0.03125, 0.0625, 0.125, 0.25, 0.3]##[0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]#[0.03125, 0.0625, 0.125, 0.25, 0.3]#[0.00097656, 0.00195312, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
                                                                    elif opt_mode in [0, 1]:
                                                                        alpha_list = [0.01, 0.001, 0.0005]

                                                                    for lr_idx in range(len(alpha_list)):

                                                                        alpha = alpha_list[lr_idx]

                                                                        new_data = []
                                                                        pref = "exp_result/catcher/100x50_stoc_legal/" \
                                                                               "REM_Dyna_mode" + str(mode) + \
                                                                               "_offline" + str(offline) + \
                                                                               "_planning" + str(num_planning) + \
                                                                               "_priThrshd" + str(0.0) + \
                                                                               "_DQNc" + str(sync) + \
                                                                               "_buffer" + str(1000) + \
                                                                               "/always_add_prot_1/"
                                                                        pref += "/random_BufferOnly_alpha" + str(alpha) + \
                                                                                "_divAF" + str(mode) + \
                                                                                "_near" + str(num_near) + \
                                                                                "_protLimit" + str(limit) + \
                                                                                "_similarity" + str(similarity) + \
                                                                                "_sampleLimit" + str(sample_limit) + \
                                                                                "_kscale" + str(kscale) + \
                                                                                "_fixCov" + str(fix_cov) + \
                                                                                "_update" + str(alg) + \
                                                                                "_lambda" + str(lambda_) + \
                                                                                "_momentum" + str(momentum) + \
                                                                                "_rms" + str(rms) + \
                                                                                "_optMode" + str(opt_mode)

                                                                        for run in range(num_run-5, num_run+1):
                                                                            accum_r = pref + "_run" + str(
                                                                                run) + ".npy"
                                                                            step_per_ep = pref + "_run" + str(
                                                                                run) + "_stepPerEp.npy"
                                                                            if os.path.isfile(accum_r):
                                                                                reward_record = np.load(accum_r)
                                                                                step_record = np.load(step_per_ep)
                                                                                total_s = -1
                                                                                last_ep = 0
                                                                                r_ep = []
                                                                                for s in step_record:
                                                                                    total_s += s
                                                                                    if last_ep != 0:
                                                                                        r_ep.append(
                                                                                            reward_record[total_s] -
                                                                                            reward_record[last_ep])
                                                                                    else:
                                                                                        r_ep.append(
                                                                                            reward_record[total_s])
                                                                                    last_ep = total_s

                                                                                r_ep = exponential_smooth(r_ep,beta=0.01)

                                                                                new_data.append(r_ep)
                                                                            else:
                                                                                not_exist.append(accum_r)

                                                                        if len(new_data) != 0:
                                                                            d = new_data
                                                                            cut = 100000
                                                                            for ep in d:
                                                                                length = len(ep)
                                                                                cut = length if length < cut else cut
                                                                            ready_plot = np.zeros((len(d), cut))
                                                                            for idx in range(len(d)):
                                                                                ready_plot[idx] = d[idx][:cut]
                                                                            learning = ready_plot[:, :]
                                                                            md = np.mean(learning, axis=0)
                                                                            lr_idx = np.where(np.array(alpha_list) == alpha)[0][0]
                                                                            if mode == 0:
                                                                                plt.plot(md, "--",
                                                                                         label="raw - alpha=" + str(
                                                                                             alpha),
                                                                                         color=color_list[
                                                                                             lr_idx % len(
                                                                                                 color_list)]
                                                                                         )
                                                                            elif mode == 17:
                                                                                plt.plot(md,
                                                                                         label="rep - alpha=" + str(
                                                                                             alpha),
                                                                                         color=color_list[
                                                                                             lr_idx % len(
                                                                                                 color_list)]
                                                                                         )

                                                                            ste = np.abs(np.std(learning, axis=0)) / np.sqrt(len(learning))
                                                                            upper = md + ste * 1
                                                                            lower = md - ste * 1
                                                                            x = np.linspace(0, len(md) - 1, len(md))
                                                                            color_idx = np.where(np.array(alpha_list) == alpha)[0][0]

                                                                            plt.fill_between(x, upper, lower,
                                                                                             facecolor=
                                                                                             facecolor_list[
                                                                                                 color_idx % len(
                                                                                                     color_list)],
                                                                                             alpha=0.3)
                                                                            plt.title("catcher "+str(mode)+" "+str(num_planning))
                                                                            plt.legend()

                                                                        else:
                                                                            print("Not found", pref)

    plt.show()
    print("Shown")

def plot_best_performance():
    num_run = 5
    not_exist = []
    #             path, mode, lr, planning
    pref_list = [["Q",  0,  0.002, 0],
                 ["er", 0,  0.002, 1],
                 ["er", 17, 0.002, 1]
                 ]

    plt.figure()
    for pref_idx in range(len(pref_list)):
        param = pref_list[pref_idx]
        pref = "exp_result/catcher/rep32/"+str(param[0])+"/REM_Dyna_mode" + str(param[1]) + \
                    "_offline1_planning"+str(param[3])+"_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" + \
                    "random_BufferOnly_alpha" + str(param[2]) + \
                    "_divAF" + str(param[1]) + \
                    "_near8_protLimit-1000.0_similarity0_sampleLimit0.0_kscale1.0_fixCov0.025_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4"
        new_data = []
        for run in range(num_run - 5, num_run + 1):
            accum_r = pref + "_run" + str(run) + ".npy"
            step_per_ep = pref + "_run" + str(run) + "_stepPerEp.npy"
            if os.path.isfile(accum_r):
                reward_record = np.load(accum_r)
                step_record = np.load(step_per_ep)
                total_s = -1
                last_ep = 0
                r_ep = []
                for s in step_record:
                    total_s += s
                    if last_ep != 0:
                        r_ep.append(reward_record[total_s] - reward_record[last_ep])
                    else:
                        r_ep.append(reward_record[total_s])
                    last_ep = total_s

                r_ep = exponential_smooth(r_ep, beta=0.01)

                new_data.append(r_ep)
            else:
                not_exist.append(accum_r)
        if len(new_data) != 0:
            d = new_data
            cut = 100000
            for ep in d:
                length = len(ep)
                cut = length if length < cut else cut
            ready_plot = np.zeros((len(d), cut))
            for idx in range(len(d)):
                ready_plot[idx] = d[idx][:cut]
            learning = ready_plot[:, :]
            md = np.mean(learning, axis=0)
            inputNN = "raw" if param[1] == 0 else "rep"
            plt.plot(md, label=str(param[0])+" "+str(inputNN)+" lr="+str(param[2]))
            ste = np.abs(np.std(learning, axis=0)) / np.sqrt(len(learning))
            upper = md + ste * 1
            lower = md - ste * 1
            x = np.linspace(0, len(md) - 1, len(md))
            plt.fill_between(x, upper, lower, alpha=0.3)
            plt.legend()

        else:
            print("Not found", pref)
    plt.show()


def paper_bw_plot():
    num_run = 100
    not_exist = []
    #             path, mode, lr, planning
    pref_list = [["Q",  0,  3e-05, 0,  -1000.0],
                 ["Q",  17, 3e-05, 0,  -1000.0],
                 ["REM", 0,  3e-05, 1, -0.4],
                 ["REM", 17,  3e-05, 1,-130.0],
                 ["LLM", 0, 3e-05, 1,  -0.05],
                 ["LLM", 17, 3e-05, 1, -6.0]
                 ]

    plt.figure()
    plt.tight_layout()
    for pref_idx in range(len(pref_list)):
        param = pref_list[pref_idx]
        alg = "random_BufferOnly" if param[0] == "Q" or param[0] == "ER" else "random"
        kscale = 1e-07 if param[0] == "REM" else 1.0
        fixCov = 0.0 if param[0] == "REM" else 0.025
        pref = "exp_result/paper/BW_"+str(param[0])+"/REM_Dyna_mode" + str(param[1]) + \
                    "_offline1_planning"+str(param[3])+"_priThrshd0.0_DQNc1_buffer1000/always_add_prot_1/" + \
                    str(alg) + "_alpha" + str(param[2]) + \
                    "_divAF" + str(param[1]) + \
                    "_near8_protLimit"+str(param[4])+"_similarity0_sampleLimit0.0_kscale"+str(kscale)+"_fixCov"+str(fixCov)+"_updateQ_lambda0.0_momentum0.0_rms0.0_optMode4_300000x100.npy"
        d = np.load(pref)
        learning = []
        for l in d:
            start = 0
            while start < len(l):
                if l[start] != 0:
                    break
                else:
                    start += 1
            print("this run starts from", start)
            # learning.append(l[start: start + 9200])
            learning.append(l[start: start + 100000])
        learning = np.array(learning)
        md = np.mean(learning, axis=0)
        inputNN = "raw" if param[1] == 0 else "rep"
        plt.plot(md, label=str(param[0]) + " " + str(inputNN) + " lr=" + str(param[2]))
        ste = np.abs(np.std(learning, axis=0)) / np.sqrt(len(learning))
        upper = md + ste * 1
        lower = md - ste * 1
        x = np.linspace(0, len(md) - 1, len(md))
        plt.fill_between(x, upper, lower, alpha=0.3)
        # plt.legend()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',mode="expand",  borderaxespad=0.)
    plt.show()


# check_weight()
# check_reward()

# check_weight()

# plot_nn_result()
# save_figures()
#remove_data()
# f = "temp_discrete/continuous_input[0.0, 1]_envSucProb1.0_legalv0_gamma[0.998, 0.8]_epoch1000_nfeature16_beta0.1/"
# plot_nn_dist_dgw(folder=f, save=False, min_clip_at=0, max_clip_at=None, switch_y=False)
# plot_recovered_state_dgw(folder=f, save=True)
# check_ground_truth_dgw()
# param_sweep_plot_nn_dist()

# for ir in [0.0]:
#     for legalv in [0]:
#         wall = False#True if legalv == 1 else False
#         for epoch in [1000]:
#             for nf in [32]:
#                 for beta in [0.01, 0.1, 1.0]:
#                     for delta in [0.01, 0.1, 1.0]:
#                         f = "temp_continuous/" + "continuous_input[" + str(ir) + \
#                             ", 1]_envSucProb1.0_legalv"+str(legalv)+"_gamma[0.998, 0.8]_epoch"+str(epoch)+"_nfeature" + str(nf) + \
#                             "_beta" + str(beta) + "_delta" + str(delta)
#                         # plot_nn_dist_cgw(folder=f, save=True, min_clip_at=0, max_clip_at=None, normalize=True, wall=wall)
#                         plot_loss(f, save=True)

f = "temp_continuous/continuous_input[0.0, 1]_envSucProb1.0_legalv1_gamma[0.998, 0.8]_epoch1000_nfeature32_beta1.0_delta0.1"
# f = "temp/"
# plot_nn_dist_cgw(folder=f, save=False, min_clip_at=None, max_clip_at=None, normalize=True, wall=False)
# plot_loss(f, save=False)

# read_img_file(betas=[0.01, 0.1, 1.0],
#               deltas=[0.01, 0.1, 1.0],
#               epoch=1000,
#               legal=1)

# plot_recovered_state_cgw(folder=f, save=False)
# check_ground_truth_cgw_xy()
# check_ground_truth_cgw_tc()
# check_ground_truth_pw()

# combine_data()
# compare_result_old()
# compare_result_new()
# cgw_training_data()

pw_show_result()
# catcher_show_result()

# visit_time()
# plot_best_performance()
folder = "exp_result/nonLinearQ_baseline/minibatch1/REM_Dyna_mode17_offline1_planning10_priThrshd0.0_DQNc1_buffer1000/"
# check_log(folder)

# paper_bw_plot()