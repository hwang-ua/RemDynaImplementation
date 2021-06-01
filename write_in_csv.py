import pandas as pd
import numpy as np
import os

def compare_result():
    agent_list = ["qLearning", "priorityER", "randomER", "remDyna_noLearnNoProto_realCov"]
    agent_name = {"qLearning": "Q_learning",
                  "priorityER": "priority_ER",
                  "randomER": "random_ER",
                  "remDyna_noLearnNoProto_realCov": "REM_Dyna_pri_pred"}
    tile_list = ["1x16tile", "4x16tile", "10x10tile"]
    standard = {"1x16tile":180,
                "4x16tile":100,
                "10x10tile":120}

    alpha_list = [0.001, 0.002, 0.004, 0.005, 0.008, 0.01, 0.016, 0.03125, 0.05, 0.0625, 0.1, 0.125, 0.25,
                  0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    near_list = [4, 8, 16, 32]
    limit_list = [0.05, 0.1, 0.25, 0.5]

    result = []
    for agent in agent_list:
        for tile in tile_list:
            for alpha in alpha_list:
                for near in near_list:
                    for limit in limit_list:
                        for run in range(1, 33):
                            file = "exp_result_cgw/"+agent+"/"+tile+"/ContinuousGridWorld_"+agent_name[agent]+"_alpha" + \
                                    str(alpha) + "_near" + str(near) + "_protLimit" + str(limit) + "_10000x" + str(run) + ".npy"
                            if os.path.isfile(file):
                                data = np.load(file)
                                print(file, "max =", np.mean(data[:, :], axis=0)[-1])

                                if agent=="remDyna_noLearnNoProto_realCov":
                                    new_line = [tile, agent_name[agent], alpha, near, limit, run, np.mean(data[:, :], axis=0)[-1]]
                                else:
                                    new_line = [tile, agent_name[agent], alpha, None, None, run, np.mean(data[:, :], axis=0)[-1]]
                                result.append(new_line)

    columns = []
    for c in range(len(result[0])):
        whole_col = []
        for l in range(len(result)):
            whole_col.append(result[l][c])
        columns.append(whole_col)

    columns_order = ['Tile Coding','Agent','Alpha', 'Num of neighbors', 'threshold', 'Avg in x runs', 'maximum cumulated reward']
    df_dict = {}
    for co in range(len(columns_order)):
        df_dict[columns_order[co]] = columns[co]

    dataframe = pd.DataFrame(df_dict)
    dataframe.to_csv("param_sweep.csv", index=False, sep=',',
                     columns=columns_order)

compare_result()
