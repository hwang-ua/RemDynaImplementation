""""
This is the experiment file for REM Dyna project
"""

import numpy as np
import sys
from utils.distance_matrix_func import *
from rl_glue import *  # Required for RL-Glue


def go_in_wall(x, y, wall_x, wall_w, hole_yl, hole_yh):
    if x > wall_x and x < (wall_x + wall_w):
        if y < hole_yl or y > hole_yh:
            return True
    else:
        return False

def add_one_gamma():
    suc_prob = 1.0
    data = np.load("random_data/graph_env_suc_prob_" + str(suc_prob) + "/cgw_noGoal_oneHot4x4_training_set_randomStart_0opt_0.998gamma_1pts_x1.npy")
    print(data.shape)
    one_ep = data[:, :2]
    gamma = 0.8
    num_tiling = 1
    num_tile = 16
    state, feature, ground_truth = preproc_cgw_data(np.array(one_ep), gamma, True, num_tile=num_tile, num_tiling=num_tiling)
    print(feature.shape, data[:, 2: 2 + num_tile*num_tiling*2].shape)
    # assert equal_array(feature, data[:, 2: 2 + num_tile**2*num_tiling])
    data = np.concatenate((data, ground_truth), axis=1)
    print(data.shape)
    np.save("random_data/graph_env_suc_prob_"+str(suc_prob)+"/cgw_noGoal_oneHot4x4_training_set_randomStart_0opt_[0.998, "+str(gamma)+"]gamma_1pts_x1.npy", data)

def remove_one_gamma():
    suc_prob = 1.0
    num_tiling = 1
    num_tile = 16

    data = np.load("random_data/graph_env_suc_prob_"+str(suc_prob)+"/cgw_noGoal_oneHot4x4_training_set_randomStart_0opt_[0.998, 0.8]gamma_1pts_x1")
    pts = data[:, : 2+num_tiling*num_tile*2]
    print("shape of pts", pts.shape)
    ground_truth = data[:, 2+num_tiling*num_tile*2*2 :]
    print("shape of gt", ground_truth.shape)
    data = np.concatenate((pts, ground_truth), axis=1)
    print(data.shape)
    np.save("random_data/graph_env_suc_prob_"+str(suc_prob)+"/cgw_noGoal_oneHot4x4_training_set_randomStart_0opt_0.8gamma_1pts_x1.npy", data)


def collect_data():
    """
    Start from every state
    """
    sys.path.append('./environment/')
    sys.path.append('./agent/')
    import json

    # jsonfile = "parameters/continuous_gridworld.json"
    jsonfile = "parameters/gridworldgraph.json"

    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    RLGlue(exp['environment'], exp['agent'])
    print("Env::", exp["environment"], ", Param:", exp["env_params"])

    env_params = {}
    if "env_params" in exp:
        env_params = exp['env_params']
    if "agent_params" in exp:
        agent_params = exp['agent_params']
    if "exp_params" in exp:
        exp_params = exp['exp_params']

    gamma = exp["agent_params"]["nn_gamma"]
    opt_prob = exp["agent_params"]["opt_prob"]
    wall_x = env_params["wall_x"]
    wall_w = env_params["wall_w"]
    hole_yl = env_params["hole_yl"]
    hole_yh = env_params["hole_yh"]

    num_tiling = 1 #agent_params["nn_num_tilings"]
    num_tile = agent_params["nn_num_tiles"]

    # path = "./random_data/"
    # file_name = "gw_random_walk_"

    if len(sys.argv) == 4:
        num_ep = int(sys.argv[1])
        num_point = int(sys.argv[2])
        file_order = sys.argv[3]
    else:
        num_ep = 1
        num_point = 1

    max_step = 50000

    # start from random point

    order_xy = np.random.random(size = num_point * 2).reshape((-1, 2))

    all_eps = np.zeros((0, 2 + num_tile**2 * num_tiling *(2+1)))
    for ind in range(len(order_xy)):
        start_x, start_y = order_xy[ind]
        print("pts =",start_x, start_y, "ind =", ind)
        if not go_in_wall(start_x, start_y, wall_x, wall_w, hole_yl, hole_yh):
            i = 0
            while i < num_ep:
                env_params["start_x"] = start_x
                env_params["start_y"] = start_y

                RL_init()

                np.random.seed(512 * i)

                dim_state = RL_env_message(["state dimension", None])
                agent_params["dim_state"] = dim_state
                num_action = RL_env_message(["num_action", None])
                agent_params["num_action"] = num_action
                RL_agent_message(["set param", agent_params])
                RL_env_message(["set param", env_params])

                one_ep = []
                num_step = 0

                info = RL_start()
                one_ep.append(info["state"])

                end_episode = False
                while not end_episode and num_step < max_step:
                    info = RL_step()
                    one_ep.append(info["state"])
                    end_episode = info["isTerminal"]

                    num_step += 1

                    if num_step % 5000 == 0:
                        print("step", num_step)

                print(">>> Ep ends. Total num of step, ", RL_num_steps())
                RL_end()

                print("1 tiling,", num_tile, "tiles")
                if type(gamma) == list:
                    state, feature, ground_truth = preproc_graph_data_multigamma(np.array(one_ep), gamma,
                                                                                 num_tile=num_tile)
                else:
                    state, feature, ground_truth = preproc_graph_data(np.array(one_ep), gamma,
                                                                      num_tile=num_tile)
                del one_ep
                f_g = np.concatenate((state, feature, ground_truth), axis=1)
                print(feature.shape, ground_truth.shape)
                print(all_eps.shape, f_g.shape)
                all_eps = np.concatenate((all_eps, f_g), axis=0)
                i += 1
        else:
            print(start_x, start_y, "is in wall")

        # save feature and ground truth
        suc_prob = env_params["suc_prob"]
        name = "random_data/graph_env_suc_prob_" + str(suc_prob) + "/cgw_noGoal_oneHot" + str(num_tile) + "x" + str(num_tile) + "_training_set_randomStart_" + \
                   str(int(opt_prob*100))+"opt_"+str(gamma)+"gamma_"+str(num_point)+"pts_x"+str(num_ep)
        np.save(name, np.array(all_eps))

collect_data()
