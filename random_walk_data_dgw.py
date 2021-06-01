""""
This is the experiment file for REM Dyna project
"""

import numpy as np
import sys
from utils.distance_matrix_func import *
import json
from rl_glue import *  # Required for RL-Glue

def check():
    data = np.load("random_data/training_set_noGoal_randomStart_0opt_[0.998, 0.99]gamma_1pts_x1.npy")
    print(data.shape)
    print(data[0, :225].sum(), data[0].sum(), data[:, 225:450].sum(), data[:, 450:].sum())

def add_one_gamma():
    data = np.load("random_data/training_set_noGoal_randomStart_0opt_0.998gamma_1pts_x1.npy")
    one_ep = data[:, :225]
    one_ep = recover_oneHot_set(one_ep, 15, 15)
    gamma = 0.8
    feature, ground_truth = preproc_dgw_data_oneHot(np.array(one_ep), 15, 15, gamma)
    data = np.concatenate((data, ground_truth), axis=1)
    np.save("random_data/training_set_noGoal_randomStart_0opt_[0.998, "+str(gamma)+"]gamma_1pts_x1.npy", data)

def collect_data():
    """
    Start from every state
    """
    sys.path.append('./environment/')
    sys.path.append('./agent/')


    jsonfile = "parameters/gridworld.json"
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

    size_x = exp["env_params"]["size_x"]
    size_y = exp["env_params"]["size_y"]
    gamma = exp["agent_params"]["nn_gamma"]
    opt_prob = exp["agent_params"]["opt_prob"]

    walls = []
    for block in env_params["walls"]:
        wx_start, wx_len, wy_start, wy_len = block
        for wx in range(wx_start, wx_start + wx_len):
            for wy in range(wy_start, wy_start + wy_len):
                walls.append([wx, wy])

    path = "./random_data/"
    file_name = "gw_random_walk_"

    if len(sys.argv) == 2:
        num_ep = int(sys.argv[1])
    else:
        num_ep = 1
    max_step = 500000

    # # start from every grid
    # order_xy = []
    # for i in range(env_params["size_x"]):
    #     for j in range(env_params["size_y"]):
    #         order_xy.append([i, j])
    # np.random.shuffle(order_xy)
    order_xy = [[np.random.randint(15), np.random.randint(15)]]

    if type(gamma) == list:
        all_eps = np.zeros((0, size_x * size_y * (1+len(gamma))))
    else:
        all_eps = np.zeros((0, size_x * size_y * 2))
    file_order = 0
    for ind in range(len(order_xy)):
        start_x, start_y = order_xy[ind]
        print(start_x, start_y)
        if [start_x, start_y] not in walls:
            i = 0
            while i < num_ep:
                env_params["start_x"] = start_x
                env_params["start_y"] = start_y

                RL_init()
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

                    if num_step%5000 == 0:
                        print("step", num_step)

                print(">>> Ep ends. Total num of step, ", RL_num_steps())
                RL_end()

                if type(gamma) == list:
                    feature, ground_truth = preproc_dgw_data_oneHot_multigamma(one_ep, size_x, size_y, gamma)
                else:
                    feature, ground_truth = preproc_dgw_data_oneHot(one_ep, size_x, size_y, gamma)
                del one_ep
                if True: #num_step < max_step:
                    # save feature and ground truth
                    print(feature.shape, ground_truth.shape)
                    f_g = np.concatenate((feature, ground_truth), axis=1)
                    all_eps = np.concatenate((all_eps, f_g), axis=0)
                    # print(all_eps.shape)
                    i += 1

        # save feature and ground truth
        np.save("random_data/fixed_env_suc_prob_1.0/dgw_training_set_noGoal_randomStart_"+str(int(opt_prob*100))+"opt_"+str(gamma)+"gamma_"+str(len(order_xy))+"pts_x"+str(num_ep)+"_x"+str(max_step), all_eps)

# add_one_gamma()
# check()
collect_data()