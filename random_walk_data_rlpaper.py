"""
This is the experiment file for REM Dyna project
"""

import numpy as np
import pickle
import sys
from utils.distance_matrix_func import *


"""
Start from every state
"""
sys.path.append('./environment/')
sys.path.append('./agent/')
import json

# load parameters
if len(sys.argv) == 4:
    alpha = float(sys.argv[1])
    num_near = int(sys.argv[2])
    add_prot_limit = float(sys.argv[3])

jsonfile = "parameters/gridworld.json"
json_dat = open(jsonfile, 'r')
exp = json.load(json_dat)
json_dat.close()

from rl_glue import *  # Required for RL-Glue

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
gamma = exp["agent_params"]["agent_gamma"]
opt_prob = exp["agent_params"]["opt_prob"]

walls = []
for block in env_params["walls"]:
    wx_start, wx_len, wy_start, wy_len = block
    for wx in range(wx_start, wx_start + wx_len):
        for wy in range(wy_start, wy_start + wy_len):
            walls.append([wx, wy])

if len(sys.argv) == 2:
    num_ep = int(sys.argv[1])
else:
    num_ep = 1
max_step = 50000


order_xy = []

# # start from every grid
# for i in range(env_params["size_x"]):
#     for j in range(env_params["size_y"]):
#         for r in range(num_ep):
#             order_xy.append([i, j])
# np.random.shuffle(order_xy)

for e in range(num_ep):
    i = np.random.randint(env_params["size_x"])
    j = np.random.randint(env_params["size_y"])
    order_xy.append([i, j])


all_eps = np.zeros((0, 4))#np.zeros((0, size_x*size_y*2))
each_tj = []
file_order = 0
for ind in range(len(order_xy)):
    start_x, start_y = order_xy[ind]
    print(start_x, start_y, ind)
    if [start_x, start_y] not in walls:
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

        print(one_ep)
        feature_s = one_ep[: -1] # one_hot_feature(one_ep[: -1], size_x, size_y)
        feature_sp = one_ep[1: ] # one_hot_feature(one_ep[1: ], size_x, size_y)
        feature = np.concatenate((feature_s, feature_sp), axis=1)

        all_eps = np.concatenate((all_eps, feature), axis=0)
        each_tj.append(one_ep)
        del one_ep
print(np.sum(all_eps[0]), np.sum(all_eps, axis=1), np.sum(all_eps))


path = "./random_data/fixed_env_suc_prob_1.0/"
file_name = "dgw_random_trajectory_randomPts"+str(num_ep)+"_step"+str(max_step)+"_opt"+str(opt_prob)+"_xyRaw_trajectoryOnly.pkl"

# save feature and ground truth
# np.save("random_data/trajectory_1000randomPts_50step_xyRaw", np.array(all_eps))
# print("File saved in random_data/trajectory_1000randomPts_50step_xyRaw")

with open(path+file_name, "wb") as f:
    pickle.dump(each_tj, f)
print("File saved in", path+file_name)