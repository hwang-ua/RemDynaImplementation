""""
This is the experiment file for REM Dyna project
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import time
import numpy as np
# import matplotlib.pyplot as plt
import os
retval = os.getcwd()
print("current dir:", retval)

import sys
sys.path.append('./environment/')
sys.path.append('./agent/')
# sys.path.append('./utils/')
# sys.path.append('./feature_model/')
import json
import utils.get_learned_representation as glr
import utils.get_learned_state as gls

np.set_printoptions(precision=3)

def fixed_param_run(env_params, agent_params, exp_params, this_run):
    print("Agent::", agent_params)
    print("Exp param::", exp_params)
    print("Exp:: control total number of steps.")

    RL_init()
    if exp_params["random_seed"] != 0:
        np.random.seed(exp_params["random_seed"] * this_run)

    dim_state = RL_env_message(["state dimension", None])
    agent_params["dim_state"] = dim_state
    num_action = RL_env_message(["num_action", None])
    agent_params["num_action"] = num_action
    agent_params["gui"] = 1

    RL_agent_message(["set param", agent_params])
    RL_env_message(["set param", env_params])

    RL_start()
    end_episode = False
    print_info = False
    c = 1
    while not end_episode:
        end_episode = RL_step()["isTerminal"]
        c += 1
        if c % 100 == 0:
            print("1st ep step", c)
    RL_end()

    chain = []
    tde = []
    for step in range(100):
        if end_episode:
            RL_end()
            print_info = True
            # input()
            info = RL_start()
        info = RL_step()
        end_episode = info["isTerminal"]

        if print_info and "other_info" in info and "plan" in info["other_info"]:
            plan_all = info["other_info"]["plan"]
            for plan in plan_all:
                for ssp in plan["sbab_list"]:
                    s, a, sp, r, g, pri, s_value, sp_value = ssp
                    trace = state_in_record(sp, chain)
                    if trace != None:
                        # tde[trace].append([sp, a, s, r, g, pri, s_value[a], sp_value[chain[trace][-2]]])
                        tde[trace].append([sp, a, s, r, g, pri, s_value[a], np.max(sp_value)])
                        chain[trace].append(a)
                        chain[trace].append(s)
                    else:
                        if state_in_wall(s):
                            # print(s, "in the wall")
                            tde.append([[sp, a, s, r, g, pri, s_value[a], np.max(sp_value)]])
                            chain.append([sp, a, s])

                    # if state_in_wall(s):
                    #     trace = state_in_record(sp, chain)
                    #     if trace != None:
                    #         chain[trace].append(s)
                    #         # print("append in old trace", len(chain[trace]))
                    #     else:
                    #         chain.append([sp, s])
                    #         # print("append a new", len(chain))
        if (step+1) % 100 == 0:
            print("step", step)

    plt.figure()
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.0, 0.0), 1.0, 1.0, fill=None))

    max_length = [0, -1]
    long_length = []
    for idx in range(len(chain)):
        if len(chain[idx]) > 5:
            long_length.append(idx)
        if len(chain[idx]) > max_length[0]:
            max_length = [len(chain[idx]), idx]

    color_list = ["orange", "green", "red", "purple"]
    ch = np.array(chain[max_length[1]])
    state, action = separate_sa(ch)

    # plt.plot(ch[:1, 0], ch[:1, 1], "o", color="blue")
    # plt.plot(ch[1:, 0], ch[1:, 1], "bo")
    # plt.plot(ch[:, 0], ch[:, 1], color='b')

    plt.plot(state[:1, 0], state[:1, 1], "o", color="b")
    plt.plot(state[:, 0], state[:, 1], color='b')
    plt.plot(state[:1, 0], state[:1, 1], "o", color="blue")
    for idx in range(1, len(state)):
        plt.plot([state[idx, 0]], [state[idx, 1]], "o", color = color_list[action[idx - 1]])


    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)
    plt.savefig("temp/maxL-"+ str(max_length[0]))

    for idx in range(len(chain)):

        plt.figure()
        currentAxis = plt.gca()
        currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
        currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
        currentAxis.add_patch(patches.Rectangle((0.0, 0.0), 1.0, 1.0, fill=None))

        ch = np.array(chain[idx])
        state, action = separate_sa(ch)

        # plt.plot(ch[:1, 0], ch[:1, 1], "o", color="blue")
        # plt.plot(ch[1:, 0], ch[1:, 1], "bo")
        # plt.plot(ch[:, 0], ch[:, 1], color='b')

        plt.plot(state[:1, 0], state[:1, 1], "o", color="blue")
        plt.plot(state[:, 0], state[:, 1], color='b')
        for i in range(1, len(state)):
            plt.plot([state[i][0]], [state[i][1]], "o", color=color_list[action[i - 1]])

        plt.xlim(left=0, right=1)
        plt.ylim(bottom=0, top=1)
        if idx in long_length:
            plt.savefig("temp/longL-" + str(idx))
        else:
            plt.savefig("temp/shortL-" + str(idx))
        plt.clf()
        plt.close()

        print("------", idx, "------")
        for info in tde[idx]:
            print(info)
        print()

def separate_sa(ch):
    action = []
    state = []
    for element in ch:
        if type(element) == int or \
            type(element) == float:
            action.append(int(element))
        else:
            state.append(element)
    return np.array(state), action

def state_in_record(s, rec):
    for i in range(len(rec)):
        if np.allclose(s, rec[i][-1]):
            return i
    return None


def state_in_wall(s):
    if 0.5 < s[0] < 0.7 and (s[1] < 0.4 or s[1] > 0.6):
        return True
    else:
        return False

jsonfile = "parameters/continuous_gridworld.json"
# jsonfile = "parameters/gridworldgraph.json"
# jsonfile = "parameters/gridworld.json"

json_dat = open(jsonfile, 'r')
exp = json.load(json_dat)
json_dat.close()

env_params = {}
if "env_params" in exp:
    env_params = exp['env_params']
if "agent_params" in exp:
    agent_params = exp['agent_params']
if "exp_params" in exp:
    exp_params = exp['exp_params']

# load parameters
if len(sys.argv) > 2:
    exp['agent'] = str(sys.argv[1])
    agent_params["alpha"] = float(sys.argv[2])
    agent_params["num_near"] = int(sys.argv[3])
    agent_params["add_prot_limit"] = float(sys.argv[4])
    this_run = int(sys.argv[5])
    if exp['agent'] == "REM_Dyna" or exp['agent'] == "REM_Dyna_deb":
        agent_params["remDyna_mode"] = int(sys.argv[6])

    elif exp['agent'] == "Q_learning":
        agent_params["qLearning_mode"] = int(sys.argv[6])
    elif exp['agent'] == "random_ER":
        agent_params["erLearning_mode"] = int(sys.argv[6])
    else:
        print("The agent doesn't have learning mode")
        # exit()

    agent_params["model_params"]["kscale"] = float(sys.argv[7])
    # agent_params["similarity_limit"] = float(sys.argv[8])
    agent_params["model_params"]["sampling_limit"] = float(sys.argv[8])
    agent_params["always_add_prot"] = int(sys.argv[9])

    agent_params["model_params"]["fix_cov"] = float(sys.argv[10])
    agent_params["model_params"]["cov"] = float(sys.argv[10])

    agent_params["alg"] = str(sys.argv[11])
    agent_params["lambda"] = float(sys.argv[12])
    agent_params["momentum"] = float(sys.argv[13])
    agent_params["rms"] = float(sys.argv[14])
    agent_params["opt_mode"] = int(sys.argv[15])
    agent_params["offline"] = int(sys.argv[16])
    agent_params["num_planning"] = int(sys.argv[17])
    agent_params["pri_thrshd"] = float(sys.argv[18])
    agent_params["len_buffer"] = int(sys.argv[19])
    agent_params["dqn_c"] = int(sys.argv[20])
else:
    this_run = 1

agent_params["div_actBit"] = agent_params["remDyna_mode"]

from rl_glue import *  # Required for RL-Glue
RLGlue(exp['environment'], exp['agent'])
print("Env::", exp["environment"], ", Param:", exp["env_params"])
fixed_param_run(env_params, agent_params, exp_params, this_run)
os.chdir(retval)
print("exp ends, cwd is", os.getcwd())
