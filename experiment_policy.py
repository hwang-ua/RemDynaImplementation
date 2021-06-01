""""
This is the experiment file for REM Dyna project
"""
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


def check_policy_change(policy, old_policy):
    if old_policy is None:
        return 10000
    else:
        assert  len(policy) == len(old_policy)
        change = 0
        for i in range(len(policy)):
            if policy[i] != old_policy[i]:
                change += 1
        return change

def check_weight_change(weight, last_weight):
    if last_weight is None:
        return 10000
    else:
        assert len(weight) == len(last_weight)
        change = np.max(np.abs(weight - last_weight))
        return change


def fixed_param_run(env_params, agent_params, exp_params, this_run):
    print("Agent::", agent_params)
    print("Exp param::", exp_params)

    num_episodes = exp_params['num_episodes']
    num_steps = exp_params['num_steps']
    num_runs = 1#exp_params['num_runs']

    control_pi = num_episodes == 0

    if control_pi:
        print("Exp:: control policy.")
        accum_r_record = []
        # test_states = []
        # test_step = 0.2
        # alist = [0.15, 0.3, 0.45, 0.75, 0.9]
        # for i in alist:#np.arange(0.0, 1.0, test_step):
        #     for j in np.arange(0.0, 1.0, test_step):
        #         test_states.append([i, j])

        run_start = time.time()
        print("run number: " + str(this_run + 1))
        RL_init()

        dim_state = RL_env_message(["state dimension", None])
        agent_params["dim_state"] = dim_state
        num_action = RL_env_message(["num_action", None])
        agent_params["num_action"] = num_action

        RL_agent_message(["set param", agent_params])
        RL_env_message(["set param", env_params])

        real_step = 0
        learning_step = 0
        agent_is_learning = False
        episode_step = 0

        accum_r = 0
        end_episode = True
        step_time = time.time()

        all_weight = np.zeros((10000, RL_agent_message(["get weight size"])))
        wind = 0
        common_name = "alg" + str(agent_params["alg"]) + \
                   "_lambda"+ str(agent_params["lambda"]) +\
                   "_lr" + str(agent_params["alpha"]) + \
                   "_optMode" + str(agent_params["opt_mode"]) + \
                   "_beta" + str(agent_params["momentum"]) + \
                   "-"+ str(agent_params["rms"])
        folder = "exp_check_weight/" + str(exp['agent'])+ "_" + str(agent_params['sample_mode'])+ "/"
        log_name = folder + "weight_log" + common_name
        result_name = folder + common_name
        if not os.path.exists(folder):
            os.makedirs(folder)

        time_notChange = 0
        # last_policy = None
        last_weight = None
        last_ep_end = 0
        each_ep_step = np.zeros(10)
        each_ep_step_ind = 0
        while time_notChange < 10 and learning_step < 50000:
            if end_episode:
            # if end_episode or episode_step == 10000:
                if real_step != 0:

                    weight = RL_agent_message(["check_weight"])
                    if wind < len(all_weight):
                        all_weight[wind] = weight
                    else:
                        all_weight = np.concatenate((all_weight, weight.reshape((1, -1))), axis=0)
                    wind += 1
                    np.save(result_name, np.array(accum_r_record))
                    np.save(log_name, all_weight)
                    print("saved log")
                    change = check_weight_change(weight, last_weight)
                    print("change of weight", change)
                    last_weight = np.copy(weight)
                    if change < 0.005:
                        time_notChange += 1
                    else:
                        time_notChange = 0
                    RL_end()
                    # print(real_step, "steps. accum_reward =", accum_r, ". time =", time.time() - start_time)
                    print(real_step, episode_step, "steps. accum_reward =", accum_r, ". time =", time.time() - start_time)

                    each_ep_step[each_ep_step_ind % 10] = real_step - last_ep_end
                    each_ep_step_ind += 1

                    for num in each_ep_step:
                        print(num, end=' ')
                    print("\n")
                    last_ep_end = real_step

                start_time = time.time()
                info = RL_start()
                ep_check_choice = np.zeros((4))

                episode_step = 0

            info = RL_step()
            episode_step += 1
            accum_r += info["reward"]
            end_episode = info["isTerminal"]

            if accum_r > 0:
                agent_is_learning = True

            if agent_is_learning:
                accum_r_record.append(accum_r)
                learning_step += 1

            real_step += 1

            if (real_step+1) % 100 == 0:
                print("=====", this_run, accum_r, real_step, learning_step, time.time() - step_time, RL_agent_message(["check model size"]),
                      info["state"], info["action"], info["reward"])
                step_time = time.time()

        print("averaged steps for last 10 ep", np.mean(np.array(each_ep_step)))
        print("One run finished:", this_run, time.time() - run_start)






jsonfile = "parameters/continuous_gridworld.json"
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
    learning_mode = int(sys.argv[2])
    agent_params["alg"] = str(sys.argv[3])
    agent_params["lambda"] = float(sys.argv[4])
    agent_params["alpha"] = float(sys.argv[5])
    agent_params["opt_mode"] = int(sys.argv[6])
    agent_params["momentum"] = float(sys.argv[7])
    agent_params["rms"] = float(sys.argv[8])
    agent_params["sample_mode"] = str(sys.argv[9])
    this_run = 1

    if exp['agent'] == 'Q_learning':
        agent_params["qLearning_mode"] = learning_mode
    elif exp["agent"] == "random_ER":
        agent_params["erLearning_mode"] = learning_mode
    else:
        print("Agent disapears")
        exit(1)
else:
    this_run = 1

from rl_glue import *  # Required for RL-Glue
RLGlue(exp['environment'], exp['agent'])
print("Env::", exp["environment"], ", Param:", exp["env_params"])

fixed_param_run(env_params, agent_params, exp_params, this_run)
os.chdir(retval)
print("exp ends, cwd is", os.getcwd())
