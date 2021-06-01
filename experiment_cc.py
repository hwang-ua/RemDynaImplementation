""""
This is the experiment file for REM Dyna project
"""
import time
import numpy as np
import torch
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

np.set_printoptions(precision=3)

def fixed_param_run(env_params, agent_params, exp_params, this_run):
    print("Agent::", agent_params)
    print("Exp param::", exp_params)

    num_episodes = exp_params['num_episodes']
    num_steps = exp_params['num_steps']
    num_runs = exp_params['num_runs']
    # which_to_rec = exp_params['which_to_rec']
    save_data = exp_params["save_data"]

    if exp['environment'] == "ContinuousGridWorld":
        num_episodes = 0

    control_step = num_episodes == 0
    if exp['environment'][:7] == "Catcher":
        control_step = True
    agent_params["environment"] = exp['environment']


    print("Exp:: control total number of steps.")
    # accum_r_record = np.zeros((1, num_steps))
    step_per_ep = []
    last_end_step = 0

    run_start = time.time()
    print("run number: " + str(this_run + 1))
    RL_init()

    # if exp_params["random_seed"] != 0:
    #     np.random.seed(exp_params["random_seed"] * this_run)
    if exp_params["random_seed"] != 0:
        np.random.seed(exp_params["random_seed"] * this_run)
        torch.manual_seed(exp_params["random_seed"] * this_run)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    RL_env_message(["set param", env_params])

    dim_state = RL_env_message(["state dimension", None])
    agent_params["dim_state"] = dim_state
    num_action = RL_env_message(["num_action", None])
    agent_params["num_action"] = num_action

    RL_agent_message(["set param", agent_params])

    # RL_agent_message(["set param", agent_params])
    # RL_env_message(["set param", env_params])

    real_step = 0
    learning_step = 0
    agent_is_learning = False

    accum_r = 0
    ep_r = 0
    end_episode = True
    step_time = time.time()
    check_choice = np.zeros((4))
    ep_check_choice = np.zeros((4))

    # stop_sign_set = False
    real_step_ep = real_step

    if control_step:
        learning_ep = 0

        while learning_step < num_steps:

            # if learning_step >= 1000:
            #     exit()

            if end_episode:
                if real_step != 0:
                    # RL_agent_step(info["reward"], info["state"])
                    step_per_ep.append(real_step - last_end_step)
                    last_end_step = real_step
                    RL_end()
                    learning_ep += 1
                    print(learning_ep, "episodes", real_step-real_step_ep, "steps. accum_reward =", accum_r, ". episode reward", ep_r, ". time =", time.time() - start_time)
                    ep_r = 0
                    real_step_ep = real_step
                start_time = time.time()
                info = RL_start()
                ep_check_choice = np.zeros((4))

            info = RL_step()
            # info = RL_step_debugging()

            accum_r += info["reward"]
            ep_r += info["reward"]
            end_episode = info["isTerminal"]

            if not agent_is_learning and accum_r != 0:
                agent_is_learning = True
                accum_r_record = np.zeros(real_step + num_steps)#np.concatenate((np.zeros(real_step), accum_r_record))

            if agent_is_learning:
                accum_r_record[real_step] = accum_r
                learning_step += 1

                # if learning_step % 1000 == 0:
                #     RL_agent_message(["change epsilon", np.max([1.0 - 0.1 * (learning_step // 1000), 0.1])])

            real_step += 1

            if (real_step + 1) % 100 == 0:
                print("=====", this_run, accum_r, real_step, str(learning_step)+"/["+str(num_steps)+"]", time.time() - step_time, RL_agent_message(["check model size"]),
                      info["state"], info["action"], info["reward"])
                step_time = time.time()
    else:
        print("control number of episodes")
        learning_ep = 0
        while learning_ep < num_episodes and learning_step < num_steps:
            if end_episode:
                if real_step != 0:
                    # RL_agent_step(info["reward"], info["state"])
                    step_per_ep.append(real_step - last_end_step)
                    last_end_step = real_step
                    RL_end()
                    learning_ep += 1
                    print(real_step - real_step_ep, "steps. accum_reward =", accum_r, ". time =",
                          time.time() - start_time)
                    real_step_ep = real_step

                start_time = time.time()
                info = RL_start()
                ep_check_choice = np.zeros((4))

            info = RL_step()
            # info = RL_step_debugging()

            accum_r += info["reward"]
            end_episode = info["isTerminal"]

            if not agent_is_learning and accum_r != 0:
                agent_is_learning = True
                accum_r_record = np.zeros(
                    real_step + num_steps)  # np.concatenate((np.zeros(real_step), accum_r_record))

            if agent_is_learning:
                accum_r_record[real_step] = accum_r
                learning_step += 1

                # if learning_step % 1000 == 0:
                #     RL_agent_message(["change epsilon", np.max([1.0 - 0.1 * (learning_step // 1000), 0.1])])

            real_step += 1

            if (real_step + 1) % 100 == 0:
                print("=====", this_run, accum_r, real_step, learning_step, time.time() - step_time,
                      RL_agent_message(["check model size"]),
                      info["state"], info["action"], info["reward"])
                step_time = time.time()

        accum_r_record = accum_r_record[:real_step - 1]

    print("One run finished:", this_run, time.time() - run_start)
    if save_data:
        if exp['agent'] == "REM_Dyna" or exp['agent'] == "REM_Dyna_deb":
            file_name = "./exp_result/" + \
                        str(exp["agent"]) + "_mode" + str(agent_params["remDyna_mode"]) + str(agent_params["representation"]) +\
                        "_offline" + str(agent_params["offline"]) + \
                        "_planning" + str(agent_params["num_planning"]) + \
                        "_priThrshd" + str(agent_params["pri_thrshd"]) + \
                        "_DQNc" + str(agent_params["dqn_c"]) + \
                        "_buffer" + str(agent_params["len_buffer"]) + "/" \
                        + "always_add_prot_" + str(agent_params["always_add_prot"]) + "/"
            # file_name = "./exp_result/" + \
            #             str(exp["agent"]) + "_mode" + str(agent_params["remDyna_mode"]) + \
            #             "_offline" + str(agent_params["offline"]) + \
            #             "_planning" + str(agent_params["num_planning"]) + \
            #             "_priThrshd" + str(agent_params["pri_thrshd"]) + \
            #             "_remModel" + \
            #             "_buffer" + str(agent_params["len_buffer"]) + "/" \
            #             + "always_add_prot_" + str(agent_params["always_add_prot"]) + "/"

        elif exp['agent'] == "Q_learning":
            file_name = "./exp_result/" + str(exp["agent"]) + "_mode" + str(agent_params["qLearning_mode"]) + "/"

        elif exp['agent'] == "random_ER":
            file_name = "./exp_result/" + str(exp["agent"]) + "_mode" + str(agent_params["erLearning_mode"]) + "/"
        elif exp['agent'] == "NN_Dyna":
            file_name = "./exp_result/" + str(exp["agent"]) + "/"
        else:
            file_name = "./exp_result/" + str(exp["agent"]) + "/"
            print("No folder name for this agent")
            exit(-1)

        if not os.path.exists(file_name):
            os.makedirs(file_name)

        if (exp['agent'] == "REM_Dyna" or exp['agent'] == "REM_Dyna_deb") and "rem_type" in agent_params:
            file_name += str(agent_params["rem_type"])
            file_name += "_alpha" + str(agent_params["alpha"]) + \
                         "_divAF" + str(agent_params["remDyna_mode"]) + \
                         "_near" + str(agent_params["num_near"]) + \
                         "_protLimit" + str(agent_params["add_prot_limit"]) + \
                         "_similarity" + str(agent_params["similarity_limit"]) + \
                         "_sampleLimit" + str(agent_params["model_params"]["sampling_limit"]) + \
                         "_kscale" + str(agent_params["model_params"]["kscale"]) + \
                         "_fixCov" + str(agent_params["model_params"]["fix_cov"]) + \
                         "_update" + str(agent_params["alg"]) + \
                         "_lambda" + str(agent_params["lambda"]) + \
                         "_momentum" + str(agent_params["momentum"]) + \
                         "_rms" + str(agent_params["rms"]) + \
                         "_optMode" + str(agent_params["opt_mode"]) + \
                         "_run" + str(this_run)
        elif exp['agent'] == "random_ER":
            file_name += "_alpha" + str(agent_params["alpha"]) + \
                         "_divAF" + str(agent_params["erLearning_mode"]) + \
                         "_update" + str(agent_params["alg"]) + \
                         "_lambda" + str(agent_params["lambda"]) + \
                         "_momentum" + str(agent_params["momentum"]) + \
                         "_rms" + str(agent_params["rms"]) + \
                         "_optMode" + str(agent_params["opt_mode"]) + \
                         "_run" + str(this_run)
        elif exp['agent'] == "Q_learning":
            file_name += "_alpha" + str(agent_params["alpha"]) + \
                         "_divAF" + str(agent_params["qLearning_mode"]) + \
                         "_update" + str(agent_params["alg"]) + \
                         "_lambda" + str(agent_params["lambda"]) + \
                         "_momentum" + str(agent_params["momentum"]) + \
                         "_rms" + str(agent_params["rms"]) + \
                         "_optMode" + str(agent_params["opt_mode"]) + \
                         "_run" + str(this_run)
        else:
            # file_name = "./exp_result/" + str(exp["agent"]) + "/"
            print("No file name for this agent")
            exit(-1)

        np.save(file_name+"_stepPerEp", step_per_ep)
        np.save(file_name, accum_r_record)
        print("data saved.", file_name + ".npy")


# jsonfile = "parameters/continuous_gridworld.json"
jsonfile = "parameters/puddle_world.json"
# jsonfile = "parameters/catcher.json"

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

    # agent_params["alg"] = str(sys.argv[11])
    agent_params["rem_type"] = str(sys.argv[11])
    agent_params["lambda"] = float(sys.argv[12])
    agent_params["momentum"] = float(sys.argv[13])
    agent_params["rms"] = float(sys.argv[14])
    agent_params["opt_mode"] = int(sys.argv[15])
    agent_params["offline"] = int(sys.argv[16])
    agent_params["num_planning"] = int(sys.argv[17])
    agent_params["pri_thrshd"] = float(sys.argv[18])
    agent_params["len_buffer"] = int(sys.argv[19])
    agent_params["dqn_c"] = int(sys.argv[20])
    agent_params["representation"] = str(sys.argv[21])
else:
    this_run = 1

agent_params["div_actBit"] = agent_params["remDyna_mode"]

from rl_glue import *  # Required for RL-Glue
RLGlue(exp['environment'], exp['agent'])
print("Env::", exp["environment"], ", Param:", exp["env_params"])
fixed_param_run(env_params, agent_params, exp_params, this_run)
os.chdir(retval)
print("exp ends, cwd is", os.getcwd())
