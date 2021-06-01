""""
This is the experiment file for REM Dyna project
"""
import time
import numpy as np
# import matplotlib.pyplot as plt
import sys
sys.path.append('./environment/')
sys.path.append('./agent/')
import json


jsonfile = "parameters/continuous_gridworld.json"
json_dat = open(jsonfile, 'r')
exp = json.load(json_dat)
json_dat.close()

from rl_glue import *  # Required for RL-Glue
RLGlue(exp['environment'], exp['agent'])
print("Env::", exp["environment"], ", Param:", exp["env_params"])

def fixed_param_run(env_params, agent_params, exp_params):

    print("Agent::", agent_params)
    print("Exp param::", exp_params)

    num_episodes = exp_params['num_episodes']
    num_steps = exp_params['num_steps']
    num_runs = 1#exp_params['num_runs']
    which_to_rec = exp_params['which_to_rec']
    save_data = exp_params["save_data"]

    control_step = num_episodes == 0

    accum_r_record = np.zeros((num_runs, num_steps))

    for run in range(num_runs):
        run_start = time.time()
        print("run number: " + str(run + 1))
        RL_init()

        dim_state = RL_env_message(["state dimension", None])
        agent_params["dim_state"] = dim_state
        num_action = RL_env_message(["num_action", None])
        agent_params["num_action"] = num_action

        RL_agent_message(["set param", agent_params])
        RL_env_message(["set param", env_params])

        step = 0
        accum_r = 0
        end_episode = True
        step_time = time.time()

        # stop_sign_set = False
        while step < num_steps:
            if end_episode:
                if step != 0:
                    # RL_agent_step(info["reward"], info["state"])
                    RL_end()
                    print(step, "steps. accum_reward =", accum_r, ". time =", time.time() - start_time)
                start_time = time.time()
                info = RL_start()
                ep_check_choice = np.zeros((4))

            # if accum_r > 50 and not stop_sign_set:
            #     RL_agent_message(["start using model"])
            #     stop_sign_set = True
            #     num_steps = step + 2000
            #     print("*** start using model ***")

            info = RL_step()
            accum_r += info["reward"]
            accum_r_record[run, step] = accum_r
            end_episode = info["isTerminal"]
            step += 1

            # check_choice[info["action"]] += 1
            # ep_check_choice[info["action"]] += 1
            if step % 100 == 0:
                print("=====", run, accum_r, step, time.time() - step_time, RL_agent_message(["check model size"]),
                      info["state"], info["action"], info["reward"])
                # print("     ", check_choice / np.sum(check_choice))
                # print("     ", ep_check_choice / np.sum(ep_check_choice))
                step_time = time.time()
        print("One run finished:", run, time.time()-run_start)
        if save_data:
            file_name = str(exp_params["folder"])  + str(exp["environment"]) + str(exp["agent"])
            if exp['agent'] == "REM_Dyna" and "rem_type" in agent_params:
                file_name += "_" + str(agent_params["rem_type"])
            # file_name += "_alpha" + str(agent_params["alpha"]) + \
            #              "_near" + str(agent_params["num_near"]) + \
            #              "_protLimit" + str(agent_params["add_prot_limit"]) + \
            #              "_run" + str(run)
            file_name += "_alpha" + str(agent_params["alpha"]) + "_divAF"+str(agent_params["div_actBit"]) + \
                         "_near" + str(agent_params["num_near"]) + \
                         "_protLimit" + str(agent_params["add_prot_limit"])
            # file_name += "_pan'sModel"
            file_name += "_" + str(num_steps) + "x" + str(num_runs) + ".npy"
            np.save(file_name, accum_r_record)
            print("data saved.", file_name + "_accumReward_" + str(num_steps) + "x" + str(num_runs) + ".npy")

env_params = {}
if "env_params" in exp:
    env_params = exp['env_params']
if "agent_params" in exp:
    agent_params = exp['agent_params']
if "exp_params" in exp:
    exp_params = exp['exp_params']

fixed_param_run(env_params, agent_params, exp_params)
    

