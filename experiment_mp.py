""""
This is the experiment file for REM Dyna project
multiprocess
"""
import multiprocessing as mp
import time
import numpy as np
# import matplotlib.pyplot as plt
import sys

sys.path.append('./environment/')
sys.path.append('./agent/')
import json

# load parameters
if len(sys.argv) == 4:
    alpha = float(sys.argv[1])
    num_near = int(sys.argv[2])
    add_prot_limit = float(sys.argv[3])

jsonfile = "parameters/continuous_gridworld.json"
json_dat = open(jsonfile, 'r')
exp = json.load(json_dat)
json_dat.close()

from rl_glue import *  # Required for RL-Glue

RLGlue(exp['environment'], exp['agent'])
print("Env::", exp["environment"], ", Param:", exp["env_params"])

def single_run(save_data, num_runs, run, agent_params, env_params):
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
    accum_r_single = np.zeros(num_steps)
    end_episode = True
    step_time = time.time()

    while step < num_steps:
        if end_episode:
            if step != 0:
                RL_end()
                print(step, "steps. accum_reward =", accum_r, ". time =", time.time() - start_time)
            start_time = time.time()
            info = RL_start()
            ep_check_choice = np.zeros((4))

        info = RL_step()
        accum_r += info["reward"]
        accum_r_single[step] = accum_r
        end_episode = info["isTerminal"]
        step += 1

        if step % 100 == 0:
            print("=====", run, accum_r, step, time.time() - step_time, RL_agent_message(["check model size"]),
                  info["state"], info["action"], info["reward"])
            step_time = time.time()
        print("One run finished:", run, time.time() - run_start)

    if save_data:
        file_name = "exp_result/" + str(exp["environment"]) + "_" + str(exp["agent"])
        if exp['agent'] == "REM_Dyna" and "rem_type" in agent_params:
            file_name += "_" + str(agent_params["rem_type"])

        file_name += "_alpha" + str(agent_params["alpha"]) + \
                     "_near" + str(agent_params["num_near"]) + \
                     "_protLimit" + str(agent_params["add_prot_limit"]) + \
                     "_run" + str(run)

        file_name += "_" + str(num_steps) + "x" + str(num_runs) + ".npy"
        np.save(file_name, accum_r_single)
        print("data saved.", file_name + "_accumReward_" + str(num_steps) + "x" + str(num_runs) + ".npy")

def fixed_param_run(env_params, agent_params, exp_params):
    print("Agent::", agent_params)
    print("Exp param::", exp_params)

    num_episodes = exp_params['num_episodes']
    num_steps = exp_params['num_steps']
    num_runs = exp_params['num_runs']
    which_to_rec = exp_params['which_to_rec']
    save_data = exp_params["save_data"]

    control_step = num_episodes == 0

    if control_step:
        print("Exp:: control total number of steps.")
        accum_r_record = np.zeros((num_runs, num_steps))

        pool = mp.Pool()
        res = pool.starmap(single_run, [(save_data, num_runs, run, agent_params, env_params) for run in range(num_runs)])
        for r in range(len(res)):
            accum_r_record[r] = res[r]

        if save_data:
            file_name = "exp_result/" + str(exp["environment"]) + "_" + str(exp["agent"])
            if exp['agent'] == "REM_Dyna" and "rem_type" in agent_params:
                file_name += "_" + str(agent_params["rem_type"])

            file_name += "_alpha" + str(agent_params["alpha"]) + \
                         "_near" + str(agent_params["num_near"]) + \
                         "_protLimit" + str(agent_params["add_prot_limit"])

            file_name += "_" + str(num_steps) + "x" + str(num_runs) + ".npy"
            np.save(file_name, accum_r_record)
            print("data saved.", file_name + "_accumReward_" + str(num_steps) + "x" + str(num_runs) + ".npy")
    #
    # else:
    #     print("Exp:: control number of episodes.")
    #     ep_step_record = np.zeros((num_runs, num_episodes))
    #     accum_r_record = np.zeros((num_runs, num_episodes, num_steps))
    #     opt_a_record = np.zeros((num_runs, num_episodes, num_steps))
    #
    #     for run in range(num_runs):
    #         print("run number: " + str(run + 1))
    #
    #         RL_init()
    #
    #         dim_state = RL_env_message(["state dimension", None])
    #         agent_params["dim_state"] = dim_state
    #         num_action = RL_env_message(["num_action", None])
    #         agent_params["num_action"] = num_action
    #
    #         RL_agent_message(["set param", agent_params])
    #         RL_env_message(["set param", env_params])
    #         for episode in range(num_episodes):
    #
    #             episode_time = time.time()
    #
    #             info = RL_start()
    #             step = 0
    #             end_episode = False
    #
    #             start_time = time.time()
    #             action_sum = 0
    #             while step < num_steps and not end_episode:
    #
    #                 step_time = time.time()
    #                 info = RL_step()
    #
    #                 # check_time =  RL_agent_message(["check time"])
    #                 # if step%1==0:
    #                 #     print("step", step, "uses time", end=" ")
    #                 #     #time = RL_agent_message(["check time"])
    #                 #     for i in RL_agent_message(["check time"]):
    #                 #         print(format(i, '.6f'), end = " ")
    #                 #     print()
    #
    #                 step += 1
    #                 end_episode = info["isTerminal"]
    #                 if not end_episode:
    #                     action_sum += info["action"]
    #                 opt_a_record[run, episode, step - 1] = action_sum / float(step)
    #                 accum_r_record[run, episode, step - 1] = RL_return()
    #
    #                 if step % 100 == 0:
    #                     print("=====", run, step, time.time() - start_time, RL_agent_message(["check model size"]),
    #                           info, RL_return())
    #                     start_time = time.time()
    #             # for i in RL_agent_message(["check total time"]):
    #             #    print(format(i, '.6f'), end=" ")
    #             # print()
    #
    #             RL_end()
    #             print(episode + 1, "episode ends. #step =", step, ", time used =", time.time() - episode_time)
    #             print("   accumulated reward:", RL_return())
    #
    #             ep_step_record[run, episode] = step
    #
    #         RL_cleanup()
    #
    #     if save_data:
    #         file_name = "exp_result/" + str(exp["environment"]) + "_" + str(exp["agent"])
    #         if exp['agent'] == "REM_Dyna" and "rem_type" in agent_params:
    #             file_name += "_" + str(agent_params["rem_type"])
    #         if "step" in which_to_rec:
    #             plt.figure(1)
    #             plt.plot(np.mean(ep_step_record, axis=0))
    #             np.save(file_name + "_step_" + str(num_steps) + "x" + str(num_runs) + ".npy", ep_step_record)
    #             print("data saved.", file_name + "_step_" + str(num_steps) + "x" + str(num_runs) + ".npy")
    #
    #         if "action" in which_to_rec:
    #             plt.figure(2)
    #             plt.plot(np.mean(opt_a_record, axis=0)[0])
    #             np.save(file_name + "_action_" + str(num_steps) + "x" + str(num_runs) + ".npy", opt_a_record)
    #             print("data saved.", file_name + "_action_" + str(num_steps) + "x" + str(num_runs) + ".npy")
    #
    #         if "return" in which_to_rec:
    #             plt.figure(3)
    #             plt.plot(np.mean(accum_r_record, axis=0)[0])
    #             np.save(file_name + "_return_" + str(num_steps) + "x" + str(num_runs) + ".npy", accum_r_record)
    #             print("data saved.", file_name + "_return_" + str(num_steps) + "x" + str(num_runs) + ".npy")
    #
    #     plt.show()


env_params = {}
if "env_params" in exp:
    env_params = exp['env_params']
if "agent_params" in exp:
    agent_params = exp['agent_params']
if "exp_params" in exp:
    exp_params = exp['exp_params']

sweep_param = exp_params["sweep_param"]
if sweep_param:
    sweep_agent = exp["sweeps"]["agent_params"]
    total_comb = 1
    for k in sweep_agent:
        total_comb *= len(sweep_agent[k])
    for index in range(total_comb):
        accum = 1
        for key in sweep_agent:
            num = len(sweep_agent[key])
            agent_params[key] = sweep_agent[key][int((index / accum) % num)]
            accum *= num
        print(index, agent_params)
        fixed_param_run(env_params, agent_params, exp_params)
else:
    if len(sys.argv) == 4:
        agent_params["alpha"] = alpha
        agent_params["num_near"] = num_near
        agent_params["add_prot_limit"] = add_prot_limit
    fixed_param_run(env_params, agent_params, exp_params)


