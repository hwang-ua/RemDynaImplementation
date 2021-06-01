""""
This is the experiment file for REM Dyna project
"""

import numpy as np
import sys
from utils.distance_matrix_func import *
from rl_glue import *  # Required for RL-Glue
import utils.tiles3 as tc


def go_in_wall(x, y, wall_x, wall_w, hole_yl, hole_yh):
    if x > wall_x and x < (wall_x + wall_w):
        if y < hole_yl or y > hole_yh:
            return True
    else:
        return False

def sort_file():
    num_ep = 50
    num_point = 200
    file_order = 25
    num_tiling = 2
    num_tile = 16
    opt_prob = 0
    gamma = 0.998
    collection = None
    name_base = "random_data/cgw_tc"+str(num_tiling)+"x"+str(num_tile)+"_training_set_allState_"+str(int(opt_prob*100))+"opt_"\
               +str(gamma)+"gamma_"
    for o in range(file_order):
        name = name_base + str(num_point)+"pts_x"+str(num_ep) + "_" + str(o)+".npy"
        temp = np.load(name)
        if collection is None:
            collection = temp
        else:
            collection = np.concatenate((collection, temp), axis=0)

    num_point *= file_order
    np.save(name_base + str(num_point) + "pts_x" + str(num_ep), collection)

def add_one_gamma():
    suc_prob = 1.0
    gamma = 0.8
    num_tiling = 32
    num_tile = 4

    data = np.load("random_data/fixed_env_suc_prob_"+str(suc_prob)+"/cgw_noGoal_separateTC"+str(num_tiling)+"x"+str(num_tile)+"_training_set_randomStart_0opt_0.998gamma_1pts_x1_x500000.npy")
    print(data.shape)
    one_ep = data[:, :2]

    state, feature, ground_truth = preproc_cgw_data(np.array(one_ep), gamma, True, num_tile=num_tile, num_tiling=num_tiling)
    # print(feature.shape, data[:, 2: 2 + num_tile*num_tiling*2].shape)
    # assert equal_array(feature, data[:, 2: 2 + num_tile**2*num_tiling])
    data = np.concatenate((data, ground_truth), axis=1)
    print(data.shape)
    np.save("random_data/fixed_env_suc_prob_"+str(suc_prob)+"/cgw_noGoal_separateTC"+str(num_tiling)+"x"+str(num_tile)+"_training_set_randomStart_0opt_[0.998, "+str(gamma)+"]gamma_1pts_x1_x500000.npy", data)

def remove_one_gamma():
    suc_prob = 1.0
    num_tiling = 32
    num_tile = 4

    data = np.load("random_data/fixed_env_suc_prob_"+str(suc_prob)+"/cgw_noGoal_separateTC"+str(num_tiling)+"x"+str(num_tile)+"_training_set_randomStart_0opt_[0.998, 0.8]gamma_1pts_x1.npy")
    pts = data[:, : 2+num_tiling*num_tile*2]
    print("shape of pts", pts.shape)
    ground_truth = data[:, 2+num_tiling*num_tile*2*2 :]
    print("shape of gt", ground_truth.shape)
    data = np.concatenate((pts, ground_truth), axis=1)
    print(data.shape)
    np.save("random_data/fixed_env_suc_prob_"+str(suc_prob)+"/cgw_noGoal_separateTC"+str(num_tiling)+"x"+str(num_tile)+"_training_set_randomStart_0opt_0.8gamma_1pts_x1.npy", data)


def cgw_collect_data():
    """
    Start from every state
    """
    sys.path.append('./environment/')
    sys.path.append('./agent/')
    import json

    jsonfile = "parameters/continuous_gridworld.json"
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

    gamma = [0.998, 0.8]#exp["agent_params"]["nn_gamma"]
    opt_prob = exp["agent_params"]["opt_prob"]
    wall_x = env_params["wall_x"]
    wall_w = env_params["wall_w"]
    hole_yl = env_params["hole_yl"]
    hole_yh = env_params["hole_yh"]

    num_tiling = 32#agent_params["nn_num_tilings"]
    num_tile = 4#agent_params["nn_num_tiles"]

    # path = "./random_data/"
    # file_name = "gw_random_walk_"

    if len(sys.argv) == 4:
        num_ep = int(sys.argv[1])
        num_point = int(sys.argv[2])
        file_order = sys.argv[3]
    else:
        num_ep = 1
        num_point = 1

    max_step = 100000
    separate = True
    sep_title = "separate" if separate else "xy"

    # start from random point

    # order_xy = np.random.random(size=num_point*2).reshape((-1, 2))
    order_xy = []
    while len(order_xy) < num_point:
        x, y = np.random.random(size=2)
        if not go_in_wall(x, y, wall_x, wall_w, hole_yl, hole_yh):
            order_xy.append([x, y])

    if separate:
        all_eps = np.zeros((0, 2 + num_tile*num_tiling*2 + num_tile*num_tiling*2)) if type(gamma) != list \
            else np.zeros((0, 2 + num_tile*num_tiling*2 + num_tile*num_tiling*2*len(gamma)))
    else:
        all_eps = np.zeros((0, 2 + num_tile**2 * num_tiling*(1+len(gamma))))

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

                if type(gamma) == list:
                    state, feature, ground_truth = preproc_cgw_data_multigamma(np.array(one_ep), gamma, True, separate,
                                                             num_tile=num_tile, num_tiling=num_tiling)
                else:
                    state, feature, ground_truth = preproc_cgw_data(np.array(one_ep), gamma, True, separate,
                                                             num_tile=num_tile, num_tiling=num_tiling)
                del one_ep

                f_g = np.concatenate((state, feature, ground_truth), axis=1)
                try:
                    all_eps = np.concatenate((all_eps, f_g), axis=0)
                except:
                    print(all_eps.shape, f_g.shape)
                    raise ValueError
                i += 1

        else:
            print(start_x, start_y, "is in wall")

    # save feature and ground truth
    suc_prob = env_params["suc_prob"]

    name = "random_data/fixed_env_suc_prob_" + str(suc_prob) + "/cgw_noGoal_"+str(sep_title)+"TC" + str(num_tiling) + "x" + str(num_tile) + "_training_set_randomStart_" + \
               str(int(opt_prob*100))+"opt_"+str(gamma)+"gamma_"+str(num_point)+"pts_x"+str(num_ep)+"_x"+str(max_step)
    if len(sys.argv) == 4:
        name += "_" + file_order
    np.save(name, np.array(all_eps))

def random_walk_with_goal():
    import pickle as pkl
    """
    Start from every state
    """
    sys.path.append('./environment/')
    sys.path.append('./agent/')
    import json

    jsonfile = "parameters/continuous_gridworld.json"
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

    opt_prob = exp["agent_params"]["opt_prob"]
    wall_x = env_params["wall_x"]
    wall_w = env_params["wall_w"]
    hole_yl = env_params["hole_yl"]
    hole_yh = env_params["hole_yh"]

    num_ep = 100
    num_point = 100

    max_step = 50000

    # start from random point
    order_xy = []
    while len(order_xy) < num_point:
        x, y = np.random.random(size=2)
        if not go_in_wall(x, y, wall_x, wall_w, hole_yl, hole_yh):
            order_xy.append([x, y])

    all_eps = []
    i = 0
    ind = 0
    total_num_step = 0
    while total_num_step < max_step:
    # while i < num_ep:
    #     i += 1
        start_x, start_y = order_xy[ind]
        if go_in_wall(start_x, start_y, wall_x, wall_w, hole_yl, hole_yh):
            print("pts =", start_x, start_y, "ind =", ind)
        ind += 1
        env_params["start_x"] = start_x
        env_params["start_y"] = start_y

        RL_init()

        num_step = 0

        dim_state = RL_env_message(["state dimension", None])
        agent_params["dim_state"] = dim_state
        num_action = RL_env_message(["num_action", None])
        agent_params["num_action"] = num_action
        RL_agent_message(["set param", agent_params])
        RL_env_message(["set param", env_params])

        one_ep = []

        info = RL_start()
        s = info["state"]
        sp = None
        r = None

        end_episode = False
        while not end_episode and num_step < max_step:
            info = RL_step()
            sp = info["state"]
            r = info["reward"]
            if r == 1:
                print(s, sp, r)
            one_ep.append([s, (sp, r)])
            s = info["state"]
            end_episode = info["isTerminal"]

            num_step += 1
            total_num_step += 1

            if num_step % 5000 == 0:
                print("step", num_step)

        print(">>> Ep ends. Total num of step, ", RL_num_steps())
        RL_end()

        all_eps.append(one_ep)
        del one_ep


    # save feature and ground truth
    suc_prob = env_params["suc_prob"]

    # name = "random_data/fixed_env_suc_prob_" + str(suc_prob) + "/cgw_withGoal_randomPoints_" + \
    #        str(int(opt_prob * 100)) + "opt_" + str(max_step)+".pkl"
    name = "cgw_withGoal_random_50000steps.pkl"
    with open(name, 'wb') as f:
        pkl.dump(all_eps, f)


def NN_model_data():
    sys.path.append('./environment/')
    sys.path.append('./agent/')
    import json

    jsonfile = "parameters/continuous_gridworld.json"
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

    wall_x = env_params["wall_x"]
    wall_w = env_params["wall_w"]
    hole_yl = env_params["hole_yl"]
    hole_yh = env_params["hole_yh"]

    num_tiling = agent_params["nn_num_tilings"]
    num_tile = agent_params["nn_num_tiles"]

    # path = "./random_data/"
    # file_name = "gw_random_walk_"

    if len(sys.argv) == 4:
        num_ep = int(sys.argv[1])
        num_point = int(sys.argv[2])
        file_order = sys.argv[3]
    else:
        num_ep = 100
        num_point = 1

    all_eps_forward = np.zeros((0, 7)) # s,a -> sp,r,g
    all_eps_backward = np.zeros((0, 5)) # sp,a -> s

    i = 0
    while i < num_ep:

        # episode
        RL_init()
        np.random.seed(512 * i)

        dim_state = RL_env_message(["state dimension", None])
        agent_params["dim_state"] = dim_state
        num_action = RL_env_message(["num_action", None])
        agent_params["num_action"] = num_action
        RL_agent_message(["set param", agent_params])
        RL_env_message(["set param", env_params])

        one_ep_s = []
        one_ep_a = []

        num_step = 0

        info = RL_start()
        one_ep_s.append(info["state"])
        one_ep_a.append(info["action"])

        end_episode = False
        while not end_episode:
            info = RL_step()

            one_ep_s.append(info["state"])
            one_ep_a.append(info["action"])

            end_episode = info["isTerminal"]

            num_step += 1

            if num_step % 5000 == 0:
                print("step", num_step)

        print(">>> Ep", i, "ends. Total num of step, ", RL_num_steps())
        RL_end()

        i += 1

        # format data
        one_ep_s = np.array(one_ep_s)
        one_ep_a = np.array(one_ep_a).reshape((-1, 1))
        forward = np.concatenate((one_ep_s[:-1], one_ep_a[:-1], one_ep_s[1:], np.zeros((len(one_ep_s)-1, 1)), 0.9*np.ones((len(one_ep_s)-1, 1))), axis=1)
        forward[-1, -2] = 1 # reward
        forward[-1, -1] = 0 # gamma
        backward = np.concatenate((one_ep_a[:-1], one_ep_s[1:], one_ep_s[:-1]), axis=1)

        all_eps_forward = np.concatenate((all_eps_forward, forward), axis=0)
        all_eps_backward = np.concatenate((all_eps_backward, backward), axis=0)

    suc_prob = env_params["suc_prob"]
    name = "random_data/fixed_env_suc_prob_" + str(suc_prob) + "/cgw_training_set_x" + str(num_ep)
    np.save(name+"_forward", all_eps_forward)
    np.save(name+"_backward", all_eps_backward)

def pw_collect_data():
    sys.path.append('./environment/')
    sys.path.append('./agent/')
    import json

    with_reward = True

    jsonfile = "parameters/continuous_gridworld.json"
    json_dat = open(jsonfile, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    RLGlue("PuddleWorld", exp['agent'])
    print("Env::", "puddleworld")

    env_params = {}
    # if "env_params" in exp:
    #     env_params = exp['env_params']
    if "agent_params" in exp:
        agent_params = exp['agent_params']
    # if "exp_params" in exp:
    #     exp_params = exp['exp_params']

    gamma = [0.998, 0.8]#exp["agent_params"]["nn_gamma"]

    if len(sys.argv) == 4:
        num_ep = int(sys.argv[1])
        num_point = int(sys.argv[2])
        file_order = sys.argv[3]
    else:
        num_ep = 1
        num_point = 1

    max_step = 100000
    tc = True
    num_tile = 4
    num_tiling = 32
    separate = True
    sep_title = "separate" if separate else "xy"

    # start from random point
    order_xy = np.random.random(size=num_point*2).reshape((-1, 2))

    if tc:
        if separate:
            all_eps = np.zeros((0, 2 + num_tile*num_tiling*2 + num_tile*num_tiling*2)) if type(gamma) != list \
                else np.zeros((0, 2 + num_tile*num_tiling*2 + num_tile*num_tiling*2*len(gamma)))
        else:
            all_eps = np.zeros((0, 2 + num_tile**2 * num_tiling*(1+len(gamma))))
    else:
        all_eps = np.zeros((0, 8))

    for ind in range(len(order_xy)):
        start_x, start_y = order_xy[ind]
        print("pts =",start_x, start_y, "ind =", ind)
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
            # RL_env_message(["set param", env_params])

            one_ep = []
            num_step = 0

            info = RL_start()
            one_ep.append(info["state"])
            reward_ep = []

            end_episode = False
            while not end_episode and num_step < max_step:
                info = RL_step()
                one_ep.append(info["state"])
                end_episode = info["isTerminal"]
                reward_ep.append(info["reward"])
                print(info)

                num_step += 1

                if num_step % 5000 == 0:
                    print("step", num_step)

            print(">>> Ep ends. Total num of step, ", RL_num_steps())
            RL_end()

            if type(gamma) == list:
                state, feature, ground_truth = preproc_cgw_data_multigamma(np.array(one_ep), gamma, tc, separate,
                                                         num_tile=num_tile, num_tiling=num_tiling)
            else:
                state, feature, ground_truth = preproc_cgw_data(np.array(one_ep), gamma, tc, separate,
                                                         num_tile=num_tile, num_tiling=num_tiling)
            del one_ep

            f_g = np.concatenate((state, feature, ground_truth), axis=1)
            try:
                all_eps = np.concatenate((all_eps, f_g), axis=0)
            except:
                print(all_eps.shape, f_g.shape)
                raise ValueError
            i += 1

    if with_reward:
        all_eps = all_eps[:-1]
        assert len(all_eps) == len(reward_ep)
        all_eps = np.concatenate((np.array(all_eps), np.array(reward_ep).reshape((-1, 1))), axis=1)

    # save feature and ground truth
    suc_prob = 1.0
    opt_prob = 0.0

    if tc:
        name = "random_data/fixed_env_suc_prob_" + str(suc_prob) + "/pw_noGoal_"+str(sep_title)+"TC" + str(num_tiling) + "x" + str(num_tile) + "_rewardScaled40_training_set_randomStart_" + \
                   str(int(opt_prob*100))+"opt_"+str(gamma)+"gamma_"+str(num_point)+"pts_x"+str(num_ep)+"_x"+str(max_step)
    else:
        name = "random_data/fixed_env_suc_prob_" + str(suc_prob) + "/pw_noGoal_raw_training_set_randomStart_" + \
                   str(int(opt_prob*100))+"opt_"+str(gamma)+"gamma_"+str(num_point)+"pts_x"+str(num_ep)+"_x"+str(max_step)

    if len(sys.argv) == 4:
        name += "_" + file_order
    np.save(name, np.array(all_eps))

def catch_collect_data():
    import environment.Catcher_dm as catch
    # generate data
    steps = 100000
    done_steps = 0
    state_dim = 10 * 5
    # data_array = np.zeros((steps,(2*state_dim)+3)) #s,a,s',r,gamma
    # data_array_sf = np.zeros((steps,(4*state_dim))) #s,rep(s),sf_g1(s),sf_g2(s)

    env = catch.Catch()
    actions = env.numAction()
    one_ep = []
    all_ep = []
    current_state = env.reset().observation.flatten()
    one_ep.append(current_state)

    while True:
        action = np.random.randint(0, actions)

        info = env.step(action)
        state = info.observation.flatten()
        reward = info.reward
        terminal = env._ball_y == env._paddle_y

        if terminal:
            all_ep.append(one_ep)
            one_ep = []
            current_state = env.reset().observation.flatten()
        else:
            current_state = state

        one_ep.append(current_state)
        done_steps += 1

        if done_steps == steps:
            if len(one_ep) != 0:
                all_ep.append(one_ep)
            break

    print("Done collecting data")

    # compute successor feature
    g1 = 0.998
    g2 = 0.8

    data_array_sf = np.zeros((0, (4 * state_dim)))  # s,rep(s),sf_g1(s),sf_g2(s)
    for ep_idx in range(len(all_ep)):
        one_ep = all_ep[ep_idx]

        state, feature, ground_truth = preproc_cgw_data_multigamma(np.array(one_ep), [g1, g2], False, False)
        f_g = np.concatenate((state, feature, ground_truth), axis=1)
        data_array_sf = np.concatenate((data_array_sf, f_g), axis=0)
        print(ep_idx)
    np.save("random_data/catcher_dm_noGoal_opt_" + str([g1, g2]) + "gamma_1pts_x1_x100000.npy", data_array_sf)

    print("Done saving data")


# sort_file()
# cgw_collect_data()
# pw_collect_data()
catch_collect_data()
# add_one_gamma()
# remove_one_gamma()
# NN_model_data()

# random_walk_with_goal()