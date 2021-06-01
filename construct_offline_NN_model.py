import utils.NN_model as nnm
import utils.tiles3 as tc
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils.get_learned_representation as glr
import utils.get_learned_state as gls

np.set_printoptions(precision=3)

num_tilings = 32
num_tiles = 4
tc_mem_size = 512
len_tc_state = num_tilings * num_tiles**2
iht = tc.IHT(tc_mem_size)

import json
jsonfile = "parameters/continuous_gridworld.json"
json_dat = open(jsonfile, 'r')
exp = json.load(json_dat)
agent_params = exp['agent_params']
model_new_encoder = glr.GetLearnedRep(2, agent_params["nn_nodes"], agent_params["nn_num_feature"],
                                       agent_params["nn_num_tilings"] * agent_params["nn_num_tiles"] * 2 * 2, agent_params["nn_lr"],
                                       agent_params["nn_lr"], agent_params["nn_weight_decay"],
                                       agent_params["nn_dec_nodes"], agent_params["nn_rec_nodes"],
                                       agent_params["optimizer"], agent_params["nn_dropout"],
                                       agent_params["nn_beta"], agent_params["nn_delta"],
                                       agent_params["nn_legal_v"], True,
                                       num_tiling=agent_params["nn_num_tilings"],
                                       num_tile=agent_params["nn_num_tiles"], constraint=True,
                                       model_path=agent_params["nn_model_path"],
                                       file_name=agent_params["nn_model_name"])
model_new_decoder = gls.GetLearnedState(agent_params["nn_num_feature"],
                                             agent_params["nn_nodes"],
                                             agent_params["nn_num_feature"],
                                             2,
                                             agent_params["nn_lr"],
                                             agent_params["nn_lr"],
                                             agent_params["nn_weight_decay"],
                                             agent_params["nn_dec_nodes"],
                                             agent_params["nn_rec_nodes"],
                                             agent_params["optimizer"],
                                             agent_params["nn_dropout"],
                                             agent_params["nn_beta"],
                                             agent_params["nn_delta"],
                                             agent_params["nn_legal_v"],
                                             True, num_tiling=agent_params["nn_num_tilings"],
                                             num_tile=agent_params["nn_num_tiles"], constraint=True,
                                             model_path=agent_params["nn_model_path"],
                                             file_name=agent_params["nn_model_name"] + "_seperateRcvs")


def representation_preprocess(state_arr):
    # rep_arr = np.zeros((len(state_arr), 32))
    # for i in range(len(state_arr)):
    #     rep_arr[i] = model_new_encoder.state_representation(state_arr[i])
    # return rep_arr
    rep_arr = model_new_encoder.state_representation_batch(state_arr)
    rep_arr = rep_arr / np.linalg.norm(rep_arr, axis=1).reshape((-1, 1))
    return rep_arr

def one_hot_action_preprocess(action):
    one_hot = np.zeros((len(action), 4))
    for i in range(len(action)):
        one_hot[i, int(action[i])] = 1
    return one_hot
def tile_coding_state(state_array):
    feature = np.zeros((len(state_array), len_tc_state))
    for idx in range(len(state_array)):
        ind = np.array(tc.tiles(iht, num_tilings, float(num_tiles) * np.array(state_array[idx])))
        feature[idx, :][ind] = 1
    return feature

def construct_offline_NN_model_seperate(node, lr, num_epochs, train_ep, batch_size, which_model="both", srg=False, gate={"f":"none","b":"none","r":"none"}):
    set_name = "random_data/fixed_env_suc_prob_1.0/cgw_training_set_x"+str(train_ep)

    forward_set = np.load(set_name + "_forward.npy", allow_pickle=True).astype(np.float32)
    backward_set = np.load(set_name + "_backward.npy", allow_pickle=True).astype(np.float32)

    # # Add illegal sequence
    # rand_s = np.random.random(size=5000).reshape((-1, 2))
    # print("\nAdding illegal seq. Before adding, training set shape", forward_set.shape, backward_set.shape)
    # for a in range(4):
    #
    #     forward_add = np.concatenate((rand_s, np.ones((len(rand_s), 1))*a, rand_s,
    #                                   np.zeros((len(rand_s), 1)), np.ones((len(rand_s), 1))*0.9), axis=1)
    #     backward_add = np.concatenate((np.ones((len(rand_s), 1))*a, rand_s, rand_s), axis=1)
    #     forward_set = np.concatenate((forward_set, forward_add), axis=0)
    #     backward_set = np.concatenate((backward_set, backward_add), axis=0)

    print("\nTraining set shape", forward_set.shape, backward_set.shape)

    one_hot_action = True
    representation = True
    if one_hot_action and representation:
        # # Train
        # f_one_hot = one_hot_action_preprocess(forward_set[:, 2])
        # # # normalize input to [-1, 1]
        # # forward_set[:, :2] = forward_set[:, :2] * 2 - 1
        # s = representation_preprocess(forward_set[:, :2])
        # sp = representation_preprocess(forward_set[:, 3:5])
        # rg = forward_set[:, 5:]
        # forward_set = np.concatenate((s, f_one_hot, sp, rg), axis=1)
        # # Train
        # b_one_hot = one_hot_action_preprocess(backward_set[:, 0])
        # # # normalize input to [-1, 1]
        # # backward_set[1:3] = backward_set[1:3] * 2 - 1
        # s = representation_preprocess(backward_set[:, 1: 3])
        # sb = representation_preprocess(backward_set[:, 3:])
        # backward_set = np.concatenate((b_one_hot, s, sb), axis=1)

        f_input = 32+4 #6 # s, one-hot a
        f_output = 32  #2 # sp
        b_input = 4+32 #6 # one-hot a, sp
        b_output = 32  #2 # s
        r_input = 32+4 #6 # s, one-hot a
        r_output = 2   #2 # r, g

    elif one_hot_action and not representation:
        f_one_hot = one_hot_action_preprocess(forward_set[:, 2])
        # # normalize input to [-1, 1]
        # forward_set[:, :2] = forward_set[:, :2] * 2 - 1
        s = forward_set[:, :2]
        sp = forward_set[:, 3:]
        forward_set = np.concatenate((s, f_one_hot, sp), axis=1)

        b_one_hot = one_hot_action_preprocess(backward_set[:, 0])
        # # normalize input to [-1, 1]
        # backward_set[1:3] = backward_set[1:3] * 2 - 1
        s = backward_set[:, 1: 3]
        sb = backward_set[:, 3:]
        backward_set = np.concatenate((b_one_hot, s, sb), axis=1)

        # print(forward_set.shape, forward_set[0])
        # print(backward_set.shape, backward_set[0])
        # exit()

        f_input = 6 # s, one-hot a
        f_output = 2 # sp
        b_input = 6 # one-hot a, sp
        b_output = 2 # s
        r_input = 6 # s, one-hot a
        r_output = 2 # r, g

    else:
        f_input = 3 # s, a
        f_output = 4 # sp
        b_input = 3 # a, sp
        b_output = 2 # s
        r_input = 3 # s, a
        r_output = 2 # r, g

    if srg:
        f_output += 2 # + r,g
        r_input = 0
        r_output = 0 # no rg models

    # tile coding feature
    # forward_set_s = tile_coding_state(forward_set[:, :2])
    # forward_set_sp = tile_coding_state(forward_set[:, f_input:f_input+2])
    # forward_set = np.concatenate((forward_set_s, forward_set[:, 2: f_input], forward_set_sp, forward_set[:, -2:]), axis=1)
    # backward_set_s = tile_coding_state(backward_set[:, b_input-2: b_input])
    # backward_set_sb = tile_coding_state(backward_set[:, -2:])
    # backward_set = np.concatenate((backward_set[:, :b_input-2], backward_set_s, backward_set_sb), axis=1)
    # f_input = len_tc_state + 4
    # f_output = len_tc_state
    # b_input = 4 + len_tc_state
    # b_output = len_tc_state
    # r_input = f_input
    # r_output = 2
    # print(forward_set.shape, backward_set.shape)

    if not srg:
        # Remove r, gamma from forward_set, construct rg set
        rg_set = np.concatenate((forward_set[:, :f_input], forward_set[:, -2:]), axis=1)
        forward_set = forward_set[:, :-2]

    print("batch size is", batch_size)

    if srg:
        model_path = "prototypes/offline_NN_separateSrg/"
    else:
        model_path = "prototypes/offline_NN_separate/"
        # model_path = "prototypes/"

    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    if srg:
        used_gate = str(gate.get("f")) + str(gate.get("b"))
    else:
        used_gate = str(gate.get("f")) + str(gate.get("b")) + str(gate.get("r"))

    model_name = "offlineNN_oneHotA" \
                 "_node" + str(node) + \
                 used_gate + \
                 "_lr" + str(lr) + \
                 "_epoch"+str(num_epochs) + \
                 "_batch"+str(batch_size) +\
                 "_trainEp"+str(train_ep)

    fnn = nnm.NNModel(f_input, node, f_output, lr, gate=gate.get("f"))
    bnn = nnm.NNModel(b_input, node, b_output, lr, gate=gate.get("b"))
    if not srg:
        rgnn = nnm.NNModel(r_input, node, r_output, lr, gate=gate.get("r"))

    # # Train
    # if which_model == "both":
    #     print("Both models")
    #     floss = fnn.training(forward_set, f_input, num_epochs, batch_size)
    #     if srg:
    #         fnn.saving(model_path, model_name + "_forwardSRG")
    #     else:
    #         fnn.saving(model_path, model_name+"_forward")
    #     np.save(model_path+model_name+"_floss", floss)
    #
    #     if not srg:
    #         rloss = rgnn.training(rg_set, r_input, num_epochs, batch_size)
    #         rgnn.saving(model_path, model_name+"_rg")
    #         np.save(model_path+model_name+"_rloss", rloss)
    #
    #     bloss = bnn.training(backward_set, b_input, num_epochs, batch_size)
    #     bnn.saving(model_path, model_name+"_backward")
    #     np.save(model_path+model_name+"_bloss", bloss)
    # elif which_model == "forward":
    #     print("Forward model only")
    #     floss = fnn.training(forward_set, f_input, num_epochs, batch_size)
    #     if srg:
    #         fnn.saving(model_path, model_name + "_forwardSrg")
    #     else:
    #         fnn.saving(model_path, model_name + "_forward")
    #     np.save(model_path + model_name + "_floss", floss)
    #
    # elif which_model == "backward":
    #     print("Backward model only")
    #     bloss = bnn.training(backward_set, b_input, num_epochs, batch_size)
    #     bnn.saving(model_path, model_name + "_backward")
    #     np.save(model_path + model_name + "_bloss", bloss)
    # elif which_model == "reward":
    #     if not srg:
    #         print("Reward+gamma model only")
    #         rloss = rgnn.training(rg_set, r_input, num_epochs, batch_size)
    #         rgnn.saving(model_path, model_name+"_rg")
    #         np.save(model_path+model_name+"_rloss", rloss)
    #     else:
    #         print("srg mode. reward and gamma are predicted by forward model")
    #         exit()
    # else:
    #     print("UNKNOWN MODEL")
    #     exit(-1)
    # print("Model saved in", model_path, model_name)


    # Test
    if (not os.path.isfile(model_path+model_name+"_floss.npy")) or \
            (not os.path.isfile(model_path+model_name+"_bloss.npy")): #or \
            #(not os.path.isfile(model_path+model_name+"_rloss.npy")):
        print("Log doesn't exit")
        return
    else:
        floss = np.load(model_path+model_name+"_floss.npy")
        bloss = np.load(model_path+model_name+"_bloss.npy")
        plt.figure()
        plt.plot(floss)
        plt.plot(bloss)
        if os.path.isfile(model_path+model_name+"_rloss.npy"):
            rloss = np.load(model_path+model_name+"_rloss.npy")
            plt.plot(rloss)
        plt.show()

    if srg:
        fnn.loading(model_path, model_name + "_forwardSrg")
        bnn.loading(model_path, model_name + "_backward")
    else:
        fnn.loading(model_path, model_name+"_forward")
        bnn.loading(model_path, model_name+"_backward")
        rgnn.loading(model_path, model_name+"_rg")

    test_name = "random_data/fixed_env_suc_prob_1.0/cgw_training_set_x1"
    ftest = np.load(test_name + "_forward.npy").astype(np.float32)
    btest = np.load(test_name + "_backward.npy").astype(np.float32)
    ftest_temp = np.copy(ftest)

    if one_hot_action and representation:
        f_one_hot = one_hot_action_preprocess(ftest[:, 2])
        if srg:
            ftest = np.concatenate((representation_preprocess(ftest[:, :2]), f_one_hot, representation_preprocess(ftest[:, 3:5]), ftest[:, 5:]), axis=1)
        else:
            ftest = np.concatenate((representation_preprocess(ftest[:, :2]), f_one_hot, representation_preprocess(ftest[:, 3:5])), axis=1)
        rgtest = np.concatenate((ftest[:, :f_input], ftest_temp[:, -2:]), axis=1)

        b_one_hot = one_hot_action_preprocess(btest[:, 0])
        btest = np.concatenate((b_one_hot, representation_preprocess(btest[:, 1:3]), representation_preprocess(btest[:, 3:])), axis=1)

    elif one_hot_action and not representation:
        f_one_hot = one_hot_action_preprocess(ftest[:, 2])
        if srg:
            ftest = np.concatenate((ftest[:, :2], f_one_hot, ftest[:, 3:]), axis=1)
        else:
            ftest = np.concatenate((ftest[:, :2], f_one_hot, ftest[:, 3:5]), axis=1)
        rgtest = np.concatenate((ftest[:, :f_input], ftest_temp[:, -2:]), axis=1)

        b_one_hot = one_hot_action_preprocess(btest[:, 0])
        btest = np.concatenate((b_one_hot, btest[:, 1:]), axis=1)


    # # tile coding state
    # ftest_s = tile_coding_state(ftest[:, :2])
    # ftest_sp = tile_coding_state(ftest[:, f_input:f_input + 2])
    # ftest = np.concatenate((ftest_s, ftest[:, 2: 6], ftest_sp, ftest_temp[:, -2:]), axis=1)
    # btest_s = tile_coding_state(btest[:, b_input - 2: b_input])
    # btest_sb = tile_coding_state(btest[:, -2:])
    # btest = np.concatenate((btest[:, :4], btest_s, btest_sb), axis=1)

    # ftest = ftest[:, :-2]
    fy, fl = fnn.test(ftest, f_input)
    by, bl = bnn.test(btest, b_input)
    if not srg:
        ry, rl = rgnn.test(rgtest, f_input)
    else:
        ry = fy[:, -2:]

    np.random.seed(1)
    test_idx = np.random.randint(0, len(ftest), size=10)
    for ti in test_idx:
        # print(ftest[ti, :f_input], "->", fy[ti], "(", ftest[ti, f_input:], ")")
        print(ftest[ti, :f_input], "->", fy[ti], "(", fy[ti][np.where(ftest[ti, f_input:] == 1)[0]], ")")
    # print(ftest[-1, :f_input], "->", fy[-1], "(", ftest[-1, f_input:], ")")
    print(ftest[-1, :f_input], "->", fy[-1], "(", fy[-1][np.where(ftest[-1, f_input:] == 1)[0]], ")")
    print("\n=============\n")
    for ti in test_idx:
        # print(btest[ti, :b_input], "->", by[ti], "(", btest[ti, b_input:], ")")
        print(btest[ti, :b_input], "->", by[ti], "(", by[ti][np.where(btest[ti, b_input:] == 1)[0]], ")")
    # print(btest[-1, :b_input], "->", by[-1], "(", btest[-1, b_input:], ")")
    print(btest[-1, :b_input], "->", by[-1], "(", by[-1][np.where(btest[-1, b_input:] == 1)[0]], ")")
    print("\n=============\n")
    for ti in test_idx:
        print(rgtest[ti, :f_input], "->", ry[ti], "(", rgtest[ti, f_input:], ")")
    print(rgtest[-1, :f_input], "->", ry[-1], "(", rgtest[-1, f_input:], ")")
    print()


    print("forward model loss", fl)
    print("backward model loss", bl)


    # plot
    color_list = ["red", "blue", "green", "orange"]

    fig = plt.figure(0)
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))

    np.random.seed(1)
    # ftest = np.random.random(size=20).reshape((-1, 2))
    # ftest = np.concatenate((ftest, np.array([[0.0, 0.02], [0.2, 1.0], [0.6, 0.2], [0.7, 0.8], [0.5, 0.8]])), axis=0)
    # ftest = np.array([[0.0, 0.02], [0.2, 1.0], [0.6, 0.2], [0.75, 0.8], [0.5, 0.8], [0.9, 1.0]])
    ftest = np.array([[0.0, 0.02], [0.2, 1.0], [0.6, 0.42], [0.75, 0.8], [0.5, 0.8], [0.9, 1.0]])
    ftest_temp = np.copy(ftest)
    if representation:
        ftest = model_new_encoder.state_representation_batch(ftest)
    for a in range(4):
        if one_hot_action:
            oha = np.zeros((len(ftest), 4))
            oha[:, a] = 1
            temp = np.concatenate((ftest, oha), axis=1)
        else:
            temp = np.concatenate((ftest, a * np.ones((len(ftest), 1))), axis=1)

        res = fnn.predict(temp)
        if srg:
            res = res[:, :-2]
        if representation:
            res = model_new_decoder.state_learned_batch(res / np.linalg.norm(res, axis=1).reshape((-1, 1)))
        plt.plot(res[:, 0], res[:, 1], ".", color=color_list[a])

    plt.plot(ftest_temp[:, 0], ftest_temp[:, 1], ".", color='black')
    plt.title("forward")

    fig = plt.figure(2)
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    plt.plot(res[:, 0], res[:, 1], ".", color="red")
    res = model_new_encoder.state_representation_batch(res)
    res = model_new_decoder.state_learned_batch(res / np.linalg.norm(res, axis=1).reshape((-1, 1)))
    plt.plot(res[:, 0], res[:, 1], ".", color="green")
    plt.title("check decoder")

    fig = plt.figure(1)
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    btest = ftest

    for a in range(4):
        if one_hot_action:
            oha = np.zeros((len(btest), 4))
            oha[:, a] = 1
            temp = np.concatenate((oha, btest), axis=1)
        else:
            temp = np.concatenate((a * np.ones((len(btest), 1)), btest), axis=1)

        res = bnn.predict(temp)
        if representation:
            res = model_new_decoder.state_learned_batch(res / np.linalg.norm(res, axis=1).reshape((-1, 1)))
        res = np.clip(res, 0, 1)
        plt.plot(res[:, 0], res[:, 1], ".", color=color_list[a])
    plt.plot(ftest_temp[:, 0], ftest_temp[:, 1], ".", color='black')
    plt.title("backward")

    plt.show()


if len(sys.argv) > 2:
    node = list(sys.argv[1].split(','))
    for i in range(len(node)):
        node[i] = int(node[i])
    lr = float(sys.argv[2])
    epoch = int(sys.argv[3])
    train_ep = int(sys.argv[4])
    which = str(sys.argv[5])
    batch_size = int(sys.argv[6])

    print("input:", node, lr, epoch, train_ep, batch_size, which)
    # construct_offline_NN_model(node, lr, epoch, train_ep, batch_size, which_model=which)
    gate = {"f": "None",
            "b": "None",
            "r": "Relu"}
    construct_offline_NN_model_seperate(node, lr, epoch, train_ep, batch_size, which_model=which, srg=False, gate=gate)

else:
    gate = {"f": "Relu",
            "b": "Relu",
            "r": "Relu"}
    for node in ["512","512,512,512"]:
    # for node in ["512,256,128"]:
        node = list(node.split(','))
        for i in range(len(node)):
            node[i] = int(node[i])
        for lr in [0.0005, 0.0001, 0.00025]:
            for train_ep in [500]:
                if train_ep == 500:
                    epoch_list = [200]
                elif train_ep == 100:
                    epoch_list = [500]
                for epoch in epoch_list:
                    for batch_size in [1024, 2048]:#[256, 512, 1024, 2048, 4096, 8192]:
                        print("input:", node, lr, epoch, train_ep, batch_size)
                        # construct_offline_NN_model(node, lr, epoch, train_ep, batch_size)
                        construct_offline_NN_model_seperate(node, lr, epoch, train_ep, batch_size, srg=False, gate=gate)
                        print("above input:", node, lr, epoch, train_ep, batch_size)
                        print("\n\n\n")
