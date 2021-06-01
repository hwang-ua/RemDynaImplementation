import sys
import os
crw = os.getcwd()
sys.path.append(crw +"/../")
print(sys.path)
os.chdir("../")


from utils.TileCoding import *
import numpy as np
import json
import utils.get_learned_representation as glr
import utils.get_learned_state as gls
import matplotlib.pyplot as plt
import matplotlib.patches as patches


jsonfile = "parameters/continuous_gridworld.json"
json_dat = open(jsonfile, 'r')
exp = json.load(json_dat)
json_dat.close()

gamma = exp["agent_params"]["nn_gamma"]
num_node = exp["agent_params"]["nn_nodes"]
num_dec_node = exp["agent_params"]["nn_dec_nodes"]
num_feature = exp["agent_params"]["nn_num_feature"]
num_rec_node = exp["agent_params"]["nn_rec_nodes"]
optimizer = exp["agent_params"]["optimizer"]
lr = exp["agent_params"]["nn_lr"]
lr_rcvs = lr  # exp["agent_params"]["nn_lr_rcvs"]
wd = exp["agent_params"]["nn_weight_decay"]
dropout = exp["agent_params"]["nn_dropout"]
num_epochs = exp["agent_params"]["nn_num_epochs"]
num_epochs_rcvs = num_epochs  # exp["agent_params"]["nn_num_epochs_rcvs"]
batch_size = exp["agent_params"]["nn_batch_size"]

beta = exp["agent_params"]["nn_beta"]
delta = exp["agent_params"]["nn_delta"]
legal_v = exp["agent_params"]["nn_legal_v"]
constraint = exp["agent_params"]["nn_constraint"]

num_tiling = 1
num_tile = 16

file_name = exp["agent_params"]["nn_model_name"]+"_continuous"

tc = TileCoding(1, num_tiling, num_tile)

len_input = num_tile*num_tiling*2
len_output = len_input * 2

num_input = 2
num_output = num_tiling * num_tile * 2 * 2

rep_model = glr.GetLearnedRep(num_input,
                              num_node,
                              num_feature,
                              num_output,
                              lr,
                              lr_rcvs,
                              wd,
                              num_dec_node,
                              num_rec_node,
                              optimizer,
                              dropout,
                              beta,
                              delta,
                              legal_v,
                              1,
                              num_tiling=num_tiling,
                              num_tile=num_tile,
                              constraint=constraint,
                              model_path="./feature_model_fixed_env/",
                              file_name="feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_legalv0_gamma[0.998, 0.8]_tc1x16_epoch1000_nfeature32_beta0.1")

rep_model_decoder = gls.GetLearnedState(2,
                                 num_node,
                                 num_feature,
                                 num_output,
                                 lr,
                                 lr,
                                 0.9,
                                 num_dec_node,
                                 num_rec_node,
                                 optimizer,
                                 0,
                                 0.1,
                                 1,
                                 0,
                                 True, num_tiling=num_tiling, num_tile=num_tile, constraint=True,
                                 model_path="./feature_model_fixed_env/",
                                 file_name="feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_legalv0_gamma[0.998, 0.8]_tc1x16_epoch1000_nfeature32_beta0.1_seperateRcvs")


def representation_same_y():
    plt.figure(0)
    x_list = np.linspace(0, 1, num=10)
    y = 0.9
    for i in range(10):
        x = x_list[i]
        print("chosen state", x, y)
        rep = rep_model.state_representation(np.array([x, y])).reshape((-1, 1))
        rep = rep / np.linalg.norm(rep)
        ax = plt.subplot(1, 10, i + 1)
        ax.set_title(str(x)[:4])
        plt.imshow(rep, cmap='Blues', vmin = -0.5, vmax = 0.5)
        plt.colorbar()
    plt.show(block=True)

def diff_representation():
    # states = np.array([[0.95, 0.9],
    #                    [0.60, 0.45],
    #                    [0.0, 0.85],
    #                    [0.75, 0.9],
    #                    [0.75, 0.9],
    #                    [0.75, 0.9]])
    #
    # next_states = np.array([[0.95, 0.95],
    #                         [0.60, 0.4],
    #                         [0.0, 0.9],
    #                         [0.7, 0.9],
    #                         [0.7, 0.9],
    #                         [0.7, 0.9]])
    #
    # test_states = np.array([[0.85, 0.9],
    #                         [0.55, 0.45],
    #                         [0.3, 0.95],
    #                         [0.75, 0.8],
    #                         [0.72, 0.7],
    #                         [0.68, 0.8]])

    color = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'black', 'grey']

    prototypes = {
        0: np.array([[0.95, 0.9], [0.95, 0.95]]),
        1: np.array([[0.60, 0.45], [0.60, 0.4]]), # between walls
        2: np.array([[0.0, 0.85], [0.0, 0.9]]),
        3: np.array([[0.75, 0.9], [0.7, 0.9]]), # in goal area
        4: np.array([[0.4, 0.9], [0.45, 0.9]]), # left side of wall
        5: np.array([[0.55, 0.45], [0.55, 0.4]]), # between walls
    }
    testset = {
        0: np.array([[0.85, 0.9], [0.95, 0.8], [0.97, 0.9]]),
        1: np.array([[0.65, 0.45], [0.63, 0.45]]),
        2: np.array([[0.3, 0.95], [0.05, 0.95], [0.03, 0.85]]),
        3: np.array([[0.75, 0.8], [0.72, 0.7], [0.68, 0.8], [0.78, 0.9]]),
        4: np.array([[0.45, 0.85], [0.4, 0.87]]),
        5: np.array([[0.51, 0.43], [0.49, 0.43]]),
    }

    plt.figure()
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((0.5, 0), 0.2, 0.4, fill=None))
    currentAxis.add_patch(patches.Rectangle((0.5, 0.6), 0.2, 0.4, fill=None))
    for id in prototypes.keys():
        proto = prototypes[id]
        all_t = testset[id]

        s = proto[0]
        sp = proto[1]
        print("\n", s, sp)

        phi_s = rep_model.state_representation(s)
        phi_s /= np.linalg.norm(phi_s)

        phi_sp = rep_model.state_representation(sp)
        phi_sp /= np.linalg.norm(phi_sp)
        print("prototype", np.linalg.norm(phi_sp - phi_s))

        plt.plot([s[0], sp[0]], [s[1], sp[1]], '.', c=color[id])
        plt.plot([s[0], sp[0]], [s[1], sp[1]], '-', c=color[id])

        for t in all_t:
            print("test", t, end=", ")
            phi_t = rep_model.state_representation(t)
            phi_t /= np.linalg.norm(phi_t)
            similarity = np.dot(phi_s, phi_t)
            print("similarity", similarity, end=", ")
            if similarity > 0.9:
                phi_tp = phi_t + phi_sp - phi_s
                phi_tp /= np.linalg.norm(phi_tp)
                phi_tp = np.clip(phi_tp, -1, 1)
                print("sampling", np.linalg.norm(phi_tp - phi_t))
                tp = rep_model_decoder.state_learned(phi_tp)

            else:
                tp = t
                print("skip this state")


            plt.plot([t[0], tp[0]], [t[1], tp[1]], 'x', c=color[id])
            plt.plot([t[0], tp[0]], [t[1], tp[1]], '-', c=color[id])

    plt.show()

# representation_same_y()
diff_representation()