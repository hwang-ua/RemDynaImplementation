from utils.TileCoding import *
from utils.recover_state import *

import numpy as np
import json

# jsonfile = "parameters/continuous_gridworld.json"
# json_dat = open(jsonfile, 'r')
# exp = json.load(json_dat)
# json_dat.close()
#
# gamma = exp["agent_params"]["nn_gamma"]
# num_node = exp["agent_params"]["nn_nodes"]
# num_dec_node = exp["agent_params"]["nn_dec_nodes"]
# num_feature = exp["agent_params"]["nn_num_feature"]
# num_rec_node = exp["agent_params"]["nn_rec_nodes"]
# optimizer = exp["agent_params"]["optimizer"]
# lr = exp["agent_params"]["nn_lr"]
# lr_rcvs = lr  # exp["agent_params"]["nn_lr_rcvs"]
# wd = exp["agent_params"]["nn_weight_decay"]
# dropout = exp["agent_params"]["nn_dropout"]
# num_epochs = exp["agent_params"]["nn_num_epochs"]
# num_epochs_rcvs = num_epochs  # exp["agent_params"]["nn_num_epochs_rcvs"]
# batch_size = exp["agent_params"]["nn_batch_size"]
#
# beta = exp["agent_params"]["nn_beta"]
# delta = exp["agent_params"]["nn_delta"]
# legal_v = exp["agent_params"]["nn_legal_v"]
# constraint = exp["agent_params"]["nn_constraint"]
#
# num_tiling = 32#exp["agent_params"]["nn_num_tilings"]
# num_tile = 4#exp["agent_params"]["nn_num_tiles"]
#
# # path = exp["agent_params"]["nn_model_path"]
# file_name = exp["agent_params"]["nn_model_name"]+"_continuous"
#
# tc = TileCoding(1, num_tiling, num_tile)
#
# len_input = num_tile*num_tiling*2
# len_output = len_input * 2

class GetLearnedState:
    def __init__(self, len_input, num_node, num_feature, len_output, lr, lr_rcvs, wd, num_dec_node, num_rec_node,
                 optimizer, dropout, beta, delta, legal_v, continuous, num_tiling, num_tile, constraint,
                 model_path, file_name, default=True):

        self.nn = RecvState(num_feature, num_rec_node, len_output, 0.9, 0.9)

        self.default = default

        self.nn.loading(model_path, file_name)

        if model_path == "feature_model_graph":
            self.one_hot = True


    def state_learned(self, state, real_state=None):
        [rep] = self.nn.test2(state.reshape((1, -1)))

        if not self.default:
            rep[0] = (rep[0] + 1)/2.0
            rep[1] = (rep[1] + 1)/2.0

            if self.one_hot:
                idx = np.where(rep == np.max(rep))[0]
                rep = np.zeros(len(rep))
                rep[idx] = 1

        return rep


    def state_learned_batch(self, state, real_state=None):
        rep = self.nn.test2(state)
        return rep