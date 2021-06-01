from utils.TileCoding import *

new_rep = True
if new_rep:
    from utils.auto_encoder_2branch import *
else:
    from utils.ann_lrpaper_good import *

import numpy as np
import json

class GetLearnedRep:

    def __init__(self, len_input, num_node, num_feature, len_output, lr, lr_rcvs, wd, num_dec_node, num_rec_node,
                 optimizer, dropout, beta, delta, legal_v, continuous, num_tiling, num_tile, constraint,
                 model_path, file_name):

        if new_rep:
            self.nn = AETraining(len_input, num_node, num_feature, len_output, lr, learning_rate_rcvs=lr_rcvs, weight_decay=wd, num_dn=num_dec_node, num_rec_node=num_rec_node, optimizer=optimizer, dropout=dropout, beta=beta, delta=delta, legal=legal_v, continuous=continuous, num_tiling=num_tiling, num_tile=num_tile, constraint=constraint)
        else:
            self.nn = NN(len_input, num_node, 32, lr, 0.9)
        self.nn.loading(model_path, file_name)

    def state_representation(self, state):

        if new_rep:
            [rep], _, _, _ = self.nn.test(state.reshape((1, -1)), len(state))
        else:
            [rep], _ = self.nn.test(state.reshape((1, -1)))
        return rep

    def state_representation_batch(self, states):
        rep, _, _, _ = self.nn.test(states, states.shape[1])
        return rep
