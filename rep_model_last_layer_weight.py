from utils.TileCoding import *
from utils.auto_encoder_2branch import *
from utils.ann_lrpaper import *

import numpy as np
import json

import torch
from torch import nn


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
lr_rcvs = lr
wd = exp["agent_params"]["nn_weight_decay"]
dropout = exp["agent_params"]["nn_dropout"]
num_epochs = exp["agent_params"]["nn_num_epochs"]
num_epochs_rcvs = num_epochs
batch_size = exp["agent_params"]["nn_batch_size"]

beta = exp["agent_params"]["nn_beta"]
delta = exp["agent_params"]["nn_delta"]
legal_v = exp["agent_params"]["nn_legal_v"]
constraint = exp["agent_params"]["nn_constraint"]

num_tiling = 32
num_tile = 4

# path = exp["agent_params"]["nn_model_path"]
file_name = exp["agent_params"]["nn_model_name"]+"_continuous"

tc = TileCoding(1, num_tiling, num_tile)

len_input = num_tile*num_tiling*2
len_output = num_tile*num_tiling * 2 * 2
ae = AETraining(2, num_node, num_feature, len_output, lr, learning_rate_rcvs=lr_rcvs, weight_decay=wd,
                     num_dn=num_dec_node, num_rec_node=num_rec_node, optimizer=optimizer, dropout=dropout, beta=beta,
                     delta=delta, legal=legal_v, continuous=1, num_tiling=num_tiling, num_tile=num_tile,
                     constraint=constraint)

ae.loading("./feature_model/", "feature_embedding_continuous_input[0.0, 1]_envSucProb1.0")

print(ae.net.de_layers[-1])
print(ae.net.de_layers[-1].weight.data.numpy().shape)
W = ae.net.de_layers[-1].weight.data.numpy()
np.save("last_layer_weight", W)
