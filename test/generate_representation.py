import sys
import os
crw = os.getcwd()
sys.path.append(crw +"/../")
print(sys.path)
from utils.auto_encoder_2branch import *
from utils.TileCoding import *
import numpy as np
import json
import time

jsonfile = "../parameters/continuous_gridworld.json"
continuous = 0 if jsonfile == "../parameters/gridworld.json" else 1
scale = 0

json_dat = open(jsonfile, 'r')
exp = json.load(json_dat)
json_dat.close()

gamma = exp["agent_params"]["nn_gamma"]
opt_prob = exp["agent_params"]["opt_prob"]
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

path = "../feature_model/distance_matrix_model-beta0.1_delta1.0_gamma[0.998, 0.8]_lr0.0001-0.0001_epoch100-100_scale0/"
file_name = exp["agent_params"]["nn_model_name"] + "_continuous"

num_tiling = exp["agent_params"]["num_tilings"]
num_tile = exp["agent_params"]["num_tiles"]

len_input = num_tiling * num_tile * 2
len_output = len_input * 2

nn = AETraining(len_input, num_node, num_feature, len_output, lr, learning_rate_rcvs=lr_rcvs,
                            weight_decay=wd, num_dn=num_dec_node, num_rec_node=num_rec_node, optimizer=optimizer,
                            dropout=dropout, beta=beta, delta=delta, legal = legal_v, continuous=True, num_tiling=num_tiling, num_tile=num_tile,
                             constraint=constraint)
nn.loading(path, file_name)
tc = TileCoding(1, num_tiling, num_tile)

test_size = 1000000
xpts = np.random.random(size=test_size).reshape((-1, 1))
ypts = np.random.random(size=test_size).reshape((-1, 1))
pts = np.concatenate((xpts, ypts), axis=1)

start = time.time()
for i in range(1, test_size+1):
    state = pts[i - 1]
    f = np.zeros(num_tile * num_tiling * 2)
    xind = tc.get_index([state[0]])
    yind = tc.get_index([state[1]])
    f[xind] = 1
    f[yind + num_tile * num_tiling] = 1
    [rep], _, _, _ = nn.test(f.reshape((1, -1)), len(f))

    if i %1000 == 0:
        print(i//1000,"th group", time.time() - start)
        start = time.time()