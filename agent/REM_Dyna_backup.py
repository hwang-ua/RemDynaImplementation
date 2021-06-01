#!/usr/bin/python3
import time
import math
import multiprocessing as mp
import numpy as np
import pickle as pkl
import utils.tiles3 as tc
import utils.TileCoding as utc
# import utils.REM_model as rem
# import utils.REM_model_first as rem
# import utils.REM_model_last as rem
# import utils.REM_model_random as rem
# import utils.REM_model_kdt as rem
LinearModel = False
if LinearModel:
    # import utils.REM_model_kdt_realCov_llm as rem
    import utils.REM_model_kdt_realCov_llm_allactions as rem
    # import utils.REM_model_kdt_realCov_flm as rem
    # import utils.REM_model_kdt_realCov_remflm as rem
else:
    import utils.REM_model_kdt_realCov as rem
# import utils.KernModelupdate as rem
# import utils.auto_encoder_2branch as atec
import utils.get_learned_representation as glr
import utils.get_learned_state as gls
import utils.get_offline_NN as gon

import matplotlib.pyplot as plt
import os

import torch
from torch import nn
import torch.nn.functional as F

np.set_printoptions(precision=3)

OLD_REM = 0
CHECK_DIST = 1
SINGLE_REP = 2
REPVF_RAWMODEL_CHECKDIST = 3
TCREPVF_RAWMODEL_CHECKDIST = 4
BIASREPVF_RAWMODEL_CHECKDIST = 5
BIASTCREPVF_RAWMODEL_CHECKDIST = 6
BIASTCREPVF_REPMODEL = 7
BIASTCREPVF_REPMODEL_CHECKDIST = 8
SINGLE_REP_CHECKDIST = 9
SINGLE_NORMREP = 10
SINGLE_NORMREP_FIXCOV = 11
TCREPVF_NORMREPMODEL_FIXCOV = 12
BIASTCREPVF_NORMREPMODEL_FIXCOV = 13
TCREPVF_NORMREPMODEL = 14
BIASTCREPVF_NORMREPMODEL = 15
NORMREPVF_RAWMODEL = 16
TCRAWVF_NORMREPMODEL = 17
TCRAWVF_NORMREPMODEL_FIXCOV = 18

TCREPVF_RAWMODEL = 19

NN_MODEL = 20
NN_REP_MODEL = 21

DQN = True
BACKPLANNING = False

raw_model_mode_list = [OLD_REM,
                       CHECK_DIST,
                       REPVF_RAWMODEL_CHECKDIST,
                       TCREPVF_RAWMODEL_CHECKDIST,
                       BIASREPVF_RAWMODEL_CHECKDIST,
                       BIASTCREPVF_RAWMODEL_CHECKDIST,
                       NORMREPVF_RAWMODEL,
                       TCREPVF_RAWMODEL]

raw_vf_mode_list = [OLD_REM,
                    CHECK_DIST,
                    # TCRAWVF_NORMREPMODEL_FIXCOV,
                    # TCRAWVF_NORMREPMODEL,
                    NN_MODEL,
                    NN_REP_MODEL]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# dqn
class DQNNet(torch.nn.Module):
    def __init__(self, dims):
        super(DQNNet, self).__init__()
        self.linear1 = torch.nn.Linear(dims[0], dims[1])
        self.linear2 = torch.nn.Linear(dims[1], dims[2])
        # self.linear3 = torch.nn.Linear(dims[2], dims[3])
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)
        # nn.init.kaiming_normal_(self.linear3.weight)

        # self.weights = []
        # for i in range(len(dims) -1):
        #     w = torch.nn.Linear(dims[i], dims[i+1])
        #     nn.init.kaiming_normal_(w.weight)
        #     self.weights.append(w)
        #     self.weights.append(nn.SELU())
        # self.weights = self.weights[:-1] # remove the gate on the last layer
        # self.net = nn.Sequential(*self.weights)

    def forward(self, x):
        x = F.selu(self.linear1(x))
        # x = self.linear1(x)
        # x = F.selu(self.linear2(x))
        x = self.linear2(x)
        # x = self.linear3(x)

        # self.net(x)
        return x

class BufferControl():
    def __init__(self, length):
    # def __init__(self, length, model):
        self.b_length = length
        self.b_empty = [i for i in range(self.b_length)]
        self.b_filled = []
        # self.model = model
        self.b_filled_length = 0

    def insert(self):
        if len(self.b_empty) == 0:
            index = self.b_filled[0]
            self.b_filled = self.b_filled[1:]
            self.b_filled.append(index)
        else:
            index = self.b_empty[0]
            if index not in self.b_filled:
                self.b_filled.append(index)
            self.b_empty = self.b_empty[1:]

        # print("---insert---")
        # print(self.b_filled[:5], self.b_filled[-5:])
        # print(self.b_empty[:5], self.b_empty[-5:], "\n")

        # if len(self.b_filled) == 1:
        #     print("Filled at: ", self.model.t)

        self.b_filled_length = len(self.b_filled)

        return index

    def remove(self, index):
        # self.b_filled.remove(index)
        self.b_empty.append(index)

        # print("---removed---")
        # print(self.b_filled[:5], self.b_filled[-5:])
        # print(self.b_empty[:5], self.b_empty[-5:], "\n")

        # if len(self.b_filled) == 0:
        #     print("Empty at: ", self.model.t)

    def force_remove(self, index):
        self.b_filled.remove(index)
        self.b_empty.append(index)
        self.b_filled_length = len(self.b_filled)

    def get_filled(self):
        return self.b_filled

    def get_filled_length(self):
        return self.b_filled_length

    def get_empty(self):
        return self.b_empty

class REM_Dyna:

    # Default values
    def __init__(self):

        self.old_weight_update = False

        self.rem_type = "random"
        self.alpha = 0.1
        self.epsilon = 0.1
        self.gamma = 0.9
        self.num_planning = 10
        self.num_branching = 1

        self.dim_state = 1
        self.action_mode = "discrete"
        self.num_action = 1
        self.action_list = [i for i in range(self.num_action)]

        self.tc_mem_size = 1024
        self.iht = tc.IHT(self.tc_mem_size)

        self.num_tilings = 1
        self.num_tiles = 1
        # self.tc = utc.TileCoding(self.dim_state, self.num_tilings, self.num_tiles, self.num_action)

        self.learning_mode = OLD_REM

        self.len_s_feature = self.num_tilings * self.num_tiles * self.dim_state
        self.len_s_rep = 1
        self.len_sa_feature = self.len_s_feature * self.num_action
        self.rep_model = None

        self.len_buffer = 1
        self.pri_thrshd = 0.1

        self.num_near = 3
        self.add_prot_limit = 0.05
        self.similarity_limit = 35.0
        self.norm_diff = 0
        # self.model_params = {"kscale":0.05}
        # self.model = rem.REM_Model(self.dim_state, self.num_near, self.add_prot_limit, self.model_params,
        #                            self.learning_mode, self.similarity_limit, self.norm_diff)
        # self.model = rem.KernModel(self.dim_state, 1000, 100, 0.01, 0.0001, 1)

        self.weight = []
        self.buffer = np.zeros((self.len_buffer, self.dim_state*2+4))#{"pri_time": np.zeros((2, self.len_buffer)), "sequence": {}}
        self.b_control = BufferControl(self.len_buffer)
        # self.b_control = BufferControl(self.len_buffer, self.model)

        self.last_state = None
        self.last_action = None

        self.state = None
        self.action = None
        self.check_time = []
        self.check_total_time = np.zeros(6)

        self.learning = False

        self.num_node = None
        self.num_dec_node = None
        self.num_feature = None
        self.num_rec_node = None
        self.optimizer = None
        self.lr = None
        self.wd = None
        self.dropout = None
        self.num_epochs = None
        self.batch_size = None
        self.beta = None
        self.delta = None
        self.legal_v = None
        self.constraint = None
        self.path = None
        self.file_name = None

        self.div_norm = None
        return

    def set_param(self, param):
        self.graph = param["graph"]

        self.gui = param["gui"]
        self.offline = param["offline"]

        self.alg = param["alg"]#'Q' or Sarsa
        self.traces_lambda = param["lambda"]
        self.opt_mode = param["opt_mode"]

        self.always_add_prot = param["always_add_prot"]

        self.rem_type = param["rem_type"]
        self.alpha = param["alpha"]
        self.epsilon = param["epsilon"]
        self.gamma = param["agent_gamma"]
        self.num_planning = param["num_planning"]
        if self.num_planning > 50:
            self.planning_steps = 10
        else:
            self.planning_steps = 1

        self.num_branching = param["num_branching"]

        self.dim_state = param["dim_state"] # 2
        self.action_mode = param["action_mode"]
        self.num_action = param["num_action"]
        self.action_list = [i for i in range(self.num_action)]

        self.num_near = param["num_near"]

        self.learning_mode = param["remDyna_mode"]

        self.div_actBit = None
        self.div_norm = None

        # dqn
        if DQN:
            # mp.set_start_method('spawn')
            self.minibatch = 1

            # binning the representation

            self.binning = 1  # 1 == no binning, original rep
            self.arrange_ind = np.zeros((self.minibatch, 32))
            for i in range(self.arrange_ind.shape[1]):
                self.arrange_ind[:, i] = self.binning * i

            self.dqn_count = 0
            self.dqn_c = param["dqn_c"]
            if self.learning_mode == 0:
                node = [2*self.binning, 512, 4]
            elif self.learning_mode == 17:
                node = [32*self.binning, 512, 4]
            else:
                print("Unknown mode for DQN")
                exit()
            self.dqn_learn = DQNNet(node).to(device)
            self.dqn_target = DQNNet(node).to(device)
            print(self.dqn_learn)
            print(self.dqn_target)
            self.dqn_learn_optimizer = torch.optim.RMSprop(self.dqn_learn.parameters(), lr = param["alpha"])
            self.dqn_loss = torch.nn.MSELoss()


        # 2， 9
        if self.learning_mode == SINGLE_REP or self.learning_mode == SINGLE_REP_CHECKDIST:
            if self.graph:
                exit(-1)

            self.len_s_feature = 2
            self.len_s_rep = param["nn_num_feature"]
            self.dim_state = param["nn_num_feature"]

            if self.opt_mode == 4:
                self.div_norm = 1

        # 10, 11
        elif self.learning_mode == SINGLE_NORMREP or self.learning_mode == SINGLE_NORMREP_FIXCOV:
            if self.graph:
                exit(-1)

            self.len_s_feature = 2
            self.len_s_rep = param["nn_num_feature"]
            self.dim_state = param["nn_num_feature"]

            if self.opt_mode == 4:
                self.div_norm = 1

        # 12, 14
        elif self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == TCREPVF_NORMREPMODEL:
            if self.graph:
                exit(-1)

            self.len_s_feature = 2
            self.rep_tilings = 1
            self.rep_tiles = 8
            self.rep_mem_size = 8
            self.len_s_rep = self.rep_mem_size * param["nn_num_feature"]
            self.dim_state = param["nn_num_feature"]
            self.rep_iht = tc.IHT(self.rep_mem_size)
            if self.opt_mode == 4:
                self.div_actBit = 32

        # 13, 15
        elif self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL:
            if self.graph:
                exit(-1)

            self.len_s_feature = 2
            self.rep_tilings = 1
            self.rep_tiles = 8
            self.rep_mem_size = 8
            self.len_s_rep = self.rep_mem_size * param["nn_num_feature"] + 1
            self.dim_state = param["nn_num_feature"]
            self.rep_iht = tc.IHT(self.rep_mem_size)
            if self.opt_mode == 4:
                self.div_actBit = 33

        # 3
        elif self.learning_mode == REPVF_RAWMODEL_CHECKDIST:
            if self.graph:
                exit(-1)

            self.len_s_feature = 2
            self.len_s_rep = param["nn_num_feature"]

            if self.opt_mode == 4:
                self.div_norm = 1

        # 16
        elif self.learning_mode == NORMREPVF_RAWMODEL:
            if self.graph:
                exit(-1)

            self.len_s_feature = 2
            self.len_s_rep = param["nn_num_feature"]

            if self.opt_mode == 4:
                self.div_norm = 1

        # 4, 19
        elif self.learning_mode == TCREPVF_RAWMODEL_CHECKDIST or \
                self.learning_mode == TCREPVF_RAWMODEL:
            if self.graph:
                exit(-1)

            self.rep_tilings = 1
            self.rep_tiles = 8
            self.rep_mem_size = 8
            self.len_s_feature = 2
            self.len_s_rep = self.rep_mem_size * param["nn_num_feature"]
            self.rep_iht = tc.IHT(self.rep_mem_size)
            if self.opt_mode == 4:
                self.div_actBit = 32

        # 5
        elif self.learning_mode == BIASREPVF_RAWMODEL_CHECKDIST:
            if self.graph:
                exit(-1)

            self.len_s_feature = 2
            self.len_s_rep = param["nn_num_feature"] + 1

            if self.opt_mode == 4:
                self.div_norm = 1

        # 6
        elif self.learning_mode == BIASTCREPVF_RAWMODEL_CHECKDIST:
            if self.graph:
                exit(-1)

            self.rep_tilings = 1
            self.rep_tiles = 8
            self.rep_mem_size = 8
            self.len_s_feature = 2
            self.len_s_rep = self.rep_mem_size * param["nn_num_feature"] + 1
            self.rep_iht = tc.IHT(self.rep_mem_size)
            if self.opt_mode == 4:
                self.div_actBit = 33

        # 7
        elif self.learning_mode == BIASTCREPVF_REPMODEL:
            if self.graph:
                exit(-1)

            self.rep_tilings = 1
            self.rep_tiles = 8
            self.rep_mem_size = 8
            self.len_s_feature = 2
            self.len_s_rep = self.rep_mem_size * param["nn_num_feature"] + 1
            self.dim_state = param["nn_num_feature"]
            self.rep_iht = tc.IHT(self.rep_mem_size)

            if self.opt_mode == 4:
                self.div_actBit = 33

        # 8
        elif self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST:
            if self.graph:
                exit(-1)

            self.rep_tilings = 1
            self.rep_tiles = 8
            self.rep_mem_size = 8
            self.len_s_feature = 2
            self.len_s_rep = self.rep_mem_size * param["nn_num_feature"] + 1
            self.dim_state = param["nn_num_feature"]
            self.rep_iht = tc.IHT(self.rep_mem_size)

            if self.opt_mode == 4:
                self.div_actBit = 33

        # 17, 18
        elif self.learning_mode == TCRAWVF_NORMREPMODEL \
                or self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
            if self.graph:
                self.tc = utc.TileCoding(2, 1, param["nn_num_tiles"])
                self.nn_num_tile = param["nn_num_tiles"]
                self.nn_num_tiling = param["nn_num_tilings"]
                assert self.nn_num_tiling == 1
                self.len_s_feature = param["nn_num_tiles"] ** 2
                self.len_s_rep = param["nn_num_tiles"] ** 2
                self.dim_state = param["nn_num_feature"]
            else:
                self.num_tilings = 32
                self.num_tiles = 4
                self.len_s_feature = 2
                self.tc_mem_size = 512
                self.len_s_rep = self.tc_mem_size
                self.dim_state = param["nn_num_feature"]
                self.iht = tc.IHT(self.tc_mem_size)
                if self.opt_mode == 4:
                    self.div_actBit = self.num_tilings

        elif self.learning_mode == NN_MODEL:
            if self.graph:
                exit(-1)
            else:
                self.num_tilings = 32
                self.num_tiles = 4
                self.len_s_feature = 2
                self.tc_mem_size = 512
                self.len_s_rep = self.tc_mem_size
                self.iht = tc.IHT(self.tc_mem_size)
                self.div_actBit = self.num_tilings
                self.div_norm = None
                self.offline_nn = gon.GetOfflineNN()
        # 21
        elif self.learning_mode == NN_REP_MODEL:
            if self.graph:
                exit(-1)
            else:
                self.num_tilings = 32
                self.num_tiles = 4
                self.len_s_feature = 2
                self.tc_mem_size = 512
                self.len_s_rep = self.tc_mem_size
                self.iht = tc.IHT(self.tc_mem_size)
                self.div_actBit = self.num_tilings
                self.div_norm = None
                self.offline_nn = gon.GetOfflineRepNN()

        # 0, 1
        else:
            if self.graph:
                self.tc = utc.TileCoding(2, 1, param["nn_num_tiles"])
                self.nn_num_tile = param["nn_num_tiles"]
                self.nn_num_tiling = param["nn_num_tilings"]
                assert self.nn_num_tiling == 1
                self.len_s_feature = param["nn_num_tiles"] ** 2
                self.len_s_rep = param["nn_num_tiles"] ** 2
                self.dim_state = self.len_s_feature

            else:
                self.num_tilings = 32
                self.num_tiles = 4
                self.tc_mem_size = 512
                self.len_s_feature = self.tc_mem_size
                self.len_s_rep = self.len_s_feature
                self.iht = tc.IHT(self.tc_mem_size)
                if self.opt_mode == 4:
                    self.div_actBit = self.num_tilings

        if self.learning_mode not in raw_model_mode_list:
            if self.graph:
                self.len_output = param["nn_num_tilings"] * param["nn_num_tiles"] ** 2
            else:
                self.len_output = 2 #param["nn_num_tilings"] * param["nn_num_tiles"] * 2 * 2
            self.rep_model_decoder = gls.GetLearnedState(self.len_s_rep,
                                                         param["nn_nodes"],
                                                         param["nn_num_feature"],
                                                         self.len_output,
                                                         param["nn_lr"],
                                                         param["nn_lr"],
                                                         param["nn_weight_decay"],
                                                         param["nn_dec_nodes"],
                                                         param["nn_rec_nodes"],
                                                         param["optimizer"],
                                                         param["nn_dropout"],
                                                         param["nn_beta"],
                                                         param["nn_delta"],
                                                         param["nn_legal_v"],
                                                         True, num_tiling=param["nn_num_tilings"], num_tile=param["nn_num_tiles"], constraint=True,
                                                         model_path=param["nn_model_path"],
                                                         file_name=param["nn_model_name"]+"_seperateRcvs_illegal")

        self.len_sa_feature = self.len_s_rep * self.num_action

        if self.div_actBit is not None:
            self.alpha = param["alpha"] / np.sqrt(float(self.div_actBit))
        else:
            self.alpha = param["alpha"]

        if self.learning_mode == CHECK_DIST or \
                self.learning_mode == SINGLE_REP or \
                self.learning_mode == SINGLE_NORMREP or\
                self.learning_mode == SINGLE_NORMREP_FIXCOV or \
                self.learning_mode == REPVF_RAWMODEL_CHECKDIST or \
                self.learning_mode == TCREPVF_RAWMODEL_CHECKDIST or \
                self.learning_mode == BIASREPVF_RAWMODEL_CHECKDIST or \
                self.learning_mode == BIASTCREPVF_RAWMODEL_CHECKDIST or \
                self.learning_mode == BIASTCREPVF_REPMODEL or \
                self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST or \
                self.learning_mode == SINGLE_REP_CHECKDIST or \
                self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == TCREPVF_NORMREPMODEL or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL or \
                self.learning_mode == NORMREPVF_RAWMODEL or \
                self.learning_mode == TCRAWVF_NORMREPMODEL or \
                self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == TCREPVF_RAWMODEL or \
                self.learning_mode == NN_REP_MODEL:
            self.num_node = param["nn_nodes"]
            self.num_dec_node = param["nn_dec_nodes"]
            self.num_feature = param["nn_num_feature"]
            self.num_rec_node = param["nn_rec_nodes"]
            self.optimizer = param["optimizer"]
            self.lr = param["nn_lr"]
            self.lr_rcvs = self.lr
            self.wd = param["nn_weight_decay"]
            self.dropout = param["nn_dropout"]
            self.num_epochs = param["nn_num_epochs"]
            self.batch_size = param["nn_batch_size"]
            self.beta = param["nn_beta"]
            self.delta = param["nn_delta"]
            self.legal_v = param["nn_legal_v"]
            self.constraint = param["nn_constraint"]
            if self.graph:
                self.num_input = param["nn_num_tilings"] * param["nn_num_tiles"]**2
                self.num_output = param["nn_num_tilings"] * param["nn_num_tiles"]**2 * 2
            else:
                self.num_input = 2
                self.num_output = param["nn_num_tilings"] * param["nn_num_tiles"] * 2 * 2
            self.continuous = True
            self.constraint = True
            self.rep_model = glr.GetLearnedRep(self.num_input,
                                               self.num_node,
                                               self.num_feature,
                                               self.num_output,
                                               self.lr,
                                               self.lr_rcvs,
                                               self.wd,
                                               self.num_dec_node,
                                               self.num_rec_node,
                                               self.optimizer,
                                               self.dropout,
                                               self.beta,
                                               self.delta,
                                               self.legal_v,
                                               self.continuous,
                                               num_tiling=param["nn_num_tilings"],
                                               num_tile=param["nn_num_tiles"],
                                               constraint=self.constraint,
                                               model_path=param["nn_model_path"],
                                               file_name=param["nn_model_name"])

            # # This block calculates covariance from training set
            # import training_set_covariance as tsc
            # data_file = "random_data/fixed_env_suc_prob_1.0/cgw_noGoal_separateTC32x4_training_set_randomStart_0opt_[0.998, 0.8]gamma_1pts_x1"
            # get_cov = tsc.TrainingSetCov(self.rep_model, self.num_feature, data_file+".npy")
            # real_cov = get_cov.get_representation_cov()
            # print(list(np.diag(real_cov)))
            # print("\n", real_cov)
            # np.save(data_file+"_normCovM", real_cov)
            # exit(0)

        # else:
            # self.tc = utc.TileCoding(self.dim_state, self.num_tilings, self.num_tiles, 1)

        self.momentum = param["momentum"]
        self.rms = param["rms"]

        if param["init_weight"] == "0":
            # self.weight = np.zeros(self.len_sa_feature)
            #self.weight = np.zeros((self.len_sa_feature))

            self.weight = nn.Linear(self.len_sa_feature, 1, bias=False)
            self.weight.weight.data.fill_(0.)

            """
            0: AdamO
            1: MarthaO
            2: FullO
            3: RMS+AMS
            4: SGD
            """
            if self.opt_mode == 0:
                ams = False
            elif self.opt_mode == 1:
                self.momentum = 0.0
                ams = False
            elif self.opt_mode == 2:
                ams = True
            elif self.opt_mode == 3:
                self.momentum = 0.0
                ams = True
            elif self.opt_mode == 4:
                print("SGD")
            else:
                print("Unknown optimizer Mode")
                exit(1)

            if self.opt_mode != 4:
                self.weight_optimizer = torch.optim.Adam(self.weight.parameters(), lr=self.alpha, amsgrad=ams,
                                                         betas=(self.momentum, self.rms))
            else:
                self.weight_optimizer = torch.optim.SGD(self.weight.parameters(), lr=self.alpha)

            if self.alg == 'Sarsa':
                self.traces = torch.from_numpy(np.zeros((self.len_sa_feature))).float()

            if self.old_weight_update == True:
                self.weight = np.zeros(self.len_sa_feature)

        elif param["init_weight"] == "1":
            self.weight = np.ones(self.len_sa_feature)
            #self.weight = np.ones((self.len_sa_feature))
        else:
            print("HAVEN't BEEN DONE YET")
            exit(-1)

        self.len_buffer = param["len_buffer"]
        self.buffer = np.zeros((self.len_buffer, self.dim_state * 2 + 4))
        self.b_control = BufferControl(self.len_buffer)

        self.pri_thrshd = param["pri_thrshd"]
        self.adpt_thrshd = False
        if self.pri_thrshd == -1:
            self.pri_thrshd = 0
            self.adpt_thrshd = True


        self.add_prot_limit = param["add_prot_limit"]
        self.similarity_limit = param["similarity_limit"]
        self.model_params = param["model_params"]
        self.model_params["num_action"] = self.num_action

        if self.learning_mode not in raw_model_mode_list:
            self.model_params["rep_model_decoder"] = self.rep_model_decoder

        self.norm_diff = param["rbf_normalize_diff"]

        if self.offline:
            print("===============")
            print("Offline learning")
            if self.learning_mode in raw_model_mode_list:
                print("Loading raw model")
                if LinearModel:
                    with open('prototypes/rem-GCov-100p-randomwalk/local_linear_model/mode0_trainingSetNormCov0.025_addProtLimit-0.025_baseOnState/model.pkl', 'rb') as f:
                        self.model = pkl.load(f)
                else:
                    with open('prototypes/rem-GCov-100p-randomwalk/rem_model/mode0_onPolicyCov_addProtLimit-0.2_kscale1e-07/model.pkl', 'rb') as f:
                        self.model = pkl.load(f)
            else:
                print("Loading rep model")
                if LinearModel:
                    print("right")
                    with open('prototypes/rem-GCov-100p-randomwalk/local_linear_model/mode10_trainingSetNormCov0.025_addProtLimit-3.0_baseOnState/model.pkl', 'rb') as f:
                    # with open('prototypes/rem-GCov-100p-randomwalk-llm/model.pkl', 'rb') as f:
                        self.model = pkl.load(f)

                    # print("number of prototypes forward", len(self.model.same_a_ind_forward[0]), len(self.model.same_a_ind_forward[1]), len(self.model.same_a_ind_forward[2]), len(self.model.same_a_ind_forward[3]))
                    # print("number of prototypes reverse", len(self.model.same_a_ind_reverse[0]), len(self.model.same_a_ind_reverse[1]), len(self.model.same_a_ind_reverse[2]), len(self.model.same_a_ind_reverse[3]))

                    # self.model.offline = True
                    # self.offline = False

                else:
                    # with open('prototypes/rem-GCov-100p-randomwalk/mode10 fixCov'+str(self.model_params["cov"])+'/model.pkl', 'rb') as f:
                    with open('prototypes/rem-GCov-100p-randomwalk/rem_model/mode10_onPolicyCov_addProtLimit-65.0_kscale1e-07/model.pkl', 'rb') as f:
                        self.model = pkl.load(f)

                    # print("number of prototypes", len(self.model.same_a_ind[0]), len(self.model.same_a_ind[1]), len(self.model.same_a_ind[2]), len(self.model.same_a_ind[3]))

                    # sig_prot_inv2 = np.load("feature_model_fixed_env/feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature4_beta1.0_inv_transition.npy")
                    # self.model.sig_prot_inv = np.zeros((self.model.seq_dim,self.model.seq_dim))
                    # self.model.sig_prot_inv[:4,:4] = sig_prot_inv2[:4,:4]
                    # self.model.sig_prot_inv[5:,5:] = sig_prot_inv2[4:,4:]

                    # # offline + learning
                    # self.model.offline = True
                    # self.offline = False

                    # from utils.recover_state import RecvState
                    # self.model.temp_decoder = RecvState(4, [32, 64, 128, 256, 512], 2, 0.9, 0.9)
                    # self.model.temp_decoder.loading("./feature_model/new_env_model/", "feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature4_beta1.0_seperateRcvs")
                    #
                    # self.model.actions_map={0:"up",1:"down",2:"right",3:"left"}

            # self.model.sample_mode = "onPolicyCov"
            # if self.learning_mode == 0:
            #     if LinearModel:
            #         self.model.sample_mode = "euclidean"
            #         self.model.const_euclidean_tree()
            #     self.model.cov = self.model_params["cov"]
            #     self.model.const_fix_cov_tree()
            #
            # elif self.learning_mode == 17 or self.learning_mode == 10:
            #     self.model.sample_mode = "fixedCov"
            #     self.model.cov = self.model_params["cov"]
            #     self.model.const_fix_cov_tree()
            # else:
            #     print("Unknown mode")
            #     exit()
            # print("number of prototypes", len(self.model.same_a_ind[0]), len(self.model.same_a_ind[1]), len(self.model.same_a_ind[2]), len(self.model.same_a_ind[3]))

            self.model.kscale = self.model_params["kscale"]

            if self.model_params["cov"] != 0:
                self.model.cov = self.model_params["cov"]
                self.model.const_fix_cov_tree()
                self.model.sample_mode = "fixedCov"
            else:
                self.model.sample_mode = "onPolicyCov"

            if LinearModel:
                self.model.sample_single_neighbour = self.model_params["sample_single_neighbour"]
                self.model.sample_weighted_mean = self.model_params["sample_weighted_mean"]

            self.offline = False
            self.model.add_prot_limit = param["add_prot_limit"]
            self.model.num_near = self.num_near
            self.model.sampling_limit = self.model_params["sampling_limit"]
            self.model.learning_mode = self.learning_mode
            self.model.rep_model = self.rep_model
            # self.model.check_sampled = True
            print("===============")
        else:
            if self.learning_mode == OLD_REM:
                self.model = rem.REM_Model(self.dim_state, self.num_near, self.add_prot_limit, self.model_params,
                                       self.learning_mode, self.similarity_limit, self.norm_diff)
            # elif self.learning_mode == TCRAWVF_NORMREPMODEL \
            #         or self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
            #     self.model = rem.REM_Model(param["nn_num_feature"], self.num_near, self.add_prot_limit, self.model_params,
            #                                self.learning_mode, self.similarity_limit, self.norm_diff,
            #                                rep_model=self.rep_model)
            else:
                self.model = rem.REM_Model(self.dim_state, self.num_near, self.add_prot_limit, self.model_params,
                                           self.learning_mode, self.similarity_limit, self.norm_diff,
                                           rep_model=self.rep_model)

        # self.b_control = BufferControl(self.len_buffer, self.model)

        # from utils.recover_state import RecvState
        # self.temp_decoder = RecvState(32, [128, 256, 512], 2, 0.9, 0.9)
        # self.temp_decoder.loading("./feature_model/", "feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_seperateRcvs")

        return

    """
    Input: [x, y]
    Return: action
    """
    def start(self, state):
        if self.graph:
            state = self._change_to_graph(state)

        if self.learning_mode == SINGLE_REP or \
                self.learning_mode == SINGLE_NORMREP or\
                self.learning_mode == SINGLE_NORMREP_FIXCOV or \
                self.learning_mode == BIASTCREPVF_REPMODEL or \
                self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST or \
                self.learning_mode == SINGLE_REP_CHECKDIST or \
                self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == TCREPVF_NORMREPMODEL or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL or \
                self.learning_mode == TCRAWVF_NORMREPMODEL or \
                self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
            state = self._state_representation(state)

        # if use nonlinear Q, normalize state to be between -1 and 1
        if DQN and self.learning_mode == OLD_REM:
            state = self._input_DQN(state)

        self.state = state
        # self.action = self._policy(state)
        self.action = self._policy(self.state, isprint=False, dqn=DQN)
        self.check_total_time = np.zeros(6)
        if self.alg == 'Sarsa':
            self.traces = torch.from_numpy(np.zeros((self.len_sa_feature))).float()
        return self.action

    """
    Input: int, [x, y]
    Return: action
    """
    def step(self, reward, state, end_of_ep=False):
        self.reward = reward
        if self.graph:
            state = self._change_to_graph(state)

        # if state[0] > 0.7:
        #     print(state)
        if reward != 0:
            self.learning = True
            gamma = 0
        else:
            gamma = self.gamma

        if self.learning_mode == SINGLE_REP or \
                self.learning_mode == SINGLE_NORMREP or \
                self.learning_mode == SINGLE_NORMREP_FIXCOV or \
                self.learning_mode == BIASTCREPVF_REPMODEL or \
                self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST or \
                self.learning_mode == SINGLE_REP_CHECKDIST or \
                self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == TCREPVF_NORMREPMODEL or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL or \
                self.learning_mode == TCRAWVF_NORMREPMODEL or \
                self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
            state = self._state_representation(state)

        if DQN and self.learning_mode == OLD_REM:
            state = self._input_DQN(state)

        # update variables
        self.last_state = self.state
        self.last_action = self.action
        self.state = state
        self.reward = reward

        other_info = None
        if self.learning:
            # use Q-learning to update weight
            if DQN:
                # self.dqn_update_learn(np.array([self._seq_to_array(self.last_state, self.last_action, self.state, reward, gamma, 0)]))
                self.dqn_update_learn(self._seq_to_array(self.last_state, self.last_action, self.state, reward, gamma, 0).reshape((1, -1)))
            else:
                self._update_weight(self.last_state, self.last_action, self.state, reward, gamma, self.alpha)

            if self.rem_type != "random_BufferOnly" and not self.offline and not self.learning_mode in [NN_MODEL, NN_REP_MODEL]:
                # update model
                # if self.learning_mode == TCRAWVF_NORMREPMODEL or self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
                #     last_rep = self._state_representation(self.last_state)
                #     rep = self._state_representation(self.state)
                #     self.model.add2Model(last_rep, self.last_action, rep, reward, gamma)
                # else:
                #     self.model.add2Model(self.last_state, self.last_action, self.state, reward, gamma)
                self.model.add2Model(self.last_state, self.last_action, self.state, reward, gamma)

            # get tde
            tde = self._get_tde(self.last_state, self.last_action, self.state, reward, gamma, dqn=DQN)

            # insert S,A into buffer
            if self.rem_type == "pri" or self.rem_type == "pri_pred":
                if np.abs(tde) > self.pri_thrshd:
                    self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma, np.abs(tde)+self.pri_thrshd)
            else:
                self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma, np.abs(tde) + self.pri_thrshd)

            # self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma, np.abs(tde)+self.pri_thrshd)

            # print("Before planning buffer", self.b_control.get_filled())

            # planning
            if not self.gui:
                # indexs, seqs = self._sample_seqs_from_buffer(self.num_planning)
                # for i in range(min(len(indexs), self.num_planning)):
                #     self._single_planning(self.alpha, self.num_planning, indexs[i], seqs[i])

                # for i in range(self.num_planning):
                #     # print("Buffer status:",len(self.b_control.get_filled()))
                #     indexs, seqs = self._sample_seqs_from_buffer(1)
                #     self._single_planning(self.alpha, self.num_planning, indexs[0], seqs[0])

                # dqn
                if DQN:
                    for _ in range(self.num_planning):#(1):
                        if self.rem_type == "random_BufferOnly":
                            indexs, seqs = self._sample_seqs_from_buffer(self.minibatch)#(self.num_planning):
                            self.dqn_update_learn(seqs)
                        elif self.rem_type == "random" or self.rem_type == "pri_pred":
                            planned = 0
                            predictions = []
                            # have_bad_planning = False
                            while planned < self.minibatch:
                                indexs, seqs = self._sample_seqs_from_buffer(1)
                                index, seq = indexs[0], seqs[0]
                                new_sequence = self._dqn_single_forward_plan(index, seq)

                                if new_sequence is not None:
                                    predictions.append(new_sequence)
                                    planned += 1
                                # if planned == 0:
                                #     have_bad_planning = True
                                #     print("0 planning now", seq)

                            # if have_bad_planning:
                            #     print("there was bad planning. Planned =", planned, seq, new_sequence, "\n")

                            self.dqn_update_learn(np.array(predictions))

                            if self.rem_type == "pri_pred" or self.rem_type == "pri":
                                preds = []
                                for seq in predictions:

                                    last_state, last_action, state, reward, gamma, _ = self._array_to_seq(seq)
                                    self._update_priority(index, last_state, last_action, state, reward, gamma)

                                    if self.rem_type == "pri_pred":
                                        predecessor = self._dqn_single_backward_plan(last_state)
                                        preds += predecessor

                        self.dqn_count += 1
                        if self.dqn_count % self.dqn_c == 0:
                            # print("sync")
                            self.synchronize_networks()

                else:
                    planning_done_steps = 0
                    while planning_done_steps < int(self.num_planning/self.planning_steps) and self.b_control.get_filled_length() > 0:
                        indexs, seqs = self._sample_seqs_from_buffer(self.planning_steps)
                        for i in range(self.planning_steps):
                            # did_planning = self._single_planning((self.alpha/np.sqrt(self.num_planning)), self.planning_steps, indexs[i], seqs[i])
                            did_planning = self._single_planning(self.alpha, self.planning_steps, indexs[i], seqs[i])
                            if did_planning:
                                planning_done_steps += 1
            else:
                other_info = {"plan": []}
                # for i in range(self.num_planning):
                #     indexs, seqs = self._sample_seqs_from_buffer(1)
                #     other_info["plan"].append(self._single_planning(self.alpha, self.num_planning, indexs[0], seqs[0]))

                planning_done_steps = 0
                while planning_done_steps < int(self.num_planning/self.planning_steps) and self.b_control.get_filled_length() > 0:
                    # indexs, seqs = self._sample_seqs_from_buffer(self.planning_steps)
                    for i in range(self.planning_steps):
                        indexs, seqs = self._sample_seqs_from_buffer(self.planning_steps)
                        info, did_planning = self._single_planning(self.alpha, self.num_planning, indexs[i], seqs[i])
                        # info, did_planning = self._single_planning(self.alpha, self.num_planning, indexs[0], seqs[0])
                        other_info["plan"].append(info)
                        if did_planning:
                            planning_done_steps += 1

                if self.learning_mode not in raw_vf_mode_list:
                    other_info["buffer"] = self._decode_all_proto(self.buffer[self.b_control.get_filled()])
                else:
                    other_info["buffer"] = self.buffer[self.b_control.get_filled()]
                # if self.learning_mode not in raw_model_mode_list:
                #     other_info["protos"] = self._decode_all_proto(self.model.get_protos())
                # else:
                #     other_info["protos"] = self.model.get_protos()
                other_info["agent_q"] = self._check_q(state)

        elif not self.learning and self.always_add_prot:
            if self.rem_type != "random_BufferOnly" and not self.offline and not self.learning_mode in [NN_MODEL, NN_REP_MODEL]:
                # if self.learning_mode == TCRAWVF_NORMREPMODEL or self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
                #     last_rep = self._state_representation(self.last_state)
                #     rep = self._state_representation(self.state)
                #     self.model.add2Model(last_rep, self.last_action, rep, reward, gamma)
                # else:
                #     self.model.add2Model(self.last_state, self.last_action, self.state, reward, gamma)
                self.model.add2Model(self.last_state, self.last_action, self.state, reward, gamma)

            tde = self._get_tde(self.last_state, self.last_action, self.state, reward, gamma, dqn=DQN)

            # insert S,A into buffer
            if self.rem_type == "pri" or self.rem_type == "pri_pred":
                if np.abs(tde) > self.pri_thrshd:
                    self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma, np.abs(tde)+self.pri_thrshd)
            else:
                self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma, np.abs(tde) + self.pri_thrshd)
            # self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma, np.abs(tde) + self.pri_thrshd)

        # choose new action
        # print("step", state, end=" ")
        self.action = self._policy(self.state, isprint=False, dqn=DQN)
        # print("")
        return self.action, other_info

    # # used for parallel planning. Not done yet
    # def _collect_minibatch(self, predict):
    #     # start = time.time()
    #     indexs, seqs = self._sample_seqs_from_buffer(1)
    #     sample = self._single_planning(self.alpha, 1, indexs[0], seqs[0])
    #     if sample is not None:
    #         last_state, state, reward, gamma, last_action = sample
    #         new_sequence = self._seq_to_array(last_state, last_action, state, reward, gamma, 0)
    #         # predict.put(new_sequence)
    #         return new_sequence
    #     # print("function time", time.time() - start)

    """
    Input: int, [x, y]
    Return: None
    """
    def end(self, reward, state):
        print("End of Episode.", self.state, self.action, state)
        self.step(reward, state, end_of_ep=True)
        return

    """
    Choose action according to given policy
    Input: [x, y]
    Return: action
    """
    def _policy(self, state, isprint=False, dqn=False):
        if self.action_mode == "discrete":
            if not self.learning:
                return np.random.choice(self.action_list)
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_list)
            else:
                if dqn:
                    # with torch.no_grad():
                    #     state = torch.from_numpy(state).float().to(device)
                    # q_values = self.dqn_learn(state).detach().cpu()
                    # _, action = torch.max(q_values, 0)
                    # action = action.data.item()
                    if self.binning != 1:
                        state = self._binning_rep(state)
                    with torch.no_grad():
                        state = torch.from_numpy(np.array(state)).float().to(device)
                        q_values = self.dqn_learn(state).detach().cpu().numpy()

                    action = self._break_tie(q_values)
                    if isprint:
                        print(q_values, action)
                    # if self.dqn_count % 100 == 0:
                    #     print("**", state, q_values, action)
                    del state, q_values
                    return action
                else:
                    return self._max_action(state, isprint)
        elif self.action_mode == "continuous":
            print("NOT DONE YET")
            return
        else:
            print("UNKNOWN ACTION MODE")
        return

    """
    Choose the optimal action
    Input: [x, y]
    Return: optimal action
    """
    def _max_action(self, state, isprint=False):
        all_choices = []
        for a in self.action_list:
            feature = self._feature_construction(state, a)
            if self.old_weight_update == True:
                all_choices.append(np.dot(self.weight, feature))
            else:
                all_choices.append(np.dot(self.weight.weight.data, feature))
        valid_index = self._break_tie(all_choices)
        if isprint and self.learning:
            print(all_choices, valid_index, end=" ")
        return valid_index

    """
    Break tie fairly
    Input: qvalue
    Return: optimal action
    """
    def _break_tie(self, xarray):
        max_v = np.max(xarray)
        valid_choices = np.where(xarray == max_v)[0]
        try:
            return np.random.choice(valid_choices)
        except:
            print(valid_choices)
            print(self.weight)
            print(xarray)
            exit(-1)
    """
    Generate learned representation
    Input: [x, y]
    Return: state representation
    """
    def _state_representation(self, state):
        rep = self.rep_model.state_representation(np.array(state))
        if (self.learning_mode == SINGLE_NORMREP or
                 self.learning_mode == SINGLE_NORMREP_FIXCOV or
                 self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or
                 self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV or
                 self.learning_mode == TCREPVF_NORMREPMODEL or
                 self.learning_mode == BIASTCREPVF_NORMREPMODEL or
                 self.learning_mode == TCRAWVF_NORMREPMODEL or
                 self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV):
            rep = rep / float(np.linalg.norm(rep))
        return rep

    def _input_DQN(self, state):
        return np.array(state) * 2 - 1

    """
    Generate feature for learning value function
    Input: [x, y], action
    Return: (s, a)-feature
    """
    def _feature_construction(self, state, action, isprint=False):
        if DQN:
            print("Using DQN. It shouldn't be in feature construction")
        #
        if self.learning_mode == SINGLE_REP or \
                self.learning_mode == SINGLE_REP_CHECKDIST:
            feature = np.zeros(self.len_sa_feature)
            feature[self.len_s_rep * int(action): self.len_s_rep * (int(action) + 1)] = state # rep

        elif self.learning_mode == SINGLE_NORMREP or \
                self.learning_mode == SINGLE_NORMREP_FIXCOV:
            feature = np.zeros(self.len_sa_feature)
            state = state / float(np.linalg.norm(state))
            feature[self.len_s_rep * int(action): self.len_s_rep * (int(action) + 1)] = state

        # 12, 14
        elif self.learning_mode ==  TCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == TCREPVF_NORMREPMODEL:
            feature = np.zeros(self.len_sa_feature)
            state = (state / float(np.linalg.norm(state)) + 1) / 2.0
            for d in range(len(state)):
                ind = np.array(tc.tiles(self.rep_iht, self.rep_tilings, [float(self.rep_tiles) * state[d]]))
                feature[self.len_s_rep * int(action) + d * self.rep_mem_size + ind] = 1

        # 13, 15
        elif self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL:
            feature = np.zeros(self.len_sa_feature)
            state = (state / float(np.linalg.norm(state)) + 1) / 2.0
            for d in range(len(state)):
                ind = np.array(tc.tiles(self.rep_iht, self.rep_tilings, [float(self.rep_tiles) * state[d]]))
                feature[self.len_s_rep * int(action) + d * self.rep_mem_size + ind] = 1
            feature[self.len_s_rep * (int(action) + 1) - 1] = 1

        elif self.learning_mode == REPVF_RAWMODEL_CHECKDIST:
            feature = np.zeros(self.len_sa_feature)
            rep = self._state_representation(state)
            feature[self.len_s_rep * int(action): self.len_s_rep * (int(action) + 1)] = rep

        elif self.learning_mode == TCREPVF_RAWMODEL_CHECKDIST or \
                self.learning_mode == TCREPVF_RAWMODEL:
            feature = np.zeros(self.len_sa_feature)
            rep = self._state_representation(state)
            rep = (rep / float(np.linalg.norm(rep)) + 1) / 2.0
            for d in range(len(rep)):
                ind = np.array(tc.tiles(self.rep_iht, self.rep_tilings, [float(self.rep_tiles) * rep[d]]))
                feature[self.len_s_rep * int(action) + d * self.rep_mem_size + ind] = 1
                # print("a =", action, "d =", d, "ind =", self.len_s_rep * int(action) + d * self.rep_mem_size + ind)

        elif self.learning_mode == BIASREPVF_RAWMODEL_CHECKDIST:
            feature = np.zeros(self.len_sa_feature)
            rep = self._state_representation(state)
            feature[self.len_s_rep * int(action): self.len_s_rep * (int(action) + 1) - 1] = rep
            feature[self.len_s_rep * (int(action) + 1) - 1] = 1

        elif self.learning_mode == BIASTCREPVF_RAWMODEL_CHECKDIST:
            feature = np.zeros(self.len_sa_feature)
            rep = self._state_representation(state)
            rep = (rep / float(np.linalg.norm(rep)) + 1) / 2.0
            for d in range(len(rep)):
                ind = np.array(tc.tiles(self.rep_iht, self.rep_tilings, [float(self.rep_tiles) * rep[d]]))
                feature[self.len_s_rep * int(action) + d * self.rep_mem_size + ind] = 1
            feature[self.len_s_rep * (int(action) + 1) - 1] = 1

        elif  self.learning_mode == BIASTCREPVF_REPMODEL:
            feature = np.zeros(self.len_sa_feature)
            state = (state / float(np.linalg.norm(state)) + 1) / 2.0
            for d in range(len(state)):
                ind = np.array(tc.tiles(self.rep_iht, self.rep_tilings, [float(self.rep_tiles) * state[d]]))
                feature[self.len_s_rep * int(action) + d * self.rep_mem_size + ind] = 1
            feature[self.len_s_rep * (int(action) + 1) - 1] = 1

        elif self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST:
            feature = np.zeros(self.len_sa_feature)
            state = (state / float(np.linalg.norm(state)) + 1) / 2.0
            for d in range(len(state)):
                ind = np.array(tc.tiles(self.rep_iht, self.rep_tilings, [float(self.rep_tiles) * state[d]]))
                feature[self.len_s_rep * int(action) + d * self.rep_mem_size + ind] = 1
            feature[self.len_s_rep * (int(action) + 1) - 1] = 1

        elif self.learning_mode == NORMREPVF_RAWMODEL:
            feature = np.zeros(self.len_sa_feature)
            rep = self._state_representation(state)
            rep = rep / float(np.linalg.norm(rep))
            feature[self.len_s_rep * int(action): self.len_s_rep * (int(action) + 1)] = rep

        elif self.learning_mode == TCRAWVF_NORMREPMODEL \
                or self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
            # if self.graph:
            #     feature = np.zeros(self.len_sa_feature)
            #     feature[self.len_s_feature * int(action): self.len_s_feature * (int(action) + 1)] = state  # one-hot encoding
            #     return feature
            state = self.rep_model_decoder.state_learned(state)
            feature = np.zeros(self.len_sa_feature)
            state = np.clip(np.array(state), 0.0, 1.0)
            ind = np.array(tc.tiles(self.iht, self.num_tilings, float(self.num_tiles) * np.array(state)))
            feature[self.len_s_rep * int(action) + ind] = 1

        elif self.learning_mode == NN_MODEL:
            feature = np.zeros(self.len_sa_feature)
            state = np.clip(np.array(state), 0.0, 1.0)
            try:
                ind = np.array(tc.tiles(self.iht, self.num_tilings, float(self.num_tiles) * np.array(state)))
            except:
                print("feature_construction")
                print(state)
                print()
            feature[self.len_s_rep * int(action) + ind] = 1

        elif self.learning_mode == NN_REP_MODEL:
            feature = np.zeros(self.len_sa_feature)
            state = np.clip(np.array(state), 0.0, 1.0)
            try:
                ind = np.array(tc.tiles(self.iht, self.num_tilings, float(self.num_tiles) * np.array(state)))
            except:
                print("feature_construction")
                print(state)
                print()
            feature[self.len_s_rep * int(action) + ind] = 1

        else:
            if self.graph:
                feature = np.zeros(self.len_sa_feature)
                feature[self.len_s_feature * int(action): self.len_s_feature * (int(action) + 1)] = state  # one-hot encoding
                return feature

            feature = np.zeros(self.len_sa_feature)
            state = np.clip(np.array(state), 0.0, 1.0)
            ind = np.array(tc.tiles(self.iht, self.num_tilings, float(self.num_tiles) * np.array(state)))
            if isprint:
                if self.reward == 1:
                    print(state, action, "\n", self.len_s_rep * int(action) + ind, "reward=1 \n")
                elif self.learning:
                    print(state, action, "\n", self.len_s_rep * int(action) + ind, "\n")

            feature[self.len_s_rep * int(action) + ind] = 1
        return feature

    """
    Update weight for learning value function
    Input: [x, y]-last, action-last, [x,y], reward, gamma, lr
    Return: None
    """
    def _update_weight(self, last_state, last_action, state, reward, gamma, alpha, current_action = None):
        # if self.learning:
        #     print("Weight", last_state, last_action, state, reward, gamma)
        last_feature = self._feature_construction(last_state, last_action, isprint=False)
        last_feature_torch = torch.from_numpy(last_feature).float()

        if self.alg == 'Sarsa':
            self.traces *= (gamma*self.traces_lambda)
            self.traces += last_feature_torch

        if current_action is None:
            feature = self._feature_construction(state, self._max_action(state))
        else:
            feature = self._feature_construction(state, current_action)

        tde = self._td_error(last_feature, feature, reward, gamma, self.weight)
        if np.abs(tde) > 0:
            self.learning = True

        if self.old_weight_update == True:
            self.weight += alpha * tde * last_feature

        else:
            self.weight_optimizer.zero_grad()
            loss = 0.0*torch.norm(self.weight.weight)#self.weight)
            loss.backward(torch.FloatTensor(np.asarray([0.0])),retain_graph=True)

            if self.div_norm:
                if self.alg == 'Sarsa':
                    self.weight.weight.grad -= (tde/torch.norm(last_feature_torch))*self.traces
                else:
                    self.weight.weight.grad -= (tde/torch.norm(last_feature_torch))*last_feature_torch
            else:
                if self.alg == 'Sarsa':
                    self.weight.weight.grad -= tde*self.traces
                else:
                    self.weight.weight.grad -= tde*last_feature_torch

            self.weight_optimizer.step()

            # if self.learning:
            #     print("after update weight TDE =", self._td_error(last_feature, feature, reward, gamma, self.weight), last_state, state, reward, gamma)
            #     exit()

            loss = 0.0*torch.norm(self.weight.weight)#self.weight)
            loss.backward()

        # if self.learning:
            # print(self.weight.weight)
            # self._max_action(last_state, isprint=True)
        return tde

    # def _update_weight(self, last_state, last_action, state, reward, gamma, alpha):
    #     last_feature = self._feature_construction(last_state, last_action)
    #     feature = self._feature_construction(state, self._max_action(state))
    #     tde = self._td_error(last_feature, feature, reward, gamma, self.weight)
    #     if tde > 0:
    #         self.learning = True
    #     if self.div_norm:
    #         alpha = alpha / np.linalg.norm(last_feature)
    #         # print("div norm", alpha, np.linalg.norm(last_feature))
    #     self.weight += alpha * tde * last_feature
    #     return

    """
    Calculate TD error given feature
    Input: feature-last, feature, reward, gamma, weight)
    Return: TD-error
    """
    def _td_error(self, last_feature, feature, reward, gamma, weight):
        # tde = reward + gamma * np.sum(weight[feature]) - np.sum(weight[last_feature])
        # tde = reward + gamma * np.dot(weight, feature) - np.dot(weight, last_feature)

        if self.old_weight_update == True:
            tde = reward + gamma * np.dot(feature, weight) \
                  - np.dot(last_feature, weight)
        else:
            tde = reward + gamma * np.dot(feature, weight.weight.data.reshape(-1)) \
                  - np.dot(last_feature, weight.weight.data.reshape(-1))
        return tde

    """
    Calculate TD error given state (x,y) or representation
    Input: [x, y]-last, action-last, [x, y], reward, gamma
    Return: TD-error
    """
    def _get_tde(self, last_state, last_action, state, reward, gamma, dqn=False):
        if dqn:
            # last_state = np.array(last_state).reshape((-1, self.dim_state))
            last_state = last_state.reshape((-1, self.dim_state))
            state = state.reshape((-1, self.dim_state))

            if self.binning != 1:
                last_state = self._binning_rep(last_state)
                state = self._binning_rep(state)

            with torch.no_grad():
                last_state = torch.from_numpy(last_state).float().to(device)
                last_action = torch.from_numpy(last_action.reshape(-1, 1)).float().type(torch.LongTensor).to(device)
                s_values = self.dqn_learn(last_state)
                prediction = s_values.gather(1, last_action).view((-1)).detach().cpu().numpy()
                #
                # gamma = torch.from_numpy(gamma).float().to(device)
                # r = torch.from_numpy(r).float().to(device)
                state = torch.from_numpy(state).float().to(device)
                sp_values = self.dqn_target(state)
                sp_q_value = sp_values.data.max(1)[0].detach().cpu().numpy()

                y = reward + gamma * sp_q_value
                tde = prediction - y
                tde = tde[0] if len(tde) == 1 else tde

            del last_state, last_action, sp_values, prediction, state, sp_q_value

        else:
            last_feature = self._feature_construction(last_state, last_action)
            feature = self._feature_construction(state, self._max_action(state))
            tde = self._td_error(last_feature, feature, reward, gamma, self.weight)
        return tde

    """
    Update sample's priority in buffer
    Input: index, [x, y]-last, action-last, [x, y], reward, gamma
    Return: None
    """
    def _update_priority(self, index, last_state, last_action, state, reward, gamma):
        new_pri = np.abs(self._get_tde(last_state, last_action, state, reward, gamma, dqn=DQN))
        # print("update", last_state, last_action, state, "new-pri="+str(new_pri))
        # print("Priority update:",self.buffer[index, -1], new_pri)

        # if new_pri > self.pri_thrshd:
        #     self.buffer[index, -1] = new_pri#+self.pri_thrshd
        # else:
        #     self.buffer[index, -1] = -10
        #     self.b_control.remove(index)
        #     # print("removing sample", last_state, state, reward, gamma)

        self.buffer[index, -1] = new_pri + self.pri_thrshd
        if new_pri < self.pri_thrshd:
            self.b_control.remove(index)

        if self.adpt_thrshd:
            self.pri_thrshd = np.mean(self.buffer[self.b_control.get_filled()][:, -1])
        return

    """
    Insert sample into buffer
    Input: [x, y]-last, action-last, [x, y], reward, gamma, TD-error
    Return: None
    """
    def _insert_seq(self, last_state, last_action, state, reward, gamma, tde):
        new_sequence = self._seq_to_array(last_state, last_action, state, reward, gamma, tde)
        index = self.b_control.insert()
        self.buffer[index] = new_sequence
        # print("insert", last_state, state, tde, index)
        if self.adpt_thrshd:
            self.pri_thrshd = np.mean(self.buffer[self.b_control.get_filled()][:, -1])
        return

    """
    Planning step
    Input: lr, number of planning, index in buffer, sasprg-array
    Return: dictionary
    """
    def _single_planning(self, alpha, n, index, seq):
        did_planning = False
        if seq is not None:
            preds = []
            succs = []

            # The following block is only used for NN offline model
            if self.learning_mode in [NN_MODEL, NN_REP_MODEL]:
                last_state, last_action, state, reward, gamma, pri = self._array_to_seq(seq)
                if self.learning_mode == NN_REP_MODEL:
                    last_state = self.rep_model.state_representation(last_state)
                    # state = self.rep_model.state_representation(state)
                sample = self.offline_nn.forward_prediction(last_state, last_action)
                if sample is not None:
                    did_planning = True
                    state, reward, gamma = sample
                    state = self.rep_model_decoder.state_learned(state/np.linalg.norm(state))
                    # print("rep model", self._array_to_seq(seq)[:2], state, reward, gamma)
                    if True: #self._check_reality(state):
                        succs.append([last_state, state])

                        # gamma = self.gamma if gamma is None else gamma
                        # reward = 1 if (0.7 <= state[0] <= 0.75 and 0.95 <= state[1] <= 1.0) else 0
                        # gamma = 0 if (0.7 <= state[0] <= 0.75 and 0.95 <= state[1] <= 1.0) else self.gamma

                        self._update_weight(last_state, last_action, state, reward, gamma, float(alpha))
                        # # add backward planning
                        # self._update_priority(index, last_state, last_action, state, reward, gamma)
                        # for a in range(self.num_action):
                        #     sbab0 = self.offline_nn.backward_prediction(a, last_state)
                        #     spbrbgb = self.offline_nn.forward_prediction(sbab0, a)
                        #     # print(last_state, sbab0, a, list(spbrbgb))
                        #     spb, rb, gb = spbrbgb
                        #     gb = self.gamma if gb is None else gb
                        #
                        #     pri = np.abs(self._get_tde(sbab0, a, last_state, rb, gb, dqn=DQN))
                        #
                        #     if True: #self._check_reality(sbab0):
                        #         preds.append([sbab0, last_state])
                        #         if True: #self._check_reality(spb):
                        #             succs.append([sbab0, spb])
                        #
                        #         if pri > self.pri_thrshd:
                        #             self._insert_seq(sbab0, a, last_state, rb, gb, pri + self.pri_thrshd)

                else:
                    self.b_control.force_remove(index)

                if self.gui:
                    return {"state": last_state, "q": self._check_q(last_state), "sbab_list": preds,
                            "succ_list": succs}, did_planning
                else:
                    return did_planning

            last_state, last_action, state, reward, gamma, pri = self._array_to_seq(seq)

            if self.rem_type == "random_BufferOnly":
                self._update_weight(last_state, last_action, state, reward, gamma, float(alpha))
                did_planning = True
            else:
                # if BACKPLANNING:
                #     # use true action, reward, gamma
                #     sample = self.model.sampleFromNext_pan(state, self.num_branching, self.num_action, config=[last_action, reward, gamma])
                #     # change return order
                #     if sample is not None:
                #         s, a, sp, r, g = sample
                #         sample = (s, sp, r, g, a)
                # else:
                #     sample = self.model.KDE_sampleSpRG(last_state, last_action)
                #
                # if DQN:
                #     if sample is None:
                #         self.b_control.force_remove(index)
                #     return sample
                sample = self.model.KDE_sampleSpRG(last_state, last_action)

                if sample is not None:
                    did_planning = True
                    ls_temp, state, reward, gamma, _ = sample

                    if BACKPLANNING:
                        last_state = ls_temp

                    # check prediction
                    # self._check_prediction(last_state, last_action, state, reward, gamma)

                    # print("sprg", seq[-1], last_state, last_action, state, reward, gamma, "tde="+str(self._get_tde(last_state, last_action, state, reward, gamma)))
                    succs.append([last_state, state])
                    gamma = self.gamma if gamma is None else gamma
                    # if gamma < 0.8:
                    #     print(ls_temp, gamma)

                    self._update_weight(last_state, last_action, state, reward, gamma, float(alpha)/np.sqrt(self.num_planning))
                    if self.rem_type == "pri" or self.rem_type == "pri_pred":
                        self._update_priority(index, last_state, last_action, state, reward, gamma)
                    if self.rem_type == "random_pred" or self.rem_type == "pri_pred":

                        sbab_list = self.model.sampleFromNext_pan(last_state, self.num_branching, self.num_action)
                        for sbab in sbab_list:
                            sbab0 = sbab[0]
                            a = sbab[1]
                            spb, rb, gb = sbab[2:]
                            gb = self.gamma if gb is None else gb

                            pri = np.abs(self._get_tde(sbab0, a, last_state, rb, gb, dqn=DQN))
                            # pri = np.abs(self._get_tde(sbab0, a, spb, rb, gb))

                            preds.append([sbab0, a, last_state, rb, gb, pri, self._check_q(sbab0), self._check_q(last_state)])
                            # succs.append([sbab0, spb])

                            # print("sbab", sbab0, a, spb, rb, gb, "pri="+str(pri))
                            if pri > self.pri_thrshd:
                                self._insert_seq(sbab0, a, last_state, rb, gb, pri + self.pri_thrshd)
                                # self._insert_seq(sbab0, a, spb, rb, gb, pri + self.pri_thrshd)
                        # input()
                else:
                    self.b_control.force_remove(index)

            if self.gui and (self.learning_mode not in raw_model_mode_list and self.learning_mode not in raw_vf_mode_list):
                return {"state": self._decode_all_proto([seq])[:, :2][0],
                        "q": self._check_q(last_state),
                        "sbab_list": self._decode_all_preds(preds),
                        "succ_list": self._decode_all_preds(succs)}, did_planning
            elif self.gui:
                return {"state": last_state, "q": self._check_q(last_state), "sbab_list": preds, "succ_list": succs}, did_planning
            else:
                return did_planning

    def _dqn_single_forward_plan(self, index, seq):
        real_s, real_a, real_sp, real_r, real_g, _ = self._array_to_seq(seq)
        if BACKPLANNING:
            # use true action, reward, gamma
            sample = self.model.sampleFromNext_pan(real_sp, self.num_branching, self.num_action,
                                                   config=[real_a, real_r, real_g])
            # change return order
            if sample is not None:
                s, a, sp, r, g = sample
                sample = (s, sp, r, g, a)
        else:
            sample = self.model.KDE_sampleSpRG(real_s, real_a)

        if sample is None:
            self.b_control.force_remove(index)
            print("Forward prediction is None", real_s)
            return sample
        else:
            last_state, state, reward, gamma, last_action = sample
            new_sequence = self._seq_to_array(last_state, last_action, state, reward, gamma, 0)
            return new_sequence

    def _dqn_single_backward_plan(self, last_state):
        sbab_list = self.model.sampleFromNext_pan(last_state, self.num_branching, self.num_action)
        predecessor = []
        for sbab in sbab_list:
            sbab0 = sbab[0]
            a = np.int64(sbab[1])
            spb, rb, gb = sbab[2:] # spb is not used. use given state instead
            gb = self.gamma if gb is None else gb
            pri = np.abs(self._get_tde(sbab0, a, last_state, rb, gb, dqn=DQN))
            predecessor.append([sbab0, last_state])
            if pri > self.pri_thrshd:
                self._insert_seq(sbab0, a, last_state, rb, gb, pri + self.pri_thrshd)
        return predecessor

    """
    Check q value of each action for a given state
    Not used for learning process
    Input: [x, y]
    Return: qvalues for all actions
    """
    def _check_q(self, state):
        qvalue = []
        for a in self.action_list:
            feature = self._feature_construction(state, a)
            # qvalue.append(np.dot(self.weight, feature))
            if self.old_weight_update == True:
                qvalue.append(np.dot(self.weight, feature))
            else:
                qvalue.append(np.dot(self.weight.weight.data, feature))
        # print(state, qvalue)
        return np.array(qvalue)

    """
    Choose sequence from buffer
    For now we use elif block
    Input: number of plannings
    Return: index in buffer, sasprg-array
    """
    def _sample_seqs_from_buffer(self, n):
        filled_ind = self.b_control.get_filled()
        filled_ind_length = self.b_control.get_filled_length()
        if self.rem_type == "random" or self.rem_type == "random_pred" or self.rem_type == "random_BufferOnly":
            indexs = np.random.choice(filled_ind, size=min(filled_ind_length, n))
            if len(indexs) == 0:
                return [], []
            return indexs, np.copy(self.buffer[indexs])
        elif self.rem_type == "pri" or self.rem_type == "pri_pred":
            # indexs = np.random.choice(filled_ind, size=min(len(filled_ind), n))
            indexs = self._sample_break_tie(self.buffer[filled_ind, -1], min(filled_ind_length, n))
            # # indexs = self._sample_break_tie(self.buffer[filled_ind, -1], n)
            if len(indexs) == 0:
                return [], []
            else:
                indexs = np.array(filled_ind)[indexs]
                seqs = np.copy(self.buffer[indexs])
                return indexs, seqs
        else:
            print("UNKNOWN TYPE IN SAMPLING")
            exit(-1)

    """
    Choose samples with highest priority
    """
    def _sample_break_tie(self, pris, num):
        indexs = []
        if num > 1:
            pris_copy = np.copy(pris)
            for i in range(num):
                indexs.append(self._break_tie(pris_copy))
                pris_copy[indexs[i]] = -1000000
        else:
            indexs.append(self._break_tie(pris))
        return np.array(indexs)

    """
    Save sample in an array
    Input: [x, y]-last, action-last, [x, y], reward, gamma, TD-error
    Return: sasprg-array
    """
    def _seq_to_array(self, last_state, last_action, state, reward, gamma, tde):
        return np.concatenate((last_state, np.array([last_action]), state,
                               np.array([reward]), np.array([gamma]), np.array([tde])),
                              axis=0)
    """
    Get sample from array
    Input: sasprg-array
    Return: [x, y]-last, action-last, [x, y], reward, gamma, TD-error
    """
    def _array_to_seq(self, seq):
        if seq.ndim == 1:
            last_state = seq[:self.dim_state]
            last_action = seq[self.dim_state]
            state = seq[self.dim_state+1: self.dim_state*2+1]
            reward = seq[self.dim_state*2+1]
            gamma = seq[self.dim_state*2+2]
            tde = seq[self.dim_state*2+3]
        else:
            last_state = seq[:, :self.dim_state]
            last_action = seq[:, self.dim_state]
            state = seq[:, self.dim_state + 1: self.dim_state * 2 + 1]
            reward = seq[:, self.dim_state * 2 + 1]
            gamma = seq[:, self.dim_state * 2 + 2]
            tde = seq[:, self.dim_state * 2 + 3]
        return last_state, last_action, state, reward, gamma, tde

    def _decode_all_proto(self, protos):
        states = np.zeros((len(protos), 7))
        for pi in range(len(protos)):
            s = self.rep_model_decoder.state_learned(protos[pi][:self.num_feature])
            a = protos[pi][self.num_feature]
            sp = self.rep_model_decoder.state_learned(protos[pi][self.num_feature+1: self.num_feature * 2 +1])
            rg = protos[pi][self.num_feature*2+1:]
            states[pi] = s[0], s[1], a, sp[0], sp[1], rg[0], rg[1]
        return states

    def _decode_all_preds(self, preds):
        states = []
        for pi in range(len(preds)):
            s1 = self.rep_model_decoder.state_learned(preds[pi][0])
            s2 = self.rep_model_decoder.state_learned(preds[pi][1])
            states.append([s1, s2])
        return states

    def _change_to_graph(self, state):
        idx = self.tc.get_index(state)
        graph = np.zeros((self.nn_num_tile ** 2))
        graph[idx] = 1
        return graph

    def _check_reality(self, state):
        x, y = state
        if 0.5 < x < 0.7:
            if y < 0.4 or y > 0.6:
                return False
        elif x > 1 or x < 0 or y > 1 or y < 0:
            return False
        return True

    # dqn
    def dqn_update_learn(self, seqs):
        s, a, sp, r, gamma, _ = self._array_to_seq(seqs)

        # binning
        if self.binning != 1:
            s = self._binning_rep(s)
            sp = self._binning_rep(sp)

        s = torch.from_numpy(s).float().to(device)
        a = torch.from_numpy(a.reshape(-1,1)).float().type(torch.LongTensor).to(device)
        s_values = self.dqn_learn(s)
        prediction = s_values.gather(1, a).view((-1)) #s_values[:, a]#
        # print(s_values.size(), a.size(), prediction.size())

        # gamma = torch.from_numpy(gamma).float().to(device)
        # r = torch.from_numpy(r).float().to(device)
        with torch.no_grad():
            gamma = torch.from_numpy(gamma).float().to(device)
            r = torch.from_numpy(r).float().to(device)
            sp = torch.from_numpy(sp).float().to(device)
            sp_values = self.dqn_target(sp)
            # vanilla DQN
            sp_q_value = sp_values.data.max(1)[0]
            # # double Q-learning DQN
            # sp_values_learn = self.dqn_learn(sp)
            # ap = sp_values_learn.data.max(1)[1].view((-1, 1))
            # sp_q_value = sp_values.gather(1, ap).view((-1))

        # print("before, t ", sp_values)
        # print("before, l ", s_values)
        y = r + gamma * sp_q_value
        loss = self.dqn_loss(prediction, y)
        self.dqn_learn_optimizer.zero_grad()
        loss.backward()
        self.dqn_learn_optimizer.step()

        del s, a, s_values, prediction, gamma, r, sp, sp_values, sp_q_value, y, loss

    def synchronize_networks(self):
        params_from = self.dqn_learn.named_parameters()
        params_to = self.dqn_target.named_parameters()

        dict_params_to = dict(params_to)

        for name, param in params_from:
            if name in dict_params_to:
                dict_params_to[name].data.copy_(param.data)
            del name, param
        del params_from, params_to

    def _check_prediction(self, last_state, last_action, state, reward, gamma):
        # check prediction
        if (self.learning_mode not in raw_model_mode_list):
            last_state = self.rep_model_decoder.state_learned(
                last_state / np.linalg.norm(last_state))
            state = self.rep_model_decoder.state_learned(state / np.linalg.norm(state))

        if BACKPLANNING:
            if 0.5 < last_state[0] < 0.7 and (
                    0 < last_state[1] < 0.4 or 0.6 < last_state[1] < 0.6):
                print("illegal prediction", last_state, int(last_action), "->", state)
        else:
            if 0.5 < state[0] < 0.7 and (
                    0 < state[1] < 0.4 or 0.6 < state[1] < 0.6):
                print("illegal prediction", last_state, int(last_action), "->", state)
            else:
                print(last_state, last_action, "->", state, reward, gamma)

    def _binning_rep(self, rep):
        if len(rep.shape) == 1:
            rep = rep.reshape((1, -1))
        feature = np.zeros((rep.shape[0], rep.shape[1] * self.binning))
        # rep_ind = (((rep + 1) / 2.0 * self.binning // 1).clip(0, self.binning - 1) + self.arrange_ind).astype(int)
        rep_ind = (((rep + 1) / 2.0 * self.binning // 1) + self.arrange_ind).astype(int)
        for i in range(feature.shape[0]):
            feature[i, rep_ind[i]] = 1
        # print(feature)
        return feature




agent = None

def agent_init():
    global agent
    agent = REM_Dyna()
    return


def agent_start(state):
    global agent
    current_action = agent.start(state)
    return current_action


def agent_step(reward, state):
    global agent
    current_action, other_info = agent.step(reward, state)
    return current_action, other_info

def agent_step_debugging(reward, state):
    global agent
    sample_sa = agent.step_debugging(reward, state)
    return sample_sa

def agent_step_debugging_update_w(seqs):
    global agent
    current_action = agent.step_debugging_update_w(seqs)
    return current_action

def agent_end(reward, state):
    global agent
    agent.end(reward, state)
    return

def agent_cleanup():
    global agent
    agent = None
    return

def agent_message(in_message):
    if in_message[0] == "set param":
        if agent != None:
            agent.set_param(in_message[1])
        else:
            print("the environment hasn't been initialized.")
    elif in_message[0] == "check time":
        return agent.check_time
    elif in_message[0] == "check total time":
        return agent.check_total_time
    elif in_message[0] == "check model size":
        return  agent.model.get_len_protos()
    elif in_message[0] == "print current value":
        return agent._max_action(agent.last_state, isprint=False)
    elif in_message[0] == "check covariance":
        print("fix cov", agent.model.cov)

        cov = agent.model.sig_prot
        cov_inv = agent.model.sig_prot_inv

        if agent.model.cov == 0:
            print("traning set cov")
            cov_inv = agent.model.cov_inv
            cov = np.linalg.inv(cov_inv)
        indices = [i for i in range(agent.model.state_dim)]
        print("s")
        print(np.diag(cov[indices][:, indices]), np.diag(cov_inv[indices][:, indices]))
        indices = [i for i in range(agent.model.state_dim + 1, agent.model.state_dim*2 + 1)]
        print("sp")
        print(np.diag(cov[indices][:, indices]), np.diag(cov_inv[indices][:, indices]))
        indices = agent.model.index_no_a
        print("ssprg")
        print(np.diag(cov[indices][:, indices]), np.diag(cov_inv[indices][:, indices]))
        indices = [i for i in range(agent.model.state_dim + 1, agent.model.seq_dim)]
        print("sprg")
        print(np.diag(cov[indices][:, indices]), np.diag(cov_inv[indices][:, indices]))
    elif in_message[0] == "change epsilon":
        agent.epsilon = in_message[1]
        print("changed epsilon", agent.epsilon)

    elif in_message[0] == "get tde":
        last_state, last_action, state, reward, gamma = in_message[1:]
        return agent._get_tde(last_state, last_action, state, reward, gamma, DQN)
    return