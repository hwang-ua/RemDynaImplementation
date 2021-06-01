# -*- coding: utf-8 -*-

import time
import math
import numpy as np
import pickle as pkl
# import random
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
    # import utils.REM_model_kdt_realCov_flm as rem
    import utils.REM_model_kdt_realCov_flm_w as rem
else:
    import utils.REM_model_kdt_realCov as rem
# import utils.KernModelupdate as rem
# import utils.auto_encoder_2branch as atec
import utils.get_learned_representation as glr
import utils.get_learned_state as gls

import os

import torch
from torch import nn

import environment.ContinuousGridWorld as world


def _previous_position(action, x, y):
    if np.random.random() > world.suc_prob:
        action = np.random.randint(0, 4)
    # up
    if action == 0:
        y -= 0.05 + np.random.normal(0, 0.01)
    # down
    elif action == 1:
        y += 0.05 + np.random.normal(0, 0.01)
    # right
    elif action == 2:
        x -= 0.05 + np.random.normal(0, 0.01)
    # left
    elif action == 3:
        x += 0.05 + np.random.normal(0, 0.01)
    else:
        print("Environment: action out of range. Action is:", action)
    return x, y

def reverse_true_model(s_list):
    sbab_list = []
    for s in s_list:
        sx, sy = s
        for action in range(4):
            new_x, new_y = _previous_position(action, sx, sy)
            if not world._go_in_wall(new_x, new_y):
                sbx = new_x
                sby = new_y
            sbx = np.clip(sbx, 0.0, 1.0)
            sby = np.clip(sby, 0.0, 1.0)
            sbab_list.append([[sbx, sby], action])
    return sbab_list


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
                    TCRAWVF_NORMREPMODEL_FIXCOV,
                    TCRAWVF_NORMREPMODEL]

DEBUGGING = True


class BufferControl():
    def __init__(self, length):
        # def __init__(self, length, model):
        self.b_length = length
        self.b_empty = [i for i in range(self.b_length)]
        self.b_filled = []
        # self.model = model

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

        return index

    def remove(self, index):
        # self.b_filled.remove(index)
        self.b_empty.append(index)

        # print("---removed---")
        # print(self.b_filled[:5], self.b_filled[-5:])
        # print(self.b_empty[:5], self.b_empty[-5:], "\n")

        # if len(self.b_filled) == 0:
        #     print("Empty at: ", self.model.t)

    def get_filled(self):
        return self.b_filled

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
        self.buffer = np.zeros(
            (self.len_buffer, self.dim_state * 2 + 4))  # {"pri_time": np.zeros((2, self.len_buffer)), "sequence": {}}
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
        self.gui = param["gui"]
        self.offline = param["offline"]

        self.alg = param["alg"]  # 'Q' or Sarsa
        self.traces_lambda = param["lambda"]
        self.opt_mode = param["opt_mode"]

        self.always_add_prot = param["always_add_prot"]

        self.rem_type = param["rem_type"]
        self.alpha = param["alpha"]
        self.epsilon = param["epsilon"]
        self.gamma = param["agent_gamma"]
        self.num_planning = param["num_planning"]
        self.num_branching = param["num_branching"]

        self.dim_state = param["dim_state"]  # 2
        self.action_mode = param["action_mode"]
        self.num_action = param["num_action"]
        self.action_list = [i for i in range(self.num_action)]

        self.num_near = param["num_near"]

        self.learning_mode = param["remDyna_mode"]

        self.div_actBit = None
        self.div_norm = None

        # 2， 9
        if self.learning_mode == SINGLE_REP or self.learning_mode == SINGLE_REP_CHECKDIST:
            self.len_s_feature = 2
            self.len_s_rep = param["nn_num_feature"]
            self.dim_state = param["nn_num_feature"]

            if self.opt_mode == 4:
                self.div_norm = 1

        # 10, 11
        elif self.learning_mode == SINGLE_NORMREP or self.learning_mode == SINGLE_NORMREP_FIXCOV:
            self.len_s_feature = 2
            self.len_s_rep = param["nn_num_feature"]
            self.dim_state = param["nn_num_feature"]

            if self.opt_mode == 4:
                self.div_norm = 1

        # 12, 14
        elif self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == TCREPVF_NORMREPMODEL:
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
            self.len_s_feature = 2
            self.len_s_rep = param["nn_num_feature"]

            if self.opt_mode == 4:
                self.div_norm = 1

        # 16
        elif self.learning_mode == NORMREPVF_RAWMODEL:
            self.len_s_feature = 2
            self.len_s_rep = param["nn_num_feature"]

            if self.opt_mode == 4:
                self.div_norm = 1

        # 4, 19
        elif self.learning_mode == TCREPVF_RAWMODEL_CHECKDIST or \
                self.learning_mode == TCREPVF_RAWMODEL:
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
            self.len_s_feature = 2
            self.len_s_rep = param["nn_num_feature"] + 1

            if self.opt_mode == 4:
                self.div_norm = 1

        # 6
        elif self.learning_mode == BIASTCREPVF_RAWMODEL_CHECKDIST:
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
            self.num_tilings = 1
            self.num_tiles = 16

            self.len_s_feature = 2
            self.tc_mem_size = 256
            self.len_s_rep = self.tc_mem_size
            self.dim_state = 2
            self.iht = tc.IHT(self.tc_mem_size)
            if self.opt_mode == 4:
                self.div_actBit = self.num_tilings


        # 0, 1
        else:
            self.num_tilings = 1
            self.num_tiles = 16
            self.tc_mem_size = 256
            self.len_s_feature = self.tc_mem_size
            self.len_s_rep = self.len_s_feature
            self.iht = tc.IHT(self.tc_mem_size)
            if self.opt_mode == 4:
                self.div_actBit = self.num_tilings

        if self.learning_mode not in raw_model_mode_list:
            self.rep_model_decoder = gls.GetLearnedState(2,
                                                         param["nn_nodes"],
                                                         param["nn_num_feature"],
                                                         32 * 4 * 2 * 2,
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
                                                         True, num_tiling=32, num_tile=4, constraint=True,
                                                         model_path="./feature_model/",
                                                         file_name="feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_seperateRcvs")

        self.len_sa_feature = self.len_s_rep * self.num_action

        if self.div_actBit is not None:
            self.alpha = param["alpha"] / np.sqrt(float(self.div_actBit))
        else:
            self.alpha = param["alpha"]

        if self.learning_mode == CHECK_DIST or \
                self.learning_mode == SINGLE_REP or \
                self.learning_mode == SINGLE_NORMREP or \
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
                self.learning_mode == TCREPVF_RAWMODEL:
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
            self.num_input = 2
            self.num_output = 32 * 4 * 2 * 2
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
                                               num_tiling=32,
                                               num_tile=4,
                                               constraint=self.constraint,
                                               model_path="./feature_model/",
                                               file_name="feature_embedding_continuous_input[0.0, 1]_envSucProb1.0")
        # else:
        # self.tc = utc.TileCoding(self.dim_state, self.num_tilings, self.num_tiles, 1)

        self.momentum = param["momentum"]
        self.rms = param["rms"]

        if param["init_weight"] == "0":
            # self.weight = np.zeros(self.len_sa_feature)
            # self.weight = np.zeros((self.len_sa_feature))

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
            # self.weight = np.ones((self.len_sa_feature))
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
        self.norm_diff = param["rbf_normalize_diff"]

        if self.offline:
            print("===============")
            print("Offline learning")
            if self.learning_mode in raw_model_mode_list:
                print("Loading raw model")
                if LinearModel:
                    with open('prototypes/raw_model_linear/model.pkl', 'rb') as f:
                        self.model = pkl.load(f)
                else:
                    with open('prototypes/raw_model_' + str(self.num_near) + 'nn_' + str(
                            self.add_prot_limit) + 'nu/model.pkl', 'rb') as f:
                        self.model = pkl.load(f)
            else:
                print("Loading rep model")
                if LinearModel:
                    with open('prototypes/representation_model_linear/model.pkl', 'rb') as f:
                        self.model = pkl.load(f)
                else:
                    with open('prototypes/representation_model_' + str(self.num_near) + 'nn_' + str(
                            self.add_prot_limit) + 'nu/model.pkl', 'rb') as f:
                        self.model = pkl.load(f)
            self.model.learning_mode = self.learning_mode
            self.model.rep_model = self.rep_model
            # self.model.sample_weighted_mean = False
            # print("Set weighted_mean flag", self.model.sample_weighted_mean)
            print("number of prototypes", len(self.model.same_a_ind[0]), len(self.model.same_a_ind[1]),
                  len(self.model.same_a_ind[2]), len(self.model.same_a_ind[3]))
            print("===============")
        else:
            if self.learning_mode == OLD_REM:
                self.model = rem.REM_Model(self.dim_state, self.num_near, self.add_prot_limit, self.model_params,
                                           self.learning_mode, self.similarity_limit, self.norm_diff)
            elif self.learning_mode == TCRAWVF_NORMREPMODEL \
                    or self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
                self.model = rem.REM_Model(param["nn_num_feature"], self.num_near, self.add_prot_limit,
                                           self.model_params,
                                           self.learning_mode, self.similarity_limit, self.norm_diff,
                                           rep_model=self.rep_model)
            else:
                self.model = rem.REM_Model(self.dim_state, self.num_near, self.add_prot_limit, self.model_params,
                                           self.learning_mode, self.similarity_limit, self.norm_diff,
                                           rep_model=self.rep_model)

        # self.b_control = BufferControl(self.len_buffer, self.model)

        # self.temp_encoder = glr.GetLearnedRep(2,[512,256,128],32,512,0.0,0.0,model_path="./feature_model/",file_name="feature_embedding_continuous_input[0.0, 1]_envSucProb1.0")
        # self.temp_decoder = gls.GetLearnedState(2,num_feature=32,num_rec_node=[128,256,512],model_path="./feature_model/",file_name="feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_seperateRcvs")
        # self.temp_encoder = glr.GetLearnedRep(self.num_input,
        #                                    self.num_node,
        #                                    self.num_feature,
        #                                    self.num_output,
        #                                    self.lr,
        #                                    self.lr_rcvs,
        #                                    self.wd,
        #                                    self.num_dec_node,
        #                                    self.num_rec_node,
        #                                    self.optimizer,
        #                                    self.dropout,
        #                                    self.beta,
        #                                    self.delta,
        #                                    self.legal_v,
        #                                    self.continuous,
        #                                    num_tiling=32,
        #                                    num_tile=4,
        #                                    constraint=self.constraint,
        #                                    model_path="./feature_model/",
        #                                    file_name="feature_embedding_continuous_input[0.0, 1]_envSucProb1.0")
        #
        # self.temp_decoder = gls.GetLearnedState(2,
        #                                         param["nn_nodes"],
        #                                         param["nn_num_feature"],
        #                                         32 * 4 * 2 * 2,
        #                                         param["nn_lr"],
        #                                         param["nn_lr"],
        #                                         param["nn_weight_decay"],
        #                                         param["nn_dec_nodes"],
        #                                         param["nn_rec_nodes"],
        #                                         param["optimizer"],
        #                                         param["nn_dropout"],
        #                                         param["nn_beta"],
        #                                         param["nn_delta"],
        #                                         param["nn_legal_v"],
        #                                         True, num_tiling=32, num_tile=4, constraint=True,
        #                                         model_path="./feature_model/",
        #                                         file_name="feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_seperateRcvs")
        # return

    """
    Input: [x, y]
    Return: action
    """

    def start(self, state):
        if self.learning_mode == SINGLE_REP or \
                self.learning_mode == SINGLE_NORMREP or \
                self.learning_mode == SINGLE_NORMREP_FIXCOV or \
                self.learning_mode == BIASTCREPVF_REPMODEL or \
                self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST or \
                self.learning_mode == SINGLE_REP_CHECKDIST or \
                self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == TCREPVF_NORMREPMODEL or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL:
            state = self._state_representation(state)
        self.state = state
        self.action = self._policy(state)
        self.check_total_time = np.zeros(6)
        if self.alg == 'Sarsa':
            self.traces = torch.from_numpy(np.zeros((self.len_sa_feature))).float()
        return self.action

    """
    Input: int, [x, y]
    Return: action
    """

    def step(self, reward, state, end_of_ep=False):
        if end_of_ep:
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
                self.learning_mode == BIASTCREPVF_NORMREPMODEL:
            state = self._state_representation(state)
        # update variables
        self.last_state = self.state
        self.last_action = self.action
        self.state = state
        self.reward = reward
        # use Q-learning to update weight
        qstart = time.time()
        tde = self._update_weight(self.last_state, self.last_action, self.state, reward, gamma, self.alpha)
        qtime = time.time() - qstart

        other_info = None
        if self.learning:
            if self.rem_type != "random_BufferOnly" and not self.offline:
                # update model
                if self.learning_mode == TCRAWVF_NORMREPMODEL or self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
                    last_rep = self._state_representation(self.last_state)
                    rep = self._state_representation(self.state)
                    self.model.add2Model(last_rep, self.last_action, rep, reward, gamma)
                else:
                    self.model.add2Model(self.last_state, self.last_action, self.state, reward, gamma)

            # get tde
            tde = self._get_tde(self.last_state, self.last_action, self.state, reward, gamma)

            # insert S,A into buffer
            # if self.rem_type == "pri" or self.rem_type == "pri_pred":
            #     if np.abs(tde) > self.pri_thrshd:
            #         self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma, np.abs(tde)+self.pri_thrshd)
            # else:
            #     self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma, np.abs(tde) + self.pri_thrshd)
            self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma,
                             np.abs(tde) + self.pri_thrshd)

            # print("Before planning buffer", self.b_control.get_filled())

            # planning
            if not self.gui:
                # indexs, seqs = self._sample_seqs_from_buffer(self.num_planning)
                # for i in range(min(len(indexs), self.num_planning)):
                #     self._single_planning(self.alpha, self.num_planning, indexs[i], seqs[i])

                if DEBUGGING:
                    all_sample = []
                    for i in range(self.num_planning):
                        indexs, seqs = self._sample_seqs_from_buffer(1)
                        if (self.learning_mode not in raw_model_mode_list) and \
                            (self.learning_mode not in raw_vf_mode_list):
                            raw_s = self.rep_model_decoder.state_learned(seqs[0][:self.dim_state])

                            all_sample.append([raw_s[0], raw_s[1], seqs[0][self.dim_state]])

                            # sbab_list = self.model.sampleFromNext_pan(seqs[0][:self.dim_state], self.num_branching, self.num_action)
                            sbab_list = reverse_true_model(seqs[0][:self.dim_state])

                            for sbab in sbab_list:
                                sb = self.rep_model_decoder.state_learned(sbab[0])
                                all_sample.append([sb[0], sb[1], sbab[1]])
                        else:
                            all_sample.append((seqs[0][:3]))
                            sbab_list = self.model.sampleFromNext_pan(seqs[0][:2], self.num_branching, self.num_action)
                            for sbab in sbab_list:
                                all_sample.append([sbab[0][0], sbab[0][1], sbab[1]])

                    return all_sample

                for i in range(self.num_planning):
                    # print("Buffer status:",len(self.b_control.get_filled()))
                    indexs, seqs = self._sample_seqs_from_buffer(1)
                    self._single_planning(self.alpha, self.num_planning, indexs[0], seqs[0])

            else:
                other_info = {"plan": []}
                for i in range(self.num_planning):
                    indexs, seqs = self._sample_seqs_from_buffer(1)
                    other_info["plan"].append(self._single_planning(self.alpha, self.num_planning, indexs[0], seqs[0]))

                if self.learning_mode not in raw_vf_mode_list:
                    other_info["buffer"] = self._decode_all_proto(self.buffer[self.b_control.get_filled()])
                else:
                    other_info["buffer"] = self.buffer[self.b_control.get_filled()]
                if self.learning_mode not in raw_model_mode_list:
                    other_info["protos"] = self._decode_all_proto(self.model.get_protos())
                else:
                    other_info["protos"] = self.model.get_protos()
                other_info["agent_q"] = self._check_q(state)

        elif not self.learning and self.always_add_prot:
            if self.rem_type != "random_BufferOnly" and not self.offline:
                if self.learning_mode == TCRAWVF_NORMREPMODEL or self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
                    last_rep = self._state_representation(self.last_state)
                    rep = self._state_representation(self.state)
                    self.model.add2Model(last_rep, self.last_action, rep, reward, gamma)
                else:
                    self.model.add2Model(self.last_state, self.last_action, self.state, reward, gamma)

            # get tde
            tde = self._get_tde(self.last_state, self.last_action, self.state, reward, gamma)

            # insert S,A into buffer
            # if self.rem_type == "pri" or self.rem_type == "pri_pred":
            #     if np.abs(tde) > self.pri_thrshd:
            #         self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma, np.abs(tde)+self.pri_thrshd)
            # else:
            #     self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma, np.abs(tde) + self.pri_thrshd)
            self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma,
                             np.abs(tde) + self.pri_thrshd)

            if DEBUGGING:
                return []

        # choose new action
        self.action = self._policy(state, False)

        # if self.learning:
        #     print("state [{:4.4f}, {:4.4f}]".format(self.last_state[0], self.last_state[1]), end=" ")
        #     self._max_action(self.last_state, True)
        #     input("interrupt")
        return self.action, other_info

    """
        Debugging function
        Input: int, [x, y]
        Return: action
    """
    """
    # ==================================================================================================================
    def step_debugging(self, reward, state, end_of_ep=False):
        # print("Beginning", np.array(state))
        if end_of_ep:
            gamma = 0
        else:
            gamma = self.gamma
        # gamma = self.gamma

        # update variables
        self.last_state = self.state
        self.last_action = self.action
        self.state = state
        self.reward = reward
        # use Q-learning to update weight

        self._update_weight(self.last_state, self.last_action, self.state, reward, gamma, self.alpha)

        if self.learning:
            # get tde
            tde = self._get_tde(self.last_state, self.last_action, self.state, reward, gamma)

            self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma,
                             np.abs(tde) + self.pri_thrshd)

            # planning
            sampled_sa = []
            for i in range(self.num_planning):
                indexs, seqs = self._sample_seqs_from_buffer(1)
                sampled_sa.append(seqs[0][:3])

        elif not self.learning and self.always_add_prot:
            # get tde
            tde = self._get_tde(self.last_state, self.last_action, self.state, reward, gamma)

            self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma,
                             np.abs(tde) + self.pri_thrshd)

            sampled_sa = []

        return sampled_sa
    """
    def step_debugging_update_w(self, real_seq):
        for seq in real_seq:
            last_state, last_action, state, reward, terminate = seq

            # if last_action == 0:
            #     assert last_state[0] == state[0]
            #     assert last_state[1] <= state[1]
            # elif last_action == 1:
            #     assert last_state[0] == state[0]
            #     assert last_state[1] >= state[1]
            # elif last_action == 2:
            #     assert last_state[0] <= state[0]
            #     assert last_state[1] == state[1]
            # elif last_action == 3:
            #     assert last_state[0] >= state[0]
            #     assert last_state[1] == state[1]
            if terminate:
                gamma = 0
            else:
                gamma = self.gamma
            # print("updating w", np.array(last_state), last_action, np.array(state), reward, gamma, self.alpha)
            if (self.learning_mode not in raw_model_mode_list) and \
                    (self.learning_mode not in raw_vf_mode_list):
                last_state = self.rep_model.state_representation(np.array(last_state))
                state = self.rep_model.state_representation(np.array(state))
            self._update_weight(last_state, last_action, state, reward, gamma, self.alpha)

        # for i in range(self.num_planning):
        #     indexs, seqs = self._sample_seqs_from_buffer(1)
        #     last_state, last_action, state, reward, gamma, pri = self._array_to_seq(seqs[0])
        #     self._update_weight(last_state, last_action, state, reward, gamma, self.alpha)

        # choose new action
        self.action = self._policy(self.state, False)
        # print("ending   ", np.array(self.state), self.action)

        return self.action
    # ==================================================================================================================

    """
    Input: int, [x, y]
    Return: None
    """

    def end(self, reward, state):
        self.step(reward, state, end_of_ep=True)
        return

    """
    Choose action according to given policy
    Input: [x, y]
    Return: action
    """

    def _policy(self, state, isprint=False):
        if self.action_mode == "discrete":
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_list)
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
        if isprint:
            print(all_choices, valid_index,
                  np.where(self._feature_construction(state, 0) == 1)[0],
                  np.where(self._feature_construction(state, 1) == 1)[0],
                  np.where(self._feature_construction(state, 2) == 1)[0],
                  np.where(self._feature_construction(state, 3) == 1)[0])
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

    """
    Generate feature for learning value function
    Input: [x, y], action
    Return: (s, a)-feature
    """

    def _feature_construction(self, state, action):
        #
        if self.learning_mode == SINGLE_REP or \
                self.learning_mode == SINGLE_REP_CHECKDIST:
            feature = np.zeros(self.len_sa_feature)
            feature[self.len_s_rep * int(action): self.len_s_rep * (int(action) + 1)] = state  # rep

        elif self.learning_mode == SINGLE_NORMREP or \
                self.learning_mode == SINGLE_NORMREP_FIXCOV:
            feature = np.zeros(self.len_sa_feature)
            state = state / float(np.linalg.norm(state))
            feature[self.len_s_rep * int(action): self.len_s_rep * (int(action) + 1)] = state

        # 12, 14
        elif self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or \
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

        elif self.learning_mode == BIASTCREPVF_REPMODEL:
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
            feature = np.zeros(self.len_sa_feature)
            state = np.clip(np.array(state), 0.0, 1.0)
            ind = np.array(tc.tiles(self.iht, self.num_tilings, float(self.num_tiles) * np.array(state)))
            feature[self.len_s_rep * int(action) + ind] = 1

        else:
            feature = np.zeros(self.len_sa_feature)
            state = np.clip(np.array(state), 0.0, 1.0)
            ind = np.array(tc.tiles(self.iht, self.num_tilings, float(self.num_tiles) * np.array(state)))
            feature[self.len_s_rep * int(action) + ind] = 1
        return feature

    """
    Update weight for learning value function
    Input: [x, y]-last, action-last, [x,y], reward, gamma, lr
    Return: None
    """

    def _update_weight(self, last_state, last_action, state, reward, gamma, alpha, current_action=None):
        # if self.learning:
        #     print("Weight", last_state, last_action, state, reward, gamma)

        # last_state = self.temp_decoder.state_learned(self.temp_encoder.state_representation(np.array(last_state)))
        # state = self.temp_decoder.state_learned(self.temp_encoder.state_representation(np.array(state)))

        last_feature = self._feature_construction(last_state, last_action)
        last_feature_torch = torch.from_numpy(last_feature).float()

        if self.alg == 'Sarsa':
            self.traces *= (gamma * self.traces_lambda)
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
            loss = 0.0 * torch.norm(self.weight.weight)  # self.weight)
            loss.backward(torch.FloatTensor(np.asarray([0.0])), retain_graph=True)

            if self.div_norm:
                if self.alg == 'Sarsa':
                    self.weight.weight.grad -= (tde / torch.norm(last_feature_torch)) * self.traces
                else:
                    self.weight.weight.grad -= (tde / torch.norm(last_feature_torch)) * last_feature_torch
            else:
                if self.alg == 'Sarsa':
                    self.weight.weight.grad -= tde * self.traces
                else:
                    self.weight.weight.grad -= tde * last_feature_torch

            self.weight_optimizer.step()

            # if self.learning:
            #     print("after update weight TDE =", self._td_error(last_feature, feature, reward, gamma, self.weight), last_state, state, reward, gamma)
            #     exit()

            loss = 0.0 * torch.norm(self.weight.weight)  # self.weight)
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

    def _get_tde(self, last_state, last_action, state, reward, gamma):
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
        new_pri = np.abs(self._get_tde(last_state, last_action, state, reward, gamma))

        # print("Priority update:",self.buffer[index, -1], new_pri)

        # if new_pri > self.pri_thrshd:
        #     self.buffer[index, -1] = new_pri#+self.pri_thrshd
        # else:
        #     self.buffer[index, -1] = -10
        #     self.b_control.remove(index)
        #     # print("removing sample", last_state, state, reward, gamma)

        self.buffer[index, -1] = new_pri  # + self.pri_thrshd
        # if new_pri < self.pri_thrshd:
        #     self.b_control.remove(index)

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

        # print("Insert:",new_sequence[-1])

        if self.adpt_thrshd:
            self.pri_thrshd = np.mean(self.buffer[self.b_control.get_filled()][:, -1])
        return

    """
    Planning step
    Input: lr, number of planning, index in buffer, sasprg-array
    Return: dictionary
    """

    def _single_planning(self, alpha, n, index, seq):
        if seq is not None:
            preds = []
            last_state, last_action, state, reward, gamma, pri = self._array_to_seq(seq)

            if self.rem_type == "random_BufferOnly":
                if self.opt_mode == 4:
                    self._update_weight(last_state, last_action, state, reward, gamma, float(alpha))
                else:
                    self._update_weight(last_state, last_action, state, reward, gamma, float(alpha))
                self._update_priority(index, last_state, last_action, state, reward, gamma)

            # ---- This block is for recreating random ER, should not be in REM Dyna code.----
            # elif self.rem_type == "random":
            #     self._update_weight(last_state, last_action, state, reward, gamma, float(alpha))
            # ---- This block is for recreating random ER performance, should not be in REM Dyna code.----

            else:

                # sample s',r,g from model
                if self.learning_mode == TCRAWVF_NORMREPMODEL or \
                        self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
                    last_rep = self._state_representation(last_state)
                    sample = self.model.KDE_sampleSpRG(last_rep, last_action)
                else:
                    sample = self.model.KDE_sampleSpRG(last_state, last_action)

                if sample is not None:
                    ls_temp, state, reward, gamma, _ = sample

                    # if self.learning:
                    #     print(state)
                    #     exit()

                    if self.learning_mode == TCRAWVF_NORMREPMODEL or \
                            self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
                        state = self.rep_model_decoder.state_learned(state)

                    # last_state = self.rep_model.state_representation(last_state)
                    # print(last_state/ np.linalg.norm(last_state))
                    # ls_temp = self.rep_model_decoder.state_learned(ls_temp/ np.linalg.norm(ls_temp))
                    # print(last_state)
                    # print(ls_temp)
                    # print()
                    # assert np.array_equal(last_state, ls_temp)
                    # assert np.array_equal(state, self._array_to_seq(seq)[2])
                    # print(state)
                    # print(self._array_to_seq(seq)[2])
                    # print()

                    # print("sprg: [{:8.4f}, {:8.4f}] {} => [{:8.4f}, {:8.4f}], {:8.4f}, {:8.4f}".format(last_state[0],last_state[1],int(last_action),state[0],state[1],reward, gamma))
                    # input("Interrupt")

                    gamma = self.gamma if gamma is None else gamma

                    if self.opt_mode == 4:
                        self._update_weight(last_state, last_action, state, reward, gamma, float(alpha))
                    else:
                        self._update_weight(last_state, last_action, state, reward, gamma, float(alpha))

                    # if self.learning:
                    #     print("state [{:4.4f}, {:4.4f}]".format(last_state[0], last_state[1]), end=" ")
                    #     self._max_action(last_state, True)

                    if self.rem_type == "pri" or self.rem_type == "pri_pred":
                        self._update_priority(index, last_state, last_action, state, reward, gamma)

                    if self.rem_type == "random_pred" or self.rem_type == "pri_pred":
                        if self.learning_mode == TCRAWVF_NORMREPMODEL or \
                                self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
                            last_rep = self._state_representation(last_state)
                            sbab_list = self.model.sampleFromNext_pan(last_rep, self.num_branching, self.num_action)
                        else:
                            sbab_list = self.model.sampleFromNext_pan(last_state, self.num_branching, self.num_action)

                        # preds = []
                        for sbab in sbab_list:

                            # real code
                            sample_b = self.model.KDE_sampleSpRG(sbab[0], sbab[1])

                            if sample_b is not None:
                                _, spb, rb, gb, _ = sample_b
                                if self.learning_mode == TCRAWVF_NORMREPMODEL or \
                                        self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
                                    spb = self.rep_model_decoder.state_learned(spb)
                                    sbab0 = self.rep_model_decoder.state_learned(sbab[0])
                                    # sbab1 = self.rep_model_decoder.state_learned(sbab[1])
                                else:
                                    sbab0 = sbab[0]
                                    # sbab1 = sbab[1]

                                # print("s   : [{:8.4f}, {:8.4f}] {} => [{:8.4f}, {:8.4f}]".format(sbab0[0], sbab0[1],
                                #                                                                  int(sbab[1]),
                                #                                                                  last_state[0],
                                #                                                                  last_state[1]))

                                gb = self.gamma if gb is None else gb

                                pri = np.abs(self._get_tde(sbab0, sbab[1], spb, rb, gb))

                                preds.append([sbab0, last_state])

                                if pri >= self.pri_thrshd:
                                    self._insert_seq(sbab0, sbab[1], spb, rb, gb, pri + self.pri_thrshd)

            if self.gui and (
                    self.learning_mode not in raw_model_mode_list and self.learning_mode not in raw_vf_mode_list):
                return {"state": self._decode_all_proto([seq])[:, :2][0],
                        "q": self._check_q(last_state),
                        "sbab_list": self._decode_all_preds(preds)}
            # elif self.gui and (self.learning_mode not in raw_model_mode_list and self.learning_mode in raw_vf_mode_list):
            #     return {"state": self._decode_all_proto([seq])[:, :2][0],
            #             "q": self._check_q(last_state),
            #             "sbab_list": self._decode_all_preds(preds)}
            elif self.gui:
                return {"state": last_state, "q": self._check_q(last_state), "sbab_list": preds}
            else:
                return

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
        if self.rem_type == "random" or self.rem_type == "random_pred" or self.rem_type == "random_BufferOnly":
            indexs = np.random.choice(filled_ind, size=min(len(filled_ind), n))
            # indexs = np.random.choice(self.buffer[filled_ind, -1], size=n)
        elif self.rem_type == "pri" or self.rem_type == "pri_pred":
            # indexs = np.random.choice(filled_ind, size=min(len(filled_ind), n))
            indexs = self._sample_break_tie(self.buffer[filled_ind, -1], min(len(filled_ind), n))
            # # indexs = self._sample_break_tie(self.buffer[filled_ind, -1], n)
        else:
            print("UNKNOWN TYPE IN SAMPLING")
            exit(-1)
        if len(indexs) == 0:
            return [], []
        else:
            indexs = np.array(filled_ind)[indexs]
            seqs = np.copy(self.buffer[indexs])

            return indexs, seqs

    """
    Choose samples with highest priority
    """

    def _sample_break_tie(self, pris, num):
        # pris = np.copy(pris)
        indexs = []
        for i in range(num):
            indexs.append(self._break_tie(pris))
            # pris[indexs[i]] = -1000000
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
        last_state = seq[:self.dim_state]
        last_action = seq[self.dim_state]
        state = seq[self.dim_state + 1: self.dim_state * 2 + 1]
        reward = seq[self.dim_state * 2 + 1]
        gamma = seq[self.dim_state * 2 + 2]
        tde = seq[self.dim_state * 2 + 3]
        return last_state, last_action, state, reward, gamma, tde

    def _decode_all_proto(self, protos):
        states = np.zeros((len(protos), 7))
        for pi in range(len(protos)):
            s = self.rep_model_decoder.state_learned(protos[pi][:32])
            a = protos[pi][32]
            sp = self.rep_model_decoder.state_learned(protos[pi][33: 65])
            rg = protos[pi][65:]
            states[pi] = s[0], s[1], a, sp[0], sp[1], rg[0], rg[1]
        return states

    def _decode_all_preds(self, preds):
        states = []
        for pi in range(len(preds)):
            s1 = self.rep_model_decoder.state_learned(preds[pi][0])
            s2 = self.rep_model_decoder.state_learned(preds[pi][1])
            states.append([s1, s2])
        return states


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
    sample_sa = agent.step(reward, state)
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
        return agent.model.get_len_protos()
    elif in_message[0] == "print current value":
        return agent._max_action(agent.last_state, isprint=True)
    return
