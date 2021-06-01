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
import utils.REM_model_kdt_realCov as rem
# import utils.REM_model_mykdt_realCov as rem

# import utils.KernModelupdate as rem
# import utils.auto_encoder_2branch as atec
import utils.get_learned_representation as glr
import os

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
TCREPVF_NORMREPMODEL_FIXCOV = 12 # tile coding rep [0,1] for VF, normalized rep [-1, 1] for model learning, fixed coviance
BIASTCREPVF_NORMREPMODEL_FIXCOV = 13

class REM_Dyna:

    # Default values
    def __init__(self):
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
        self.tc = utc.TileCoding(self.dim_state, self.num_tilings, self.num_tiles, self.num_action)

        """
        mode = 0: old rem dyna
        mode = 1: use tile coding for value function, save xy in model, sample xy in model, use Gaussian Kernel
                  check distance from s to sampled predecessor and successor 
        mode = 2: use learned representation for both value function and sampling
                  use Euclidean distance 
        """
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
        self.model_params = {"kscale":0.05}
        self.model = rem.REM_Model(self.dim_state, self.num_near, self.add_prot_limit, self.model_params,
                                   self.learning_mode, self.similarity_limit, self.norm_diff)
        # self.model = rem.KernModel(self.dim_state, 1000, 100, 0.01, 0.0001, 1)

        self.weight = []
        self.b_time = 0
        self.buffer = np.zeros((self.len_buffer, self.dim_state*2+4))#{"pri_time": np.zeros((2, self.len_buffer)), "sequence": {}}
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
        self.always_add_prot = param["always_add_prot"]

        self.rem_type = param["rem_type"]
        self.alpha = param["alpha"]
        self.epsilon = param["epsilon"]
        self.gamma = param["agent_gamma"]
        self.num_planning = param["num_planning"]
        self.num_branching = param["num_branching"]

        self.dim_state = param["dim_state"] # 2
        self.action_mode = param["action_mode"]
        self.num_action = param["num_action"]
        self.action_list = [i for i in range(self.num_action)]

        self.num_near = param["num_near"]
        # self.tc_mem_size = param["tc_mem_size"]
        # with open("./utils/iht/iht_"+str(self.tc_mem_size)+".pkl", 'rb') as f:
        #     self.iht = pkl.load(f) #tc.IHT(self.tc_mem_size)

        # self.num_tilings = param["num_tilings"]
        # self.num_tiles = param["num_tiles"]

        self.learning_mode = param["remDyna_mode"]

        self.div_actBit = None
        self.div_norm = None

        # 2， 9
        if self.learning_mode == SINGLE_REP or self.learning_mode == SINGLE_REP_CHECKDIST:
            self.num_tilings = 32
            self.num_tiles = 4
            self.len_s_feature = 2 #self.num_tilings * self.num_tiles * self.dim_state
            self.len_s_rep = param["nn_num_feature"]
            self.dim_state = param["nn_num_feature"]
            self.div_actBit = None
            self.div_norm = 1

        # 10, 11
        elif self.learning_mode == SINGLE_NORMREP or self.learning_mode == SINGLE_NORMREP_FIXCOV:
            self.num_tilings = 32
            self.num_tiles = 4
            self.len_s_feature = 2 #self.num_tilings * self.num_tiles * self.dim_state
            self.len_s_rep = param["nn_num_feature"]
            self.dim_state = param["nn_num_feature"]
            self.div_actBit = None
            self.div_norm = 1

        # 12
        elif self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV:
            self.num_tilings = 32
            self.num_tiles = 4
            self.len_s_feature = 2
            self.rep_tilings = 1
            self.rep_tiles = 8
            self.rep_mem_size = 8
            self.len_s_rep = self.rep_mem_size * param["nn_num_feature"]
            self.dim_state = param["nn_num_feature"]
            with open("./utils/iht/iht_" + str(self.rep_mem_size) + ".pkl", 'rb') as f:
                self.rep_iht = pkl.load(f)
            self.div_actBit = 32
            self.div_norm = None

        # 13
        elif self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV:
            self.num_tilings = 32
            self.num_tiles = 4
            self.len_s_feature = 2
            self.rep_tilings = 1
            self.rep_tiles = 8
            self.rep_mem_size = 8
            self.len_s_feature = 2
            self.len_s_rep = self.rep_mem_size * param["nn_num_feature"] + 1
            self.dim_state = param["nn_num_feature"]
            with open("./utils/iht/iht_" + str(self.rep_mem_size) + ".pkl", 'rb') as f:
                self.rep_iht = pkl.load(f)
            self.div_actBit = 33
            self.div_norm = None

        # 3
        elif self.learning_mode == REPVF_RAWMODEL_CHECKDIST:
            self.num_tilings = 32
            self.num_tiles = 4
            self.len_s_feature = 2 #self.num_tilings * self.num_tiles * self.dim_state
            self.len_s_rep = param["nn_num_feature"]
            self.div_actBit = None
            self.div_norm = 1

        # 4
        elif self.learning_mode == TCREPVF_RAWMODEL_CHECKDIST:
            self.num_tilings = 32
            self.num_tiles = 4
            self.rep_tilings = 1
            self.rep_tiles = 8
            self.rep_mem_size = 8
            self.len_s_feature = 2#self.num_tilings * self.num_tiles * self.dim_state
            self.len_s_rep = self.rep_mem_size * param["nn_num_feature"]
            with open("./utils/iht/iht_" + str(self.rep_mem_size) + ".pkl", 'rb') as f:
                self.rep_iht = pkl.load(f)  # tc.IHT(self.tc_mem_size)
            self.div_actBit = 32
            self.div_norm = None

        # 5
        elif self.learning_mode == BIASREPVF_RAWMODEL_CHECKDIST:
            self.num_tilings = 32
            self.num_tiles = 4
            self.len_s_feature = 2#self.num_tilings * self.num_tiles * self.dim_state
            self.len_s_rep = param["nn_num_feature"] + 1
            self.div_actBit = None
            self.div_norm = 1

        # 6
        elif self.learning_mode == BIASTCREPVF_RAWMODEL_CHECKDIST:
            self.num_tilings = 32
            self.num_tiles = 4
            self.rep_tilings = 1
            self.rep_tiles = 8
            self.rep_mem_size = 8
            self.len_s_feature = 2 #self.num_tilings * self.num_tiles * self.dim_state
            self.len_s_rep = self.rep_mem_size * param["nn_num_feature"] + 1
            with open("./utils/iht/iht_" + str(self.rep_mem_size) + ".pkl", 'rb') as f:
                self.rep_iht = pkl.load(f)  # tc.IHT(self.tc_mem_size)
            self.div_actBit = 33
            self.div_norm = None

        # 7
        elif self.learning_mode == BIASTCREPVF_REPMODEL:
            self.num_tilings = 32
            self.num_tiles = 4
            self.rep_tilings = 1
            self.rep_tiles = 8
            self.rep_mem_size = 8
            self.len_s_feature = 2 #self.num_tilings * self.num_tiles * self.dim_state
            self.len_s_rep = self.rep_mem_size * param["nn_num_feature"] + 1
            self.dim_state = param["nn_num_feature"]
            with open("./utils/iht/iht_" + str(self.rep_mem_size) + ".pkl", 'rb') as f:
                self.rep_iht = pkl.load(f)  # tc.IHT(self.tc_mem_size)
            self.div_actBit = 33
            self.div_norm = None

        # 8
        elif self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST:
            self.num_tilings = 32
            self.num_tiles = 4
            self.rep_tilings = 1
            self.rep_tiles = 8
            self.rep_mem_size = 8
            self.len_s_feature = 2#self.num_tilings * self.num_tiles * self.dim_state
            self.len_s_rep = self.rep_mem_size * param["nn_num_feature"] + 1
            self.dim_state = param["nn_num_feature"]
            with open("./utils/iht/iht_" + str(self.rep_mem_size) + ".pkl", 'rb') as f:
                self.rep_iht = pkl.load(f)  # tc.IHT(self.tc_mem_size)
            self.div_actBit = 33
            self.div_norm = None

        # 0, 1
        else:
            self.num_tilings = 1
            self.num_tiles = 16
            self.tc_mem_size = param["tc_mem_size"]
            self.len_s_feature = param["tc_mem_size"]  #self.num_tilings * self.num_tiles ** self.dim_state #
            self.len_s_rep = self.len_s_feature
            self.div_actBit = self.num_tilings
            self.div_norm = None

        self.len_sa_feature = self.len_s_rep * self.num_action

        if self.div_actBit is not None:
            self.alpha = param["alpha"] / float(self.div_actBit)
        else:
            self.alpha = param["alpha"]

        # self.tc = utc.TileCoding(self.dim_state, self.num_tilings, self.num_tiles, self.num_action)
        if self.learning_mode == SINGLE_REP or \
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
                self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV:
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
            self.path = "./feature_model/without batchNorm illegal_v xy_input/distance_matrix_model-beta0.1_delta1.0_gamma[0.998, 0.8]_lr0.0001-0.0001_epoch100-100_scale0/"
            self.num_input = self.len_s_feature
            self.num_output = self.num_tilings * self.num_tiles * 2 * 2
            self.continuous = True
            self.constraint = True
            self.file_name = param["nn_model_name"] + "_continuous"
            self.tc = utc.TileCoding(1, self.num_tilings, self.num_tiles, 1)
            self.rep_model = glr.GetLearnedRep(self.num_input, self.num_node, self.num_feature, self.num_output, self.lr,
                                               self.lr_rcvs, self.wd, self.num_dec_node, self.num_rec_node,
                                               self.optimizer, self.dropout, self.beta, self.delta, self.legal_v,
                                               self.continuous, num_tiling=self.num_tilings, num_tile=self.num_tiles, constraint=self.constraint,
                                               model_path=self.path, file_name=self.file_name)
        else:
            self.tc = utc.TileCoding(self.dim_state, self.num_tilings, self.num_tiles, 1)

        if param["init_weight"] == "0":
            self.weight = np.zeros(self.len_sa_feature)
            #self.weight = np.zeros((self.len_sa_feature))
        elif param["init_weight"] == "1":
            self.weight = np.ones(self.len_sa_feature)
            #self.weight = np.ones((self.len_sa_feature))
        else:
            print("HAVEN't BEEN DONE YET")
            exit(-1)
        self.len_buffer = param["len_buffer"]
        self.pri_thrshd = param["pri_thrshd"]

        self.b_time = 0
        self.buffer = np.zeros((self.len_buffer, self.dim_state * 2 + 4))

        self.add_prot_limit = param["add_prot_limit"]
        self.similarity_limit = param["similarity_limit"]
        self.model_params = param["model_params"]
        self.norm_diff = param["rbf_normalize_diff"]

        if self.learning_mode == OLD_REM:
            self.model = rem.REM_Model(self.dim_state, self.num_near, self.add_prot_limit, self.model_params,
                                   self.learning_mode, self.similarity_limit, self.norm_diff)
        else:
            self.model = rem.REM_Model(self.dim_state, self.num_near, self.add_prot_limit, self.model_params,
                                       self.learning_mode, self.similarity_limit, self.norm_diff,
                                       rep_model=self.rep_model)
        return

    """
    Input: [x, y]
    Return: action
    """
    def start(self, state):
        if self.learning_mode == SINGLE_REP or \
                self.learning_mode == SINGLE_NORMREP or\
                self.learning_mode == SINGLE_NORMREP_FIXCOV or \
                self.learning_mode == BIASTCREPVF_REPMODEL or \
                self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST or \
                self.learning_mode == SINGLE_REP_CHECKDIST or \
                self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV:
            state = self._state_representation(state)
        self.state = state
        self.action = self._policy(state)
        self.check_total_time = np.zeros(6)
        return self.action

    """
    Input: int, [x, y]
    Return: action
    """
    def step(self, reward, state):
        if self.learning_mode == SINGLE_REP or \
                self.learning_mode == SINGLE_NORMREP or \
                self.learning_mode == SINGLE_NORMREP_FIXCOV or \
                self.learning_mode == BIASTCREPVF_REPMODEL or \
                self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST or \
                self.learning_mode == SINGLE_REP_CHECKDIST or \
                self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV:
            state = self._state_representation(state)
        # update variables
        self.last_state = self.state
        self.last_action = self.action
        self.state = state
        self.reward = reward
        # use Q-learning to update weight
        qstart = time.time()
        self._update_weight(self.last_state, self.last_action, self.state, reward, self.gamma, self.alpha)
        qtime = time.time() - qstart

        other_info = None
        if self.learning:
            if self.rem_type != "random_BufferOnly":
                # update model
                self.model.add2Model(self.last_state, self.last_action, self.state, reward, self.gamma)

            # get tde
            tde = self._get_tde(self.last_state, self.last_action, self.state, reward, self.gamma)

            # insert S,A into buffer
            self._insert_seq(self.last_state, self.last_action, self.state, reward, self.gamma, np.abs(tde)+self.pri_thrshd)

            # planning
            other_info = {"plan":[]}
            indexs, seqs = self._sample_seqs_from_buffer(self.num_planning)
            for i in range(self.num_planning):
                other_info["plan"].append(self._single_planning(self.alpha, self.num_planning, indexs[i], seqs[i]))
            other_info["buffer"] = self.buffer[:min(self.b_time, self.len_buffer)]
            other_info["protos"] = self.model.get_protos()
        elif not self.learning and self.always_add_prot:
            if self.rem_type != "random_BufferOnly":
                # update model even when the agent is not learning
                self.model.add2Model(self.last_state, self.last_action, self.state, reward, self.gamma)
                other_info = {"protos": self.model.get_protos()}

        # choose new action
        self.action = self._policy(state, False)
        return self.action, other_info

    """
    Input: int, [x, y]
    Return: None
    """
    def end(self, reward, state):
        self.step(reward, state)
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
            # TODO
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
            # all_choices.append(np.sum(self.weight[feature]))
            all_choices.append(np.dot(self.weight,feature))
            # if math.isnan(all_choices[-1]):
            #     print("_max_action: w^Tf=nan\n weight and feature are: ", self.weight,feature)
        valid_index = self._break_tie(all_choices)
        if isprint:
            print(all_choices, valid_index)
        return valid_index

    """
    Break tie fairly
    Input: qvalue
    Return: optimal action
    """
    def _break_tie(self, xarray):
        max_v = np.max(xarray)
        valid_choices = np.where(xarray == max_v)[0]
        return np.random.choice(valid_choices)

    """
    Generate learned representation
    Input: [x, y]
    Return: state representation
    """
    def _state_representation(self, state):
        # f = np.zeros(self.len_s_feature)
        # xind = self.tc.get_index([state[0]])
        # yind = self.tc.get_index([state[1]])
        # f[xind] = 1
        # f[yind + self.num_tiles * self.num_tilings] = 1
        # [rep], _, _, _ = self.rep_model.test(f.reshape((1, -1)), len(f))
        rep = self.rep_model.state_representation(np.array(state))
        if self.learning_mode == SINGLE_NORMREP or \
                self.learning_mode == SINGLE_NORMREP_FIXCOV or \
                self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV:
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
            feature[self.len_s_rep * int(action): self.len_s_rep * (int(action) + 1)] = state # rep

        elif self.learning_mode == SINGLE_NORMREP or \
                self.learning_mode == SINGLE_NORMREP_FIXCOV:
            feature = np.zeros(self.len_sa_feature)
            state = state / float(np.linalg.norm(state))
            feature[self.len_s_rep * int(action): self.len_s_rep * (int(action) + 1)] = state

        # 12
        elif self.learning_mode ==  TCREPVF_NORMREPMODEL_FIXCOV:
            feature = np.zeros(self.len_sa_feature)
            state = (state / float(np.linalg.norm(state)) + 1) / 2.0
            for d in range(len(state)):
                ind = np.array(tc.tiles(self.rep_iht, self.rep_tilings, [float(self.rep_tiles) * state[d]]))
                feature[self.len_s_rep * int(action) + d * self.rep_mem_size + ind] = 1

        # 13
        elif self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV:
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

        elif self.learning_mode == TCREPVF_RAWMODEL_CHECKDIST:
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

        else:
            # indices = tc.tiles(self.iht, self.num_tilings, float(self.num_tiles) * np.array(state), [action])
            feature = np.zeros(self.len_sa_feature)

            # xind = self.tc.get_index([state[0]])
            # yind = self.tc.get_index([state[1]])
            # xind = np.array(tc.tiles(self.iht, self.num_tilings, float(self.num_tiles) * np.array([state[0]])))
            # yind = np.array(tc.tiles(self.iht, self.num_tilings, float(self.num_tiles) * np.array([state[1]])))
            # feature[self.len_s_rep * int(action) + xind] = 1
            # feature[self.len_s_rep * int(action) + self.len_s_rep // self.dim_state + yind] = 1

            #ind = self.tc.get_index(state)
            state = np.clip(np.array(state), 0.0, 1.0)
            ind = np.array(tc.tiles(self.iht, self.num_tilings, float(self.num_tiles) * np.array(state)))
            feature[self.len_s_rep * int(action) + ind] = 1
        return feature

    """
    Update weight for learning value function
    Input: [x, y]-last, action-last, [x,y], reward, gamma, lr
    Return: None
    """
    def _update_weight(self, last_state, last_action, state, reward, gamma, alpha):
        last_feature = self._feature_construction(last_state, last_action)
        feature = self._feature_construction(state, self._max_action(state))
        tde = self._td_error(last_feature, feature, reward, gamma, self.weight)
        if tde > 0:
            self.learning = True
        if self.div_norm:
            alpha = alpha / np.linalg.norm(last_feature)
            # print("div norm", alpha, np.linalg.norm(last_feature))
        self.weight += alpha * tde * last_feature
        return

    """
    Calculate TD error given feature
    Input: feature-last, feature, reward, gamma, weight)
    Return: TD-error
    """
    def _td_error(self, last_feature, feature, reward, gamma, weight):
        # tde = reward + gamma * np.sum(weight[feature]) - np.sum(weight[last_feature])
        tde = reward + gamma * np.dot(weight, feature) - np.dot(weight, last_feature)
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
        new_tde = self._get_tde(last_state, last_action, state, reward, gamma)
        self.buffer[index, -1] = np.abs(new_tde)+self.pri_thrshd
        return

    """
    Insert sample into buffer
    Input: [x, y]-last, action-last, [x, y], reward, gamma, TD-error
    Return: None
    """
    def _insert_seq(self, last_state, last_action, state, reward, gamma, tde):
        new_sequence = self._seq_to_array(last_state, last_action, state, reward, gamma, tde)
        self.buffer[self.b_time % self.len_buffer, :] = new_sequence
        self.b_time += 1
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
                self._update_weight(last_state, last_action, state, reward, gamma, float(alpha) / np.sqrt(n))
                self._update_priority(index, last_state, last_action, state, reward, gamma)
            else:
                # sample s',r,g from model
                sample = self.model.KDE_sampleSpRG(last_state, last_action)
                if sample is not None:
                    _, state, reward, gamma, _ = sample
                    gamma = self.gamma if gamma is None else gamma
                    self._update_weight(last_state, last_action, state, reward, gamma, float(alpha) / np.sqrt(n))
                    self._update_priority(index, last_state, last_action, state, reward, gamma)

                    if self.rem_type == "random_pred" or self.rem_type == "pri_pred":
                        sbab_list = self.model.sampleFromNext_pan(last_state, self.num_branching, self.num_action)
                        # preds = []
                        for sbab in sbab_list:
                            preds.append([sbab[0], last_state])
                            sample_b = self.model.KDE_sampleSpRG(sbab[0], sbab[1])
                            if sample_b is not None:
                                _, spb, rb, gb, _ = sample_b
                                gb = self.gamma if gb is None else gb
                                pri = np.abs(self._get_tde(sbab[0], sbab[1], spb, rb, gb))
                                if pri >= self.pri_thrshd:
                                    self._insert_seq(sbab[0], sbab[1], spb, rb, gb, pri + self.pri_thrshd)
            return {"state": last_state, "q": self._check_q(last_state), "sbab_list": preds}

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
            qvalue.append(np.dot(self.weight, feature))
        return np.array(qvalue)

    """
    Choose sequence from buffer
    For now we use elif block
    Input: number of plannings
    Return: index in buffer, sasprg-array
    """
    def _sample_seqs_from_buffer(self, n):
        if self.rem_type == "random" or self.rem_type == "random_pred" or self.rem_type == "random_BufferOnly":
            indexs = np.random.randint(0, min(self.b_time, self.len_buffer), size=n)
        elif self.rem_type == "pri" or self.rem_type == "pri_pred":
            indexs = self._sample_break_tie(self.buffer[:min(self.b_time, self.len_buffer), -1], min(self.b_time, n))
            if self.b_time < n:
                indexs = np.array(list(indexs)*(n//self.b_time+1))[:n]
        else:
            print("UNKNOWN TYPE")
        seqs = np.copy(self.buffer[indexs, :])
        return indexs, seqs

    """
    Choose samples with highest priority
    """
    def _sample_break_tie(self, pris, num):
        pris = np.copy(pris)
        indexs = []
        for i in range(num):
            indexs.append(self._break_tie(pris))
            pris[indexs[i]] = -1000000
            # indexs.append(self._break_tie(pris))
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
        state = seq[self.dim_state+1: self.dim_state*2+1]
        reward = seq[self.dim_state*2+1]
        gamma = seq[self.dim_state*2+2]
        tde = seq[self.dim_state*2+3]
        return last_state, last_action, state, reward, gamma, tde

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
    return