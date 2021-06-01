# -*- coding: utf-8 -*-

import time
import numpy as np
import random
import pickle as pkl

import utils.tiles3 as tc
import utils.TileCoding as utc

import utils.auto_encoder_2branch as atec
import utils.get_learned_representation as glr

import torch
from torch import nn

OLD_Q = 0
SINGLE_REP = 2
TCREP = 3

class Q_learning():

    # Default values
    def __init__(self):
        self.alpha = 0.1
        self.epsilon = 0.1
        self.gamma = 0.9

        self.dim_state = 1
        self.action_mode = "discrete"
        self.num_action = 1
        self.action_list = [i for i in range(self.num_action)]

        self.tc_mem_size = 1024
        self.iht = tc.IHT(self.tc_mem_size)
        self.num_tilings = 1
        self.num_tiles = 1

        self.weight = None
        self.last_state = None
        self.last_action = None

        self.state = None
        self.action = None
        self.check_time = []
        self.check_total_time = np.zeros(6)

        self.learning = False
        return

    def set_param(self, param):
        self.alg = param["alg"]#'Q' or Sarsa
        self.traces_lambda = param["lambda"]

        self.learning_mode = param["qLearning_mode"]

        self.epsilon = param["epsilon"]
        self.gamma = param["agent_gamma"]

        self.dim_state = param["dim_state"]
        self.action_mode = param["action_mode"]
        self.num_action = param["num_action"]
        self.action_list = [i for i in range(self.num_action)]

        self.div_actBit = None
        self.div_norm = None

        if self.learning_mode == SINGLE_REP:
            # self.num_tilings = 32
            # self.num_tiles = 4
            self.len_s_feature = 2
            self.len_s_rep = param["nn_num_feature"]
            self.dim_state = self.len_s_rep
            self.div_norm = 1

        elif self.learning_mode == TCREP:
            # self.num_tilings = 32
            # self.num_tiles = 4
            self.rep_tilings = 1
            self.rep_tiles = 8
            self.rep_mem_size = 8
            self.len_s_feature = 2
            self.len_s_rep = self.rep_mem_size * param["nn_num_feature"]
            self.rep_iht = tc.IHT(self.rep_mem_size)
            self.div_actBit = 32

        else:
            self.num_tilings = 1
            self.num_tiles = 10
            self.tc_mem_size = 100
            self.iht = tc.IHT(self.tc_mem_size)
            self.len_s_feature = self.tc_mem_size  # self.num_tilings * self.num_tiles ** self.dim_state #
            self.len_s_rep = self.len_s_feature
            self.div_actBit = self.num_tilings

        self.len_sa_feature = self.len_s_rep * self.num_action

        if self.div_actBit is not None:
            self.alpha = param["alpha"] / np.sqrt(float(self.div_actBit))
        else:
            self.alpha = param["alpha"]

        if self.learning_mode == SINGLE_REP or \
                self.learning_mode == TCREP:
            self.num_node = param["nn_nodes"]
            self.num_dec_node = param["nn_dec_nodes"]
            self.num_feature = param["nn_num_feature"]
            self.num_rec_node = param["nn_rec_nodes"]
            self.optimizer = param["optimizer"]
            self.lr = param["nn_lr"]
            self.wd = param["nn_weight_decay"]
            self.dropout = param["nn_dropout"]
            self.num_epochs = param["nn_num_epochs"]
            self.batch_size = param["nn_batch_size"]
            self.beta = param["nn_beta"]
            self.delta = param["nn_delta"]
            self.legal_v = param["nn_legal_v"]
            self.constraint = param["nn_constraint"]
            self.path = "./feature_model/"
            self.num_input = self.len_s_feature
            self.num_output = 32 * 4 * 2 * 2
            self.file_name = "feature_embedding_continuous_input[0.0, 1]_envSucProb1.0"
            # self.tc = utc.TileCoding(1, 32, 4, 1)
            self.rep_model = glr.GetLearnedRep(self.num_input, self.num_node, self.num_feature, self.num_output,
                                               self.lr,
                                               self.lr, self.wd, self.num_dec_node, self.num_rec_node,
                                               self.optimizer, self.dropout, self.beta, self.delta, self.legal_v,
                                               True, num_tiling=32, num_tile=4,
                                               constraint=self.constraint,
                                               model_path=self.path, file_name=self.file_name)
            print("Number of input for representation model:", self.num_input)

        self.momentum = param["momentum"]
        self.rms = param["rms"]
        self.opt_mode = param["opt_mode"]

        if param["init_weight"] == "0":
            # self.weight = np.zeros(self.len_sa_feature)
            #self.weight = np.zeros((self.len_sa_feature))
            self.weight = nn.Linear(self.len_sa_feature, 1, bias=False)
            self.weight.weight.data.fill_(0.)
            # nn.init.normal_(self.weight.weight, mean=0, std=0.001)

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
        elif param["init_weight"] == "1":
            self.weight = np.ones(self.len_sa_feature)
            #self.weight = np.ones((self.len_sa_feature))
        else:
            print("HAVEN't BEEN DONE YET")
            exit(-1)

        return

    def start(self, state):
        if self.learning_mode == SINGLE_REP:
            state = self._state_representation(state)
        self.state = state
        self.action = self._policy(state)
        self.check_total_time = np.zeros(6)

        if self.alg == 'Sarsa':
            self.traces = torch.from_numpy(np.zeros((self.len_sa_feature))).float()
        return self.action

    def step(self, reward, state):
        # update variables
        if self.learning_mode == SINGLE_REP:
            state = self._state_representation(state)
        self.last_state = self.state
        self.last_action = self.action
        self.state = state
        self.reward = reward

        if self.alg == 'Sarsa':
            # choose new action
            cstart = time.time()
            self.action = self._policy(state)
            ctime = time.time() - cstart

            # use Q-learning to update weight
            qstart = time.time()
            self._update_weight(self.last_state, self.last_action, self.state, reward, self.gamma, self.alpha, self.action)
            qtime = time.time() - qstart

        else:
            # use Q-learning to update weight
            qstart = time.time()
            self._update_weight(self.last_state, self.last_action, self.state, reward, self.gamma, self.alpha)
            qtime = time.time() - qstart
            # choose new action
            cstart = time.time()
            self.action = self._policy(state)
            ctime = time.time() - cstart



        return self.action, None

    def end(self, reward, state):
        self.step(reward, state)
        return

    def _policy(self, state):
        if self.action_mode == "discrete":
            if random.random() < self.epsilon:
                return np.random.randint(0, len(self.action_list))
            else:
                return self._max_action(state)
        elif self.action_mode == "continuous":
            # TODO
            print("NOT DONE YET")
            return
        else:
            print("UNKNOWN ACTION MODE")
        return

    def _max_action(self, state):
        all_choices = []
        for a in self.action_list:
            feature = self._feature_construction(state, a)
            all_choices.append(np.dot(self.weight.weight.data, feature))
        valid_index = self._break_tie(all_choices)
        return valid_index

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
        rep = self.rep_model.state_representation(np.array(state))
        return rep

    # return activated indices
    def _feature_construction(self, state, action):
        if self.learning_mode == SINGLE_REP:
            feature = np.zeros(self.len_sa_feature)
            feature[self.len_s_rep * int(action): self.len_s_rep * (int(action) + 1)] = state # rep

        elif self.learning_mode == TCREP:
            feature = np.zeros(self.len_sa_feature)
            rep = self._state_representation(state)
            rep = (rep / float(np.linalg.norm(rep)) + 1) / 2.0
            for d in range(len(rep)):
                ind = np.array(tc.tiles(self.rep_iht, self.rep_tilings, [float(self.rep_tiles) * rep[d]]))
                feature[self.len_s_rep * int(action) + d * self.rep_mem_size + ind] = 1

        else:
            feature = np.zeros(self.len_sa_feature)
            ind = np.array(tc.tiles(self.iht, self.num_tilings, float(self.num_tiles) * np.array(state)))
            feature[self.len_s_rep * int(action) + ind] = 1
        return feature

    def _update_weight(self, last_state, last_action, state, reward, gamma, alpha, current_action = None):
        last_feature = self._feature_construction(last_state, last_action)
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
        self.weight_optimizer.zero_grad()
        loss = 0.0*torch.norm(self.weight.weight)
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

        loss = 0.0*torch.norm(self.weight.weight)#self.weight)
        loss.backward()
        return

    def _td_error(self, last_feature, feature, reward, gamma, weight):
        tde = reward + gamma * np.dot(feature, weight.weight.data.reshape(-1)) - np.dot(last_feature, weight.weight.data.reshape(-1))
        return tde

    def check_policy(self, data):
        policy = []
        for state in data:
            a = self._max_action(state)
            policy.append(a)
        return policy

    def check_weight(self):
        return (self.weight.weight.data).numpy()

agent = None


def agent_init():
    global agent
    agent = Q_learning()
    return


def agent_start(state):
    global agent
    current_action = agent.start(state)
    return current_action


def agent_step(reward, state):
    global agent
    current_action = agent.step(reward, state)
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
            print("the agent hasn't been initialized.")
    if in_message[0] == "check_policy":
        return agent.check_policy(in_message[1])
    if in_message[0] == "check_weight":
        return agent.check_weight()
    return
