# -*- coding: utf-8 -*-
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

OLD_ER = 0
SINGLE_REP = 2
TCREP = 3
BIASTCREP = 4

class RandomER:

    # Default values
    def __init__(self):
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

        self.len_buffer = 1
        self.num_near = 3
        self.add_prot_limit = 0.8

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
        return

    def set_param(self, param):
        self.graph = param["graph"]

        self.alg = param["alg"]#'Q' or Sarsa
        self.traces_lambda = param["lambda"]
        self.opt_mode = param["opt_mode"]

        self.learning_mode = param["erLearning_mode"]
        self.sample_mode = "random"#param["sample_mode"]

        self.epsilon = param["epsilon"]
        self.gamma = param["agent_gamma"]

        self.num_planning = param["num_planning"]
        self.num_branching = param["num_branching"]

        self.dim_state = param["dim_state"]
        self.action_mode = param["action_mode"]
        self.num_action = param["num_action"]
        self.action_list = [i for i in range(self.num_action)]

        self.num_near = param["num_near"]

        self.div_actBit = None
        self.div_norm = None

        if self.learning_mode == SINGLE_REP:
            if self.graph:
                exit(-1)
            self.len_s_feature = 2
            self.len_s_rep = param["nn_num_feature"]
            self.dim_state = param["nn_num_feature"]
            self.div_norm = None

        elif self.learning_mode == TCREP:
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

        elif self.learning_mode == BIASTCREP:
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

        else:
            if self.graph:
                self.tc = utc.TileCoding(2, 1, param["nn_num_tiles"])
                self.nn_num_tile = param["nn_num_tiles"]
                self.nn_num_tiling = 1
                self.len_s_feature = param["nn_num_tiles"] ** 2
                self.len_s_rep = param["nn_num_tiles"] ** 2

                self.dim_state = self.len_s_feature
            else:
                self.num_tilings = 32
                self.num_tiles = 4
                self.tc_mem_size = 512
                self.iht = tc.IHT(self.tc_mem_size)
                self.len_s_feature = self.tc_mem_size
                self.len_s_rep = self.len_s_feature

                if self.opt_mode == 4:
                    self.div_actBit = self.num_tilings

        self.len_sa_feature = self.len_s_rep * self.num_action

        if self.div_actBit is not None:
            self.alpha = param["alpha"] / np.sqrt(float(self.div_actBit))
        else:
            self.alpha = param["alpha"]

        if self.learning_mode == SINGLE_REP or \
                self.learning_mode == TCREP or \
                self.learning_mode == BIASTCREP:
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
            self.file_name = param["nn_model_name"] + "_continuous"

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

        if param["init_weight"] == "0":
            self.weight = nn.Linear(self.len_sa_feature, 1, bias=False)
            self.weight.weight.data.fill_(0.)

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
            self.weight = np.ones(self.tc_mem_size)
        else:
            print("HAVEN't BEEN DONE YET")
            exit(-1)

        self.len_buffer = param["len_buffer"]

        self.b_time = 0
        self.buffer = np.zeros((self.len_buffer, self.dim_state * 2 + 4))

        self.add_prot_limit = param["add_prot_limit"]

        return

    def start(self, state):
        if self.graph:
            state = self._change_to_graph(state)

        if self.learning_mode == SINGLE_REP or \
                self.learning_mode == TCREP or \
                self.learning_mode == BIASTCREP:
            state = self._state_representation(state)

        self.state = state
        self.action = self._policy(state)
        self.check_total_time = np.zeros(6)
        if self.alg == 'Sarsa':
            self.traces = torch.from_numpy(np.zeros((self.len_sa_feature))).float()
        return self.action

    def step(self, reward, state):
        if self.graph:
            state = self._change_to_graph(state)
        if self.learning_mode == SINGLE_REP or \
                self.learning_mode == TCREP or \
                self.learning_mode == BIASTCREP:
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

        other_info = {"plan":[]}
        if True:
            # get priority
            gstart = time.time()
            tde = self._get_tde(self.last_state, self.last_action, self.state, reward, self.gamma)
            gtime = time.time() - gstart

            # insert S,A into buffer
            istart = time.time()
            self._insert_seq(self.last_state, self.last_action, self.state, reward, self.gamma, tde)
            itime = time.time() - istart

            # planning
            if self.learning:
                pstart = time.time()
                current_planning_steps = min(self.b_time, self.len_buffer, self.num_planning)
                for _ in range(current_planning_steps):
                    other_info["plan"].append(self._single_planning(self.alpha, self.num_planning))
                ptime = time.time() - pstart

        # choose new action
        cstart = time.time()
        self.action = self._policy(state)
        ctime = time.time() - cstart

        other_info["buffer"] = self.buffer[:min(self.b_time, self.len_buffer)]

        # if self.learning:
        #     print(self._check_q(self.last_state))
        return self.action, None

    def end(self, reward, state):
        print("End of episode.", self.state, self.action)
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
            # all_choices.append(np.sum(self.weight[feature]))
        valid_index = self._break_tie(all_choices)
        return valid_index

    def _break_tie(self, xarray):
        max_v = np.max(xarray)
        valid_choices = np.where(xarray == max_v)[0]
        return np.random.choice(valid_choices)

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
            state = (state / float(np.linalg.norm(state)) + 1) / 2.0
            for d in range(len(state)):
                ind = np.array(tc.tiles(self.rep_iht, self.rep_tilings, [float(self.rep_tiles) * state[d]]))
                feature[self.len_s_rep * int(action) + d * self.rep_mem_size + ind] = 1

        elif self.learning_mode == BIASTCREP:
            feature = np.zeros(self.len_sa_feature)
            state = (state / float(np.linalg.norm(state)) + 1) / 2.0
            for d in range(len(state)):
                ind = np.array(tc.tiles(self.rep_iht, self.rep_tilings, [float(self.rep_tiles) * state[d]]))
                feature[self.len_s_rep * int(action) + d * self.rep_mem_size + ind] = 1
            feature[self.len_s_rep * (int(action) + 1) - 1] = 1

        else:
            if self.graph:
                feature = np.zeros(self.len_sa_feature)
                feature[
                self.len_s_feature * int(action): self.len_s_feature * (int(action) + 1)] = state  # one-hot encoding
                return feature

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
        if np.abs(tde) > 0 and not self.learning:
            self.learning = True

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
                # if self.learning:
                    # print(tde)
                self.weight.weight.grad -= tde*last_feature_torch

        self.weight_optimizer.step()

        loss = 0.0*torch.norm(self.weight.weight)#self.weight)
        loss.backward()

        return

    def _td_error(self, last_feature, feature, reward, gamma, weight):
        tde = reward + gamma * np.dot(feature, weight.weight.data.reshape(-1)) \
              - np.dot(last_feature, weight.weight.data.reshape(-1))
        # tde = reward + gamma * np.sum(weight[feature]) - np.sum(weight[last_feature])
        return tde

    def _get_tde(self, last_state, last_action, state, reward, gamma):
        last_feature = self._feature_construction(last_state, last_action)
        feature = self._feature_construction(state, self._max_action(state))
        tde = self._td_error(last_feature, feature, reward, gamma, self.weight)
        return tde

    def _update_priority(self, index, last_state, last_action, state, reward, gamma):
        new_tde = self._get_tde(last_state, last_action, state, reward, gamma)
        self.buffer[index, -1] = new_tde
        return

    def _insert_seq(self, last_state, last_action, state, reward, gamma, tde):
        new_sequence = self._seq_to_array(last_state, last_action, state, reward, gamma, tde)
        self.buffer[self.b_time % self.len_buffer, :] = new_sequence
        self.b_time += 1
        return

    def _single_planning(self, alpha, n):
        index, seq = self._sample_seq_from_buffer()
        if seq is not None:
            last_state, last_action, state, reward, gamma, tde = self._array_to_seq(seq)
            # print('Priority: ',tde)
            # self._update_weight(last_state, last_action, state, reward, gamma, float(alpha) / np.sqrt(n))
            self._update_weight(last_state, last_action, state, reward, gamma, float(alpha))
            self._update_priority(index, last_state, last_action, state, reward, gamma)
            return {"state": last_state, "q": self._check_q(last_state)}

    def _check_q(self, state):
        qvalue = []
        for a in range(self.num_action):
            feature = self._feature_construction(state, a)
            qvalue.append(np.dot(self.weight.weight.data, feature))
            # qvalue.append(np.sum(self.weight[feature]))
        return np.array(qvalue)

    def _sample_seq_from_buffer(self):
        if self.sample_mode == "random":
            index = np.random.randint(0, min(self.b_time, self.len_buffer))
            seq = np.copy(self.buffer[index, :])
        elif self.sample_mode == "priority":
            index = self._sample_break_tie(self.buffer[:min(self.b_time, self.len_buffer), -1])
            seq = np.copy(self.buffer[index, :])
        return index, seq

    def _sample_break_tie(self, pris):
        pris = np.copy(pris)
        index = self._break_tie(pris)
        return index

    def _seq_to_array(self, last_state, last_action, state, reward, gamma, tde):
        return np.concatenate((last_state, np.array([last_action]), state,
                               np.array([reward]), np.array([gamma]), np.array([tde])),
                              axis=0)

    def _array_to_seq(self, seq):
        last_state = seq[:self.dim_state]
        last_action = seq[self.dim_state]
        state = seq[self.dim_state+1: self.dim_state*2+1]
        reward = seq[self.dim_state*2+1]
        gamma = seq[self.dim_state*2+2]
        tde = seq[self.dim_state*2+3]
        return last_state, last_action, state, reward, gamma, tde

    def check_weight(self):
        return (self.weight.weight.data).numpy()

    def _change_to_graph(self, state):
        idx = self.tc.get_index(state)
        graph = np.zeros((self.nn_num_tile ** 2))
        graph[idx] = 1
        return graph

agent = None


def agent_init():
    global agent
    agent = RandomER()
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
            print("the environment hasn't been initialized.")
    elif in_message[0] == "check time":
        return agent.check_time
    elif in_message[0] == "get weight size":
        return agent.len_sa_feature
    elif in_message[0] == "check total time":
        return agent.check_total_time
    elif in_message[0] == "check model size":
        return  # agent.model.b
    elif in_message[0] == "check_weight":
        return agent.check_weight()
    return
