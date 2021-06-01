#!/usr/bin/python3

import time
import math
import numpy as np
import pickle as pkl
import utils.tiles3 as tc
import utils.TileCoding as utc
import utils.get_learned_representation as glr
import utils.get_learned_state as gls
import utils.get_offline_NN as gon
import os

import torch
from torch import nn

np.set_printoptions(precision=3)


class BufferControl():
    def __init__(self, length):
    # def __init__(self, length, model):
        self.b_length = length
        self.b_empty = [i for i in range(self.b_length)]
        self.b_filled = []
        # self.model = model
        self.b_filled_length = None

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

class NN_Dyna:

    # Default values
    def __init__(self):

        return

    def set_param(self, param):
        self.gamma = param["agent_gamma"]
        self.alpha = param["alpha"]
        self.epsilon = param["epsilon"]
        self.pri_thrshd = param["pri_thrshd"]
        self.learning = False
        self.num_planning = param["num_planning"]
        if self.num_planning > 50:
            self.planning_steps = 10
        else:
            self.planning_steps = 1

        self.dim_state = 2#param["nn_num_feature"]

        self.len_buffer = param["len_buffer"]
        self.buffer = np.zeros((self.len_buffer, self.dim_state * 2 + 4))
        self.b_control = BufferControl(self.len_buffer)

        self.num_action = param["num_action"]
        self.action_list = [i for i in range(self.num_action)]

        self.num_tilings = 1
        self.num_tiles = 10
        self.len_s_feature = 2
        self.tc_mem_size = 100
        self.len_s_rep = self.tc_mem_size
        self.iht = tc.IHT(self.tc_mem_size)
        self.div_actBit = self.num_tilings
        self.div_norm = None

        self.len_output = 2
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
                                                     True, num_tiling=param["nn_num_tilings"],
                                                     num_tile=param["nn_num_tiles"], constraint=True,
                                                     model_path=param["nn_model_path"],
                                                     file_name=param["nn_model_name"] + "_seperateRcvs")
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

        self.len_sa_feature = self.len_s_rep * self.num_action
        self.weight = nn.Linear(self.len_sa_feature, 1, bias=False)
        self.weight.weight.data.fill_(0.)

        self.weight_optimizer = torch.optim.SGD(self.weight.parameters(), lr=self.alpha)

        self.offline_nn = gon.GetOfflineNN()
        return

    """
    Input: [x, y]
    Return: action
    """
    def start(self, state):
        self.state = state
        self.action = self._policy(state)
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

        # update variables


        self.last_state = self.state
        self.last_action = self.action
        self.state = state
        self.reward = reward
        # use Q-learning to update weight
        tde = self._update_weight(self.last_state, self.last_action, self.state, reward, gamma, self.alpha)

        other_info = None
        # get tde
        tde = self._get_tde(self.last_state, self.last_action, self.state, reward, gamma)
        self._insert_seq(self.last_state, self.last_action, self.state, reward, gamma, np.abs(tde)+self.pri_thrshd)

        if self.learning:
            # planning
            planning_done_steps = 0
            while planning_done_steps < int(self.num_planning/self.planning_steps) and self.b_control.get_filled_length() > 0:
                indexs, seqs = self._sample_seqs_from_buffer(self.planning_steps)
                for i in range(self.planning_steps):
                    # did_planning = self._single_planning((self.alpha/np.sqrt(self.num_planning)), self.planning_steps, indexs[i], seqs[i])
                    did_planning = self._single_planning(self.alpha, self.planning_steps, indexs[i], seqs[i])
                    if did_planning:
                        planning_done_steps += 1

        # choose new action
        self.action = self._policy(state, False)

        return self.action, other_info

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
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_list)
        else:
            return self._max_action(state, isprint)

    """
    Choose the optimal action
    Input: [x, y]
    Return: optimal action
    """
    def _max_action(self, state, isprint=False):
        all_choices = []
        for a in self.action_list:
            feature = self._feature_construction(state, a)
            all_choices.append(np.dot(self.weight.weight.data, feature))
        valid_index = self._break_tie(all_choices)
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
        rep = rep / float(np.linalg.norm(rep))
        return rep

    """
    Generate feature for learning value function
    Input: [x, y], action
    Return: (s, a)-feature
    """
    def _feature_construction(self, state, action):
        # state = self.rep_model_decoder.state_learned(state)
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
    def _update_weight(self, last_state, last_action, state, reward, gamma, alpha, current_action = None):

        last_feature = self._feature_construction(last_state, last_action)
        last_feature_torch = torch.from_numpy(last_feature).float()

        if current_action is None:
            feature = self._feature_construction(state, self._max_action(state))
        else:
            feature = self._feature_construction(state, current_action)

        tde = self._td_error(last_feature, feature, reward, gamma, self.weight)
        if np.abs(tde) > 0:
            self.learning = True

        else:
            self.weight_optimizer.zero_grad()
            loss = 0.0*torch.norm(self.weight.weight)#self.weight)
            loss.backward(torch.FloatTensor(np.asarray([0.0])),retain_graph=True)

            if self.div_norm:
                self.weight.weight.grad -= (tde/torch.norm(last_feature_torch))*last_feature_torch
            else:
                self.weight.weight.grad -= tde*last_feature_torch

            self.weight_optimizer.step()

            loss = 0.0*torch.norm(self.weight.weight)#self.weight)
            loss.backward()
        return tde

    """
    Calculate TD error given feature
    Input: feature-last, feature, reward, gamma, weight)
    Return: TD-error
    """
    def _td_error(self, last_feature, feature, reward, gamma, weight):
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
        self.buffer[index, -1] = new_pri #+ self.pri_thrshd
        return

    """
    Insert sample into buffer
    Input: [x, y]-last, action-last, [x, y], reward, gamma, TD-error
    Return: None
    """
    def _insert_seq(self, last_state, last_action, state, reward, gamma, tde):
        new_sequence = self._seq_to_array(last_state, last_action, state, reward, gamma, tde)
        index = self.b_control.insert()
        # print(self.buffer.shape, len(new_sequence))
        self.buffer[index] = new_sequence
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
            last_state, last_action, state, reward, gamma, pri = self._array_to_seq(seq)

            sample = self.offline_nn.forward_prediction(last_state, last_action)
            if sample is not None:
                did_planning = True
                state, reward, gamma= sample
                succs.append([last_state, state])
                gamma = self.gamma if gamma is None else gamma
                self._update_weight(last_state, last_action, state, reward, gamma, float(alpha))
                self._update_priority(index, last_state, last_action, state, reward, gamma)
                for a in range(self.num_action):
                    sbab0 = self.offline_nn.backward_prediction(a, last_state)
                    spbrbgb = self.offline_nn.forward_prediction(sbab0, a)
                    spb, rb, gb = spbrbgb
                    gb = self.gamma if gb is None else gb

                    pri = np.abs(self._get_tde(sbab0, a, last_state, rb, gb))

                    preds.append([sbab0, last_state])
                    succs.append([sbab0, spb])

                    if pri >= self.pri_thrshd:
                        self._insert_seq(sbab0, a, last_state, rb, gb, pri + self.pri_thrshd)
            else:
                self.b_control.force_remove(index)
        return did_planning

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
        indexs = self._sample_break_tie(self.buffer[filled_ind, -1], min(filled_ind_length, n))
        if len(indexs) ==0:
            return [], []
        else:
            indexs = np.array(filled_ind)[indexs]
            seqs = np.copy(self.buffer[indexs])

            return indexs, seqs

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
        last_state = seq[:self.dim_state]
        last_action = seq[self.dim_state]
        state = seq[self.dim_state+1: self.dim_state*2+1]
        reward = seq[self.dim_state*2+1]
        gamma = seq[self.dim_state*2+2]
        tde = seq[self.dim_state*2+3]
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



agent = None


def agent_init():
    global agent
    agent = NN_Dyna()
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
        return  None
    elif in_message[0] == "print current value":
        return agent._max_action(agent.last_state, isprint=True)
    elif in_message[0] == "check covariance":
        return None
    return
