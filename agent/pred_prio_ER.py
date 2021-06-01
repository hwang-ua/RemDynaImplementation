# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import time
import numpy as np
import random
import utils.tiles3 as tc
# import utils.REM_model as rem
import utils.KernModelupdate as rem


class RandomER:

    # Default values
    def __init__(self):
        self.alpha = 0.1
        self.epsilon = 0.1
        self.gamma = 0.9
        self.num_planning = 10
        self.num_branching = 1
        self.pri_threshold = 0.0001

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
        # self.model = rem.REM_Model(self.dim_state, self.num_near, self.add_prot_limit)
        self.model = rem.KernModel(self.dim_state, 1000, 100, 0.01, 0.0001, 1)

        self.weight = []
        self.b_time = 0
        self.buffer = {"pri_time": np.zeros((2, self.len_buffer)), "sequence": {}}
        self.last_state = None
        self.last_action = None

        self.state = None
        self.action = None
        self.check_time = []
        self.check_total_time = np.zeros(6)

        self.learning = True#False
        return

    def set_param(self, param):
        self.alpha = param["alpha"]
        self.epsilon = param["epsilon"]
        self.gamma = param["gamma"]
        self.num_planning = param["num_planning"]
        self.num_branching = param["num_branching"]
        self.pri_threshold = param["pri_threshold"]

        self.dim_state = param["dim_state"]
        self.action_mode = param["action_mode"]
        self.num_action = param["num_action"]
        self.action_list = [i for i in range(self.num_action)]

        self.num_near = param["num_near"]
        self.tc_mem_size = param["tc_mem_size"]
        self.iht = tc.IHT(self.tc_mem_size)
        self.num_tilings = param["num_tilings"]
        self.num_tiles = param["num_tiles"]

        if param["init_weight"] == "0":
            self.weight = np.zeros(self.tc_mem_size)
        elif param["init_weight"] == "1":
            self.weight = np.ones(self.tc_mem_size)
        else:
            print("HAVEN't BEEN DONE YET")
            exit(-1)

        self.len_buffer = param["len_buffer"]

        self.b_time = 0
        self.buffer = {"pri_time": np.zeros((2, self.len_buffer)), "sequence": {}}

        self.add_prot_limit = param["add_prot_limit"]
        # self.model = rem.REM_Model(self.dim_state, self.num_near, self.add_prot_limit)
        self.model = rem.KernModel(self.dim_state, 1000, 100, 0.01, 0.0001, 1)
        return

    def start(self, state):
        self.state = state
        self.action = self._policy(state)
        self.check_total_time = np.zeros(6)
        return self.action

    def step(self, reward, state):
        # update variables
        self.last_state = self.state
        self.last_action = self.action
        self.state = state
        self.reward = reward

        # use Q-learning to update weight
        qstart = time.time()
        self._update_weight(self.last_state, self.last_action, self.state, reward, self.gamma, self.alpha)
        qtime = time.time() - qstart

        if self.learning:
            # get priority
            gstart = time.time()
            priority = self._get_priority(self.last_state, self.last_action, self.state, reward, self.gamma)
            gtime = time.time() - gstart

            # insert S,A into buffer
            istart = time.time()
            self._insert_sa(self.last_state, self.last_action, self.state, reward, self.gamma, priority)
            itime = time.time() - istart

            # change the priority of the sequence at time t-1
            self._set_sh_priority(self.b_time, priority)

            # planning
            pstart = time.time()
            for _ in range(self.num_planning):
                self._single_planning(self.alpha, self.num_planning)
            ptime = time.time() - pstart

        # choose new action
        cstart = time.time()
        self.action = self._policy(state)
        ctime = time.time() - cstart

        return self.action

    def end(self, reward, state):
        self.step(reward, state)
        return

    def _policy(self, state):
        if self.action_mode == "discrete":
            if random.random() < self.epsilon:
                return random.randint(0, len(self.action_list))
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
            all_choices.append(np.sum(self.weight[feature]))
        valid_index = self._break_tie(all_choices)
        return valid_index

    def _break_tie(self, xarray):
        max_v = np.max(xarray)
        valid_choices = np.where(xarray == max_v)[0]
        return np.random.choice(valid_choices)

    # return activated indices
    def _feature_construction(self, state, action):
        indices = tc.tiles(self.iht, self.num_tilings, float(self.num_tiles) * np.array(state), [action])
        return indices

    def _update_weight(self, last_state, last_action, state, reward, gamma, alpha):
        last_feature = self._feature_construction(last_state, last_action)
        feature = self._feature_construction(state, self._max_action(state))
        tde = self._td_error(last_feature, feature, reward, gamma, self.weight)
        if tde > 0:
            self.learning = True
        self.weight[last_feature] += alpha * tde
        return

    def _td_error(self, last_feature, feature, reward, gamma, weight):
        tde = reward + gamma * np.sum(weight[feature]) - np.sum(weight[last_feature])
        return tde

    def _get_priority(self, last_state, last_action, state, reward, gamma):
        last_feature = self._feature_construction(last_state, last_action)
        feature = self._feature_construction(state, self._max_action(state))
        tde = self._td_error(last_feature, feature, reward, gamma, self.weight)
        priority = abs(tde) + self.pri_threshold
        return priority

    def _update_priority(self, index, last_state, last_action, state, reward, gamma):
        new_pri = self._get_priority(last_state, last_action, state, reward, gamma)
        self.buffer["pri_time"][0, index] = new_pri
        return

    def _insert_sa(self, last_state, last_action, state, reward, gamma, priority):
        new_sequence = (last_state, last_action, state, reward, gamma)
        self.buffer["pri_time"][0:, self.b_time % self.len_buffer] = [priority, self.b_time]
        self.buffer["sequence"][self.b_time % self.len_buffer] = new_sequence
        self.b_time += 1
        return

    def _set_sh_priority(self, b_time, priority):
        if b_time >= 2:
            try:
                np.where(self.buffer["pri_time"][1] == b_time - 2)[0][0]
            except:
                print(self.buffer)
                print(self.buffer["pri_time"])
                print(self.buffer["pri_time"].shape)
                print(b_time, self.b_time)
            index = np.where(self.buffer["pri_time"][1] == b_time-2)[0][0]
            self.buffer["pri_time"][0, index] = priority
        return


    def _single_planning(self, alpha, n):
        h_pri_index, h_pri_seq = self._sample_seq_from_buffer()

        if h_pri_seq is not None:
            last_state, last_action, state, reward, gamma = h_pri_seq
            self._update_weight(last_state, last_action, state, reward, gamma, float(alpha) / np.sqrt(n))
            self._update_priority(h_pri_index, last_state, last_action, state, reward, gamma)
        return

    def _sample_seq_from_buffer(self):
        h_pri_index = self._break_tie(self.buffer["pri_time"][0])
        # self.b_empty_keys.append(h_pri_index)
        h_pri_seq = self.buffer["sequence"][h_pri_index]
        return h_pri_seq, h_pri_index

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
    elif in_message[0] == "check total time":
        return agent.check_total_time
    elif in_message[0] == "check model size":
        return  # agent.model.b
    return