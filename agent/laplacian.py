# -*- coding: utf-8 -*-

import time
import numpy as np
import random
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


class GraphLaplacian:

    # Default values
    def __init__(self):
        self.last_feature = None

        self.alpha = 0.1
        self.epsilon = 0
        self.gamma = 1
        self.dim_state = 1
        self.num_action = 1
        self.action_list = []
        self.num_tilings = 1
        self.num_tiles = 1
        self.tc = utc.TileCoding(self.dim_state, self.num_tilings, self.num_tiles, self.num_action)
        self.len_weight = (self.num_tilings * (self.num_tiles**self.dim_state) * self.num_action)
        self.weight = np.zeros(self.len_weight)

        self.active_set = set()
        self.embd_dim = 10
        self.evl_max_bound = 100
        self.uk = np.zeros((self.len_weight, self.embd_dim))

        return

    def set_param(self, param):
        self.alpha = param["alpha"]
        self.epsilon = param["epsilon"]
        self.gamma = param["gamma"]

        self.dim_state = param["dim_state"]
        self.num_action = param["num_action"]
        self.action_mode = param["action_mode"]
        self.num_action = param["num_action"]
        self.action_list = [i for i in range(self.num_action)]

        self.num_tilings = param["num_tilings"]
        self.num_tiles = param["num_tiles"]

        if param["div_actBit"]:
            self.alpha = param["alpha"] / float(self.num_tilings)
        else:
            self.alpha = param["alpha"]

        self.tc = utc.TileCoding(self.dim_state, self.num_tilings, self.num_tiles, self.num_action)
        self.len_weight = (self.num_tilings * (self.num_tiles**self.dim_state) * self.num_action)

        if param["init_weight"] == "0":
            # self.weight = np.zeros(self.tc_mem_size)
            self.weight = np.zeros(self.len_weight)
        elif param["init_weight"] == "1":
            # self.weight = np.ones(self.tc_mem_size)
            self.weight = np.ones(self.len_weight)
        else:
            print("HAVEN't BEEN DONE YET")
            exit(-1)

        self.active_set = set()
        self.embd_dim = param["embedding_dim"]
        self.evl_max_bound = param["eigenvalue_upper_bound"]
        self.uk = np.zeros((self.len_weight, self.embd_dim)) # size d * k

        return

    def start(self, state):
        self.last_feature = self._feature_construction(state)
        for act in self.last_feature:
            if act not in self.active_set:
                self.active_set.add(act)
                self.uk[act, :] = np.random.normal(0, 1.0/np.sqrt(self.len_weight), self.embd_dim)
                print(np.sum(self.uk[act]))
        print("=========", np.sum(self.uk))
        return self._policy(state)

    def step(self, reward, state):
        feature = self._feature_construction(state)
        for act in feature:
            if act not in self.active_set:
                self.active_set.add(act)
                self.uk[act, :] = np.random.normal(0, 1.0/np.sqrt(self.len_weight), self.embd_dim)
                print(np.sum(self.uk[act]))
        print("=========", np.sum(self.uk))
        last_f = self._complete_feature(self.last_feature)
        f = self._complete_feature(feature)
        x = last_f - f

        # update U_k with gradient in Appendix 10
        a = np.dot(x, x) - 2 * self.evl_max_bound # 1
        uk_x = np.dot(self.uk.T, x) # 2
        uk_uk_x = np.dot(self.uk, uk_x) # 3
        x_x_uk = 2 * np.outer(x, uk_x) # 4
        uk_uk_uk = np.dot(self.uk, np.dot(self.uk.T, self.uk)) # 5

        # print(np.sum(uk_uk_uk - self.uk))

        ukukuk_uk = 2 * self.evl_max_bound ** 2 * (uk_uk_uk - self.uk) # 6
        x_uk_uk_uk = np.dot(uk_uk_x.T, self.uk) # 7
        x_x_uk_uk_uk = np.outer(x, x_uk_uk_uk) # 8
        uk_uk_x_x_uk = np.outer(uk_uk_x, uk_x.T) # 9
        # 10: plug in the expressions into the gradient function to obtain the Jacobian with respect to uk
        gradient = a * (x_x_uk_uk_uk + uk_uk_x_x_uk - x_x_uk) + ukukuk_uk
        self.uk -= self.alpha * gradient

        return self._policy(state), None

    def end(self, reward, state):
        self.step(reward, state)
        return

    def _policy(self, state):
        return np.random.randint(0, self.num_action)

    def _laplacian(self, state):
        return

    def _complete_feature(self, feature):
        f = np.zeros(self.len_weight)
        for i in feature:
            f[i] = 1
        return f

    # return activated indices
    def _feature_construction(self, state, action=None):
        if action == None:
            indices = self.tc.get_index(np.array(state))
        else:
            indices = self.tc.get_index(np.array(state), action)
        return indices

agent = None


def agent_init():
    global agent
    agent = GraphLaplacian()
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
        return  #agent.model.get_len_protos()
    return