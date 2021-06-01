# -*- coding: utf-8 -*-

import time
import numpy as np
import random
# import sys
# sys.path.append('../utils/')
# import os
# print(os.getcwd())
import utils.tiles3 as tc
import utils.KernModelupdate as km


class OptAction():

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

        self.weight = []
        self.last_state = None
        self.last_action = None

        self.state = None
        self.action = None
        self.check_time = []
        self.check_total_time = np.zeros(6)

        self.opt_a = 2

        self.learning = False
        return

    def set_param(self, param):
        self.alpha = param["alpha"]
        self.epsilon = param["epsilon"]
        self.gamma = param["gamma"]

        self.dim_state = param["dim_state"]
        self.action_mode = param["action_mode"]
        self.num_action = param["num_action"]
        self.action_list = [i for i in range(self.num_action)]

        self.tc_mem_size = param["tc_mem_size"]
        self.iht = tc.IHT(self.tc_mem_size)
        self.num_tilings = param["num_tilings"]
        self.num_tiles = param["num_tiles"]

        self.opt_a = param["opt_a"]

        if param["init_weight"] == "0":
            self.weight = np.zeros(self.tc_mem_size)
        elif param["init_weight"] == "1":
            self.weight = np.ones(self.tc_mem_size)
        else:
            print("HAVEN't BEEN DONE YET")
            exit(-1)

        return

    def start(self, state):
        self.state = state
        self.action = self.opt_a
        self.check_total_time = np.zeros(6)
        return self.action

    def step(self, reward, state):
        # update variables

        self.last_state = self.state
        self.last_action = self.action
        self.state = state
        self.reward = reward

        # choose new action
        cstart = time.time()
        self.action = self.opt_a
        ctime = time.time() - cstart

        return self.action, None

    def end(self, reward, state):
        self.step(reward, state)
        return



agent = None


def agent_init():
    global agent
    agent = OptAction()
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
    # agent = REM_Dyna()
    agent = None
    return


def agent_message(in_message):
    if in_message[0] == "set param":
        if agent != None:
            agent.set_param(in_message[1])
        else:
            print("the environment hasn't been initialized.")
    if in_message[0] == "check time":
        return agent.check_time
    if in_message[0] == "check total time":
        return agent.check_total_time
    return
