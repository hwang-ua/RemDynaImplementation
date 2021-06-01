# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import time
import numpy as np
import random
import utils.tiles3 as tc
# import utils.REM_model as rem
import utils.KernModelupdate as rem


class OptimalCgwAgent:

    # Default values
    def __init__(self):
        self.num_action = 4
        self.action_list = [i for i in range(self.num_action)]
        self.policy = {}
        return

    def set_param(self, param):
        self.num_action = param["num_action"]
        self.action_list = [i for i in range(self.num_action)]
        self.policy = {}
        self.opt_prob = param["opt_prob"]
        self.policy = param["opt_a"]
        return

    def start(self, state):
        self.action = self._policy(state)
        return self.action

    def step(self, reward, state):
        self.action = self._policy(state)
        return self.action, None

    def end(self, reward, state):
        return

    def _policy(self, state):
        if np.random.random() <= self.opt_prob:
            x, y = state
            for rule in self.policy:
                range_x, range_y, choice = rule
                if x >= range_x[0] and x <= range_x[1] and y >= range_y[0] and y <= range_y[1]:
                    action = choice[np.random.randint(len(choice))]
        else:
            action = np.random.randint(self.num_action)
        return action


agent = None

def agent_init():
    global agent
    agent = OptimalCgwAgent()
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