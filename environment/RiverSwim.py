# -*- coding: utf-8 -*-

import numpy as np
import random


class RiverSwim():
    def __init__(self):

        # self.STEPS_LIMIT = 200#params["max_steps"]
        self.pos = 0.0
        self.swimRightStay = 0.6
        self.swimRightUp = 0.35
        self.swimRightDown = 0.05
        self.S1swimRightUp = 0.6
        self.SNswimRightDown = 0.4
        self.stepSize = 0.1
        self.noise = self.stepSize / 5.0

    def scaleObs(self, obs):
        return np.array([obs[0]])

    def getObs(self):
        return np.array([self.pos])

    # reset environment
    # return position
    def reset(self):
        self.n = 0
        self.pos = 0.0
        return self.scaleObs(self.getObs())

    # default setting of max_steps
    # def defaults(self):
    #    return {
    #        "max_steps": 20000
    #    }

    def model(self, s, a):
        x = self.pos if s is None else s[0]
        stepSize = self.stepSize + random.uniform(-self.noise, self.noise)
        if a == 0:
            x = x - stepSize
        elif a == 1:
            flip = random.random()
            if self.pos < self.stepSize / 2.0:  # first state in chain
                if flip > self.S1swimRightUp:
                    x = x + stepSize
            if 1.0 - x < self.stepSize / 2.0:  # end of chain
                if flip <= self.SNswimRightDown:
                    x = x - stepSize
            else:  # middle of chain
                if flip <= self.swimRightDown:
                    x = x - stepSize
                elif flip > self.swimRightDown + self.swimRightStay:
                    x = x + stepSize
        x = np.clip(x, 0., 1.)
        return (np.array([x]), self.rewardFunction(x), None)

    def model_predecessor(self, s):
        x = s[0]
        stepSize = self.stepSize + random.uniform(-self.noise, self.noise)
        a = 0 if random.random() > 0.5 else 1
        if a == 0:
            x = x + stepSize
        elif a == 1:
            flip = random.random()
            if 1.0 - (x + stepSize) < self.stepSize / 2.0:
                if flip <= self.SNswimRightDown:
                    x = x + stepSize
            if (x - stepSize) < self.stepSize / 2.0 and flip > self.S1swimRightUp:  # first state in chain
                x = x - stepSize
            else:  # middle of chain
                if flip <= self.swimRightDown:
                    x = x + stepSize
                elif flip > self.swimRightDown + self.swimRightStay:
                    x = x - stepSize
        x = np.clip(x, 0., 1.)
        _, r, _ = self.model(np.array([x]), a)
        return (np.array([x]), a, r, s, None)

    def step(self, a):
        sp, reward, _ = self.model(None, a)
        self.pos = sp[0]
        return (self.scaleObs(self.getObs()), reward, False, a)

    def rewardFunction(self, x):
        if 1.0 - x < self.stepSize / 2.0:
            # print 'reward is 1'
            return 1.0
        if x < self.stepSize / 2.0:
            return 5.0 / 1000.0
        return 0.0

    def numObservations(self):
        return 1

    def numActions(self):
        return 2

    def set_param(self, params):
        # self.STEPS_LIMIT = params["max_steps"]
        return


env = None


def env_init():
    global env
    env = RiverSwim()
    return


def env_start():
    global env, current_state
    current_state = env.reset()  # position
    return current_state


def env_step(action):
    global env, current_state
    info = env.step(action)
    step_info = {"state": info[0], "reward": info[1], "isTerminal": info[2]}
    return step_info


def env_end(action):
    # TODO
    return


def env_cleanup():
    global env, current_state
    current_state = env.reset()
    return


def env_message(in_message):  # returns string, in_message: string
    if in_message[0] == "set param":
        if env is not None:
            env.set_param(in_message[1])
        else:
            print("the environment hasn't been initialized.")
    elif in_message[0] == "state dimension":
        return env.numObservations()
    elif in_message[0] == "num_action":
        return env.numActions()
    return ""
