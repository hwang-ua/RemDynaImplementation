import numpy as np
import random
import math

class MountainCar():
    def __init__(self):
        self.MIN_POS = -1.2
        self.MAX_POS = 0.6
        self.MAX_VEL = 0.07 #the negative of this is also the minimum velocity
        self.MIN_VEL = -0.07
        self.GOAL_POS = 0.5
        
        self.pos = -0.5
        self.vel = 0.0
        
        #self.num_step = 1000#params['totalSamples']
        self.sparse = True#params['sparse']

    def scaleObs(self, obs):
        return np.array([(obs[0] - self.MIN_POS)/(self.MAX_POS-self.MIN_POS), (obs[1] - self.MIN_VEL)/(self.MAX_VEL-self.MIN_VEL)])

    def unscaleObs(self, obs):
        return np.array([(obs[0] * (self.MAX_POS - self.MIN_POS)) + self.MIN_POS, (obs[1] * (self.MAX_VEL - self.MIN_VEL)) + self.MIN_VEL])

    def getObs(self):
        return np.array([self.pos, self.vel])

    def reset(self):
        self.pos = random.uniform(-.4, -.6)
        # self.pos = -0.5
        self.vel = 0.0
        return self.scaleObs(self.getObs())

    # Uses the true model to give S' and R from an S and A
    def model(self, s, a):
        pos = s[0]
        vel = s[1]
        vel = vel + float(a-1)*0.001 + math.cos(3.0*self.pos)*(-0.0025)
        if vel > self.MAX_VEL:
            vel = self.MAX_VEL
        if vel < self.MIN_VEL:
            vel = self.MIN_VEL
        pos = pos + vel
        if pos > self.MAX_POS:
            pos = self.MAX_POS
        if pos < self.MIN_POS:
            pos = self.MIN_POS
        if pos == self.MIN_POS and vel < 0:
            vel = 0.0

        done = pos >= self.GOAL_POS
        reward = -1
        if self.sparse:
            reward = 1 if done else 0
        return (np.array([pos, vel]), reward, done)

    def step(self, a):
        obs, reward, done = self.model(self.getObs(), a)
        scaled_obs = self.scaleObs(obs)
        self.pos = obs[0]
        self.vel = obs[1]
        return (scaled_obs, reward, done, "")

    def numObservations(self):
        return 2

    def numActions(self):
        return 3
    
    def set_param(self, params):
        # self.STEPS_LIMIT = params["max_steps"]
        #self.num_step = params['num_step']
        self.sparse = params['sparse']
        return


env = None

def env_init():
    global env
    env = MountainCar()
    return 
    
def env_start():
    global env, current_state
    current_state = env.reset() # position
    return current_state

def env_step(action):
    global env, current_state
    info = env.step(action)
    step_info = {}
    step_info["state"] = info[0]
    step_info["reward"] = info[1]
    step_info["isTerminal"] = info[2]
    return step_info

def env_end(action):
    # Nothing happens here
    return

def env_cleanup():
    global env, current_state
    current_state = env.reset()
    return

def env_message(in_message): # returns string, in_message: string
    if in_message[0] == "set param":
        if env != None:
            env.set_param(in_message[1])
        else:
            print("the environment hasn't been initialized.")
    elif in_message[0] == "state dimension":
        return env.numObservations()
    elif in_message[0] == "num_action":
        return env.numActions()
    return ""
