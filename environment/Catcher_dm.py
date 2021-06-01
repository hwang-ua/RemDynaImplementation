# pylint: disable=g-bad-file-header
# Copyright 2019 The dm_env Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Catch reinforcement learning environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dm_env
from dm_env import specs
import numpy as np

_ACTIONS = (-1, 0, 1)  # Left, no-op, right.


class Catch(dm_env.Environment):
    """A Catch environment built on the `dm_env.Environment` class.
    The agent must move a paddle to intercept falling balls. Falling balls only
    move downwards on the column they are in.
    The observation is an array shape (rows, columns), with binary values:
    zero if a space is empty; 1 if it contains the paddle or a ball.
    The actions are discrete, and by default there are three available:
    stay, move left, and move right.
    The episode terminates when the ball reaches the bottom of the screen.
    """

    def __init__(self, rows=10, columns=5, seed=1, teleport=False):
        """Initializes a new Catch environment.
        Args:
          rows: number of rows.
          columns: number of columns.
          seed: random seed for the RNG.
        """
        self._rows = rows
        self._columns = columns
        # self._rng = np.random.RandomState(seed)
        self._board = np.zeros((rows, columns), dtype=np.float32)
        self._ball_x = None
        self._ball_y = None
        self._paddle_x = None
        self._paddle_y = self._rows - 1
        self._reset_next_step = True

        # self.paddle_width = 5
        # self.ball_width = 2
        # self.ball_height = 10

        self.teleport = teleport
        self.teleport_in_r = int(rows*0.2)
        self.teleport_from_c = int(columns*0.2)
        # print(self.teleport_in_r,self.teleport_from_c)

    def set_param(self, param):
        self._rows = param["rows"]
        self._columns = param["columns"]
        # self._rng = np.random.RandomState(seed)
        self._board = np.zeros((self._rows, self._columns), dtype=np.float32)
        self._ball_x = None
        self._ball_y = None
        self._paddle_x = None
        self._paddle_y = self._rows - 1
        self._reset_next_step = True

        # self.paddle_width = 5
        # self.ball_width = 2
        # self.ball_height = 10

        self.teleport = param["teleport"]
        self.teleport_in_r = int(self._rows * 0.2)
        self.teleport_from_c = int(self._columns * 0.2)
        # print(self.teleport_in_r,self.teleport_from_c)

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        # self._ball_x = self._rng.randint(self._columns)
        self._ball_x = np.random.randint(self._columns)
        # self._ball_x = np.random.randint(self._columns-self.ball_width)
        # self._ball_y = 0
        self._ball_y = np.random.randint(int(0.3*(self._rows)))
        self._paddle_x = self._columns // 2

        if self.teleport:
            while (self._ball_y == self.teleport_in_r and self._ball_x == self.teleport_from_c):
                self._ball_x = np.random.randint(self._columns)

        return dm_env.restart(self._observation())

    def step(self, action):
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        # Move the paddle.
        dx = _ACTIONS[action]
        self._paddle_x = np.clip(self._paddle_x + dx, 0, self._columns - 1)

        # Drop the ball.
        self._ball_y += 1

        if self.teleport:
            while (self._ball_y == self.teleport_in_r and self._ball_x == self.teleport_from_c):
                self._ball_x = np.random.randint(self._columns)

        # Check for termination.
        if self._ball_y == self._paddle_y:
            reward = 1. if self._paddle_x == self._ball_x else -1.
            self._reset_next_step = True
            # if reward == -1:
            #     self._reset_next_step = True
            # else:
            #     self.reset()
            return dm_env.termination(reward=reward, observation=self._observation())
        else:
            return dm_env.transition(reward=0., observation=self._observation())

    def observation_spec(self):
        """Returns the observation spec."""
        return specs.BoundedArray(shape=self._board.shape, dtype=self._board.dtype,
                                  name="board", minimum=0, maximum=1)

    def action_spec(self):
        """Returns the action spec."""
        return specs.DiscreteArray(
            dtype=int, num_values=len(_ACTIONS), name="action")

    def _observation(self):
        self._board.fill(0.)
        self._board[self._ball_y, self._ball_x] = 1.
        self._board[self._paddle_y, self._paddle_x] = 1.

        # for i in range(self._ball_y,self._ball_y+self.ball_height):
        #     for j in range(self._ball_x,self._ball_x+self.ball_width):
        #         self._board[j, i] = 1.
        # for i in range(self._paddle_x-(self.paddle_width/2),self._paddle_x+(self.paddle_width/2)):
        #     self._board[self._paddle_y, self.i] = 1.

        return self._board.copy()

    def numObservations(self):
        return self._rows * self._columns

    def numAction(self):
        return len(_ACTIONS)



def env_init():
    global env
    env = Catch(rows=100, columns=50)
    return

def env_start():
    global env, current_state
    current_state = env.reset()  # position
    return current_state.observation.flatten()

# import matplotlib.pyplot as plt
def env_step(action):
    global env, current_state
    info = env.step(action)
    step_info = {}
    step_info["state"] = info.observation.flatten()
    step_info["reward"] = info.reward
    step_info["isTerminal"] = env._ball_y == env._paddle_y
    return step_info

def env_end(action):
    # Nothing happens here
    return

def env_cleanup():
    global env, current_state
    current_state = env.reset().observation.flatten()
    return

def env_message(in_message):  # returns string, in_message: string
    if in_message[0] == "state dimension":
        return env.numObservations()
    elif in_message[0] == "num_action":
        return env.numAction()
    elif in_message[0] == "set param":
        env.set_param(in_message[1])
    elif in_message[0] == "set_state":
        env.state = np.array([in_message[1][0], in_message[1][1]])


"""
if __name__ == '__main__':

    #generate data
    steps = 100000
    done_steps = 0
    columns = 50
    rows = 100
    teleport = True
    state_dim = rows*columns
    data_array = np.zeros((steps,(2*state_dim)+3)) #s,a,s',r,gamma
    data_array_sf = np.zeros((steps,(4*state_dim))) #s,rep(s),sf_g1(s),sf_g2(s)

    env = Catch(rows = rows, columns=columns, teleport=teleport)
    actions = env.numAction()

    current_state = env.reset().observation.flatten()
    while True:
        print(done_steps, current_state.shape)

        action = np.random.randint(0, actions)

        info = env.step(action)
        state = info.observation.flatten()
        reward = info.reward
        terminal = env._ball_y == env._paddle_y

        if terminal:
            gamma = 0.0
        else:
            gamma = 1.0

        data_array[done_steps,0:state_dim] = current_state.flatten()
        data_array[done_steps,state_dim] = action
        data_array[done_steps,state_dim+1:(2*state_dim)+1] = state.flatten()
        data_array[done_steps,(2*state_dim)+1] = reward
        data_array[done_steps,(2*state_dim)+2] = gamma

        data_array_sf[done_steps,0:state_dim] = current_state.flatten()
        data_array_sf[done_steps,state_dim:2*state_dim] = current_state.flatten()

        done_steps += 1

        if done_steps == steps:
            break

        if terminal:
            current_state = env.reset().observation.flatten()
        else:
            current_state = state

    print("Done collecting data")

    #compute successor feature
    acc_features_g1 = np.zeros(state_dim)
    acc_features_g2 = np.zeros(state_dim)
    g1 = 0.998
    g2  = 0.8

    features = data_array_sf[steps-1, state_dim:2*state_dim]
    acc_features_g1[:] = features
    acc_features_g2[:] = features
    data_array_sf[steps-1,2*state_dim:3*state_dim] = acc_features_g1
    data_array_sf[steps-1,3*state_dim:4*state_dim] = acc_features_g2

    for i in range(steps-2,-1,-1):

            print(i)

            features = data_array_sf[i, state_dim:2*state_dim]

            acc_features_g1 *= g1
            acc_features_g1 += features

            acc_features_g2 *= g2
            acc_features_g2 += features

            data_array_sf[i,2*state_dim:3*state_dim] = acc_features_g1
            data_array_sf[i,3*state_dim:4*state_dim] = acc_features_g2

    np.save("../random_data/catcher_dm_"+str(rows)+"r"+str(columns)+"c_noGoal_opt_"+str([0.998, 0.8])+"gamma_1pts_x1_x"+str(steps)+"-stoc-tel.npy",data_array_sf)

    print("Done saving data")
"""