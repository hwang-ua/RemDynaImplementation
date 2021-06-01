import numpy as np

def mergeRight(d1, d2):
    for k in d1:
        d2[k] = d1[k]
    return d2

class ContinuousGridWorld:
    def __init__(self):
        self.line_number = 0
        self.record = np.load("random_data/fixed_env_suc_prob_1.0/cgw_noGoal_separateTC32x4_training_set_randomStart_0opt_[0.998, 0.8]gamma_1pts_x1.npy")
        self.length = len(self.record)
        return

    def set_param(self, params):
        self.line_number = 0
        return

    def reset(self):
        self.line_number = 0
        x, y = self._next_position()
        return [x, y]

    def step(self, action):
        x, y = self._next_position()
        reward, terminate = self._check_goal(x, y)
        return ([x, y], reward, terminate)

    def _next_position(self):
        x, y = self.record[self.line_number, :2]
        self.line_number += 1
        return x, y

    def _check_goal(self, x, y):
        if self.line_number == self.length:
            terminate = True
        else:
            terminate = False

        if 0.7 <= x <= 0.75 and 0.95 <= y <= 1.0:
            return 1, terminate
        else:
            return 0, terminate

    def numObservations(self):
        return 2

    def numActions(self):
        return 4


env = None


def env_init():
    global env
    env = ContinuousGridWorld()
    return


def env_start():
    global env, current_state
    current_state = env.reset()  # position
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


def env_message(in_message):  # returns string, in_message: string
    if in_message[0] == "state dimension":
        return env.numObservations()
    elif in_message[0] == "num_action":
        return env.numActions()
    elif in_message[0] == "set param":
        env.set_param(in_message[1])
    elif in_message[0] == "sample_random":
        while True:
            x = np.random.uniform(low=0.0, high=1.0)
            y = np.random.uniform(low=0.0, high=1.0)
            # x = np.random.uniform(low=0.0, high=0.1)
            # y = np.random.uniform(low=0.9, high=1.0)
            if not env._go_in_wall(x, y):
                env.x = x
                env.y = y
                return ([x,y])
            # if env._go_in_wall(x, y):
            #     print("In wall:,",x,y)
            # return ([x,y])
    elif in_message[0] == "sample_random_around":
        while True:
            x = np.random.uniform(low=in_message[1][0], high=in_message[2][0])
            y = np.random.uniform(low=in_message[2][1], high=in_message[2][1])
            # x = np.random.uniform(low=0.0, high=0.1)
            # y = np.random.uniform(low=0.9, high=1.0)
            if not env._go_in_wall(x, y):
                env.x = x
                env.y = y
                return ([x,y])
            # return ([x,y])
    elif in_message[0] == "set_state":
        env.x = in_message[1][0]
        env.y = in_message[1][1]

    elif in_message[0] == "pseudo_model":

        buffer_x = in_message[1][0]
        buffer_y = in_message[1][1]
        buffer_a = in_message[1][2]

        new_x, new_y = env._next_position(buffer_a, buffer_x, buffer_y)
        if not env._go_in_wall(new_x, new_y):
            if new_x >= env.goal_x[0] and new_x <= env.goal_x[1] \
                    and new_y >= env.goal_y[0] and new_y <= env.goal_y[1]:
                reward = 1
                terminate = 1
            else:
                reward = 0
                terminate = 0
            return new_x, new_y, reward, terminate
        else:
            return buffer_x, buffer_y, 0, 0

    else:
        print("Unknown request")
    return ""
