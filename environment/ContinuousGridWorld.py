import numpy as np

def mergeRight(d1, d2):
    for k in d1:
        d2[k] = d1[k]
    return d2

class ContinuousGridWorld:
    def __init__(self):

        self.x = 0.0
        self.y = 1.0

        self.goal_x = [0.95, 1.0]
        self.goal_y = [0.95, 1.0]

        self.num_step = 0

        self.wall_x = 0.5
        self.wall_w = 0.2
        self.hole_yl = 0.4
        self.hole_yh = 0.6
        self.new_yl = 0.8
        self.new_yh = 1.0
        self.suc_prob = 0.9
        self.change_time = 1000000
        self.sparse = 1

        self.params = None
        return

    def set_param(self, params):
        self.params = params
        self.x = params["start_x"]
        self.y = params["start_y"]
        self.goal_x = params["goal_x"]
        self.goal_y = params["goal_y"]
        self.wall_x = params["wall_x"]
        self.wall_w = params["wall_w"]
        self.hole_yl = params["hole_yl"]
        self.hole_yh = params["hole_yh"]

        self.new_yl = params["new_yl"]
        self.new_yh = params["new_yh"]

        self.suc_prob = params["suc_prob"]

        self.change_time = params["change_time"]
        self.sparse = params["sparse"]

        self.num_step = 0
        return

    def reset(self):
        self.x = self.params["start_x"]
        self.y = self.params["start_y"]
        self.hole_yl = self.params["hole_yl"]
        self.hole_yh = self.params["hole_yh"]
        return [self.x, self.y]

    def step(self, action):
        x, y = self._next_position(action, self.x, self.y)
        self.x, self.y = x, y
        # if not self._go_in_wall(x, y):
        #     self.x = x
        #     self.y = y
        #     # pass
        # else:
        #     print("GET IN WALL:", x, y, self.x, self.y)
        self.x = np.clip(self.x, 0.0, 1.0)
        self.y = np.clip(self.y, 0.0, 1.0)

        if self.num_step == self.change_time:
            self._change_hole(self.new_yl, self.new_yh)
        self.num_step += 1

        reward, terminate = self._check_goal()

        return ([self.x, self.y], reward, terminate)

    def _next_position(self, action, x, y):
        if np.random.random() > self.suc_prob:
            action = np.random.randint(0, 4)
        # up
        if action == 0:
            y += 0.05 + np.random.normal(0, 0.01)
            if x > self.wall_x and x < (self.wall_x + self.wall_w) and \
                    y > self.hole_yh:
                y = self.hole_yh
        # down
        elif action == 1:
            y -= 0.05 + np.random.normal(0, 0.01)
            if x > self.wall_x and x < (self.wall_x + self.wall_w) and \
                    y < self.hole_yl:
                y = self.hole_yl
        # right
        elif action == 2:
            x += 0.05 + np.random.normal(0, 0.01)
            if (y > self.hole_yh or y < self.hole_yl) and \
                    (x > self.wall_x and x < self.wall_x + self.wall_w):
                x = self.wall_x
        # left
        elif action == 3:
            x -= 0.05 + np.random.normal(0, 0.01)
            if (y > self.hole_yh or y < self.hole_yl) and \
                    (x > self.wall_x and x < self.wall_x + self.wall_w):
                x = self.wall_x + self.wall_w
        else:
            print("Environment: action out of range. Action is:", action)

        return x, y

    def _go_in_wall(self, x, y):
        if x > self.wall_x and x < (self.wall_x + self.wall_w):
            if y < self.hole_yl or y > self.hole_yh:
                return True
        else:
            return False

    def _change_hole(self, new_yl, new_yh):
        self.hole_yl = new_yl
        self.hole_yh = new_yh
        self.params["hole_yl"] = new_yl
        self.params["hole_yh"] = new_yh
        print("changing pos of hole", self.params["hole_yl"], self.params["hole_yh"])
        return

    def _check_goal(self):
        if self.sparse:
            if self.x >= self.goal_x[0] and self.x <= self.goal_x[1]\
                    and self.y >= self.goal_y[0] and self.y <= self.goal_y[1]:
                return 1, 1
            else:
                return 0, 0
        else:
            if self.x >= self.goal_x[0] and self.x <= self.goal_x[1]\
                    and self.y >= self.goal_y[0] and self.y <= self.goal_y[1]:
                return 0, 1
            else:
                return -1, 0

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
