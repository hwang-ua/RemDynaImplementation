import numpy as np


class GridWorld:
    def __init__(self):
        self.x = 0
        self.y = 0

        self.size_x = 30
        self.size_y = 30

        self.goal_y = 0
        self.goal_x = -1

        self.num_step = 0

        self.walls = []

        self.suc_prob = 0.9
        self.change_time = 1000000
        self.sparse = 1

        self.params = None
        return

    def set_param(self, params):
        self.params = params

        self.x = params["start_x"]
        self.y = params["start_y"]

        self.size_x = params["size_x"]
        self.size_y = params["size_y"]

        self.goal_x = params["goal_x"]
        self.goal_y = params["goal_y"]

        self.walls = []
        for block in params["walls"]:
            wx_start, wx_len, wy_start, wy_len = block
            for wx in range(wx_start, wx_start+wx_len):
                for wy in range(wy_start, wy_start+wy_len):
                    self.walls.append([wx, wy])

        self.suc_prob = params["suc_prob"]
        self.change_time = params["change_time"]

        self.sparse = params["sparse"]
        return

    def reset(self):
        self.x = self.params["start_x"]
        self.y = self.params["start_y"]

        self.size_x = self.params["size_x"]
        self.size_y = self.params["size_y"]

        self.goal_x = self.params["goal_x"]
        self.goal_y = self.params["goal_y"]

        self.walls = []
        for block in self.params["walls"]:
            wx_start, wx_len, wy_start, wy_len = block
            for wx in range(wx_start, wx_start + wx_len):
                for wy in range(wy_start, wy_start + wy_len):
                    self.walls.append([wx, wy])

        self.suc_prob = self.params["suc_prob"]
        self.change_time = self.params["change_time"]

        self.sparse = self.params["sparse"]
        return [self.x, self.y]

    def step(self, action):
        x, y = self._next_position(action, self.x, self.y)

        if not self._go_in_wall(x, y):
            self.x = x
            self.y = y

        self.x = np.clip(self.x, 0, self.size_x - 1)
        self.y = np.clip(self.y, 0, self.size_y - 1)

        if self.num_step == self.change_time:
            self._change_hole()
        self.num_step += 1

        reward, terminate = self._check_goal()

        return ([self.x, self.y], reward, terminate)

    def _next_position(self, action, x, y):
        if np.random.random() > self.suc_prob:
            action = np.random.randint(0, 4)
        # down
        if action == 0:
            y += 1
        # up
        elif action == 1:
            y -= 1
        # right
        elif action == 2:
            x += 1
        # left
        elif action == 3:
            x -= 1
        else:
            print("Out of range.")

        return x, y

    def _go_in_wall(self, x, y):
        if [x, y] in self.walls:
            return True
        else:
            return False

    def _change_hole(self):
        self.walls = []
        for block in self.params["new_walls"]:
            wx_start, wx_len, wy_start, wy_len = block
            for wx in range(wx_start, wx_start+wx_len):
                for wy in range(wy_start, wy_start+wy_len):
                    self.walls.append([wx, wy])
        return

    def _check_goal(self):
        if self.sparse:
            if self.x == self.goal_x and self.y == self.goal_y:
                return 1, 1
            else:
                return 0, 0
        else:
            if self.x == self.goal_x and self.y == self.goal_y:
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
    env = GridWorld()
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
    else:
        print("Unknown request")
    return ""