import numpy as np

class puddle():
    def __init__(self, headX, headY, tailX, tailY, radius, length, axis):
        self.headX = headX
        self.headY = headY
        self.tailX = tailX
        self.tailY = tailY
        self.radius = radius
        self.length = length
        self.axis = axis

    def getDistance(self, xCoor, yCoor):

        if self.axis == 0:
            u = (xCoor - self.tailX)/self.length
        else:
            u = (yCoor - self.tailY)/self.length

        dist = 0.0

        if u < 0.0 or u > 1.0:
            if u < 0.0:
                dist = np.sqrt(np.power((self.tailX - xCoor),2) + np.power((self.tailY - yCoor),2));
            else:
                dist = np.sqrt(np.power((self.headX - xCoor),2) + np.power((self.headY - yCoor),2));
        else:
            x = self.tailX + u * (self.headX - self.tailX);
            y = self.tailY + u * (self.headY - self.tailY);

            dist = np.sqrt(np.power((x - xCoor),2) + np.power((y - yCoor),2));

        if dist < self.radius:
            return (self.radius - dist)
        else:
            return 0

class puddleworld():
    def __init__(self, normalized=False):
        self.num_action = 4
        self.num_state = 2
        self.state = None
        self.puddle1 = puddle(0.45,0.75,0.1,0.75,0.1,0.35,0)
        self.puddle2 = puddle(0.45,0.8,0.45,0.4,0.1,0.4,1)

        self.pworld_min_x = 0.0
        self.pworld_max_x = 1.0
        self.pworld_min_y = 0.0
        self.pworld_max_y = 1.0
        self.pworld_mid_x = (self.pworld_max_x - self.pworld_min_x)/2.0
        self.pworld_mid_y = (self.pworld_max_y - self.pworld_min_y)/2.0

        self.goalDimension = 0.05
        self.defDisplacement = 0.05

        self.sigma = 0.01

        self.goalXCoor = self.pworld_max_x - self.goalDimension #1000#
        self.goalYCoor = self.pworld_max_y - self.goalDimension #1000#
        self.normalized = normalized

        self.wasReset = False

    def internal_reset(self):
        if not self.wasReset:
            self.state = np.random.uniform(low=0.0, high=0.1, size=(2,))

            reset = False
            while not reset:
                self.state[0] = np.random.uniform(low=0, high=1)
                self.state[1] = np.random.uniform(low=0, high=1)
                if not self._terminal():
                    reset = True
            print("\nStart state:", self.state)
            self.wasReset = True
        return self._get_ob()

    def reset(self):
        self.wasReset = False
        return self.internal_reset()

    def _get_ob(self):
        if self.normalized:
            s = self.state
            s0 = (s[0] - self.pworld_mid_x) * 2.0
            s1 = (s[1] - self.pworld_mid_y) * 2.0
            return np.array([s0, s1])
        else:
            s = self.state
            return np.array([s[0], s[1]])

    def _terminal(self):
        s = self.state
        return bool((s[0] >= self.goalXCoor) and (s[1] >= self.goalYCoor))

    def _reward(self,x,y,terminal):
        if terminal:
            return -1
        reward = -1
        dist = self.puddle1.getDistance(x, y)
        reward += (-400. * dist)
        dist = self.puddle2.getDistance(x, y)
        reward += (-400. * dist)
        reward = reward
        return reward

    def step(self,a):
        s = self.state

        xpos = s[0]
        ypos = s[1]

        n = np.random.normal(scale=self.sigma)

        if a == 0: #up
            ypos += self.defDisplacement+n
        elif a == 1: #down
            ypos -= self.defDisplacement+n
        elif a == 2: #right
            xpos += self.defDisplacement+n
        else: #left
            xpos -= self.defDisplacement+n

        if xpos > self.pworld_max_x:
            xpos = self.pworld_max_x
        elif xpos < self.pworld_min_x:
            xpos = self.pworld_min_x

        if ypos > self.pworld_max_y:
            ypos = self.pworld_max_y
        elif ypos < self.pworld_min_y:
            ypos = self.pworld_min_y

        s[0] = xpos
        s[1] = ypos
        self.state = s

        terminal = self._terminal()
        reward = self._reward(xpos,ypos,terminal) / 40.0

        # if terminal:
        #     self.reset()

        return (self._get_ob(), reward, terminal, {})

    def numObservations(self):
        return 2

    def numActions(self):
        return 4




def env_init():
    global env
    env = puddleworld()
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
        print("Puddle world: no parameter setting")
        # env.set_param(in_message[1])
    # elif in_message[0] == "sample_random":
    #     while True:
    #         x = np.random.uniform(low=0.0, high=1.0)
    #         y = np.random.uniform(low=0.0, high=1.0)
    #         # x = np.random.uniform(low=0.0, high=0.1)
    #         # y = np.random.uniform(low=0.9, high=1.0)
    #         if not env._go_in_wall(x, y):
    #             env.x = x
    #             env.y = y
    #             return ([x,y])
    #         # if env._go_in_wall(x, y):
    #         #     print("In wall:,",x,y)
    #         # return ([x,y])
    # elif in_message[0] == "sample_random_around":
    #     while True:
    #         x = np.random.uniform(low=in_message[1][0], high=in_message[2][0])
    #         y = np.random.uniform(low=in_message[2][1], high=in_message[2][1])
    #         # x = np.random.uniform(low=0.0, high=0.1)
    #         # y = np.random.uniform(low=0.9, high=1.0)
    #         if not env._go_in_wall(x, y):
    #             env.x = x
    #             env.y = y
    #             return ([x,y])
    #         # return ([x,y])
    elif in_message[0] == "set_state":
        env.state = np.array([in_message[1][0], in_message[1][1]])