import numpy as np
from ple.games.catcher import Catcher
from ple import PLE
import os

######## catcher ########
def get_ob(state):
    return np.array([state['player_x'], state['player_vel'], state['fruit_x'], state['fruit_y']])

def get_ob_normalize(state):
    np_state = np.array([state['player_x'], state['player_vel'], state['fruit_x'], state['fruit_y']])
    np_state[0] = (np_state[0] - 26) / 26
    np_state[1] = (np_state[1]) / 8
    np_state[2] = (np_state[2] - 26) / 26
    np_state[3] = (np_state[3] - 20) / 45
    return np_state

class catcher():
    def __init__(self, init_lives=3):
        self.catcherOb = Catcher(init_lives=init_lives)
        self.pOb = PLE(self.catcherOb, fps=30, state_preprocessor=get_ob_normalize, display_screen=False)
        self.num_action = 2
        self.actions = self.pOb.getActionSet()
        self.num_state = 4
        self.pOb.init()

    def setseed(self, value):
        self.pOb.rng.seed(value)
        return 0

    def reset(self):
        self.pOb.reset_game()
        return self.pOb.getGameState()

    def step(self, a):
        reward = self.pOb.act(self.actions[a])
        terminal = self.pOb.game_over()
        return (self.pOb.getGameState(), reward, terminal, {})

    def close(self):
        return


######## catcher-3 actions ########
# def catcher3_get_ob(state):
#    return np.array([state['player_x'], state['player_vel'], state['fruit_x'], state['fruit_y']])

class catcher3():
    def __init__(self, init_lives=3, normalize=True, display=False):
        self.catcherOb = Catcher(init_lives=init_lives)
        if display is False:
            # do not open a pygame window
            os.putenv('SDL_VIDEODRIVER', 'fbcon')
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        if normalize:
            self.pOb = PLE(self.catcherOb, fps=30, state_preprocessor=get_ob_normalize, display_screen=display)
        else:
            self.pOb = PLE(self.catcherOb, fps=30, state_preprocessor=get_ob, display_screen=display)
        self.actions = [100, 97, None] # self.pOb.getActionSet()

        self.num_action = 3
        self.num_state = 4

        self.image_based = True
        self.image_processor = True#False

        if self.image_processor:
            self.frame_length = 2
            self.frame_history = []


        self.wasReset = False

        self.pOb.init()

    def _get_image(self):
        '''
        return a np array with shape = [64, 64, 3]
        '''
        frame_visual = self.pOb.getScreenGrayscale()  # Visual State
        frame_visual = np.reshape(frame_visual, (64, 64, 1)) / 255

        return frame_visual


    def _processor_image(self):
        '''
        return a np array with shape = [64, 64, 3]
        '''
        frame_visual = self._get_image()
        self.frame_history.append(frame_visual)
        if len(self.frame_history) < self.frame_length:
            while len(self.frame_history) < self.frame_length:  # first accumulate enough frames
                action = np.random.randint(0, self.num_action)
                self.pOb.act(action)
                frame_visual = self._get_image()
                self.frame_history.append(frame_visual)
        elif len(self.frame_history) > self.frame_length:
            del self.frame_history[0]
        return np.array(self.frame_history)

    def setseed(self, value):
        self.pOb.rng.seed(value)
        return 0

    def internal_reset(self):
        if not self.wasReset:
            self.pOb.reset_game()
            self.wasReset = True
        if self.image_based:
            if self.image_processor:
                self.frame_history = []
                return self._processor_image()
            else:
                return self._get_image()
        else:
            return self.pOb.getGameState()

    def reset(self):
        self.wasReset = False
        return self.internal_reset()

    def step(self, a):
        reward = self.pOb.act(self.actions[a])
        terminal = self.pOb.game_over()

        # if terminal:
        #     state = self.reset()
        # else:
        if self.image_based:
            if self.image_processor:
                state = self._processor_image()
            else:
                state = self._get_image()
        else:
            state = self.pOb.getGameState()

        return (state, reward, terminal, {})

    def close(self):
        return

    def numObservations(self):
        if self.image_processor:
            return 8192
        else:
            return 4096


def env_init():
    global env
    env = catcher3(init_lives=1)
    return

def env_start():
    global env, current_state
    current_state = env.reset().flatten()  # position
    return current_state

# import matplotlib.pyplot as plt
def env_step(action):
    global env, current_state
    info = env.step(action)
    step_info = {}
    step_info["state"] = info[0].flatten()

    # reshaped_img = step_info["state"].reshape((2,64,64))[0]
    # for line in range(len(reshaped_img)):
    #     one_idx = np.where(reshaped_img[line] == 1)[0]
    #     if len(one_idx) != 0:
    #         print(line, one_idx)
    # print("*")
    # reshaped_img = step_info["state"].reshape((2,64,64))[1]
    # for line in range(len(reshaped_img)):
    #     one_idx = np.where(reshaped_img[line] == 1)[0]
    #     if len(one_idx) != 0:
    #         print(line, one_idx)
    # print()

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
        return env.num_action
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

# if __name__ == '__main__':
#
#     #generate data
#     steps = 100000
#     done_steps = 0
#     state_dim = 2*64*64*1
#     data_array = np.zeros((steps,(2*state_dim)+3)) #s,a,s',r,gamma
#     data_array_sf = np.zeros((steps,(4*state_dim))) #s,rep(s),sf_g1(s),sf_g2(s)
#
#     env = catcher3(init_lives=1)
#     actions = env.num_action
#
#     current_state = env.reset()
#     while True:
#         print(done_steps)
#
#         action = np.random.randint(0, actions)
#
#         state, reward, terminal, _ = env.step(action)
#
#         if terminal:
#             gamma = 0.0
#         else:
#             gamma = 1.0
#
#         data_array[done_steps,0:state_dim] = current_state.flatten()
#         data_array[done_steps,state_dim] = action
#         data_array[done_steps,state_dim+1:(2*state_dim)+1] = state.flatten()
#         data_array[done_steps,(2*state_dim)+1] = reward
#         data_array[done_steps,(2*state_dim)+2] = gamma
#
#         data_array_sf[done_steps,0:state_dim] = current_state.flatten()
#         data_array_sf[done_steps,state_dim:2*state_dim] = current_state.flatten()
#
#         done_steps += 1
#
#         if done_steps == steps:
#             break
#
#         if terminal:
#             current_state = env.reset()
#         else:
#             current_state = state
#
#     print("Done collecting data")
#
#     #compute successor feature
#     acc_features_g1 = np.zeros(state_dim)
#     acc_features_g2 = np.zeros(state_dim)
#     g1 = 0.998
#     g2  = 0.8
#
#     features = data_array_sf[steps-1, state_dim:2*state_dim]
#     acc_features_g1[:] = features
#     acc_features_g2[:] = features
#     data_array_sf[steps-1,2*state_dim:3*state_dim] = acc_features_g1
#     data_array_sf[steps-1,3*state_dim:4*state_dim] = acc_features_g2
#
#     for i in range(steps-2,-1,-1):
#
#             print(i)
#
#             features = data_array_sf[i, state_dim:2*state_dim]
#
#             acc_features_g1 *= g1
#             acc_features_g1 += features
#
#             acc_features_g2 *= g2
#             acc_features_g2 += features
#
#             data_array_sf[i,2*state_dim:3*state_dim] = acc_features_g1
#             data_array_sf[i,3*state_dim:4*state_dim] = acc_features_g2
#
#     np.save("../random_data/catcher_noGoal_opt_"+str([0.998, 0.8])+"gamma_1pts_x1_x100000.npy",data_array_sf)
#
#     print("Done saving data")
