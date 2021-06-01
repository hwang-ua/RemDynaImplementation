#!/usr/bin/python3

import tkinter as tk
import numpy as np
import time
# import matplotlib.pyplot as plt
import sys
sys.path.append('./environment/')
sys.path.append('./agent/')

import json
jsonfile = "parameters/continuous_gridworld.json"
json_dat = open(jsonfile, 'r')
exp = json.load(json_dat)
json_dat.close()
print("Env::", exp["environment"], ", Param:", exp["env_params"])
print("Agent::", exp["agent"], ", Param:", exp["agent_params"])
print("Exp param::", exp["exp_params"])

from rl_glue import *

"""
input:
    experiment: the experiment object
    start: starting point of the agent.
            A numpy array containing the coordinate of the starting point. The coordinate should be in the range 0 to 1.
    goals: goal of the agent. A numpy array specifying the area of goal. There can be more than one goal in the experiment
            The goal should be a rectangle which is specified as the coordinates of its bottom left corner and its top right corner.
            (i.e. assume the goal is a square area on top right corner of the canvas, and the length of the square is 0.05,
            then the input is a numpy array [[0.95, 0.95, 1.0, 1.0]]. If there are more than one goal, the numpy assay will be 
            [[a1,b1,c1,d1], [a2,b2,c2,d2], ...])
            The coordinate should be in the range 0 to 1.
    walls: walls in the grid world. A numpy array containing all walls' coordinates.
            Each wall should be a rectangle area, and is specified as the coordinates of its bottom left corner and its top right corner.
            (i.e. a numpy array [[0.5, 0.0, 0.7, 0.8], [0.2, 0.4, 0.4, 0.5]] means there are 2 walls on canvas.
            The first wall's bottom left corner is on (0.5, 0.0), its up right corner is on (0.7, 0.8).)
            The coordinate should be in the range 0 to 1.
"""

class ContinuousGWGUI:

    def __init__(self, experiment,
                 start=np.array([0.0, 1.0]),
                 goals=np.array([[0.95, 0.95, 1.0, 1.0]]),
                 walls=np.array([[0.5, 0.0, 0.7, 0.8]]),
                 time_change = None,
                 new_walls = None,
                 ):

        self.exp = experiment
        self.start = start
        self.goals = goals
        self.walls = walls
        self.time_change = time_change
        self.new_walls = new_walls
        self.t = 0

        self.buffer_id = {}
        self.proto_id = {}
        self.qvalue_id = {}
        self.wall_ps = []

        # control the scale of the line showing the q-value of each s,a pair
        self.line_scale = 10
        # how long will a symbol shown on canvas
        self.line_time = 700

        # height and width of the canvas
        self.h = 500
        self.w = 500

        # edge of canvas
        self.x0, self.y0 = 10,10

        self.window = tk.Tk()
        self.window.title('continuous grid world')
        self.window.geometry(str(self.w+2*self.x0)+"x"+str(self.h+20*self.x0))

        # draw self.canvas
        self.canvas = tk.Canvas(self.window, bg='white', height=self.h+self.x0*2, width=self.w+self.y0*2)
        rect = self.canvas.create_rectangle(self.x0, self.y0, self.x0+self.w, self.y0+self.h)
        self.canvas.pack()

        # decide colors of elements
        self.start_color = "dark orange"
        self.goals_color = "SpringGreen3"
        self.agent_color = "tomato"
        self.plan_pos_color = "orange"
        self.plan_pref_color = "red"
        self.buffer_color = "lightskyblue"
        self.proto_pos_color = "light yellow"
        self.proto_line_color = "light grey"
        self.deepest_q = self.canvas.winfo_rgb(self.goals_color)

        # draw grid world
        rect = self.canvas.create_rectangle(self.x0, self.y0, self.x0+self.w, self.y0+self.h)

        # draw start point and goal
        start[0] = self.x0 + start[0] * self.w
        start[1] = self.y0 + self.h - start[1] * self.h
        start_p = self.canvas.create_oval(start[0]-self.w/100, start[1]-self.w/100,
                                          start[0]+self.w/100, start[1]+self.h/100, fill=self.goals_color)

        for g in self.goals:
            g[0] = self.x0 + g[0] * self.w
            g[1] = self.y0 + self.h - g[1] * self.h
            g[2] = self.x0 + g[2] * self.w
            g[3] = self.y0 + self.h - g[3] * self.h
            g_p = self.canvas.create_rectangle(g[0], g[1], g[2], g[3], fill = self.goals_color)

        # draw walls
        self.wall_ps = []
        for wall in walls:
            wx0 = self.x0 + wall[0] * self.w
            wy0 = self.y0 + self.h - wall[1] * self.h
            wx1 = self.x0 + wall[2] * self.w
            wy1 = self.y0 + self.h - wall[3] * self.h
            self.wall_ps.append(self.canvas.create_rectangle(wx0, wy0, wx1, wy1, fill = "black"))

        # draw agent
        self.agent = self.canvas.create_oval(start[0] - self.w/100, start[1] - self.w/100,
                                             start[0] + self.w/100, start[1] + self.h/100, fill=self.agent_color)
        self.current_pos = np.array([start[0], start[1]])

        # add button
        self.run_button_button = tk.Button(self.window, text='single run', command=self.single_run).pack()
        self.ep_button = tk.Button(self.window, text='single episode', command=self.single_ep).pack()
        self.step_button = tk.Button(self.window, text='single step', command=self.single_step).pack()
        self.exit_button = tk.Button(self.window, text='exit', command=self.close).pack()

        # show windows
        self.window.mainloop()

    def close(self):
        self.window.quit()
        self.window.destroy()
        exit(0)

    """
    start a single run
    return after one run ends
    """
    def single_run(self):
        self.t = 0
        end = False
        while not end:
            info = self.single_ep()
            end = info["end_run"]
        return

    """
    start a single episode
    return after one episode ends
    """
    def single_ep(self):
        end = False
        while not end:
            info = self.single_step()
            end = info["isTerminal"]
            if self.t % 1000 == 0:
                print("=====", self.t, "steps, # prototype =", RL_agent_message(["check model size"]))
        return info

    """
    start a single step
    return after one step ends
    """
    def single_step(self):
        # call the step function in experiment class, and get info dictionary
        info = self.exp.step()
        # move agent to the current position
        self.move_agent(info["state"])

        # check if there is any other information
        if "other_info" in info.keys() and info["other_info"] is not None:
            other_info = info["other_info"]
            if "plan" in other_info.keys() and other_info["plan"] is not None:
                self.show_plan_info(other_info["plan"])
            if "buffer" in other_info.keys() and other_info["buffer"] is not None:
                self.show_buffer_info(other_info["buffer"])
            if "protos" in other_info.keys() and other_info["protos"] is not None:
                self.show_proto_info(other_info["protos"])
            # show agent q value
            if "agent_q" in other_info.keys():
                self.show_agent_qvalue(info["state"], other_info["agent_q"])

        self.t += 1
        if self.t == self.time_change:
            self._change_wall()
        return info

    def _change_wall(self):
        for wid in self.wall_ps:
            self.canvas.delete(wid)
        self.wall_ps = []
        for wall in self.new_walls:
            wx0 = self.x0 + wall[0] * self.w
            wy0 = self.y0 + self.h - wall[1] * self.h
            wx1 = self.x0 + wall[2] * self.w
            wy1 = self.y0 + self.h - wall[3] * self.h
            self.wall_ps.append(self.canvas.create_rectangle(wx0, wy0, wx1, wy1, fill="black"))
        self.canvas.update()
        return

    """
    input: a list containing dictionarys. 
            state: current position of the agent
            q: q-value of the s-a pairs. Order of actions is: up, down, right, left
            i.e.[{"state": [0.5, 0.7], "q": [0.1, 0.2, 0.3, 0.4]}, {...}, ...]
    """
    def show_plan_info(self, plan_info):
        for p in plan_info:
            pos = self.state2coord(p["state"])
            qvalue = p["q"]

            # qsum = np.sum(qvalue)
            # if qsum == 0:
            #     scale = 1
            # elif qsum < 1:
            #     scale = qvalue /
            # else:
            #     scale = qvalue / np.sum(qvalue)
            range = np.max(qvalue) - np.min(qvalue)
            if range != 0:
                scale = (qvalue - np.min(qvalue)) / range
            else:
                scale = np.zeros(len(qvalue))

            canvas_id = self.canvas.create_oval(pos[0]-2, pos[1]-2, pos[0]+2, pos[1]+2, fill=self.plan_pos_color)
            self.canvas.after(self.line_time, self.canvas.delete, canvas_id)
            # s-a pair: up
            canvas_id = self.canvas.create_line(pos[0], pos[1], pos[0], int(pos[1] - scale[0] * self.line_scale),
                                                width=1, fill=self.plan_pref_color)
            self.canvas.after(self.line_time, self.canvas.delete, canvas_id)
            # s-a pair: down
            canvas_id = self.canvas.create_line(pos[0], pos[1], pos[0], int(pos[1] + scale[1] * self.line_scale),
                                                width=1, fill=self.plan_pref_color)
            self.canvas.after(self.line_time, self.canvas.delete, canvas_id)
            # s-a pair: right
            canvas_id = self.canvas.create_line(pos[0], pos[1], int(pos[0] + scale[2] * self.line_scale), pos[1],
                                                width=1, fill=self.plan_pref_color)
            self.canvas.after(self.line_time, self.canvas.delete, canvas_id)
            # s-a pair: left
            canvas_id = self.canvas.create_line(pos[0], pos[1], int(pos[0] - scale[3] * self.line_scale), pos[1],
                                                width=1, fill=self.plan_pref_color)
            self.canvas.after(self.line_time, self.canvas.delete, canvas_id)

            self.canvas.update()

            if "sbab_list" in p.keys():
                sbab_list = p["sbab_list"]
                for sbab in sbab_list:
                    pos = self.state2coord(sbab[0])
                    pos2 = self.state2coord(sbab[1])

                    canvas_id = self.canvas.create_oval(pos[0] - 2, pos[1] - 2, pos[0] + 2, pos[1] + 2,
                                                        fill="black")
                    self.canvas.after(self.line_time, self.canvas.delete, canvas_id)

                    canvas_id = self.canvas.create_oval(pos2[0] - 2, pos2[1] - 2, pos2[0] + 2, pos2[1] + 2,
                                                        fill="grey")
                    self.canvas.after(self.line_time, self.canvas.delete, canvas_id)

                    canvas_id = self.canvas.create_line(pos[0], pos[1], pos2[0], pos2[1],
                                                        width=1, fill="grey")
                    self.canvas.after(self.line_time, self.canvas.delete, canvas_id)

            self.canvas.update()

            if "succ_list" in p.keys():
                succ_list = p["succ_list"]
                for succ in succ_list:
                    pos = self.state2coord(succ[0])
                    pos2 = self.state2coord(succ[1])

                    canvas_id = self.canvas.create_oval(pos2[0] - 2, pos2[1] - 2, pos2[0] + 2, pos2[1] + 2,
                                                        fill="forestgreen")
                    self.canvas.after(self.line_time, self.canvas.delete, canvas_id)

                    canvas_id = self.canvas.create_line(pos[0], pos[1], pos2[0], pos2[1],
                                                        width=1, fill="lightgreen")
                    self.canvas.after(self.line_time, self.canvas.delete, canvas_id)
                    canvas_id = self.canvas.create_oval(pos[0] - 2, pos[1] - 2, pos[0] + 2, pos[1] + 2,
                                                        fill="darkgreen")
                    self.canvas.after(self.line_time, self.canvas.delete, canvas_id)

            self.canvas.update()

        return

    """
    show all states in buffer
    input: a numpy array, containing states in buffer
            i.e. [[0.1, 0.2], [0.5, 0.6], ...]
    """
    def show_buffer_info(self, buffer_info):
        new_buffer_id = set()
        temp_buffer_id = self.buffer_id.copy()
        for b in buffer_info:
            pos = tuple(self.state2coord(b))
            new_buffer_id.add(pos)

        for bi in temp_buffer_id:
            if bi not in new_buffer_id:
                canvas_id = self.buffer_id.pop(bi, None)
                self.canvas.delete(canvas_id)

        for nbi in new_buffer_id:
            if nbi not in self.buffer_id:
                canvas_id = self.canvas.create_text(nbi[0], nbi[1], fill=self.buffer_color,font="Times 10 bold", text="x")
                self.buffer_id[nbi] = canvas_id

        self.canvas.update()
        return
    # def show_buffer_info(self, buffer_info):
    #     for b in buffer_info:
    #         pos = self.state2coord(b)
    #         # draw point
    #         canvas_id = self.canvas.create_text(pos[0], pos[1], fill="blue",font="Times 10 bold", text="x")
    #         self.canvas.after(self.line_time, self.canvas.delete, canvas_id)
    #     self.canvas.update()
    #     return

    """
    show all prototypes in model
    input: a numpy array, containing all (s,a,s',r,gamma) sequences saved in model
            i.e. [[0.10, 0.20, 2, 0.15, 0.20, 0, 0.9],
                  [0.96, 0.91, 0, 0.96, 0.96, 1, 0.9],
                  ...
                 ]
           (You don't need to contain all these info, (s,a,s') sequece also works)
    """
    def show_proto_info(self, proto_info):
        new_proto_id = set()
        temp_proto_id = self.proto_id.copy()
        for p in proto_info:
            # If you have a sequence in different format, please modify these lines
            pos_start = tuple(self.state2coord(p[0:2]))
            pos_end = tuple(self.state2coord(p[3:5]))
            new_proto_id.add(tuple([pos_start, pos_end]))

        for pi in temp_proto_id:
            if pi not in new_proto_id:
                canvas_ids = self.proto_id.pop(pi, None)
                for cid in canvas_ids:
                    self.canvas.delete(cid)

        for npi in new_proto_id:
            if npi not in self.proto_id:
                pos_start = npi[0]
                pos_end = npi[1]
                # draw s
                sid = self.canvas.create_oval(pos_start[0]-2, pos_start[1]-2, pos_start[0]+2,
                                                    pos_start[1]+2, fill=self.proto_pos_color)
                self.canvas.tag_lower(sid)
                # draw s'
                spid = self.canvas.create_oval(pos_end[0] - 2, pos_end[1] - 2, pos_end[0] + 2,
                                                    pos_end[1] + 2, fill=self.proto_pos_color)
                self.canvas.tag_lower(spid)

                # connect s and s'
                lid = self.canvas.create_line(pos_start[0], pos_start[1], pos_end[0], pos_end[1], fill=self.proto_line_color)
                self.canvas.tag_lower(lid)

                self.proto_id[npi] = [sid, spid, lid]
        self.canvas.update()
        return
    # def show_proto_info(self, proto_info):
    #     for p in proto_info:
    #         # If you have a sequence in different format, please modify these lines
    #         pos_start = self.state2coord(p[0:2])
    #         pos_end = self.state2coord(p[3:5])
    #
    #         # draw s
    #         canvas_id = self.canvas.create_oval(pos_start[0]-2, pos_start[1]-2, pos_start[0]+2,
    #                                             pos_start[1]+2, fill='dark orange')
    #         self.canvas.after(self.line_time, self.canvas.delete, canvas_id)
    #
    #         # draw s'
    #         canvas_id = self.canvas.create_oval(pos_end[0] - 2, pos_end[1] - 2, pos_end[0] + 2,
    #                                             pos_end[1] + 2, fill='dark orange')
    #         self.canvas.after(self.line_time, self.canvas.delete, canvas_id)
    #
    #         # connect s and s'
    #         canvas_id = self.canvas.create_line(pos_start[0], pos_start[1], pos_end[0], pos_end[1])
    #         self.canvas.after(self.line_time, self.canvas.delete, canvas_id)
    #
    #     self.canvas.update()
    #     return

    def show_agent_qvalue(self, state, qvalue):
        pos = self.state2coord(state)

        sumq = np.sum(qvalue)
        if sumq != 0:
            scale = qvalue / np.sum(qvalue)
        else:
            scale = np.ones(4)

        # draw state
        canvas_id = self.canvas.create_oval(pos[0] - 2, pos[1] - 2, pos[0] + 2, pos[1] + 2, fill=self.plan_pos_color)
        self.canvas.after(self.line_time, self.canvas.delete, canvas_id)
        # s-a pair: up
        canvas_id = self.canvas.create_line(pos[0], pos[1], pos[0], int(pos[1] - scale[0] * self.line_scale),
                                            width=1, fill=self.plan_pref_color)
        self.canvas.after(self.line_time, self.canvas.delete, canvas_id)
        # s-a pair: down
        canvas_id = self.canvas.create_line(pos[0], pos[1], pos[0], int(pos[1] + scale[1] * self.line_scale),
                                            width=1, fill=self.plan_pref_color)
        self.canvas.after(self.line_time, self.canvas.delete, canvas_id)
        # s-a pair: right
        canvas_id = self.canvas.create_line(pos[0], pos[1], int(pos[0] + scale[2] * self.line_scale), pos[1],
                                            width=1, fill=self.plan_pref_color)
        self.canvas.after(self.line_time, self.canvas.delete, canvas_id)
        # s-a pair: left
        canvas_id = self.canvas.create_line(pos[0], pos[1], int(pos[0] - scale[3] * self.line_scale), pos[1],
                                            width=1, fill=self.plan_pref_color)
        self.canvas.after(self.line_time, self.canvas.delete, canvas_id)

        self.canvas.update()
        return

    """
    input: coordinate in environment (from 0 to 1)
    output: coordinate on canvas 
    """
    def state2coord(self, state):
        coord = np.zeros((2))
        coord[0] = self.x0 + state[0] * self.w
        coord[1] = self.y0 + self.h - state[1] * self.h
        return coord

    """
    input: coordinate on canvas
    output: coordinate in environment (from 0 to 1) 
    """
    def coord2state(self, coord):
        state = np.zeros((2))
        state[0] = (coord[0] - self.x0) / self.w
        state[1] = (coord[1] - self.y0 - self.h) / (-1 * self.h)
        return state

    """
    update agent's position on canvas
    input: state of agent
    """
    def move_agent(self, new_p):
        new_p = self.state2coord(new_p)
        dxy = new_p - self.current_pos
        self.canvas.move(self.agent, dxy[0], dxy[1])
        self.current_pos = new_p
        self.canvas.update()
        return


"""
This is an experiment class
Based on RL-glue

input: 
    env_params, agent_params, exp_params are dictionaries containing parameters used in the experiment.
    You can modify this part as what you need
"""
class Experiment:
    def __init__(self, env_params, agent_params, exp_params):
        self.env_params = env_params
        self.agent_params = agent_params
        self.exp_params = exp_params
        self.num_episodes = exp_params['num_episodes']
        self.num_steps = exp_params['num_steps']
        self.num_runs = exp_params['num_runs']
        self.which_to_rec = exp_params['which_to_rec']
        self.save_data = exp_params["save_data"]
        self.control_step = self.num_episodes == 0
        self.accum_r_record = np.zeros((self.num_runs, self.num_steps))
        self.count_step = 0
        self.count_run = 0
        self.end_run = True
        self.end_ep = True
        return

    """
    prepare for a new run
    set parameters in the agent and environment
    """
    def init_run(self):
        RL_init()
        np.random.seed(512)
        dim_state = RL_env_message(["state dimension", None])
        self.agent_params["dim_state"] = dim_state
        num_action = RL_env_message(["num_action", None])
        self.agent_params["num_action"] = num_action
        RL_agent_message(["set param", self.agent_params])
        RL_env_message(["set param", self.env_params])
        print("ALL params have been set.")
        self.count_step = 0
        self.accum_r = 0
        self.end_run = False
        return

    def run(self):
        while not self.end_run:
            self.episode()
        return

    def start_ep(self):
        info = RL_start()
        info["isTerminal"] = False
        self.end_ep = False
        return info

    def episode(self):
        while self.count_step < self.num_steps:
            self.step()
        return

    def step(self):
        # check if a new run should be started
        if self.end_run:
            print("calling init_run")
            self.init_run()

        # check if a new episode should be started
        if self.end_ep:
            print("calling init_ep")
            info = self.start_ep()
        else:
            info = RL_step()
            self.accum_r += info["reward"]
            self.accum_r_record[self.count_run, self.count_step] = self.accum_r
            self.end_ep = info["isTerminal"]
        self.count_step += 1

        # when it is the end of one episode
        if self.end_ep and self.count_step != 0:
            RL_end()
            self.end_ep = True
            # info["ep_end"] = True
            print(self.count_step, "steps. accum_reward =", self.accum_r)
        # when the total number of step gets to the limit of number of step,
        # both run and episode end
        if self.count_step >= self.num_steps:
            print("total number of steps is", self.count_step, ". Run ends")
            self.end_ep = True
            self.end_run = True
            info["isTerminal"] = True

        # add end_run in info dict
        info["end_run"] = self.end_run
        return info

env_params = {}
agent_params = {}
exp_params = {}
if "env_params" in exp:
    env_params = exp['env_params']
if "agent_params" in exp:
    agent_params = exp['agent_params']
if "exp_params" in exp:
    exp_params = exp['exp_params']


if len(sys.argv) > 2:
    exp['agent'] = str(sys.argv[1])
    agent_params["alpha"] = float(sys.argv[2])
    agent_params["num_near"] = int(sys.argv[3])
    agent_params["add_prot_limit"] = float(sys.argv[4])
    this_run = int(sys.argv[5])
    if exp['agent'] == "REM_Dyna" or exp['agent'] == "REM_Dyna_deb":
        agent_params["remDyna_mode"] = int(sys.argv[6])
    elif exp['agent'] == "Q_learning":
        agent_params["qLearning_mode"] = int(sys.argv[6])
    elif exp['agent'] == "random_ER":
        agent_params["erLearning_mode"] = int(sys.argv[6])
    else:
        print("The agent doesn't have learning mode")
        # exit()

    agent_params["model_params"]["kscale"] = float(sys.argv[7])
    # agent_params["similarity_limit"] = float(sys.argv[8])
    agent_params["model_params"]["sampling_limit"] = float(sys.argv[8])
    agent_params["always_add_prot"] = int(sys.argv[9])

    agent_params["model_params"]["fix_cov"] = float(sys.argv[10])
    agent_params["model_params"]["cov"] = float(sys.argv[10])

    agent_params["alg"] = str(sys.argv[11])
    agent_params["lambda"] = float(sys.argv[12])
    agent_params["momentum"] = float(sys.argv[13])
    agent_params["rms"] = float(sys.argv[14])
    agent_params["opt_mode"] = int(sys.argv[15])
    agent_params["offline"] = int(sys.argv[16])
    agent_params["num_planning"] = int(sys.argv[17])

else:
    this_run = 1

agent_params["gui"] = True
agent_params["div_actBit"] = agent_params["remDyna_mode"]

RLGlue(exp['environment'], exp['agent'])

exp_obj = Experiment(env_params, agent_params, exp_params)
# gui = ContinuousGWGUI(exp_obj)
gui = ContinuousGWGUI(exp_obj,
                      walls=np.array([[0.5, 0.0, 0.7, 0.4], [0.5, 0.6, 0.7, 1.0]]),
                      goals=np.array([[0.7, 0.95, 0.75, 1.0]]),
                      time_change=10000000,
                      new_walls=np.array([[0.5, 0.0, 0.7, 0.8]])
                      )