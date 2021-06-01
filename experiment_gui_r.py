#!/usr/bin/python3
import os
import pickle
import copy
import tkinter as tk
import numpy as np
import time

LinearModel = True
env = "pw" # "ww"
rep_type = "AE" # "noCons", "new"
TC = True
with_reward = True

if rep_type == "AE":
    nn_sign = "_AE"
else:
    nn_sign = ""
rep_sign = "_" + rep_type

if LinearModel:
    import utils.REM_model_kdt_realCov_llm_allactions as rem
else:
    import utils.REM_model_kdt_realCov as rem
# import utils.REM_model_kdt_realCov_llm as rem
# import utils.REM_model_kdt_realCov_flm as rem
# import utils.REM_model_kdt_realCov_remflm as rem
import utils.get_learned_representation as glr
import utils.get_learned_state as gls
# import matplotlib.pyplot as plt
import sys
sys.path.append('./environment/')
sys.path.append('./agent/')

import json

if env == "pw":
    jsonfile = "parameters/puddle_world.json"
else:
    jsonfile = "parameters/continuous_gridworld.json"

json_dat = open(jsonfile, 'r')
exp = json.load(json_dat)
json_dat.close()

from rl_glue import *
RLGlue(exp['environment'], exp['agent'])
print("Environment:", exp['environment'])

OLD_REM = 0
CHECK_DIST = 1
SINGLE_REP = 2
REPVF_RAWMODEL_CHECKDIST = 3
TCREPVF_RAWMODEL_CHECKDIST = 4
BIASREPVF_RAWMODEL_CHECKDIST = 5
BIASTCREPVF_RAWMODEL_CHECKDIST = 6
BIASTCREPVF_REPMODEL = 7
BIASTCREPVF_REPMODEL_CHECKDIST = 8
SINGLE_REP_CHECKDIST = 9
SINGLE_NORMREP = 10
SINGLE_NORMREP_FIXCOV = 11
TCREPVF_NORMREPMODEL_FIXCOV = 12
BIASTCREPVF_NORMREPMODEL_FIXCOV = 13
TCREPVF_NORMREPMODEL = 14
BIASTCREPVF_NORMREPMODEL = 15
NORMREPVF_RAWMODEL = 16

"""
This is an experiment class
Based on RL-glue

input:
    env_params, agent_params, exp_params are dictionaries containing parameters used in the experiment.
    You can modify this part as what you need
"""
class Experiment:
    def __init__(self, env_params, exp_params):
        self.env_params = env_params
        self.agent_params = agent_params
        self.exp_params = exp_params
        # self.num_steps = exp_params['num_steps']
        self.num_steps = 1000
        return

    """
    prepare for a new run
    set parameters in the agent and environment
    """
    def init_run(self):
        RL_init()
        dim_state = RL_env_message(["state dimension", None])
        self.agent_params["dim_state"] = dim_state
        self.num_action = RL_env_message(["num_action", None])
        self.agent_params["num_action"] = self.num_action
        self.agent_params["model_params"]["num_action"] = self.num_action

        RL_agent_message(["set param", self.agent_params])
        RL_env_message(["set param", self.env_params])
        print("ALL params have been set.")

        # # Wall world
        if jsonfile == "parameters/continuous_gridworld.json":
            if rep_type == "AE":
                len_output = 2
            else:
                len_output = self.agent_params["nn_num_tilings"] * self.agent_params["nn_num_tiles"] * 2 * 2
            self.rep_model = glr.GetLearnedRep(2, self.agent_params["nn_nodes"+nn_sign], self.agent_params["nn_num_feature"], len_output, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=self.agent_params["nn_num_tilings"], num_tile=self.agent_params["nn_num_tiles"], constraint=True, model_path=self.agent_params["nn_model_path"],file_name=self.agent_params["nn_model_name"])
            # self.rep_model = glr.GetLearnedRep(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=self.agent_params["nn_num_tilings"], num_tile=self.agent_params["nn_num_tiles"], constraint=True, model_path=self.agent_params["nn_model_path"],file_name=self.agent_params["nn_model_name"])
        # Puddle world
        elif jsonfile == "parameters/puddle_world.json":
            if rep_type != "AE" and TC and not with_reward:
                len_output = self.agent_params["nn_num_tilings"] * self.agent_params["nn_num_tiles"] * 2 * 2
            elif rep_type == "AE":
                len_output = 2
            elif rep_type != "AE":
                len_output = self.agent_params["nn_num_tilings"] * self.agent_params["nn_num_tiles"] * 2 * 2 + 1
            self.rep_model = glr.GetLearnedRep(2, self.agent_params["nn_nodes"+nn_sign], self.agent_params["nn_num_feature"], len_output, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"+nn_sign], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=1, num_tile=1, constraint=True, model_path=self.agent_params["nn_model_path"],file_name=self.agent_params["nn_model_name"+rep_sign])
            # self.rep_model = glr.GetLearnedRep(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=1, num_tile=1, constraint=True, model_path=self.agent_params["nn_model_path"],file_name=self.agent_params["nn_model_name"])

        # Catcher
        # self.rep_model = glr.GetLearnedRep(8192, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], 8192*2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=self.agent_params["nn_num_tilings"], num_tile=self.agent_params["nn_num_tiles"], constraint=True, model_path=self.agent_params["nn_model_path"],file_name=self.agent_params["nn_model_name"])

        # # Wall world
        # # self.rep_model_decoder = gls.GetLearnedState(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], self.agent_params["nn_num_tilings"] * self.agent_params["nn_num_tiles"] * 2 * 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=self.agent_params["nn_num_tilings"], num_tile=self.agent_params["nn_num_tiles"], constraint=True, model_path=self.agent_params["nn_model_path"], file_name=self.agent_params["nn_model_name"]+"_seperateRcvs")
        # self.rep_model_decoder = gls.GetLearnedState(self.agent_params["nn_num_feature"],
        #                                                  self.agent_params["nn_nodes"],
        #                                                  self.agent_params["nn_num_feature"],
        #                                                  2,
        #                                                  self.agent_params["nn_lr"],
        #                                                  self.agent_params["nn_lr"],
        #                                                  self.agent_params["nn_weight_decay"],
        #                                                  self.agent_params["nn_dec_nodes"],
        #                                                  self.agent_params["nn_rec_nodes"],
        #                                                  self.agent_params["optimizer"],
        #                                                  self.agent_params["nn_dropout"],
        #                                                  self.agent_params["nn_beta"],
        #                                                  self.agent_params["nn_delta"],
        #                                                  self.agent_params["nn_legal_v"],
        #                                                  True, num_tiling=self.agent_params["nn_num_tilings"], num_tile=self.agent_params["nn_num_tiles"], constraint=True,
        #                                                  model_path=self.agent_params["nn_model_path"],
        #                                                  file_name=self.agent_params["nn_model_name"]+"_seperateRcvs_illegal")
        self.rep_model_decoder = None
        self.agent_params["model_params"]["rep_model_decoder"] = self.rep_model_decoder

        #rem---[-1.0,1.0]
        # self.rep_model = glr.GetLearnedRep(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], 32 * 4 * 2 * 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=32, num_tile=4, constraint=True, model_path="./feature_model/", file_name="feature_embedding_continuous_input[-1.0, 1]_envSucProb1.0")
        # self.rep_model_decoder = gls.GetLearnedState(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], 32 * 4 * 2 * 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=32, num_tile=4, constraint=True, model_path="./feature_model/", file_name="feature_embedding_continuous_input[-1.0, 1]_envSucProb1.0_seperateRcvs", default=False)

        # self.rep_model = glr.GetLearnedRep(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], 32 * 4 * 2 * 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=32, num_tile=4, constraint=True, model_path="./feature_model/", file_name="feature_embedding_lplc_continuous_envSucProb1.0", default=False)
        # self.rep_model_decoder = gls.GetLearnedState(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], 32 * 4 * 2 * 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=32, num_tile=4, constraint=True, model_path="./feature_model/", file_name="feature_embedding_lplc_continuous_envSucProb1.0_seperateRcvs", default=False)

        return

    """
    Generate learned representation
    Input: [x, y]
    Return: state representation
    """
    def _state_representation(self, state, model=None):

        state_new = copy.deepcopy(state)

        #remove if rem[0.0,1.0]
        # state_new[0] = (-1.0+(state_new[0]*2.0))
        # state_new[1] = (-1.0+(state_new[1]*2.0))

        if model == None:
            rep = self.rep_model.state_representation(np.array(state_new))
        else:
            rep = model.state_representation(np.array(state_new))

        self.learning_mode = self.agent_params["remDyna_mode"]
        # if self.learning_mode == SINGLE_NORMREP or \
        #         self.learning_mode == SINGLE_NORMREP_FIXCOV or \
        #         self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or \
        #         self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV or \
        #         self.learning_mode == TCREPVF_NORMREPMODEL or \
        #         self.learning_mode == BIASTCREPVF_NORMREPMODEL:
        #     rep = rep / float(np.linalg.norm(rep))
        rep = rep / float(np.linalg.norm(rep))
        return rep


    def run_model_learning(self):
        np.random.seed(seed=1000)
        self.init_run()
        # self.prototypes_x = {}
        # self.prototypes_xdash = {}
        # for act in range(self.num_action):
        for act in range(1):
            print(act)
            prototypes_x = []
            prototypes_xdash = []
            self.model = rem.REM_Model(self.agent_params["nn_num_feature"], self.agent_params["num_near"], self.agent_params["add_prot_limit"], self.agent_params["model_params"], self.agent_params["remDyna_mode"], 35.0, 0, rep_model=self.rep_model)
            # for step in range(self.num_steps):
            step = 1
            while True:
                original_state = RL_env_message(["sample_random"])
                state = self._state_representation(original_state)
                action = np.random.randint(self.num_action)
                # action = act
                result = RL_env_step(action)
                r = result["reward"]
                original_next_state = result['state']
                next_state = self._state_representation(original_next_state)
                if result["isTerminal"]:
                    g = 0
                else:
                    g = 0.9
                self.model.add2Model(state, action, next_state, r, g)
                if self.model.get_added_prototype():
                    prototypes_x.append(original_state)
                    prototypes_xdash.append(original_next_state)
                    print(step, len(prototypes_x))
                    if len(prototypes_x) == 100:
                        break
                step += 1
                if (step%1000) == 0:
                    print(step)
                if step == 100000:
                    break
            # self.prototypes_x[act] = prototypes_x
            # self.prototypes_xdash[act] = prototypes_xdash
            # print(len(prototypes_x))
            # with open('prototypes/rem-FCov/'+str(act)+'s.pkl', 'wb') as f:
            with open('prototypes/rem-CCov-temp/s.pkl', 'wb') as f:
                pickle.dump(prototypes_x, f)
            # with open('prototypes/rem-FCov/'+str(act)+'sdash.pkl', 'wb') as f:
            with open('prototypes/rem-CCov-temp/sdash.pkl', 'wb') as f:
                pickle.dump(prototypes_xdash, f)

    def run_model_learning_sarsalambda_policy(self):
        np.random.seed(seed=1000)
        self.init_run()
        self.prototypes_x = {}
        self.prototypes_xdash = {}
        self.model = rem.REM_Model(self.agent_params["nn_num_feature"], self.agent_params["num_near"], self.agent_params["add_prot_limit"], self.agent_params["model_params"], self.agent_params["remDyna_mode"], 35.0, 0, rep_model=self.rep_model)
        weight = np.zeros((32*4))
        traces = np.zeros((32*4))
        values = np.zeros((4))
        for ep in range(1):
            # print(weight)
            print("episode:",ep)
            original_state = RL_env_start()
            state = self._state_representation(original_state)
            if np.random.uniform() < 0.1:
                action = np.random.randint(self.num_action)
            else:
                for act in range(4):
                    values[act] = np.dot(weight[(32*act):(32*(act+1))], state)
                max_v = np.max(values)
                action = np.random.choice(np.where(values == max_v)[0])
            step = 1
            while True:
                result = RL_env_step(action)
                r = result["reward"]
                original_next_state = result['state']
                next_state = self._state_representation(original_next_state)
                if result["isTerminal"]:
                    g = 0
                else:
                    g = 0.9
                self.model.add2Model(state, action, next_state, r, g)
                estimate = np.dot(weight[(32*action):(32*(action+1))], state)
                traces[(32*action):(32*(action+1))] += state
                if not result["isTerminal"]:
                    if np.random.uniform() < 0.1:
                        action_next = np.random.randint(self.num_action)
                    else:
                        for act in range(4):
                            values[act] = np.dot(weight[(32*act):(32*(act+1))], next_state)
                        max_v = np.max(values)
                        action_next = np.random.choice(np.where(values == max_v)[0])
                    tde = r + (g*values[action_next]) - estimate
                else:
                    tde = r - estimate
                weight += (tde*(0.001/np.linalg.norm(state))*traces)
                traces *= (0.9*g)
                if self.model.get_added_prototype():
                    if action in self.prototypes_x:
                        self.prototypes_x[action].append(original_state)
                        self.prototypes_xdash[action].append(original_next_state)
                    else:
                        self.prototypes_x[action]=[original_state]
                        self.prototypes_xdash[action]=[original_next_state]
                    print(step, action, len(self.prototypes_x[act]))
                    quit = True
                    for act in range(4):
                        if (act in self.prototypes_x) and (len(self.prototypes_x[act]) < 500):
                            quit = False
                action = action_next
                state = next_state
                if quit:
                    break
                step += 1
                if (step%1000) == 0:
                    print(step)
                if step == 50000 or result["isTerminal"]:
                    break
                else:
                    original_state = original_next_state
            if quit:
                break
        for act in range(4):
            if act in self.prototypes_x:
                with open('prototypes/rem-GCov-100p-sarsalambda-1NN-modelLearning/'+str(act)+'s.pkl', 'wb') as f:
                    pickle.dump(self.prototypes_x[act], f)
                with open('prototypes/rem-GCov-100p-sarsalambda-1NN-modelLearning/'+str(act)+'sdash.pkl', 'wb') as f:
                    pickle.dump(self.prototypes_xdash[act], f)

        with open('prototypes/rem-GCov-100p-sarsalambda-1NN-modelLearning/model.pkl', 'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

    def run_model_learning_random_policy(self, folder):
        if LinearModel:
            self.run_LLM_model_learning_random_policy(folder)
        else:
            self.run_REM_model_learning_random_policy(folder)

    # For REM
    def run_REM_model_learning_random_policy(self, folder):
        np.random.seed(seed=512)
        self.init_run()
        self.prototypes_x = {}
        self.prototypes_xdash = {}

        # representation
        self.model = rem.REM_Model(self.agent_params["nn_num_feature"], self.agent_params["num_near"], self.agent_params["add_prot_limit"], self.agent_params["model_params"], self.agent_params["remDyna_mode"], 35.0, 0, rep_model=self.rep_model)
        # raw
        # self.model = rem.REM_Model(2, self.agent_params["num_near"], self.agent_params["add_prot_limit"], self.agent_params["model_params"], self.agent_params["remDyna_mode"], 0, 0)

        for ep in range(100):
            start = time.time()
            print("episode:",ep)
            original_state = RL_env_start()
            # original_state = RL_env_message(["sample_random"])
            step = 1
            while True:
                # representation
                state = self._state_representation(original_state)
                # # raw
                # state = np.array(original_state)

                action = np.random.randint(self.num_action)
                result = RL_env_step(action)
                r = result["reward"]
                original_next_state = result['state']

                # representation
                next_state = self._state_representation(original_next_state)
                # raw
                # next_state = np.array(original_next_state)

                if result["isTerminal"]:
                    g = 0
                else:
                    g = 1.0
                self.model.add2Model(state, action, next_state, r, g)
                # print("Added:",self.rep_model_decoder.state_learned(state),action,self.rep_model_decoder.state_learned(next_state))
                # self.model.update_rem_onlyLearnPrototypes(state, action, next_state, r, g)
                if self.model.get_added_prototype():
                    if action in self.prototypes_x:
                        self.prototypes_x[action].append(original_state)
                        self.prototypes_xdash[action].append(original_next_state)
                    else:
                        self.prototypes_x[action]=[original_state]
                        self.prototypes_xdash[action]=[original_next_state]
                    print(step, action, len(self.prototypes_x[action]))
                    quit = True
                    for act in range(4):
                        if (act in self.prototypes_x) and (len(self.prototypes_x[act]) < 50000):
                            quit = False
                if quit:
                    break
                step += 1
                if (step%1000) == 0:
                    print(step)
                # if step == 10000 or result["isTerminal"]:
                if result["isTerminal"]:
                    break
                else:
                    original_state = original_next_state
            if quit:
                break
            print("1 ep time =", time.time() - start)
        with open(folder+'model.pkl', 'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)
        for act in range(4):
            with open(folder+str(act)+'s.pkl', 'wb') as f:
                pickle.dump(self.prototypes_x[act], f)
            with open(folder+str(act)+'sdash.pkl', 'wb') as f:
                pickle.dump(self.prototypes_xdash[act], f)

    # For LLM
    def run_LLM_model_learning_random_policy(self, folder):
        np.random.seed(seed=512)
        self.init_run()

        self.prototypes_forward_x = []
        self.prototypes_forward_xdash = []
        self.prototypes_reverse_x = []
        self.prototypes_reverse_xdash = []

        self.model = rem.REM_Model(self.agent_params["nn_num_feature"],
                                   self.agent_params["num_near"],
                                   self.agent_params["add_prot_limit"],
                                   self.agent_params["model_params"],
                                   self.agent_params["remDyna_mode"],
                                   self.agent_params["similarity_limit"],
                                   0,
                                   rep_model=self.rep_model)
        # self.model = rem.REM_Model(2,
        #                            self.agent_params["num_near"],
        #                            self.agent_params["add_prot_limit"],
        #                            self.agent_params["model_params"],
        #                            self.agent_params["remDyna_mode"],
        #                            self.agent_params["similarity_limit"],
        #                            0,
        #                            rep_model=self.rep_model)
        for ep in range(100):
            print("episode:",ep)
            start = time.time()
            original_state = RL_env_start()
            # original_state = RL_env_message(["sample_random"])
            step = 1
            while True:
                state = self._state_representation(original_state)
                # state = np.array(original_state)

                action = np.random.randint(self.num_action)
                result = RL_env_step(action)
                r = result["reward"]
                original_next_state = result['state']

                next_state = self._state_representation(original_next_state)
                # next_state = np.array(original_next_state)

                if result["isTerminal"]:
                    g = 0
                else:
                    g = 1.0
                self.model.add2Model(state, action, next_state, r, g)

                if self.model.get_added_prototype_forward():
                    self.prototypes_forward_x.append(original_state)
                    self.prototypes_forward_xdash.append(original_next_state)
                    print(step, "forward", len(self.prototypes_forward_x))

                if self.model.get_added_prototype_reverse():
                    self.prototypes_reverse_x.append(original_state)
                    self.prototypes_reverse_xdash.append(original_next_state)
                    print(step, "reverse", len(self.prototypes_reverse_x))

                # if quit:
                #     break
                step += 1
                if (step%1000) == 0:
                    print(step)
                # if step == 10000 or result["isTerminal"]:
                if result["isTerminal"]:
                    break
                else:
                    original_state = original_next_state
            # if quit:
            #     break
            print("running time", time.time() - start)

        with open(folder+'model.pkl', 'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

        with open(folder+'-forward-s.pkl', 'wb') as f:
            pickle.dump(self.prototypes_forward_x, f)
        with open(folder+'-forward-sdash.pkl', 'wb') as f:
            pickle.dump(self.prototypes_forward_xdash, f)
        with open(folder+'-reverse-s.pkl', 'wb') as f:
            pickle.dump(self.prototypes_reverse_x, f)
        with open(folder+'-reverse-sdash.pkl', 'wb') as f:
            pickle.dump(self.prototypes_reverse_xdash, f)

        arr = self.model.check_model_for_action()
        np.save(folder+"trained_time", arr)


    def run_model_sampling_random_policy(self, folder, folder2):

        with open(folder+'model.pkl', 'rb') as f:
            self.model = pickle.load(f)

        np.random.seed(seed=512)
        self.init_run()
        self.prototypes_x = {}
        self.prototypes_xprev = {}
        for ep in range(1):
            print("episode:",ep)
            original_state = RL_env_start()
            step = 1
            while True:
                # state = self._state_representation(original_state)
                state = original_state
                action = np.random.randint(self.num_action)
                result = RL_env_step(action)
                r = result["reward"]
                original_next_state = result['state']
                # next_state = self._state_representation(original_next_state)
                next_state = original_next_state
                if result["isTerminal"]:
                    g = 0
                else:
                    g = 0.9
                sampled_state = self.rep_model_decoder.state_learned(self.model._sample_predecessor_for_action(action, next_state, None))
                # sampled_state = self.model._sample_predecessor_for_action(action, next_state, None)
                if action in self.prototypes_x:
                    self.prototypes_x[action].append(original_next_state)
                    self.prototypes_xprev[action].append(sampled_state)
                else:
                    self.prototypes_x[action]=[original_next_state]
                    self.prototypes_xprev[action]=[sampled_state]
                print(step, action, len(self.prototypes_x[action]))
                step += 1
                if (step%1000) == 0:
                    print(step)
                if step == 10000 or result["isTerminal"]:
                    break
                else:
                    original_state = original_next_state
        for act in range(4):
            with open(folder2+str(act)+'splan.pkl', 'wb') as f:
                pickle.dump(self.prototypes_x[act], f)
            with open(folder2+str(act)+'sprev.pkl', 'wb') as f:
                pickle.dump(self.prototypes_xprev[act], f)

    def run_model_sampling_forward_random_policy(self):

        with open('prototypes/rem-GCov-100p-randomwalk/model.pkl', 'rb') as f:
            self.model = pickle.load(f)

        np.random.seed(seed=1000)
        self.init_run()
        self.prototypes_x = {}
        self.prototypes_xnext = {}
        for ep in range(1):
            print("episode:",ep)
            original_state = RL_env_start()
            step = 1
            while True:
                state = self._state_representation(original_state)
                action = np.random.randint(self.num_action)
                result = RL_env_step(action)
                r = result["reward"]
                original_next_state = result['state']
                next_state = self._state_representation(original_next_state)
                if result["isTerminal"]:
                    g = 0
                else:
                    g = 0.9
                sampled_state = self.rep_model_decoder.state_learned(self.model.sample_sprg(state, action)[2])
                if action in self.prototypes_x:
                    self.prototypes_x[action].append(original_state)
                    self.prototypes_xnext[action].append(sampled_state)
                else:
                    self.prototypes_x[action]=[original_state]
                    self.prototypes_xnext[action]=[sampled_state]
                print(step, action, len(self.prototypes_x[action]))
                step += 1
                if (step%1000) == 0:
                    print(step)
                if step == 10000 or result["isTerminal"]:
                    break
                else:
                    original_state = original_next_state
        for act in range(4):
            with open('sampling/rem-GCov-100p-randomwalk-forwardsampling/'+str(act)+'splan.pkl', 'wb') as f:
                pickle.dump(self.prototypes_x[act], f)
            with open('sampling/rem-GCov-100p-randomwalk-forwardsampling/'+str(act)+'snext.pkl', 'wb') as f:
                pickle.dump(self.prototypes_xnext[act], f)


    def run_model_sampling_forward_fixed_data(self, folder, folder2):

        with open(folder+'model.pkl', 'rb') as f:
            self.model = pickle.load(f)
            self.model.offline = True

        np.random.seed(seed=512)
        self.init_run()
        self.prototypes_x_backup = copy.deepcopy(self.prototypes_x)
        self.prototypes_x = {}
        self.prototypes_xnext = {}
        for act in range(4):
            for original_state in self.prototypes_x_backup[act]:
                # state = self._state_representation(original_state)
                state = original_state
                action = act
                # sampled_state = self.rep_model_decoder.state_learned(self.model.sample_sprg(state, action)[2])
                sampled_state = self.model.sample_sprg(state, action)[2]
                if action in self.prototypes_x:
                    self.prototypes_x[action].append(original_state)
                    self.prototypes_xnext[action].append(sampled_state)
                else:
                    self.prototypes_x[action]=[original_state]
                    self.prototypes_xnext[action]=[sampled_state]
                print(action, len(self.prototypes_x[action]))
        for act in range(4):
            with open(folder2+str(act)+'splan.pkl', 'wb') as f:
                pickle.dump(self.prototypes_x[act], f)
            with open(folder2+str(act)+'snext.pkl', 'wb') as f:
                pickle.dump(self.prototypes_xnext[act], f)

    def run_model_decoder(self, folder):
        np.random.seed(seed=512)
        self.init_run()
        # self.prototypes_x = {}
        # self.prototypes_xdash = {}
        # for act in range(self.num_action):
        for act in range(1):
            print(act)
            prototypes_x = []
            prototypes_xdash = []

            step = 1
            error = 0.0
            while True:
                original_state = RL_env_message(["sample_random"])
                state = self._state_representation(original_state)
                rec_state = self.rep_model_decoder.state_learned(state)
                if (rec_state[0] < 0) or (rec_state[1] < 0):
                    print(original_state,rec_state)
                error += np.power(np.linalg.norm(rec_state-original_state),2)
                prototypes_x.append(original_state)
                prototypes_xdash.append(rec_state)
                step += 1
                if (step%1000) == 0:
                    print(step)
                if step == 100:
                    break
            # self.prototypes_x[act] = prototypes_x
            # self.prototypes_xdash[act] = prototypes_xdash
            # print(len(prototypes_x))
            # with open(folder+str(act)+'s.pkl', 'wb') as f:
            with open(folder+'s.pkl', 'wb') as f:
                pickle.dump(prototypes_x, f)
            # with open(fodler+str(act)+'sdash.pkl', 'wb') as f:
            with open(folder+'srec.pkl', 'wb') as f:
                pickle.dump(prototypes_xdash, f)

        print("Decoder error:",error/step)


    def run_model_knn(self, folder="temp", folder2="temp", data=None):

        with open(folder+'model.pkl', 'rb') as f:
            self.model = pickle.load(f)

        np.random.seed(seed=512)
        self.init_run()

        state_dim = self.model.state_dim

        for statenum in range(len(data)):
            print("---")
            print("State:",data[statenum])
            for act in range(4):
                print("Action:",act)
                state = self._state_representation(data[statenum])
                action = act

                #sample next state
                occupied = [i for i in range(0, state_dim+1)]
                seq = self.model._refill_seq(np.concatenate((state, np.array([action]))), occupied)
                kernel, nz_ind = self.model._kernel_sa(seq)
                # kernel, nz_ind = self.model._kernel_sa(seq,unit_tree=True)
                knn = self.model.prot_array[nz_ind, :32]

                #sample prev state
                # occupied = [i for i in range(state_dim, state_dim * 2 + 1)]
                # seq = self.model._refill_seq(np.concatenate((np.array([action]), state)), occupied)
                # kernel, nz_ind = self.model._kernel_asp(seq)
                # knn = self.model.prot_array[nz_ind, 33:65]

                state_knn = []
                print(len(knn))
                pos = 0
                for nb in knn:
                    sn = self.rep_model_decoder.state_learned(nb)
                    print("Neighbour:",sn,"From:",self.rep_model_decoder.state_learned(self.model.prot_array[nz_ind[pos], :32]),"To:",self.rep_model_decoder.state_learned(self.model.prot_array[nz_ind[pos], 33:65]))
                    pos += 1
                    state_knn.append(sn)

                state_knn.append(data[statenum])
                dataHere = np.array(state_knn)

                with open(folder2+str(statenum)+str(act)+'.pkl', 'wb') as f:
                    pickle.dump(dataHere, f)

    # def run_model_sampling_fixed_states(self, data, num_samples, folder):
    #
    #     with open(folder + '/model.pkl', 'rb') as f:
    #         self.model_new = pickle.load(f)
    #         self.model_new.offline = True
    #         self.model_new.cov = self.agent_params["model_params"]["cov"]
    #         self.model_new.sampling_limit = agent_params["model_params"]["sampling_limit"]
    #
    #     self.model_new_encoder = glr.GetLearnedRep(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], self.agent_params["nn_num_tilings"] * self.agent_params["nn_num_tiles"] * 2 * 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=self.agent_params["nn_num_tilings"], num_tile=self.agent_params["nn_num_tiles"], constraint=True, model_path=self.agent_params["nn_model_path"], file_name=self.agent_params["nn_model_name"])
    #     self.model_new_decoder = gls.GetLearnedState(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=self.agent_params["nn_num_tilings"], num_tile=self.agent_params["nn_num_tiles"], constraint=True, model_path=self.agent_params["nn_model_path"], file_name=self.agent_params["nn_model_name"]+"_seperateRcvs_illegal")
    #
    #     np.random.seed(seed=1000)
    #     self.init_run()
    #     statenum = 0
    #     for state in data:
    #         print("State:",state)
    #         for action in range(4):
    #         # for action in range(1):
    #             print("Action:",action)
    #             self.samples_world = []
    #             self.samples_new = []
    #             self.samples_old = []
    #             self.samples_lap = []
    #             for step in range(1):
    #                 print(step,end=" ")
    #
    #                 RL_env_message(["set_state",state])
    #
    #                 result = RL_env_step(action)
    #
    #                 # new_rem_next_state = self.model_new.sample_sprg(np.array(state), action)
    #                 prediction = self.model_new.sample_sprg(self._state_representation(state, model=self.model_new_encoder), action)
    #                 new_rem_next_state = self.model_new_decoder.state_learned(prediction[2])
    #                 print(state, action, new_rem_next_state, prediction[3], prediction[4])
    #                 print(action, new_rem_next_state)
    #                 self.samples_new.append(new_rem_next_state)
    #             self.samples_new.append(state)
    #             print(self.samples_new)
    #             with open(folder + str(statenum) + str(action) + 'forward.pkl', 'wb') as f:
    #                 pickle.dump(self.samples_new, f)
    #
    #             #     prediction = self.model_new.sampleFromNext_pan(np.array(state), 1, self.num_action, config=[action, 0, 0.9])
    #             #     # prediction = self.model_new.sampleFromNext_pan(self._state_representation(state, model=self.model_new_encoder), 1, self.num_action, config=[action, 0, 0.9])
    #             #     s, a, sp, r, g = prediction
    #             #     pred_state = s
    #             #     # pred_state = self.model_new_decoder.state_learned(s)
    #             #     print(pred_state, a, state, r, g)
    #             #     self.samples_new.append(pred_state)
    #             # self.samples_new.append(state)
    #             # with open(folder +str(statenum)+str(action)+'backward.pkl', 'wb') as f:
    #             #     pickle.dump(self.samples_new, f)
    #
    #
    #             # print(self.samples_world)
    #             # with open('sampling-vis/world-forwardsampling/'+str(statenum)+str(action)+'.pkl', 'wb') as f:
    #             #     pickle.dump(self.samples_world, f)
    #             # with open(folder +str(statenum)+str(action)+'.pkl', 'wb') as f:
    #             #     pickle.dump(self.samples_new, f)
    #             # with open('sampling-vis/old_rem-forwardsampling/'+str(statenum)+str(action)+'.pkl', 'wb') as f:
    #                 # pickle.dump(self.samples_old, f)
    #             # with open('sampling-vis/lapInput-forwardsampling/'+str(statenum)+str(action)+'.pkl', 'wb') as f:
    #                 # pickle.dump(self.samples_lap, f)
    #         print()
    #         statenum += 1

    def run_model_sampling_fixed_states(self, data, num_samples, folder):

        with open(folder + '/model.pkl', 'rb') as f:
            self.model_new = pickle.load(f)
            self.model_new.offline = True
            self.model_new.kscale = self.agent_params["model_params"]["kscale"]
            # # fixed cov sampling
            # self.model_new.cov = self.agent_params["model_params"]["cov"]
            # self.model_new.const_fix_cov_tree()

        # self.model_new_encoder = glr.GetLearnedRep(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], self.agent_params["nn_num_tilings"] * self.agent_params["nn_num_tiles"] * 2 * 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=self.agent_params["nn_num_tilings"], num_tile=self.agent_params["nn_num_tiles"], constraint=True, model_path=self.agent_params["nn_model_path"], file_name=self.agent_params["nn_model_name"])
        self.model_new_encoder = glr.GetLearnedRep(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], 2 * 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=self.agent_params["nn_num_tilings"], num_tile=self.agent_params["nn_num_tiles"], constraint=True, model_path=self.agent_params["nn_model_path"], file_name=self.agent_params["nn_model_name"])
        # self.model_new_decoder = gls.GetLearnedState(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=self.agent_params["nn_num_tilings"], num_tile=self.agent_params["nn_num_tiles"], constraint=True, model_path=self.agent_params["nn_model_path"], file_name=self.agent_params["nn_model_name"]+"_seperateRcvs_illegal")

        # with open('prototypes/rem-GCov-100p-randomwalk-OR/model.pkl', 'rb') as f:
        # self.model_old = pickle.load(f)
        # self.model_new.kscale = 1
        # self.model_new.num_near = 2
        # self.model_new.fix_cov = 0.0001

        # with open('prototypes/rem-GCov-100p-randomwalk-lapInput/model.pkl', 'rb') as f:
        # self.model_lap = pickle.load(f)
        # self.model_new.kscale = 1
        # self.model_new.num_near = 2
        # self.model_new.fix_cov = 0.0001

        # self.model_lap_encoder = glr.GetLearnedRep(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], 32 * 4 * 2 * 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=32, num_tile=4, constraint=True, model_path="./feature_model/", file_name="feature_embedding_lplc_continuous_envSucProb1.0", default=False)

        # self.model_lap_decoder = gls.GetLearnedState(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], 32 * 4 * 2 * 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=32, num_tile=4, constraint=True, model_path="./feature_model/", file_name="feature_embedding_lplc_continuous_envSucProb1.0_seperateRcvs", default=False)

        np.random.seed(seed=1000)
        self.init_run()
        statenum = 0
        for state in data:
            print("State:", state)
            self.samples_world = state
            for action in range(4):
                # for action in range(1):
                print("Action:", action)
                self.samples_new = []
                self.samples_old = []
                self.samples_lap = []
                for step in range(100):
                    print(step, end=" ")

                    RL_env_message(["set_state", state])

                    result = RL_env_step(action)
                    # original_next_state = result['state']

                    # res = self.model_new.sample_sprg(self._state_representation(state, model=self.model_new_encoder), action)
                    res = self.model_new.sample_sprg(np.array(state), action)
                    if res is not None:
                        # new_rem_next_state = self.model_new_decoder.state_learned(res[2])
                        new_rem_next_state = res[2]
                        self.samples_new.append(new_rem_next_state)
                    else:
                        self.samples_new.append([-1, -1])

                    # lap_next_state = self.model_lap_decoder.state_learned(self.model_lap.sample_sprg(self._state_representation(state, model=self.model_lap_encoder), action)[2])
                    # self.samples_lap.append(lap_next_state)

                print()

                # print(self.samples_world)
                with open(folder + str(statenum) + '-state.pkl', 'wb') as f:
                    pickle.dump(self.samples_world, f)
                with open(folder + str(statenum) + str(action) + '-sample.pkl', 'wb') as f:
                    pickle.dump(self.samples_new, f)
                # with open('sampling-vis/old_rem-forwardsampling/'+str(statenum)+str(action)+'.pkl', 'wb') as f:
                # pickle.dump(self.samples_old, f)
                # with open('sampling-vis/lapInput-forwardsampling/'+str(statenum)+str(action)+'.pkl', 'wb') as f:
                # pickle.dump(self.samples_lap, f)
            statenum += 1


    def learn_linear_model(self, folder='temp', num_samples=1000):

        with open(folder+'model.pkl', 'rb') as f:
            self.model = pickle.load(f)

        self.rep_model_decoder = gls.GetLearnedState(2, self.agent_params["nn_nodes"], self.agent_params["nn_num_feature"], self.agent_params["nn_num_tilings"] * self.agent_params["nn_num_tiles"] * 2 * 2, self.agent_params["nn_lr"], self.agent_params["nn_lr"], self.agent_params["nn_weight_decay"], self.agent_params["nn_dec_nodes"], self.agent_params["nn_rec_nodes"], self.agent_params["optimizer"], self.agent_params["nn_dropout"], self.agent_params["nn_beta"], self.agent_params["nn_delta"], self.agent_params["nn_legal_v"], True, num_tiling=self.agent_params["nn_num_tilings"], num_tile=self.agent_params["nn_num_tiles"], constraint=True, model_path=self.agent_params["nn_model_path"], file_name=self.agent_params["nn_model_name"]+"_seperateRcvs")

        state_dim = self.agent_params["nn_num_feature"]

        pos = 15
        print("Prototype:",self.rep_model_decoder.state_learned(self.model.prot_array[pos,:state_dim]), self.model.prot_array[pos,state_dim],self.rep_model_decoder.state_learned(self.model.prot_array[pos,state_dim+1:(state_dim*2)+1]))
        # exit()

        np.random.seed(seed=512)
        self.init_run()

        #for a particular prototype
        self.model.prot_array_model_forward[0].reset_model()
        indices = [i for i in range(state_dim)]
        _, covmat_inv = self.model._cal_covmat_inv(indices)
        prot_seq = self.model.prot_array[0, indices]

        #learn model
        for i in range(1000):
            original_state = RL_env_message(["sample_random"])
            state = self._state_representation(original_state)
            # state = original_state
            # state = np.array(original_state)
            action = 0
            result = RL_env_step(action)
            r = result["reward"]
            original_next_state = result['state']
            next_state = self._state_representation(original_next_state)
            # next_state = original_next_state
            # next_state = np.array(original_next_state)
            if result["isTerminal"]:
                g = 0
            else:
                g = 0.9

            #for a particular prototype
            last_state_copy = np.copy(state)
            last_state_copy = np.insert(last_state_copy,0,1.0)
            state_copy = np.copy(next_state)
            state_copy = np.insert(state_copy,0,1.0)
            reward = r
            gamma = g
            # seq = self.model._seq2array(state, action, next_state, reward, gamma)[indices]
            # diff = seq - prot_seq
            # rho = np.exp(-1 * diff.dot(covmat_inv).dot(diff.T))
            # rho = 1
            # self.model.prot_array_model_forward[0].update_model(rho*last_state_copy, rho*state, rho*reward, rho*gamma)
            self.model.prot_array_model_forward[0].update_model(last_state_copy, state, reward, gamma)

            # self.model.update_rem_onlyUpdatePrototypes(state, action, next_state, r, g)

        # with open(folder+'model.pkl', 'wb') as f:
        #     pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

        # #evaluate model
        sum_err_state = 0.
        sum_err_reward = 0.
        sum_err_gamma = 0.

        world_samples = []
        predict_samples = []

        np.random.seed(seed=512)

        for i in range(10):
            original_state = RL_env_message(["sample_random"])
            state = self._state_representation(original_state)
            action = 3
            result = RL_env_step(action)
            r = result["reward"]
            original_next_state = result['state']
            next_state = self._state_representation(original_next_state)
            if result["isTerminal"]:
                g = 0
            else:
                g = 0.9

            #for a particular prototype
            last_state_copy = np.copy(state)
            last_state_copy = np.insert(last_state_copy,0,1.0)
            predictions = self.model.prot_array_model_forward[0].predict(last_state_copy)

            next_state_pred, reward_pred, gamma_pred = predictions[:state_dim], predictions[state_dim], predictions[state_dim+1]

            sum_err_state += np.power(np.linalg.norm(next_state_pred-next_state), 2)
            sum_err_reward += np.power(reward_pred-r, 2)
            sum_err_gamma += np.power(gamma_pred-g, 2)

            print(original_next_state, self.rep_model_decoder.state_learned(next_state_pred))

            # world_samples.append(original_next_state)
            # predict_samples.append(self.rep_model_decoder.state_learned(next_state_pred))

        print("Avg. error: ", (sum_err_state/num_samples),(sum_err_reward/num_samples),(sum_err_gamma/num_samples))
        #
        # with open('lm-vis/world.pkl', 'wb') as f:
        #     pickle.dump(world_samples, f)
        # with open('lm-vis/sample.pkl', 'wb') as f:
        #     pickle.dump(predict_samples, f)

    def run_matrix_inverse_calc(self, folder='temp', num_samples=1000):
        np.random.seed(seed=512)
        self.init_run()

        # dim_here = 5
        dim_here = 4
        cov_mat = np.zeros((dim_here,dim_here))

        inv_mat = np.eye((dim_here)) * 1.0

        tempd1 = np.zeros((dim_here, 1))
        temp1d = np.zeros((1, dim_here))
        tempdd = np.zeros((dim_here, dim_here))

        for i in range(num_samples):
            original_state = RL_env_message(["sample_random"])
            # original_state = RL_env_message(["sample_random_around",[0.1,0.1],[0.15,0.15]])
            state = self._state_representation(original_state)

            # state = np.insert(state,0,1.0)
            state_temp = state[...,np.newaxis]

            cov_mat += state_temp.dot(state_temp.T)

            #Ainv u
            inv_mat.dot(state_temp, out=tempd1)
            #v^T Ainv
            state_temp.T.dot(inv_mat,out=temp1d)
            #Ainv u v^T Ainv
            tempd1.dot(temp1d, out=tempdd)
            #1.0 + v^T Ainv u
            denominator = 1.0 + temp1d.dot(state_temp)
            #update
            inv_mat -= ((1.0/denominator)*tempdd)

        cov_mat /= num_samples
        cor_mat = np.linalg.inv(cov_mat)

        print(cov_mat)
        print(inv_mat)
        print(cor_mat)

        np.save("temp_r/real_cov_local_poor.npy",cov_mat)
        np.save("temp_r/real_cor_local_poor.npy",cor_mat)

        # with open(folder+'.pkl', 'wb') as f:
        #     np.save(f,inv_mat)

    def run_matrix_inverse_calc_transition(self, folder='temp', num_samples=1000):
        np.random.seed(seed=512)
        self.init_run()

        # dim_here = 5
        dim_here = (4*2)+2

        sum = np.zeros((dim_here,1))
        cov_mat = np.zeros((dim_here,dim_here))

        inv_mat = np.eye((dim_here)) * 1.0

        tempd1 = np.zeros((dim_here, 1))
        temp1d = np.zeros((1, dim_here))
        tempdd = np.zeros((dim_here, dim_here))

        for i in range(num_samples):
            original_state = RL_env_message(["sample_random"])
            # original_state = RL_env_message(["sample_random_around",[0.1,0.1],[0.15,0.15]])
            state = self._state_representation(original_state)
            state = state/np.linalg.norm(state)

            action = np.random.randint(self.num_action)
            result = RL_env_step(action)
            r = result["reward"]
            if result["isTerminal"]:
                gamma = 0.0
            else:
                gamma = 0.9
            original_next_state = result['state']
            next_state = self._state_representation(original_next_state)
            next_state = state/np.linalg.norm(next_state)

            state = np.append(state,next_state)
            state = np.append(state,r)
            state = np.append(state,gamma)

            state_temp = state[...,np.newaxis]

            sum =  ((i*sum) + state_temp)/(i+1)
            cov_mat = ((i*cov_mat) + state_temp.dot(state_temp.T))/(i+1)

            # #Ainv u
            # inv_mat.dot(state_temp, out=tempd1)
            # #v^T Ainv
            # state_temp.T.dot(inv_mat,out=temp1d)
            # #Ainv u v^T Ainv
            # tempd1.dot(temp1d, out=tempdd)
            # #1.0 + v^T Ainv u
            # denominator = 1.0 + temp1d.dot(state_temp)
            # #update
            # inv_mat -= ((1.0/denominator)*tempdd)

        cov_mat /= num_samples
        sum /= num_samples

        cov_mat = cov_mat - sum

        inv_mat = np.linalg.inv(cov_mat)

        np.save(folder,inv_mat)

    def run_predecessor_chaining_recursive(self, state_rep, actions_map):
        sampled_states = []
        for action in range(0,4):
            state = self.model._sample_predecessor_for_action(action, state_rep, None)
            # state = state/np.linalg.norm(state)
            # sampled_state = self.rep_model_decoder.state_learned(state)
            sampled_state = state
            sampled_states.append(sampled_state)
            print(actions_map[action], "<-", sampled_state, end=" ")
        print()
        return sampled_states

    def run_predecessor_chaining(self, folder):

        with open(folder+'model.pkl', 'rb') as f:
            self.model = pickle.load(f)
            self.model.sig_prot_inv = np.eye((self.model.seq_dim))*0.01
            for act in range(4):
                self.model._kdtree_const(act)
            # self.model.sample_single_neighbour = True
            # self.model.sample_weighted_mean = False

        np.random.seed(seed=512)
        self.init_run()

        state = np.array([0.75,0.95])
        actions_map={0:"up",1:"down",2:"right",3:"left"}

        sampled_states = []
        sampled_states.append(state)

        for i in range(24):
            sampled_states_new = []
            # print(sampled_states)
            for state in sampled_states:
                state = np.clip(np.array(state), 0.0, 1.0)
                print(state,"<-",end=" ")
                # state_rep = self._state_representation(state)
                state_rep = state
                sampled_states_new.extend(self.run_predecessor_chaining_recursive(state_rep,actions_map))
            print()
            sampled_states = sampled_states_new

    def temp_function(self, folder):

        np.random.seed(seed=512)
        self.init_run()

        with open(folder+'model.pkl', 'rb') as f:
            self.model = pickle.load(f)
            self.model.offline = True

        np.random.seed(seed=1000)
        self.init_run()
        self.prototypes_x = {}
        self.prototypes_xprev = {}
        sum = 0
        center_state = [0.2,0.3]
        center_rep = self._state_representation(np.array(center_state))
        cov = np.eye(4)*0.01
        for i in range(100):
            original_state = RL_env_message(["sample_random_around",[0.2-0.0625,0.2+0.0625],[0.3-0.0625,0.3+0.0625]])
            state = self._state_representation(original_state)
            for action in range(4):
                kernel = self.model.sample_sprg(state, action)
                # if np.max(kernel) < 0.9:
                print(i,original_state,kernel)
                #     sum += 1
            # print(original_state,state.dot(cov.dot(center_rep.T)))
        print(sum)

        # for i in range(self.model.b):
        #     print("State:", i, self.rep_model_decoder.state_learned(np.array(self.model.prot_array[i,:4])), self.rep_model_decoder.state_learned(np.array(self.model.prot_array[i,5:9])), np.linalg.norm(self.model.prot_array[i,:4]),np.linalg.norm(self.model.prot_array[i,5:9]))


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
    exp_params['agent'] = str(sys.argv[1])
    agent_params["alpha"] = float(sys.argv[2])
    agent_params["num_near"] = int(sys.argv[3])
    agent_params["add_prot_limit"] = float(sys.argv[4])
    agent_params["remDyna_mode"] = int(sys.argv[6])
    agent_params["model_params"]["kscale"] = float(sys.argv[7])
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
print("Env::", exp["environment"], ", Param:", exp["env_params"])
print("Agent::", exp["agent"], ", Param:", exp["agent_params"])
print("Exp param::", exp["exp_params"])

exp = Experiment(env_params, exp_params)
# exp.run_model_learning()
# exp.run_model_decoder(folder='reconstruction/supervisedRep/')

# for training offline model. mode 10 for learning, mode 17 for experiment
if env == "pw" and with_reward:
    adding = "_TCRscale40"
elif env == "pw" and TC:
    adding = "_TC"
else:
    adding = ""

folder = "prototypes/rem-GCov-100p-randomwalk/" + rep_type + "_" + env + "_mode" + str(agent_params["remDyna_mode"]) + "_trainingSetNormCov" + \
         str(agent_params["model_params"]["cov"])+"_addProtLimit"+str(agent_params["add_prot_limit"]) + \
         "_kscale" + str(agent_params["model_params"]["kscale"]) + "/"
if not os.path.isdir(folder):
    os.mkdir(folder)
exp.run_model_learning_random_policy(folder=folder)

# exp.run_model_sampling_random_policy(folder='prototypes/rem-GCov-100p-randomwalk-llm-xy/',folder2='sampling/rem-GCov-100p-randomwalk-llm-xy/')
# exp.run_model_sampling_forward_fixed_data(folder='prototypes/rem-GCov-100p-randomwalk-llm-xy/',folder2='sampling/rem-GCov-100p-randomwalk-llm-xy-forwardsampling/')
# exp.run_model_sampling_forward_random_policy()
# exp.run_model_learning_sarsalambda_policy()
# exp.run_model_knn(folder="prototypes/rem-GCov-100p-randomwalk-flm/", folder2="prototypes-knn/rem-GCov-100p-randomwalk-flm/", data=[[0.05,0.95],[0.5,0.2],[0.75,0.95],[0.5,0.5],[0.2,0.5]])

f = "prototypes/rem-GCov-100p-randomwalk/local_linear_model/pw_mode0_trainingSetNormCov0.025_addProtLimit-0.05_kscale1.0/"
# f = "prototypes/rem-GCov-100p-randomwalk/rem_model/mode10_trainingSetNormCov0.0_addProtLimit-65.0_kscale1e-07/"
# exp.run_model_sampling_fixed_states(data=[[0.05,0.95],[0.5,0.2],[0.75,0.95],[0.5,0.5],[0.2,0.5],
#                                           [0.7, 0.8], [0.75, 0.8], [0.47, 0.8], [0.47, 0.2], [0.7, 0.2],
#                                           [0.65, 0.8], [0.71, 0.93], [0.78, 0.97], [0.5, 1.0]], num_samples=1, folder = f)

# exp.run_model_sampling_fixed_states(data=[[0.05,0.95]], num_samples=1)
# exp.learn_linear_model(folder='prototypes/rem-GCov-100p-randomwalk-llm/', num_samples=10000)
# exp.run_matrix_inverse_calc_transition(folder='feature_model/new_env_model/feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature4_beta1.0_inv_transition', num_samples=100000)
# exp.run_predecessor_chaining(folder="prototypes/rem-GCov-100p-randomwalk/local_linear_model/mode0_trainingSetNormCov0.0_addProtLimit-0.2/")
# exp.temp_function(folder='prototypes/rem-GCov-100p-randomwalk/')
