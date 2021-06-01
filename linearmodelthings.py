import numpy as np
import utils.get_learned_representation as glr
import utils.get_learned_state as gls

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
RLGlue(exp['environment'], exp['agent'])

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
    agent_params["similarity_limit"] = float(sys.argv[8])
    agent_params["always_add_prot"] = int(sys.argv[9])
    agent_params["model_params"]["fix_cov"] = float(sys.argv[10])
    agent_params["alg"] = str(sys.argv[11])
    agent_params["lambda"] = float(sys.argv[12])
    agent_params["momentum"] = float(sys.argv[13])
    agent_params["rms"] = float(sys.argv[14])
    agent_params["opt_mode"] = int(sys.argv[15])

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
        RL_agent_message(["set param", self.agent_params])
        RL_env_message(["set param", self.env_params])
        print("ALL params have been set.")

exp = Experiment(env_params, exp_params)
np.random.seed(seed=1000)
exp.init_run()

model_new_encoder = glr.GetLearnedRep(2, exp.agent_params["nn_nodes"], exp.agent_params["nn_num_feature"], 32 * 4 * 2 * 2, exp.agent_params["nn_lr"], exp.agent_params["nn_lr"], exp.agent_params["nn_weight_decay"], exp.agent_params["nn_dec_nodes"], exp.agent_params["nn_rec_nodes"], exp.agent_params["optimizer"], exp.agent_params["nn_dropout"], exp.agent_params["nn_beta"], exp.agent_params["nn_delta"], exp.agent_params["nn_legal_v"], True, num_tiling=32, num_tile=4, constraint=True, model_path="./feature_model/new_env_model/", file_name="feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature32_beta0.1", default=True)

model_new_decoder = gls.GetLearnedState(2, exp.agent_params["nn_nodes"], exp.agent_params["nn_num_feature"], 32 * 4 * 2 * 2, exp.agent_params["nn_lr"], exp.agent_params["nn_lr"], exp.agent_params["nn_weight_decay"], exp.agent_params["nn_dec_nodes"], exp.agent_params["nn_rec_nodes"], exp.agent_params["optimizer"], exp.agent_params["nn_dropout"], exp.agent_params["nn_beta"], exp.agent_params["nn_delta"], exp.agent_params["nn_legal_v"], True, num_tiling=32, num_tile=4, constraint=True, model_path="./feature_model/new_env_model/", file_name="feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature32_beta0.1_seperateRcvs")

#create w matrix
# w = model_new_encoder.nn.net.de_layers[-1].weight.detach().numpy()
# w_bias = model_new_encoder.nn.net.de_layers[-1].bias.detach().numpy()
# w = np.insert(w,0,w_bias,axis=1)
# np.save('feature_model/new_env_model/feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature32_beta0.1_w.npy',w)

#create wpsuedoInv
w = np.load('feature_model/new_env_model/feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature32_beta0.1_w.npy')
# wpsuedoinv = np.linalg.inv(w.T.dot(w)).dot(w.T)
# wpsuedoinvw = wpsuedoinv.dot(w)
# np.save('feature_model/new_env_model/feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature32_beta0.1_wPsuedoInv.npy',wpsuedoinv)
# np.save('feature_model/new_env_model/feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature32_beta0.1_wPsuedoInvw.npy',wpsuedoinvw)

wpsuedoinv = np.load('feature_model/new_env_model/feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature32_beta0.1_wPsuedoInv.npy')

for i in range(10):

    print("---")

    state_from = RL_env_message(["sample_random"])
    state_from_rep = model_new_encoder.state_representation(np.array(state_from))
    state_from_rep = state_from_rep/np.linalg.norm(state_from_rep)
    state_from_rep_bias = np.insert(state_from_rep,0,1.0)
    result = RL_env_step(3)
    state_to = result['state']
    state_to_rep = model_new_encoder.state_representation(np.array(state_to))
    state_to_rep = state_to_rep/np.linalg.norm(state_to_rep)
    state_to_rep_bias = np.insert(state_to_rep,0,1.0)

    print("Model:", state_from, state_to)

    # state_from_sample = [i + 0.02 for i in state_from]
    low_bd = [i-0.05 if i > 0.0 else 0.0 for i in state_from]
    up_bd = [i+0.05 if i < 1.0 else 1.0 for i in state_from]
    state_from_sample = RL_env_message(["sample_random_around",low_bd,up_bd])
    state_from_sample_rep = model_new_encoder.state_representation(np.array(state_from_sample))
    state_from_sample_rep = state_from_sample_rep/np.linalg.norm(state_from_sample_rep)
    state_from_sample_rep_bias = np.insert(state_from_sample_rep,0,1.0)
    RL_env_message(["set_state",state_from_sample])
    result = RL_env_step(3)
    state_to_sample_real = result['state']

    print("Sample:", state_from_sample, state_to_sample_real)

    print("with inverse:",end=' ')
    temp = w.dot(state_from_sample_rep_bias)
    nexts = wpsuedoinv.dot(temp + w.dot(state_to_rep_bias - state_from_rep_bias))
    nexts = nexts[1:]
    print(model_new_decoder.state_learned(nexts), end=' ')
    nexts = nexts/np.linalg.norm(nexts)
    print(model_new_decoder.state_learned(nexts))

    print("without inverse:",end=' ')
    nexts = state_from_sample_rep + (state_to_rep - state_from_rep)
    print(model_new_decoder.state_learned(nexts), end=' ')
    nexts = nexts/np.linalg.norm(nexts)
    print(model_new_decoder.state_learned(nexts))

    # print("---")
    # diff = wpsuedoinv.dot(w.dot(state_to_rep - state_from_rep))
    # print(model_new_decoder.state_learned(diff))
    # diff = diff/np.linalg.norm(diff)
    # print(model_new_decoder.state_learned(diff))
    #
    # print("---")
    # diff = (state_to_rep - state_from_rep)
    # print(model_new_decoder.state_learned(diff))
    # diff = diff/np.linalg.norm(diff)
    # print(model_new_decoder.state_learned(diff))
