import numpy as np
import time
import sklearn.neighbors as skln
import sklearn.preprocessing as sklp
from utils.get_learned_representation import *

import matplotlib.pylab as plt
import utils.get_learned_state as gls


OLD_REM = 0
CHECK_DIST = 1

SINGLE_REP = 2
SINGLE_REP_CHECKDIST = 9

REPVF_RAWMODEL_CHECKDIST = 3
TCREPVF_RAWMODEL_CHECKDIST = 4

TCREPVF_RAWMODEL = 19

BIASREPVF_RAWMODEL_CHECKDIST = 5
BIASTCREPVF_RAWMODEL_CHECKDIST = 6


BIASTCREPVF_REPMODEL = 7
BIASTCREPVF_REPMODEL_CHECKDIST = 8

SINGLE_NORMREP = 10
SINGLE_NORMREP_FIXCOV = 11

TCREPVF_NORMREPMODEL_FIXCOV = 12 # tile coding rep [0,1] for VF, normalized rep [-1, 1] for model learning, fixed coviance
BIASTCREPVF_NORMREPMODEL_FIXCOV = 13
TCREPVF_NORMREPMODEL = 14
BIASTCREPVF_NORMREPMODEL = 15

NORMREPVF_RAWMODEL = 16

TCRAWVF_NORMREPMODEL = 17
TCRAWVF_NORMREPMODEL_FIXCOV = 18


raw_model_mode_list = [OLD_REM,
                       CHECK_DIST,
                       REPVF_RAWMODEL_CHECKDIST,
                       TCREPVF_RAWMODEL_CHECKDIST,
                       BIASREPVF_RAWMODEL_CHECKDIST,
                       BIASTCREPVF_RAWMODEL_CHECKDIST,
                       NORMREPVF_RAWMODEL,
                       TCREPVF_RAWMODEL]

class Forward_model:
    def __init__(self, state_dim, reward_dim, gamma_dim, kscale, num_action, prototype):
        self.state_dim = state_dim
        self.state_dim_here = state_dim + 1
        self.reward_dim = reward_dim
        self.gamma_dim = gamma_dim
        self.kscale = kscale
        self.num_action = num_action

        self.reset_model()

        self.prototype = prototype


    def reset_model(self):

        self.ssT_inv_tempd1 = np.zeros((self.state_dim_here, 1))
        self.ssT_inv_temp1d = np.zeros((1, self.state_dim_here))
        self.ssT_inv_tempdd = np.zeros((self.state_dim_here, self.state_dim_here))

        self.nexts_w = {}
        self.reward_w = {}
        self.gamma_w = {}
        self.ssT_inv = {}
        self.sT_nexts = {}
        self.sT_reward = {}
        self.sT_gamma = {}
        self.count = {}
        self.diff_sum = {}
        self.diff_mean = {}
        self.diff_cov_first = {}
        self.diff_cov = {}
        self.current_state = {}
        self.next_state = {}
        self.next_state_axis = {}
        self.trained = {}

        for i in range(self.num_action):

            self.nexts_w[i] = np.zeros((self.state_dim_here, self.state_dim))
            self.reward_w[i] = np.zeros((self.state_dim_here, self.reward_dim))
            self.gamma_w[i] = np.zeros((self.state_dim_here, self.gamma_dim))

            self.ssT_inv[i] = np.eye((self.state_dim_here)) * self.kscale

            self.sT_nexts[i] = np.zeros((self.state_dim_here, self.state_dim))
            self.sT_reward[i] = np.zeros((self.state_dim_here, self.reward_dim))
            self.sT_gamma[i] = np.zeros((self.state_dim_here, self.gamma_dim))

            self.count[i] = 0
            self.diff_sum[i] = np.zeros((self.state_dim, 1))
            self.diff_mean[i] = np.zeros((self.state_dim, 1))
            self.diff_cov_first[i] = np.zeros((self.state_dim, self.state_dim))
            self.diff_cov[i] = np.zeros((self.state_dim, self.state_dim))

            self.current_state[i] = None
            self.next_state[i] = None
            self.next_state_axis[i] = None

            self.trained[i] = 0

    def update_model(self, state, state_next, reward, gamma, action):
        state = np.array(state)
        state_next = np.array(state_next)

        if self.current_state[action] is None:
            self.current_state[action] = state[1:]
            self.next_state[action] = state_next
            self.next_state_axis[action] = self.next_state[action][...,np.newaxis]

        # #predict based on diff
        # state[1:] = state[1:] - self.current_state[action]
        # state_next = state_next - self.next_state[action]

        state_temp = state[...,np.newaxis]
        state_next_temp = state_next[...,np.newaxis]

        #Sherman Morrison update of inverse
        #Ainv u
        self.ssT_inv[action].dot(state_temp, out=self.ssT_inv_tempd1)
        #v^T Ainv
        state_temp.T.dot(self.ssT_inv[action],out=self.ssT_inv_temp1d)
        #Ainv u v^T Ainv
        self.ssT_inv_tempd1.dot(self.ssT_inv_temp1d, out=self.ssT_inv_tempdd)
        #1.0 + v^T Ainv u
        denominator = 1.0 + self.ssT_inv_temp1d.dot(state_temp)
        #update
        self.ssT_inv[action] -= ((1.0/denominator)*self.ssT_inv_tempdd)

        #update targets
        self.sT_nexts[action] += state_temp.dot(state_next_temp.T)
        self.sT_reward[action] += (state_temp*reward)
        self.sT_gamma[action] += (state_temp*gamma)

        #update weights
        self.ssT_inv[action].dot(self.sT_nexts[action], out=self.nexts_w[action])
        self.ssT_inv[action].dot(self.sT_reward[action], out=self.reward_w[action])
        self.ssT_inv[action].dot(self.sT_gamma[action], out=self.gamma_w[action])

        self.trained[action] += 1

        self.count[action] += 1
        diff = state_next - state[1:]
        diff = diff[...,np.newaxis]
        self.diff_sum[action] += diff
        self.diff_mean[action] = self.diff_sum[action] / float(self.count[action])
        self.diff_cov_first[action] = ((self.count[action] - 1.0) * self.diff_cov_first[action] + np.outer(diff, diff)) / float(self.count[action])
        self.diff_cov[action] = self.diff_cov_first[action] - np.outer(self.diff_mean[action], self.diff_mean[action])

    def predict(self, state, action):

        # #predict based on diff
        # state[1:] = state[1:] - self.current_state[action]

        state_temp = state[...,np.newaxis]

        nexts = self.nexts_w[action].T.dot(state_temp)
        # #predict based on diff
        # nexts += self.next_state_axis[action]
        reward = self.reward_w[action].flatten().dot(state)
        gamma = self.gamma_w[action].flatten().dot(state)

        self.reward = reward
        self.gamma = gamma

        nexts = np.append(nexts,reward)
        nexts = np.append(nexts,gamma)

        return nexts.flatten()

    def sample(self, state, action):
        nexts = np.zeros((self.state_dim+2, 1))
        nexts[:self.state_dim,0] = np.random.multivariate_normal(state/np.linalg.norm(state), self.diff_cov[action] + np.eye(self.state_dim) * self.kscale, 1)[0]
        nexts[self.state_dim,0] = self.reward[action]
        nexts[self.state_dim,0] = self.gamma[action]

        return nexts

class Reverse_model:
    def __init__(self, state_dim, kscale, num_action, prototype):
        self.state_dim = state_dim
        self.state_dim_here = state_dim + 1
        self.kscale = kscale
        self.num_action = num_action

        self.reset_model()

        self.prototype = prototype

    def reset_model(self):

        self.snextsnextT_inv_tempd1 = np.zeros((self.state_dim_here, 1))
        self.snextsnextT_inv_temp1d = np.zeros((1, self.state_dim_here))
        self.snextsnextT_inv_tempdd = np.zeros((self.state_dim_here, self.state_dim_here))

        self.prevs_w = {}
        self.snextsnextT_inv = {}
        self.snextT_prevs = {}
        self.count = {}
        self.diff_sum = {}
        self.diff_mean = {}
        self.diff_cov_first = {}
        self.diff_cov = {}
        self.current_state = {}
        self.prev_state = {}
        self.prev_state_axis = {}

        self.trained = {}

        for i in range(self.num_action):

            self.prevs_w[i] = np.zeros((self.state_dim_here, self.state_dim))

            self.snextsnextT_inv[i] = np.eye((self.state_dim_here)) * self.kscale
            self.snextT_prevs[i] = np.zeros((self.state_dim_here, self.state_dim))

            self.count[i] = 0
            self.diff_sum[i] = np.zeros((self.state_dim, 1))
            self.diff_mean[i] = np.zeros((self.state_dim, 1))
            self.diff_cov_first[i] = np.zeros((self.state_dim, self.state_dim))
            self.diff_cov[i] = np.zeros((self.state_dim, self.state_dim))

            self.current_state[i] = None
            self.prev_state[i] = None

            self.trained[i] = 0

    def update_model(self, state, state_prev, action):
        state = np.array(state)
        state_prev = np.array(state_prev)

        if self.current_state[action] is None:
            self.current_state[action] = state[1:]
            self.prev_state[action] = state_prev
            self.prev_state_axis[action] = self.prev_state[action][...,np.newaxis]

        # #predict based on diff
        # state[1:] = state[1:] - self.current_state[action]
        # state_prev = state_prev - self.prev_state[action]

        state_temp = state[...,np.newaxis]
        state_prev_temp = state_prev[...,np.newaxis]

        #Sherman Morrison update of inverse
        #Ainv u
        self.snextsnextT_inv[action].dot(state_temp, out=self.snextsnextT_inv_tempd1)
        #v^T Ainv
        state_temp.T.dot(self.snextsnextT_inv[action],out=self.snextsnextT_inv_temp1d)
        #Ainv u v^T Ainv
        self.snextsnextT_inv_tempd1.dot(self.snextsnextT_inv_temp1d, out=self.snextsnextT_inv_tempdd)
        #1.0 + v^T Ainv u
        denominator = 1.0 + self.snextsnextT_inv_temp1d.dot(state_temp)
        #update
        self.snextsnextT_inv[action] -= ((1.0/denominator)*self.snextsnextT_inv_tempdd)

        #update targets
        self.snextT_prevs[action] += state_temp.dot(state_prev_temp.T)

        #update weights
        self.snextsnextT_inv[action].dot(self.snextT_prevs[action], out=self.prevs_w[action])

        self.trained[action] += 1

        self.count[action] += 1
        diff = state_prev - state[1:]
        diff = diff[...,np.newaxis]
        self.diff_sum[action] += diff
        self.diff_mean[action] = self.diff_sum[action] / float(self.count[action])
        self.diff_cov_first[action] = ((self.count[action] - 1.0) * self.diff_cov_first[action] + np.outer(diff, diff)) / float(self.count[action])
        self.diff_cov[action] = self.diff_cov_first[action] - np.outer(self.diff_mean[action], self.diff_mean[action])

    def predict(self, state, action):

        # #predict based on diff
        # state[1:] = state[1:] - self.current_state[action]

        state_temp = state[...,np.newaxis]

        prevs = self.prevs_w[action].T.dot(state_temp)
        # #predict based on diff
        # prevs += self.prev_state_axis[action]

        return prevs.flatten()

    def sample(self, state, action):
        prevs = np.random.multivariate_normal(state/np.linalg.norm(state), self.diff_cov[action] + np.eye(self.state_dim) * self.kscale, 1)[0]

        return prevs

class REM_Model:
    def __init__(self, state_dim, num_near, add_prot_limit, model_params, learning_mode, similarity_limit, norm_diff, num_action=4, rep_model=None):

        print("\nREM model parameter")
        print("state dimension", state_dim)
        print("num near", num_near)
        print("add prot limit", add_prot_limit)
        print("model params", model_params)
        print("learning mode", learning_mode)
        print("similarity limit", similarity_limit)
        print("norm diffrence kernel", norm_diff)
        print("\n")
        self.sample_mode = "fixedCov"
        self.state_dim = state_dim
        self.seq_dim = state_dim * 2 + 3  # s, a, s', r, gamma
        self.action_ind = state_dim
        self.num_action = num_action
        self.index_no_a = [i for i in range(self.seq_dim)]
        self.index_no_a.remove(self.action_ind)

        self.t_forward = 0
        self.sum_array_forward = np.zeros((self.state_dim))
        self.sig_prot_forward = np.zeros((self.state_dim, self.state_dim))
        self.sig_prot_first_forward = np.zeros((self.state_dim, self.state_dim))
        self.sig_prot_inv_forward = np.zeros((self.state_dim, self.state_dim))

        self.t_reverse = 0
        self.sum_array_reverse = np.zeros((self.state_dim))
        self.sig_prot_reverse = np.zeros((self.state_dim, self.state_dim))
        self.sig_prot_first_reverse = np.zeros((self.state_dim, self.state_dim))
        self.sig_prot_inv_reverse = np.zeros((self.state_dim, self.state_dim))

        self.prot_len = 20000
        self.prot_array_forward = np.empty((self.prot_len, self.state_dim))
        self.prot_array_reverse = np.empty((self.prot_len, self.state_dim))
        self.prot_array_model_forward = {}
        self.prot_array_model_reverse = {}

        self.b_forward = 0  # number of prototype in model
        self.b_reverse = 0  # number of prototype in model

        self.b_a = set()
        self.kscale = model_params["kscale"]

        self.add_prot_limit = add_prot_limit
        self.num_near = num_near

        self.running_time = {"cons":0.0,
                             "search":0.0}

        self.learning_mode = learning_mode
        if self.learning_mode == SINGLE_NORMREP_FIXCOV or \
                self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
            self.fix_cov = model_params["fix_cov"]
        else:
            self.fix_cov = 0

        self.similarity_limit = float(similarity_limit)

        self.depth_limit = 10

        self.rep_model = rep_model

        self.s_tree = None
        self.sp_tree = None
        self.s_tree_euclidean = None
        self.sp_tree_euclidean = None
        self.s_tree_fixedCov = None
        self.sp_tree_fixedCov = None

        self.sp_allA_tree = None
        self.sprg_tree = None

        self.sample_single_neighbour = model_params["sample_single_neighbour"]
        self.sample_weighted_mean = model_params["sample_weighted_mean"]
        self.offline = False

        # from utils.recover_state import RecvState
        # self.temp_decoder = RecvState(state_dim, [32, 64, 128, 256, 512], 2, 0.9, 0.9)
        # self.temp_decoder.loading("./feature_model/new_env_model/", "feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature4_beta1.0_seperateRcvs")
        #
        # self.actions_map={0:"up",1:"down",2:"right",3:"left"}

        # self.rep_model_decoder = model_params["rep_model_decoder"]
        self.sampling_limit = model_params["sampling_limit"]
        self.check_sampled = False
        self.cov = model_params["cov"]
        return

    """
    Decide whether a sample should be added into model or not.
    If yes, add sample into model and update variables
    """
    def add2Model(self, last_state, last_action, state, reward, gamma):
        self.update_rem(last_state, last_action, state, reward, gamma)
        return

    """
    Generate (s', r, \gamma) given s and a
    """
    def KDE_sampleSpRG(self, last_state, last_action):
        # for i in range(5):
        #     print("Step:",i)
        sample = self.sample_sprg(last_state, last_action)
        if sample is not None:
            last_state, last_action, state, reward, gamma = sample
            # if np.max(state) > 1 or np.min(state) < -1:
            #     print("Warning: forward prediction out of range")
            #     state = np.clip(state, -1, 1)
            return (last_state, state, reward, gamma, last_action)
        else:
            return None

    """
    Generate predecessor states give current state
    """
    def sampleFromNext_pan(self, state, f, num_action, config=None):
        if config is not None:
            last_action, true_reward, true_gamma = config
            pred_s = self._sample_predecessor_for_action(last_action, state, f)
            # if np.max(pred_s) > 1 or np.min(pred_s) < -1:
            #     print("Warning: backward prediction out of range")
            #     pred_s = np.clip(pred_s, -1, 1)
            if pred_s is not None:
                predecessor = (pred_s, last_action, state, true_reward, true_gamma)
            else:
                predecessor = None
            return predecessor
        else:
            predecessor_list = self._sample_predecessor(num_action, state, f)
            new_pred_list = []
            for p in predecessor_list:
                if p is not None:
                    last_state, last_action, _, reward, gamma = p
                    # if np.max(last_state) > 1 or np.max(last_state) < 1:
                    #     print("Warning: backward prediction out of range")
                    #     last_state = np.clip(last_state, -1, 1)
                    new_p = (last_state, last_action, state, reward, gamma)
                    new_pred_list.append(new_p)
            return new_pred_list


    # Algorithm 7
    def update_rem(self, last_state, last_action, state, reward, gamma):
        """
        If mode == SINGL_REP save representation in model
        Else, save (x, y)
        """
        # if self.learning_mode == SINGLE_REP or \
        #         self.learning_mode == SINGLE_NORMREP or \
        #         self.learning_mode == SINGLE_NORMREP_FIXCOV or \
        #         self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or \
        #         self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV or \
        #         self.learning_mode == BIASTCREPVF_REPMODEL or \
        #         self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST or \
        #         self.learning_mode == SINGLE_REP_CHECKDIST or \
        #         self.learning_mode == TCREPVF_NORMREPMODEL or \
        #         self.learning_mode == BIASTCREPVF_NORMREPMODEL or \
        #         self.learning_mode == TCRAWVF_NORMREPMODEL or \
        #         self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
        #     assert len(last_state) == self.state_dim
        #     assert self.state_dim == 32
        # else:
        #     assert len(last_state) == 2

        seq = self._seq2array(last_state, last_action, state, reward, gamma)

        last_state_copy = np.copy(last_state)
        last_state_copy = np.insert(last_state_copy,0,1.0)
        state_copy = np.copy(state)
        state_copy = np.insert(state_copy,0,1.0)

        # if last_action not in self.b_a:
        #     self._update_prototype(seq)
        #     self.b_a.add(last_action)
        if self.b_forward == 0:
             self._update_prototype(seq)
        else:

            rho, nnear_ind = self._sa_n_near(seq)
            rhor, nnear_ind2 = self._asp_n_near(seq)

            pos = 0
            for i in nnear_ind:
                self.prot_array_model_forward[i].update_model(last_state_copy, state, reward, gamma, last_action)
                # self.prot_array_model_forward[i].update_model(rho[pos]*last_state_copy, rho[pos]*state, rho[pos]*reward, rho[pos]*gamma)
                pos += 1
            pos = 0
            for i in nnear_ind2:
                self.prot_array_model_reverse[i].update_model(state_copy, last_state, last_action)
                # self.prot_array_model_reverse[i].update_model(rhor[pos]*state_copy, rhor[pos]*last_state)
                pos += 1

            self._update_prototype(seq)

        if self.added_prototype_forward:
            self.prot_array_model_forward[self.b_forward-1].update_model(last_state_copy, state, reward, gamma, last_action)
        if self.added_prototype_reverse:
            self.prot_array_model_reverse[self.b_reverse-1].update_model(state_copy, last_state, last_action)

        return

    """
    Update number of samples that the model has seen
    Update the average and covariance matrix of samples
    """
    def _update_t_mu_cov(self, seq, model="forward"):
        if model == "forward":
            self.t_forward += 1
            self.sum_array_forward += seq
            mu_prot = self.sum_array_forward / float(self.t_forward)
            self.sig_prot_first_forward = ((self.t_forward - 1.0) * self.sig_prot_first_forward + np.outer(seq, seq)) / float(self.t_forward)
            self.sig_prot_forward = (self.sig_prot_first_forward - np.outer(mu_prot, mu_prot))
            self.sig_prot_inv_forward = np.linalg.inv(self.sig_prot_forward + (np.eye(len(seq)) * self.kscale))
        elif model == "reverse":
            self.t_reverse += 1
            self.sum_array_reverse += seq
            mu_prot = self.sum_array_reverse / float(self.t_reverse)
            self.sig_prot_first_reverse = ((self.t_reverse - 1.0) * self.sig_prot_first_reverse + np.outer(seq, seq)) / float(self.t_reverse)
            self.sig_prot_reverse = (self.sig_prot_first_reverse - np.outer(mu_prot, mu_prot))
            self.sig_prot_inv_reverse = np.linalg.inv(self.sig_prot_reverse + (np.eye(len(seq)) * self.kscale))
        return

    """
    Construct new kd tree (for offline model only)
    """
    def const_fix_cov_tree(self):

        leaf_size = int(np.ceil(self.b_forward / 2.0))

        X = self.prot_array_forward[:self.b_forward]
        indices = [i for i in range(self.state_dim)]
        _, covmat_inv = self._cal_covmat_inv(indices, sample_mode="fixedCov")
        self.s_tree_fixedCov = skln.BallTree(X, leaf_size=leaf_size,
                                                     metric=skln.DistanceMetric.get_metric('mahalanobis', VI=covmat_inv))

        leaf_size = int(np.ceil(self.b_reverse / 2.0))
        X = self.prot_array_reverse[:self.b_reverse]
        indices = [i for i in range(self.state_dim + 1, self.state_dim * 2 + 1)]
        _, covmat_inv = self._cal_covmat_inv(indices, sample_mode="fixedCov")
        self.sp_tree_fixedCov = skln.BallTree(X, leaf_size=leaf_size,
                                             metric=skln.DistanceMetric.get_metric('mahalanobis', VI=covmat_inv))

    def const_euclidean_tree(self):

        leaf_size = int(np.ceil(self.b_forward / 2.0))

        X = self.prot_array_forward[:self.b_forward]
        indices = [i for i in range(self.state_dim)]
        self.s_tree_euclidean = skln.BallTree(X, leaf_size=leaf_size,
                                              metric=skln.DistanceMetric.get_metric('euclidean'))

        leaf_size = int(np.ceil(self.b_reverse / 2.0))
        X = self.prot_array_reverse[:self.b_reverse]
        indices = [i for i in range(self.state_dim + 1, self.state_dim * 2 + 1)]
        self.sp_tree_euclidean = skln.BallTree(X, leaf_size=leaf_size,
                                               metric=skln.DistanceMetric.get_metric('euclidean'))

    """
    Update kd tree
    """

    def _kdtree_const_forward(self):
        leaf_size = int(np.ceil(self.b_forward / 2.0))
        X = self.prot_array_forward[:self.b_forward]

        indices = [i for i in range(self.state_dim)]
        # Used
        _, covmat_inv = self._cal_covmat_inv(indices, sample_mode="onPolicyCov", covariance_array=self.sig_prot_inv_forward)
        self.s_tree = skln.BallTree(X, leaf_size=leaf_size,
                                    metric=skln.DistanceMetric.get_metric('mahalanobis', VI=covmat_inv))
        self.s_tree_euclidean = skln.BallTree(X, leaf_size=leaf_size,
                                              metric=skln.DistanceMetric.get_metric('euclidean'))

        _, covmat_inv = self._cal_covmat_inv(indices, sample_mode="fixedCov")
        self.s_tree_fixedCov = skln.BallTree(X, leaf_size=leaf_size,
                                             metric=skln.DistanceMetric.get_metric('mahalanobis', VI=covmat_inv))
        return

    def _kdtree_const_reverse(self):

        leaf_size = int(np.ceil(self.b_reverse / 2.0))
        X = self.prot_array_reverse[:self.b_reverse]

        indices = [self.state_dim+1+i for i in range(self.state_dim)]

        # Used
        _, covmat_inv = self._cal_covmat_inv(indices, sample_mode="onPolicyCov", covariance_array=self.sig_prot_inv_reverse)
        self.sp_tree = skln.BallTree(X, leaf_size=leaf_size,
                                     metric=skln.DistanceMetric.get_metric('mahalanobis', VI=covmat_inv))
        self.sp_tree_euclidean = skln.BallTree(X, leaf_size=leaf_size,
                                               metric=skln.DistanceMetric.get_metric('euclidean'))

        _, covmat_inv = self._cal_covmat_inv(indices, sample_mode="fixedCov")
        self.sp_tree_fixedCov = skln.BallTree(X, leaf_size=leaf_size,
                                              metric=skln.DistanceMetric.get_metric('mahalanobis',
                                                                                    VI=covmat_inv))
        return

    # Algorithm 8
    """
    Check whether the new sample satisfies the requirment (distance of knn > threshold)
    If yes, add the sample in to model (new prototype)
    """
    def _update_prototype(self, seq):
        #forward model update
        add_prot = False
        action = seq[self.action_ind]
        seq_state = [i for i in range(self.state_dim)]

        if self.b_forward == 0:
            add_prot = True
        else:
            # same_a = self.same_a_ind_forward[action]
            indices = [i for i in range(self.state_dim)]
            _, covmat_inv = self._cal_covmat_inv(indices, sample_mode=self.sample_mode, covariance_array=self.sig_prot_inv_forward)

            #1-NN
            if self.sample_mode == "fixedCov":
                _, [index] = self.s_tree_fixedCov.query([seq[seq_state]], k=1)
            else:
                _, [index] = self.s_tree.query([seq[seq_state]], k=1)

            prototypes = self.prot_array_forward[index]
            seq_mat_diff = prototypes - np.outer(np.ones(len(index)), seq[seq_state])
            prototypes_transition = -1*np.diag((seq_mat_diff).dot(covmat_inv).dot(seq_mat_diff.T))

            if np.max(prototypes_transition) < self.add_prot_limit:
                add_prot = True

        self.added_prototype_forward = False
        if add_prot:
            self.added_prototype_forward = True
            if self.b_forward < self.prot_len:
                self.prot_array_forward[self.b_forward] = seq[seq_state]
                self.prot_array_model_forward[self.b_forward] = Forward_model(self.state_dim, 1, 1, self.kscale, self.num_action, seq[seq_state])
            else:
                self.prot_array_forward = np.concatenate((self.prot_array_forward, np.array([seq[seq_state]])), axis=0)
                self.prot_array_model_forward[self.b_forward] = Forward_model(self.state_dim, 1, 1, self.kscale, self.num_action, seq[seq_state])

            self.b_forward += 1

            # prototypes covariance
            self._update_t_mu_cov(seq[seq_state], model="forward")
            self._kdtree_const_forward()

        #reverse model update

        add_prot = False
        if self.b_reverse == 0:
            add_prot = True
        else:
            seq_state = [self.state_dim+1+i for i in range(self.state_dim)]

            indices = [self.state_dim + 1 + i for i in range(self.state_dim)]
            _, covmat_inv = self._cal_covmat_inv(indices, sample_mode=self.sample_mode, covariance_array=self.sig_prot_inv_reverse)

            #1-NN
            if self.sample_mode == "fixedCov":
                _, [index] =  self.sp_tree_fixedCov.query([seq[seq_state]], k=1)
            else:
                _, [index] =  self.sp_tree.query([seq[seq_state]], k=1)

            prototypes = self.prot_array_reverse[index]
            seq_mat_diff = prototypes - np.outer(np.ones(len(index)), seq[seq_state])
            prototypes_transition = -1*np.diag((seq_mat_diff).dot(covmat_inv).dot(seq_mat_diff.T))
            if np.max(prototypes_transition) < self.add_prot_limit:
                add_prot = True

        self.added_prototype_reverse = False
        if add_prot:
            self.added_prototype_reverse = True
            if self.b_reverse < self.prot_len:
                self.prot_array_reverse[self.b_reverse] = seq[seq_state]
                self.prot_array_model_reverse[self.b_reverse] = Reverse_model(self.state_dim, self.kscale, self.num_action, seq[seq_state])
            else:
                self.prot_array_reverse = np.concatenate((self.prot_array_reverse, np.array([seq[seq_state]])), axis=0)
                self.prot_array_model_reverse[self.b_reverse] = Reverse_model(self.state_dim, self.kscale, self.num_action, seq[seq_state])

            self.b_reverse += 1

            # prototypes covariance
            self._update_t_mu_cov(seq[seq_state], model="reverse")
            self._kdtree_const_reverse()

        return

    # Algorithm 6
    """
    sample predecessor state given current state and previous action
    """
    def _sample_s(self, last_action, state, seq, depth=0, eval_mode=True):
        if depth >= self.depth_limit:
            return None

        if self.b_reverse == 0:
            return None

        else:
            kernel, nz_ind = self._kernel_asp(seq, sample_mode=self.sample_mode, mode="sample")
            if self.offline:
                if(np.sum(kernel)) == 0:
                    kernel = np.ones(len(kernel)) / float(len(kernel))
                else:
                    kernel /= np.sum(kernel)
            else:
                if eval_mode:
                    rel_pos = np.where(kernel>self.sampling_limit)[0]
                    kernel = kernel[rel_pos]
                    nz_ind = nz_ind[rel_pos]
                    if len(rel_pos) == 0:
                        # print("Backward: No nearest neighbor")
                        return None
                    kernel /= np.sum(kernel)
                else:
                    if(np.sum(kernel)) == 0:
                        kernel = np.ones(len(kernel)) / float(len(kernel))
                    else:
                        kernel /= np.sum(kernel)

            state_copy = np.copy(state)
            state_copy = np.insert(state_copy,0,1.0)
            predictions = {}
            target_mu = np.zeros((self.state_dim))
            pos = 0
            for i in nz_ind:
                predictions[i] = self.prot_array_model_reverse[i].predict(state_copy, last_action)
                target_mu += (kernel[pos]*predictions[i])
                pos += 1

            if self.sample_weighted_mean:
                sampled_s = target_mu
            elif self.sample_single_neighbour:
                s_ind = np.random.choice(nz_ind, 1, p=kernel)[0]
                sampled_s = predictions[s_ind]
                # sampled_s = self.prot_array_model_reverse[s_ind].sample(predictions[i], last_action)
                # sampled_s = self.prot_array_model_reverse[s_ind].sample(target_mu, last_action)
            else:
                s_ind = np.random.choice(nz_ind, 1, p=kernel)[0]
                target_mu = predictions[s_ind]

                cov_mu = np.zeros((self.state_dim,self.state_dim))
                #conditional covariance
                pos = 0
                for i in nz_ind:
                    diff = predictions[i]-target_mu
                    cov_mu += ((kernel[pos]*diff).dot(diff.T))
                    pos += 1

                if self.fix_cov == 0:
                    sampled_s = np.random.multivariate_normal(target_mu, cov_mu + np.eye(cov_mu.shape[0]) * self.kscale)
                else:
                    sampled_s = np.random.multivariate_normal(target_mu, np.eye(cov_mu.shape[0]) * self.fix_cov)

            if self.learning_mode not in raw_model_mode_list:
                sampled_s /= np.linalg.norm(sampled_s)


            """
            Run this block only for mode==CHECK_DIST
            """
            if self.learning_mode == CHECK_DIST or \
                    self.learning_mode == REPVF_RAWMODEL_CHECKDIST or \
                    self.learning_mode == TCREPVF_RAWMODEL_CHECKDIST or \
                    self.learning_mode == BIASREPVF_RAWMODEL_CHECKDIST or \
                    self.learning_mode == BIASTCREPVF_RAWMODEL_CHECKDIST:

                satisfied = self._check_distance(state, sampled_s)
                if not satisfied:
                    sampled_s = self._sample_s(last_action, state, depth=depth + 1)

            elif self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST or \
                    self.learning_mode == SINGLE_REP_CHECKDIST:
                satisfied = self._check_distance(state, sampled_s, rep=True)
                if not satisfied:
                    sampled_s = self._sample_s(last_action, state, depth=depth + 1)

            return sampled_s  # sampled_s_list


    # Algorithm 5
    """
    sample (s', r, \gamma) given current state and action
    """
    def sample_sprg(self, last_state, last_action, depth=0, eval_mode=True, return_knn=False):
        if depth >= self.depth_limit:
            return None

        if self.b_forward == 0:
            return None

        else:
            occupied = [i for i in range(self.state_dim + 1)]
            seq = self._refill_seq(np.concatenate((last_state, np.array([last_action]))), occupied)

            kernel, nz_ind = self._kernel_sa(seq, sample_mode=self.sample_mode, mode="sample")

            if self.offline:
                if(np.sum(kernel)) == 0:
                    kernel = np.ones(len(kernel)) / float(len(kernel))
                else:
                    kernel /= np.sum(kernel)
            else:
                if eval_mode:
                    # if np.max(kernel) < 0.9:
                    #     # print("None similar sprg", kernel)
                    #     # print("None similar sprg")
                    #     return None
                    # # else:
                    # #     print("Sampling sprg", kernel)

                    rel_pos = np.where(kernel>self.sampling_limit)[0]
                    kernel = kernel[rel_pos]
                    nz_ind = nz_ind[rel_pos]

                    """
                    Check
                    """
                    if self.check_sampled:
                        candidates = self.prot_array_forward[nz_ind]
                        # ls = self.rep_model_decoder.state_learned(last_state)
                        ls = last_state
                        print(ls, end=" -> ")
                        for c in candidates:
                            # print(self.rep_model_decoder.state_learned(c) - ls, end=", ")
                            print(c, c - ls, end=", ")
                        print()


                    if len(rel_pos) == 0:
                        # print("Forward: No nearest neighbor")
                        return None

                    kernel /= np.sum(kernel)
                else:
                    if(np.sum(kernel)) == 0:
                        kernel = np.ones(len(kernel)) / float(len(kernel))
                    else:
                        kernel /= np.sum(kernel)

            last_state_copy = np.copy(last_state)
            last_state_copy = np.insert(last_state_copy,0,1.0)
            predictions = {}
            target_mu = np.zeros((self.state_dim+2))
            pos = 0

            for i in nz_ind:
                predictions[i] = self.prot_array_model_forward[i].predict(last_state_copy, last_action)
                # if eval_mode:
                #     print("Successor:", i, kernel[pos], self.temp_decoder.test2(self.prot_array[i][:self.state_dim]), self.temp_decoder.test2(last_state), self.actions_map[last_action], self.temp_decoder.test2(predictions[i][:self.state_dim]),predictions[i][self.state_dim],predictions[i][self.state_dim])
                target_mu += (kernel[pos]*predictions[i])
                pos += 1
            # if eval_mode:
            #     input()

            # s_ind = np.random.choice(nz_ind, 1, p=kernel)[0]
            # s_ind = np.random.choice(nz_ind, 1)[0]
            # target_mu = predictions[s_ind]

            if self.sample_weighted_mean:
                sampled_sprg = target_mu
            elif self.sample_single_neighbour:
                s_ind = np.random.choice(nz_ind, 1, p=kernel)[0]
                sampled_sprg = predictions[s_ind]
                # sampled_sprg = self.prot_array_model_forward[s_ind].sample(predictions[i], last_action)
                # sampled_sprg = self.prot_array_model_forward[s_ind].sample(target_mu, last_action)
            else:
                s_ind = np.random.choice(nz_ind, 1, p=kernel)[0]
                target_mu = predictions[s_ind]
                cov_mu = np.zeros((self.state_dim+2,self.state_dim+2))
                #conditional covariance
                pos = 0
                for i in nz_ind:
                    diff = predictions[i]-target_mu
                    cov_mu += ((kernel[pos]*diff).dot(diff.T))
                    pos += 1

                if self.fix_cov == 0:
                    sampled_sprg = np.random.multivariate_normal(target_mu, cov_mu + np.eye(cov_mu.shape[0]) * self.kscale, 1)[0]
                else:
                    sampled_sprg = np.random.multivariate_normal(target_mu, np.eye(cov_mu.shape[0]) * self.fix_cov, 1)[0]

            state, reward, gamma = sampled_sprg[:self.state_dim], \
                                   sampled_sprg[self.state_dim], \
                                   sampled_sprg[self.state_dim + 1]

            if self.learning_mode not in raw_model_mode_list:
                state /= np.linalg.norm(state)

            """
            Run this block for checking the similarity between sampled s' and s.
            If similarity is larger than threshold, then return
            Otherwise, sample again
            Tha deepest depth is 10 (can try at most 10 times)
            """
            if self.learning_mode == CHECK_DIST or \
                    self.learning_mode == REPVF_RAWMODEL_CHECKDIST or \
                    self.learning_mode == TCREPVF_RAWMODEL_CHECKDIST or \
                    self.learning_mode == BIASREPVF_RAWMODEL_CHECKDIST or \
                    self.learning_mode == BIASTCREPVF_RAWMODEL_CHECKDIST:

                satisfied = self._check_distance(state, last_state)
                if not satisfied:
                    sprg = self.sample_sprg(last_state, last_action, depth=depth, eval_mode=eval_mode)
                    if sprg is not None:
                        (last_state, last_action, state, reward, gamma) = sprg

            elif self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST or \
                    self.learning_mode == SINGLE_REP_CHECKDIST:
                satisfied = self._check_distance(state, last_state, rep=True)
                if not satisfied:
                    sprg = self.sample_sprg(last_state, last_action, depth=depth, eval_mode=eval_mode)
                    if sprg is not None:
                        (last_state, last_action, state, reward, gamma) = sprg

            # if eval_mode:
            #     print("Successor:", self.temp_decoder.test2(last_state), self.actions_map[last_action], self.temp_decoder.test2(state), reward, gamma)
                # succ = self.temp_decoder.test2(state)
                # if (succ[0] > (0.5+0.05) and succ[0] < (0.7-0.05)) and (succ[1] < (0.4-0.05) or succ[1] > (0.6+0.05)):
                #     print("Succ wallllll!!!!", succ)
                #     input()
            if return_knn:
                knn = self.prot_array_forward[nz_ind, :self.state_dim]
                # print(knn)
                return last_state, last_action, state, reward, gamma, knn

            # print("prediction", last_state, last_action, state, reward, gamma)
            return (last_state, last_action, state, reward, gamma)

    # if distance satisfies requirment, return True, otherwise return False
    """
    Given 2 states, get learned representation for them and calculate similarity using \phi(s1)^T * \phi(s2)
    If similarity is larger than threshold then it satisfies requirment
    """
    def _check_distance(self, s1, s2, rep=False):
        if not rep:
            rep1 = self.rep_model.state_representation(s1)
            rep2 = self.rep_model.state_representation(s2)
        else:
            rep1, rep2 = s1, s2
        similarity = np.dot(rep1, rep2)
        if similarity > self.similarity_limit:
            return True
        else:
            return False

    """
    Fill 0 into empty position
    """
    def _refill_seq(self, existing, occupied):
        seq = np.zeros((self.seq_dim))
        seq[occupied] = existing
        return seq

    # Algorithm 4
    """
    given s, for each possible previous action a-, sample predecessor state s-, and it's next state.
    Input all possible actions, state s
    Return (s-, a-, s-', r, \gamma)
    """
    def _sample_predecessor(self, num_action, state, f):
        predecessor_list = []
        occupied = [i for i in range(self.action_ind, self.state_dim * 2 + 1)]
        seq = self._refill_seq(np.concatenate((np.array([0]), state)), occupied)
        # sum_prob_sp = np.sum(self._kernel_sp(seq)[0])
        for last_action in range(num_action):
            seq[self.action_ind] = last_action
            # if (self._prob_a_sp(last_action, state, sum_prob_sp, seq)/sum_prob_sp) != 0:
            #     print(last_action)
            #     exit()
            #     pred_s = self._sample_s(last_action, state, seq)
            #     if pred_s is not None:
            #         predecessor_list.append(self.sample_sprg(pred_s, last_action))
            pred_s = self._sample_s(last_action, state, seq)
            if pred_s is not None:
                tuple = self.sample_sprg(pred_s, last_action, eval_mode=False)
                # tuple = self.sample_sprg(pred_s, last_action)
                if tuple is not None:
                    predecessor_list.append(tuple)
                    # print("backward", tuple)
        return predecessor_list

    def _sample_predecessor_for_action(self, action, state, f):
        predecessor_list = []
        occupied = [i for i in range(self.action_ind, self.state_dim * 2 + 1)]
        seq = self._refill_seq(np.concatenate((np.array([action]), state)), occupied)
        # sum_prob_sp = np.sum(self._kernel_sp(seq)[0])
        # if self._prob_a_sp(action, state, sum_prob_sp, seq) != 0:
        pred_s = self._sample_s(action, state, seq)
        return pred_s

    """
    save sample into array
    """
    def _seq2array(self, last_state, last_action, state, reward, gamma):
        return np.concatenate((last_state, np.array([last_action]), state, np.array([reward]), np.array([gamma])),
                              axis=0)


    """
    Find similarity between 2 samples / prototypes

    If there is no prototype that has same action with the sample, there's no similar prototype

    If the number of prototypes that have same action with the sample is less than k (k nearest neighbor),
    then all these prototypes are in knn.
    """

    """similarity between 2 (s,a) pairs"""
    def _kernel_sa(self, seq, sample_mode="fixedCov", mode="learn"):
        indices = [i for i in range(self.state_dim)]
        action = seq[self.action_ind]

        if self.b_forward <= self.num_near:
            index = []
            for k in range(self.b_forward):
                if mode == "sample":
                    if self.prot_array_model_forward[k].trained[action] > 0:
                        index.append(k)
                else:
                    index.append(k)
            if index == []:
                return np.array([]), np.array([])
            # index = np.array([i for i in range(self.b_forward)])

            seq = np.outer(np.ones(len(index)), seq)
            seq_s = seq[:, indices]
            proto_s = self.prot_array_forward[index]
            diffs = seq_s - proto_s

            if sample_mode=="euclidean":
                k_s = np.exp(-1 * np.linalg.norm(diffs, axis=1))
                # k_s = np.linalg.norm(diffs, axis=1)
                # k_s = (k_s.max() - k_s) / k_s.sum()
            elif sample_mode == "fixedCov":
                _, covmat_inv = self._cal_covmat_inv(indices, sample_mode="fixedCov")
                k_s = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            else:
                _, covmat_inv = self._cal_covmat_inv(indices, sample_mode="onPolicyCov", covariance_array=self.sig_prot_inv_forward)
                k_s = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

            pos = np.array(index)

        else:
            start = time.time()
            if sample_mode=="euclidean":
                _, [index] = self.s_tree_euclidean.query([seq[indices]], k=self.num_near)
            elif sample_mode == "fixedCov":
                _, [index] = self.s_tree_fixedCov.query([seq[indices]], k=self.num_near)
            else:
                _, [index] = self.s_tree.query([seq[indices]], k=self.num_near)
            self.running_time["search"] += time.time() - start

            if mode == "sample":
                index_temp = []
                for k in index:
                    if self.prot_array_model_forward[k].trained[action] > 0:
                        index_temp.append(k)
                index = index_temp
                if index == []:
                    return np.array([]), np.array([])

            seq = np.outer(np.ones(len(index)), seq)
            seq_s = seq[:, indices]
            proto_s = self.prot_array_forward[index]
            diffs = seq_s - proto_s
            if sample_mode=="euclidean":
                k_s = np.exp(-1 * np.linalg.norm(diffs, axis=1))
                # k_s = np.linalg.norm(diffs, axis=1)
                # k_s = (k_s.max() - k_s) / k_s.sum()
            elif sample_mode == "fixedCov":
                _, covmat_inv = self._cal_covmat_inv(indices, sample_mode="fixedCov")
                k_s = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            else:
                _,covmat_inv = self._cal_covmat_inv(indices, sample_mode="onPolicyCov", covariance_array=self.sig_prot_inv_forward)
                k_s = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

            pos = np.array(index)
        return k_s, pos


    """similarity between 2 (a, s') pairs"""
    def _kernel_asp(self, seq, sample_mode="fixedCov", mode="learn"):
        indices = [i for i in range(self.state_dim + 1, self.state_dim * 2 + 1)]
        action = seq[self.action_ind]

        if self.b_reverse < self.num_near:

            index = []
            # for k in range(self.b_forward):
            for k in range(self.b_reverse):
                if mode == "sample":
                    if self.prot_array_model_reverse[k].trained[action] > 0:
                        index.append(k)
                else:
                    index.append(k)
            if index == []:
                return np.array([]), np.array([])
            # index = np.array([i for i in range(self.b_reverse)])

            seq = np.outer(np.ones(len(index)), seq)
            seq_sp = seq[:, indices]
            proto_sp = self.prot_array_reverse[index]
            diffs = seq_sp - proto_sp
            if sample_mode == "euclidean":
                k_sp = np.exp(-1 * np.linalg.norm(diffs, axis=1))
                # k_sp = np.linalg.norm(diffs, axis=1)
                # k_sp = (k_sp.max() - k_sp) / k_sp.sum()
            elif sample_mode == "fixedCov":
                _, covmat_inv = self._cal_covmat_inv(indices, sample_mode="fixedCov")
                k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            else:
                _,covmat_inv = self._cal_covmat_inv(indices, sample_mode="onPolicyCov", covariance_array=self.sig_prot_inv_reverse)
                k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

            # k_sp = np.copy(np.diag(seq_sp.dot(proto_sp.T)))

            pos = np.array(index)

        else:
            start = time.time()
            if sample_mode == "euclidean":
                _, [index] = self.sp_tree_euclidean.query([seq[indices]], k=self.num_near)
            elif sample_mode == "fixedCov":
                _, [index] = self.sp_tree_fixedCov.query([seq[indices]], k=self.num_near)
            else:
                _, [index] = self.sp_tree.query([seq[indices]], k=self.num_near)
            self.running_time["search"] += time.time() - start

            if mode == "sample":
                index_temp = []
                for k in index:
                    if self.prot_array_model_reverse[k].trained[action] > 0:
                        index_temp.append(k)
                index = index_temp
                if index == []:
                    return np.array([]), np.array([])

            seq = np.outer(np.ones(len(index)), seq)
            seq_sp = seq[:, indices]
            proto_sp = self.prot_array_reverse[index]
            diffs = seq_sp - proto_sp

            if sample_mode == "euclidean":
                k_sp = np.exp(-1 * np.linalg.norm(diffs, axis=1))
                # k_sp = np.linalg.norm(diffs, axis=1)
                # k_sp = (k_sp.max() - k_sp) / k_sp.sum()
            elif sample_mode == "fixedCov":
                _, covmat_inv = self._cal_covmat_inv(indices, sample_mode="fixedCov")
                k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            else:
                _,covmat_inv = self._cal_covmat_inv(indices, sample_mode="onPolicyCov", covariance_array=self.sig_prot_inv_reverse)
                k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

            pos = np.array(index)

        return k_sp, pos

    """similarity between 2 (s', r, g) pairs"""
    # def _kernel_sprg(self, seq, knn_ind = None, sample_mode=False):
    #     indices = [i for i in range(self.state_dim + 1, self.seq_dim)]
    #
    #     # if knn_ind is not None:
    #     seq = np.outer(np.ones(len(knn_ind)), seq)
    #     seq_sprg = seq[:, indices]
    #     protos_sprg = self.prot_array[knn_ind][:, indices]
    #
    #     diffs = seq_sprg - protos_sprg
    #     if sample_mode:
    #         k_sprg = np.exp(-1 * np.linalg.norm(diffs, axis=1))
    #     else:
    #         _, covmat_inv = self._cal_covmat_inv(indices)
    #         k_sprg = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
    #
    #     # k_sprg = np.copy(np.diag(seq_sprg.dot(proto_sprg.T)))
    #
    #     # if(np.sum(k_sprg)) == 0:
    #         # k_sprg = np.ones(len(k_sprg)) / float(len(k_sprg))
    #     # else:
    #     #     k_sprg /= np.sum(k_sprg)
    #
    #     return k_sprg, knn_ind

    """similarity between 2 s'"""
    # def _kernel_sp(self, seq, sample_mode=False):
    #     indices = [i for i in range(self.state_dim + 1, self.state_dim * 2 + 1)]
    #     if self.b <= self.num_near:
    #         seq = np.outer(np.ones(self.b), seq)
    #         seq_sp = seq[:, indices]
    #         protos_sp = self.prot_array[:self.b, indices]
    #         diffs = seq_sp - protos_sp
    #
    #         if sample_mode:
    #             k_sp = np.exp(-1 * np.linalg.norm(diffs, axis=1))
    #         else:
    #             _, covmat_inv = self._cal_covmat_inv(indices)
    #             k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
    #
    #         pos = np.array([i for i in range(self.b)])
    #
    #     else:
    #         seq = np.outer(np.ones(self.num_near), seq)
    #         seq_sp = seq[:, indices]
    #         _, covmat_inv = self._cal_covmat_inv(indices)
    #
    #         start = time.time()
    #         _, [index] = self.sp_allA_tree.query([seq_sp[0]], k=self.num_near)
    #         self.running_time["search"] += time.time() - start
    #
    #         protos_sp = self.prot_array[index][:, indices]
    #         diffs = seq_sp - protos_sp
    #         k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
    #
    #         pos = index
    #
    #     return k_sp, index

    """
    Get covariance matrix and its inverse
    """

    def _cal_covmat_inv(self, indices=[], sample_mode="fixedCov", covariance_array=None):
        if sample_mode == "onPolicyCov":
            return None, covariance_array
        elif sample_mode == "fixedCov":
            covmat_inv = np.eye(len(indices))*(1.0/self.cov)
            return None, covmat_inv

    """
    Find k nearest neighbors of (s,a)
    """
    def _sa_n_near(self, seq, sample_mode="fixedCov"):
        kernel, nz_ind = self._kernel_sa(seq, sample_mode=sample_mode)
        return kernel, nz_ind

    """
    Find k nearest neighbors of (a, s')
    """
    def _asp_n_near(self, seq, sample_mode="fixedCov"):
        kernel, nz_ind = self._kernel_asp(seq, sample_mode=sample_mode)
        return kernel, nz_ind

    """
    Calculate probability P(a|s')
    """
    def _prob_a_sp(self, last_action, state, sum_prob_sp, seq):
        sum_prob_asp = np.sum(self._kernel_asp(seq)[0])
        # if sum_prob_sp == 0:
        #     return 1
        # else:
        #     return sum_prob_asp / sum_prob_sp
        return sum_prob_asp
    """
    Get all prototypes currently in the model
    """
    # def get_protos(self):
    #     return self.prot_array[:self.b]

    """
    Get number of prototypes
    """
    def get_len_protos(self):
        return [self.b_forward, self.b_reverse]

    """
    Check running time
    Not used for learning process
    """
    def _check_time(self):
        res = self.running_time.copy()
        for key in self.running_time.keys():
            self.running_time[key] = 0.0
        return res

    def get_added_prototype_forward(self):
        return self.added_prototype_forward
    def get_added_prototype_reverse(self):
        return self.added_prototype_reverse

    def check_model_for_action(self):
        check_trained = np.zeros((len(self.prot_array_model_forward.keys()) * self.num_action, self.state_dim+2))
        for k in self.prot_array_model_forward.keys():
            for action in range(self.num_action):
                if self.prot_array_model_forward[k].trained[action] == 0:
                    print(self.prot_array_model_forward[k].prototype, action, self.prot_array_model_forward[k].trained[action])
                check_trained[k*self.num_action + action, :self.state_dim] = self.prot_array_model_forward[k].prototype
                check_trained[k*self.num_action + action, self.state_dim] = action
                check_trained[k*self.num_action + action, self.state_dim+1] = self.prot_array_model_forward[k].trained[action]
        print("maximum training time", np.max(check_trained[:, self.state_dim+1]))
        print("minimum training time", np.min(check_trained[:, self.state_dim+1]))
        print("averaged", np.mean(check_trained[:, self.state_dim+1]))
        return check_trained
