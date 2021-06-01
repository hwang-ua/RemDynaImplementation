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

rep_checkDist_mode_list = [TCRAWVF_NORMREPMODEL]

class Forward_model:
    def __init__(self, state_dim, reward_dim, gamma_dim, kscale):
        self.state_dim = state_dim
        self.state_dim_here = state_dim + 1
        self.reward_dim = reward_dim
        self.gamma_dim = gamma_dim
        self.kscale = kscale
        # self.kscale = 1e-3
        self.kscale_inv = 1.0

        self.reset_model()


    def reset_model(self):

        # self.nexts_w = np.zeros((self.state_dim_here, self.state_dim))
        self.reward_w = np.zeros((self.state_dim_here, self.reward_dim))
        self.gamma_w = np.zeros((self.state_dim_here, self.gamma_dim))

        self.ssT_inv = np.eye((self.state_dim_here)) * self.kscale_inv

        self.ssT_inv_tempd1 = np.zeros((self.state_dim_here, 1))
        self.ssT_inv_temp1d = np.zeros((1, self.state_dim_here))
        self.ssT_inv_tempdd = np.zeros((self.state_dim_here, self.state_dim_here))

        # self.sT_nexts = np.zeros((self.state_dim_here, self.state_dim))
        self.sT_reward = np.zeros((self.state_dim_here, self.reward_dim))
        self.sT_gamma = np.zeros((self.state_dim_here, self.gamma_dim))

        self.state_diff = None

        self.count = 0
        self.diff_sum = np.zeros((self.state_dim, 1))
        self.diff_mean = np.zeros((self.state_dim, 1))
        self.diff_cov_first = np.zeros((self.state_dim, self.state_dim))
        self.diff_cov = np.zeros((self.state_dim, self.state_dim))

        # from utils.recover_state import RecvState
        # self.temp_decoder = RecvState(self.state_dim, [32, 64, 128, 256, 512], 2, 0.9, 0.9)
        # self.temp_decoder.loading("./feature_model/new_env_model/", "feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature4_beta1.0_seperateRcvs")

    # def update_model(self, state, state_next, reward, gamma):
    def update_model(self, state, state_next, reward, gamma, kernel=None):
        state = np.array(state)
        state_next = np.array(state_next)

        state_temp = state[...,np.newaxis]
        state_next_temp = state_next[...,np.newaxis]

        #Sherman Morrison update of inverse
        #Ainv u
        self.ssT_inv.dot(state_temp, out=self.ssT_inv_tempd1)
        #v^T Ainv
        state_temp.T.dot(self.ssT_inv,out=self.ssT_inv_temp1d)
        #Ainv u v^T Ainv
        self.ssT_inv_tempd1.dot(self.ssT_inv_temp1d, out=self.ssT_inv_tempdd)
        #1.0 + v^T Ainv u
        denominator = 1.0 + self.ssT_inv_temp1d.dot(state_temp)
        #update
        self.ssT_inv -= ((1.0/denominator)*self.ssT_inv_tempdd)

        #CF inverse
        # self.ssT_inv = np.linalg.inv(state_temp.dot(state_temp.T) + np.eye(self.state_dim_here))

        #update targets
        # self.sT_nexts += state_temp.dot(state_next_temp.T)
        self.sT_reward += (state_temp*reward)
        self.sT_gamma += (state_temp*gamma)

        #update weights
        # self.ssT_inv.dot(self.sT_nexts, out=self.nexts_w)
        self.ssT_inv.dot(self.sT_reward, out=self.reward_w)
        self.ssT_inv.dot(self.sT_gamma, out=self.gamma_w)

        if self.state_diff is None:
            self.state_diff = state_next - state[1:]

            # self.state_diff = np.insert(self.state_diff,0,0.0)
            # self.w = np.load('feature_model/new_env_model/feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature32_beta0.1_w.npy')
            # self.wpsuedoinv = np.load('feature_model/new_env_model/feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature32_beta0.1_wPsuedoInv.npy')

        if kernel == None:
            self.count += 1
            diff = state_next - state[1:]
            diff = diff[...,np.newaxis]
            self.diff_sum += diff
            self.diff_mean = self.diff_sum / float(self.count)
            self.diff_cov_first = ((self.count - 1.0) * self.diff_cov_first + np.outer(diff, diff)) / float(self.count)
            self.diff_cov = self.diff_cov_first - np.outer(self.diff_mean, self.diff_mean)
        else:
            self.count += kernel
            diff = state_next - state[1:]
            diff = diff[...,np.newaxis]

            # self.diff_sum += diff
            # self.diff_mean = self.diff_sum / float(self.count)
            # self.diff_cov_first = ((self.count - 1.0) * self.diff_cov_first + np.outer(kernel*diff, diff)) / float(self.count)

            # self.diff_sum = (self.count - kernel) * self.diff_sum + (kernel*diff)
            # self.diff_mean = self.diff_sum / float(self.count)
            # self.diff_cov_first = ((self.count - kernel) * self.diff_cov_first + np.outer(kernel*diff, diff)) / float(self.count)
            # self.diff_cov = self.diff_cov_first - np.outer(self.diff_mean, self.diff_mean)

            self.diff_sum += (kernel*diff)
            self.diff_mean = self.diff_sum
            self.diff_cov_first += (kernel*np.outer(diff, diff))
            self.diff_cov = self.diff_cov_first - np.outer(self.diff_mean, self.diff_mean)


    def sample(self, state, weighted_mu, target_mu, mean, cov):
        nexts = np.zeros((self.state_dim+2, 1))

        # nexts1 = (state + self.state_diff).flatten()
        nexts1 = (state + self.diff_mean.flatten()).flatten()
        # nexts1 = (weighted_mu[:self.state_dim] + self.diff_mean.flatten()).flatten()

        state_copy = np.insert(state,0,1.0)

        # if (nexts1[:self.state_dim]).dot(mean[:self.state_dim]) < 0.9:
        if (nexts1[:self.state_dim]/np.linalg.norm(nexts1[:self.state_dim])).dot(mean[:self.state_dim]/np.linalg.norm(mean[:self.state_dim])) < 0.9:
        # if (nexts1[:self.state_dim]).dot(mean[:self.state_dim]) > 1.0:
        # if (weighted_mu[:self.state_dim]).dot(mean[:self.state_dim]) < 0.9:
            # print("REM!",(nexts1[:self.state_dim]).dot(mean[:self.state_dim]), self.temp_decoder.test2(state))
            # input()
            nexts[:self.state_dim,0] = np.random.multivariate_normal(target_mu[:self.state_dim], cov[:self.state_dim,:self.state_dim] + np.eye(self.state_dim) * self.kscale, 1)[0]
            nexts[self.state_dim,0] = np.random.normal(target_mu[self.state_dim], cov[self.state_dim,self.state_dim] + self.kscale, 1)[0]
            nexts[self.state_dim+1,0] = np.random.normal(target_mu[self.state_dim+1], cov[self.state_dim+1,self.state_dim+1] + self.kscale, 1)[0]
        else:
            # print("FLM!",(nexts1[:self.state_dim]).dot(mean[:self.state_dim]), self.temp_decoder.test2(state))
            # input()
            nexts[:self.state_dim,0] = np.random.multivariate_normal(nexts1/np.linalg.norm(nexts1), self.diff_cov + np.eye(self.state_dim) * self.kscale, 1)[0]
            # nexts[:self.state_dim,0] = np.random.multivariate_normal(weighted_mu[:self.state_dim], self.diff_cov + np.eye(self.state_dim) * self.kscale, 1)[0]
            # nexts[:self.state_dim,0] = weighted_mu[:self.state_dim]
            # nexts[:self.state_dim,0] = nexts1
            nexts[self.state_dim,0] = self.reward_w.flatten().dot(state_copy)
            nexts[self.state_dim+1,0] = self.gamma_w.flatten().dot(state_copy)

        # print("Inner:",nexts[:self.state_dim,0].dot(mean[:self.state_dim,0]))

        return nexts.flatten()

    def predict(self, state):
        state_temp = state[...,np.newaxis]

        nexts = (state[1:] + self.state_diff).flatten()
        reward = self.reward_w.flatten().dot(state)
        gamma = self.gamma_w.flatten().dot(state)

        nexts = np.append(nexts,reward)
        nexts = np.append(nexts,gamma)

        return nexts.flatten()



class Reverse_model:
    def __init__(self, state_dim, kscale):
        self.state_dim = state_dim
        self.state_dim_here = state_dim + 1
        self.kscale = kscale
        # self.kscale = 1e-3
        self.kscale_inv = 1.0

        self.reset_model()

    def reset_model(self):

        self.state_diff = None

        self.count = 0
        self.diff_sum = np.zeros((self.state_dim, 1))
        self.diff_mean = np.zeros((self.state_dim, 1))
        self.diff_cov_first = np.zeros((self.state_dim, self.state_dim))
        self.diff_cov = np.zeros((self.state_dim, self.state_dim))

        # from utils.recover_state import RecvState
        # self.temp_decoder = RecvState(self.state_dim, [32, 64, 128, 256, 512], 2, 0.9, 0.9)
        # self.temp_decoder.loading("./feature_model/new_env_model/", "feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature4_beta1.0_seperateRcvs")


    # def update_model(self, state, state_prev):
    def update_model(self, state, state_prev, kernel=None):
        state = np.array(state)
        state_prev = np.array(state_prev)

        state_temp = state[...,np.newaxis]
        state_prev_temp = state_prev[...,np.newaxis]

        if self.state_diff is None:
            self.state_diff = state_prev - state[1:]

            # self.state_diff = np.insert(self.state_diff,0,0.0)
            # self.w = np.load('feature_model/new_env_model/feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature32_beta0.1_w.npy')
            # self.wpsuedoinv = np.load('feature_model/new_env_model/feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature32_beta0.1_wPsuedoInv.npy')

        if kernel == None:
            self.count += 1
            diff = state_prev - state[1:]
            diff = diff[...,np.newaxis]
            self.diff_sum += diff
            self.diff_mean = self.diff_sum / float(self.count)
            self.diff_cov_first = ((self.count - 1.0) * self.diff_cov_first + np.outer(diff, diff)) / float(self.count)
            self.diff_cov = self.diff_cov_first - np.outer(self.diff_mean, self.diff_mean)
        else:
            self.count += kernel
            diff = state_prev - state[1:]
            diff = diff[...,np.newaxis]

            # self.diff_sum += (kernel*diff)
            # self.diff_mean = self.diff_sum / float(self.count)
            # self.diff_cov_first = ((self.count - 1.0) * self.diff_cov_first + np.outer(kernel*diff, diff)) / float(self.count)

            # self.diff_sum = (self.count - kernel) * self.diff_sum + (kernel*diff)
            # self.diff_mean = self.diff_sum / float(self.count)
            # self.diff_cov_first = ((self.count - kernel) * self.diff_cov_first + np.outer(kernel*diff, diff)) / float(self.count)
            # self.diff_cov = self.diff_cov_first - np.outer(self.diff_mean, self.diff_mean)

            self.diff_sum += (kernel*diff)
            self.diff_mean = self.diff_sum
            self.diff_cov_first += np.outer(kernel*diff, diff)
            self.diff_cov = self.diff_cov_first - np.outer(self.diff_mean, self.diff_mean)

    def sample(self, state, weighted_mu, target_mu, mean, cov):

        # prevs = (state + self.state_diff).flatten()
        prevs = (state + self.diff_mean.flatten()).flatten()
        # prevs = (weighted_mu + self.diff_mean.flatten()).flatten()

        # if (prevs).dot(mean) < 0.9:
        if (prevs/np.linalg.norm(prevs)).dot(mean/np.linalg.norm(mean)) < 0.9:
        # if (prevs).dot(mean) > 1.0:
        # if (weighted_mu).dot(mean) < 0.9:
            # print("REM!",(weighted_mu).dot(mean),self.temp_decoder.test2(state))
            # input()
            prevs = np.random.multivariate_normal(target_mu, cov + np.eye(self.state_dim) * self.kscale, 1)[0]
            # prevs = np.random.multivariate_normal(mean, cov + np.eye(self.state_dim) * self.kscale, 1)[0]
        else:
            # print("FLM!",(weighted_mu).dot(mean),self.temp_decoder.test2(state))
            # input()
            prevs = np.random.multivariate_normal(prevs/np.linalg.norm(prevs), self.diff_cov + np.eye(self.state_dim) * self.kscale, 1)[0]
            # prevs = np.random.multivariate_normal(weighted_mu, self.diff_cov + np.eye(self.state_dim) * self.kscale, 1)[0]
            # prevs = weighted_mu
            # prevs = prevs

        # print("Inner:",prevs.dot(mean))

        return prevs


    def predict(self, state):
        state_temp = state[...,np.newaxis]

        prevs = (state[1:] + self.state_diff).flatten()

        # prevs = self.wpsuedoinv.dot(self.w.dot(state) + self.w.dot(self.state_diff))[1:].flatten()

        return prevs.flatten()

class REM_Model:
    def __init__(self, state_dim, num_near, add_prot_limit, model_params, learning_mode, similarity_limit, norm_diff,
                 rep_model=None):

        print("\nREM model parameter")
        print("state dimension", state_dim)
        print("num near", num_near)
        print("add prot limit", add_prot_limit)
        print("model params", model_params)
        print("learning mode", learning_mode)
        print("similarity limit", similarity_limit)
        print("norm diffrence kernel", norm_diff)
        print("\n")
        self.state_dim = state_dim
        self.seq_dim = state_dim * 2 + 3  # s, a, s', r, gamma
        self.action_ind = state_dim
        self.index_no_a = [i for i in range(self.seq_dim)]
        self.index_no_a.remove(self.action_ind)

        # the action is not included
        self.sum_array = np.zeros((self.seq_dim))
        self.sig_prot = np.zeros((self.seq_dim, self.seq_dim))
        self.sig_prot_first = np.zeros((self.seq_dim, self.seq_dim))
        self.sig_prot_inv = np.zeros((self.seq_dim, self.seq_dim))

        self.prot_len = 20000
        self.c = np.ones((self.prot_len))
        self.cr = np.ones((self.prot_len))
        self.prot_array = np.empty((self.prot_len, self.seq_dim))
        self.prot_array_model_forward = {}
        self.prot_array_model_reverse = {}

        self.b = 0  # number of prototype in model
        self.b_a = set()
        self.t = 0  # number of sequence sent to model (added & not added)
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

        self.s_tree = {}
        self.sp_tree = {}
        self.ssprg_tree = {}

        self.sp_allA_tree = None
        self.sprg_tree = None

        self.same_a_ind = {}

        # from utils.recover_state import RecvState
        # self.temp_decoder = RecvState(state_dim, [32, 64, 128, 256, 512], 2, 0.9, 0.9)
        # self.temp_decoder.loading("./feature_model/new_env_model/", "feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature4_beta1.0_seperateRcvs")
        #
        # self.actions_map={0:"up",1:"down",2:"right",3:"left"}

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
            return (last_state, state, reward, gamma, last_action)
        else:
            return None

    """
    Generate predecessor states give current state
    """
    def sampleFromNext_pan(self, state, f, num_action):
        predecessor_list = self._sample_predecessor(num_action, state, f)
        new_pred_list = []
        for p in predecessor_list:
            if p is not None:
                last_state, last_action, state, reward, gamma = p
                new_p = (last_state, last_action, reward, state, gamma)
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
        #     assert self.state_dim == 4
        # else:
        #     assert len(last_state) == 2

        seq = self._seq2array(last_state, last_action, state, reward, gamma)
        self._update_t_mu_cov(seq)

        last_state_copy = np.copy(last_state)
        last_state_copy = np.insert(last_state_copy,0,1.0)
        state_copy = np.copy(state)
        state_copy = np.insert(state_copy,0,1.0)

        if last_action not in self.b_a:
            self._update_prototype(seq)
            self.b_a.add(last_action)
        else:
            rho, nnear_ind = self._sa_n_near(seq)
            rhor, nnear_ind2 = self._asp_n_near(seq)

            # rel_pos = np.where(rho>0.5)[0]
            # rho = np.array(rho[rel_pos])
            # nnear_ind = np.array(nnear_ind[rel_pos])
            #
            # rel_pos = np.where(rhor>0.5)[0]
            # rhor = np.array(rhor[rel_pos])
            # nnear_ind2 = np.array(nnear_ind2[rel_pos])

            self.c[nnear_ind] = np.multiply((1.0 - rho), self.c[nnear_ind]) + \
                                np.multiply(rho, self._kernel_sprg(seq, knn_ind = nnear_ind)[0])
            self.cr[nnear_ind2] = np.multiply((1.0 - rhor), self.cr[nnear_ind2]) + \
                                 np.multiply(rhor, self._kernel_sa(seq, knn_ind = nnear_ind2)[0])

            pos = 0
            for i in nnear_ind:
                    self.prot_array_model_forward[i].update_model(last_state_copy, state, reward, gamma)
                    # self.prot_array_model_forward[i].update_model(last_state_copy, state, reward, gamma, rho[pos])
                    pos += 1
            pos = 0
            for i in nnear_ind2:
                self.prot_array_model_reverse[i].update_model(state_copy, last_state)
                # self.prot_array_model_reverse[nnear_ind2[i]].update_model(state_copy, last_state, rhor[pos])
                pos += 1

            self._update_prototype(seq)

        if self.added_prototype:
            self.prot_array_model_forward[self.b-1].update_model(last_state_copy, state, reward, gamma)
            self.prot_array_model_reverse[self.b-1].update_model(state_copy, last_state)
            # self.prot_array_model_forward[self.b-1].update_model(last_state_copy, state, reward, gamma, 1.0)
            # self.prot_array_model_reverse[self.b-1].update_model(state_copy, last_state, 1.0)

        return

    """
    Update number of samples that the model has seen
    Update the average and covariance matrix of samples
    """
    def _update_t_mu_cov(self, seq):
        self.t += 1
        self.sum_array += seq
        mu_prot = self.sum_array / float(self.t)
        self.sig_prot_first = ((self.t - 1.0) * self.sig_prot_first + np.outer(seq, seq)) / float(self.t)
        self.sig_prot = (self.sig_prot_first - np.outer(mu_prot, mu_prot))
        self.sig_prot_inv = np.linalg.inv(self.sig_prot + (np.eye(len(seq)) * self.kscale))
        return

    """
    Update kd tree
    """
    def _kdtree_const(self, action):
        X_ind = self.same_a_ind[action]
        leaf_size = int(np.ceil(len(X_ind) / 2.0))
        X = self.prot_array[X_ind]

        start_time = time.time()

        indices = [i for i in range(self.state_dim)]
        _, covmat_inv = self._cal_covmat_inv(indices)
        # covmat_inv = 1*np.eye(len(indices))
        # Used
        self.s_tree[action] = skln.BallTree(X[:, indices], leaf_size=leaf_size,
                                                   metric=skln.DistanceMetric.get_metric('mahalanobis', VI=covmat_inv))

        indices = [i for i in range(self.state_dim + 1, self.state_dim * 2 + 1)]
        _, covmat_inv = self._cal_covmat_inv(indices)
        # covmat_inv = 1*np.eye(len(indices))
        # Used
        self.sp_tree[action] = skln.BallTree(X[:, indices], leaf_size=leaf_size,
                                                   metric=skln.DistanceMetric.get_metric('mahalanobis', VI=covmat_inv))

        indices = self.index_no_a
        _, covmat_inv = self._cal_covmat_inv(indices)
        # Used
        self.ssprg_tree[action] = skln.BallTree(X[:, indices], leaf_size=leaf_size,
                                             metric=skln.DistanceMetric.get_metric('mahalanobis', VI=covmat_inv))

        # indices = [i for i in range(self.state_dim + 1, self.seq_dim)]
        # _, covmat_inv = self._cal_covmat_inv(indices)
        # self.sprg_tree = skln.BallTree(self.prot_array[:self.b, indices], leaf_size=leaf_size,
        #                                metric=skln.DistanceMetric.get_metric('mahalanobis', VI=covmat_inv))

        # indices = [i for i in range(self.state_dim + 1, self.state_dim * 2 + 1)]
        # _, covmat_inv = self._cal_covmat_inv(indices)
        # # Used
        # self.sp_allA_tree = skln.BallTree(self.prot_array[:self.b, indices], leaf_size=leaf_size,
        #                                metric=skln.DistanceMetric.get_metric('mahalanobis', VI=covmat_inv))

        self.running_time["cons"] += time.time() - start_time
        return

    # Algorithm 8
    """
    Check whether the new sample satisfies the requirment (distance of knn > threshold)
    If yes, add the sample in to model (new prototype)
    """
    def _update_prototype(self, seq):
        add_prot = False
        action = seq[self.action_ind]

        seq_noaction = seq[self.index_no_a]

        if action not in self.same_a_ind.keys():
            add_prot = True
        else:
            same_a = self.same_a_ind[action]
            _, covmat_inv = self._cal_covmat_inv(self.index_no_a)

            # if len(same_a) >= self.num_near:
            #     _, [index] = self.ssprg_tree[action].query([seq[self.index_no_a]], k=self.num_near)
            # else:
            #     _, [index] = self.ssprg_tree[action].query([seq[self.index_no_a]], k=len(same_a))

            #1-NN
            _, [index] = self.ssprg_tree[action].query([seq[self.index_no_a]], k=1)

            prototypes = self.prot_array[np.array(same_a)][index][:,self.index_no_a]
            seq_mat_diff = prototypes - np.outer(np.ones(len(index)), seq[self.index_no_a])
            prototypes_transition = -1*np.diag((seq_mat_diff).dot(covmat_inv).dot(seq_mat_diff.T))

            if np.max(prototypes_transition) < self.add_prot_limit:
                add_prot = True

        self.added_prototype = False
        if add_prot:
            self.added_prototype = True
            if self.b < self.prot_len:
                self.prot_array[self.b] = seq
                self.prot_array_model_forward[self.b] = Forward_model(self.state_dim, 1, 1, self.kscale)
                self.prot_array_model_reverse[self.b] = Reverse_model(self.state_dim, self.kscale)
            else:
                self.prot_array = np.concatenate((self.prot_array, np.array([seq])), axis=0)
                self.prot_array_model_forward[self.b] = Forward_model(self.state_dim, 1, 1, self.kscale)
                self.prot_array_model_reverse[self.b] = Reverse_model(self.state_dim, self.kscale)
                self.c = np.concatenate((self.c, np.ones(1)), axis=0)
                self.cr = np.concatenate((self.cr, np.ones(1)), axis=0)

            # print("Adding:",self.temp_decoder.test2(self.prot_array[self.b,:32]),self.temp_decoder.test2(self.prot_array[self.b,33:65]))

            if action in self.same_a_ind.keys():
                self.same_a_ind[action].append(self.b)
            else:
                self.same_a_ind[action] = [self.b]

            self.b += 1

            # self._update_t_mu_cov(seq)
            # self._kdtree_const(action)

        self._kdtree_const(action)
        return

    # Algorithm 5
    """
    sample (s', r, \gamma) given current state and action
    """
    def sample_sprg(self, last_state, last_action, depth=0, eval_mode=True):
        if depth >= self.depth_limit:
            return None

        if self.b == 0:
            return None

        else:
            occupied = [i for i in range(self.state_dim + 1)]
            seq = self._refill_seq(np.concatenate((last_state, np.array([last_action]))), occupied)

            kernel, nz_ind = self._kernel_sa(seq)

            if eval_mode:
                if np.max(kernel) < 0.9:
                    # print("None similar sprg", kernel)
                    # print("None similar sprg")
                    return None
                # else:
                #     print("Sampling sprg", kernel)

                rel_pos = np.where(kernel>0.5)
                kernel = kernel[rel_pos]
                nz_ind = nz_ind[rel_pos]

            betas = np.multiply(self.c[nz_ind], kernel)
            n_sa = np.sum(betas)
            if n_sa != 0 and not np.isnan(n_sa):
                betas /= float(n_sa)

                #original
                mu = np.average(self.prot_array[nz_ind, self.state_dim+1:],weights=np.transpose(betas),axis=0)
                # mu[:self.state_dim] = mu[:self.state_dim] /np.linalg.norm(mu[:self.state_dim])
                diff = self.prot_array[nz_ind, self.state_dim + 1: ] - mu
                cov = (diff * betas[:, None]).T.dot(diff)

                s_ind = np.random.choice(nz_ind, 1, p=betas)[0]

                # sampled_sprg = np.random.multivariate_normal(target_mu, cov + np.eye(cov.shape[0]) * self.kscale, 1)[0]

                target_mu = self.prot_array[s_ind, self.state_dim + 1: ]

                last_state_copy = np.copy(last_state)
                last_state_copy = np.insert(last_state_copy,0,1.0)
                weighted_mu = np.zeros((self.state_dim+2))
                pos = 0
                for i in nz_ind:
                    weighted_mu += (betas[pos]*self.prot_array_model_forward[i].predict(last_state_copy))
                    pos += 1

                sampled_sprg = self.prot_array_model_forward[s_ind].sample(last_state, weighted_mu, target_mu, mu, cov)

                state, reward, gamma = sampled_sprg[:self.state_dim], \
                                       sampled_sprg[self.state_dim], \
                                       sampled_sprg[self.state_dim + 1]

                if self.learning_mode not in raw_model_mode_list:
                    state[:self.state_dim] /= np.linalg.norm(state[:self.state_dim])

                # real_last_state = self.temp_decoder.test2(last_state)
                # real_state = self.temp_decoder.test2(state)
                # if (real_last_state[0]<=0.5) and (real_state[0]>=0.7):
                    # print("Forward:",real_last_state,real_state)

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
                        sprg = self.sample_sprg(last_state, last_action, depth=depth+1, eval_mode=eval_mode)
                        if sprg is not None:
                            (last_state, last_action, state, reward, gamma) = sprg

                elif self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST or \
                        self.learning_mode == SINGLE_REP_CHECKDIST or \
                        self.learning_mode in rep_checkDist_mode_list:
                    satisfied = self._check_distance(state, last_state, rep=True)
                    if not satisfied:
                        sprg = self.sample_sprg(last_state, last_action, depth=depth+1, eval_mode=eval_mode)
                        if sprg is not None:
                            (last_state, last_action, state, reward, gamma) = sprg

                # if eval_mode:
                #     print("Successor:", self.temp_decoder.test2(last_state), self.actions_map[last_action], self.temp_decoder.test2(state))
                #     # succ = self.temp_decoder.test2(state)
                #     # if (succ[0] > (0.5+0.05) and succ[0] < (0.7-0.05)) and (succ[1] < (0.4-0.05) or succ[1] > (0.6+0.05)):
                #     #     print("Succ wallllll!!!!", succ)
                #     input()

                return (last_state, last_action, state, reward, gamma)
            else:
                # print("sprg return None")
                return None

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
            #     pred_s = self._sample_s(last_action, state, seq)
            #     if pred_s is not None:
            #         predecessor_list.append(self.sample_sprg(pred_s, last_action))
            pred_s = self._sample_s(last_action, state, seq)
            if pred_s is not None:
                tuple = self.sample_sprg(pred_s, last_action, eval_mode=False)
                if tuple is not None:
                    predecessor_list.append(tuple)
        return predecessor_list

    def _sample_predecessor_for_action(self, action, state, f):
        predecessor_list = []
        occupied = [i for i in range(self.action_ind, self.state_dim * 2 + 1)]
        seq = self._refill_seq(np.concatenate((np.array([action]), state)), occupied)
        # sum_prob_sp = np.sum(self._kernel_sp(seq)[0])
        # if self._prob_a_sp(action, state, sum_prob_sp, seq) != 0:
        pred_s = self._sample_s(action, state, seq)

        return pred_s


    # Algorithm 6
    """
    sample predecessor state given current state and previous action
    """
    def _sample_s(self, last_action, state, seq, depth=0):

        if depth >= self.depth_limit:
            return None

        if self.b == 0:
            return None

        else:

            kernel, nz_ind = self._kernel_asp(seq)

            # if np.max(kernel) < 0.9:
            #     # print("None similar s", kernel)
            #     # print("None similar s")
            #     return None
            # # else:
            # #     print("Sampling s", kernel)

            betas = np.multiply(self.cr[nz_ind], kernel)
            n_spa = np.sum(betas)

            if n_spa != 0 and not np.isnan(n_spa):
                betas = betas / float(n_spa)
            else:
                # print("return None")
                return None

            # print(betas)

            #original
            mu = np.average(self.prot_array[nz_ind, :self.state_dim],weights=np.transpose(betas),axis=0)
            diff = self.prot_array[nz_ind, :self.state_dim] - mu
            cov = (diff * betas[:, None]).T.dot(diff)  # n_spa

            s_ind = np.random.choice(nz_ind, 1, p=betas)[0]

            # sampled_s = np.random.multivariate_normal(target_mu, cov + np.eye(cov_on_ind.shape[0]) * self.kscale)

            target_mu = self.prot_array[s_ind, :self.state_dim]

            state_copy = np.copy(state)
            state_copy = np.insert(state_copy,0,1.0)
            weighted_mu = np.zeros((self.state_dim))
            pos = 0
            for i in nz_ind:
                weighted_mu += (betas[pos]*self.prot_array_model_reverse[i].predict(state_copy))
                pos += 1

            sampled_s = self.prot_array_model_reverse[s_ind].sample(state, weighted_mu, target_mu, mu, cov)

            if self.learning_mode not in raw_model_mode_list:
                sampled_s /= np.linalg.norm(sampled_s)

            # print("Predecessor:",self.temp_decoder.test2(state),self.actions_map[last_action],self.temp_decoder.test2(sampled_s))
            # pred = self.temp_decoder.test2(sampled_s)
            # if (pred[0] > (0.5+0.05) and pred[0] < (0.7-0.05)) and (pred[1] < (0.4-0.05) or pred[1] > (0.6+0.05)):
            #     print("Pred wallllll!!!!",pred)
            #     input()

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
                    sampled_s = self._sample_s(last_action, state, seq, depth=depth + 1)

            elif self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST or \
                    self.learning_mode == SINGLE_REP_CHECKDIST or \
                    self.learning_mode in rep_checkDist_mode_list:
                satisfied = self._check_distance(state, sampled_s, rep=True)
                if not satisfied:
                    sampled_s = self._sample_s(last_action, state, seq, depth=depth + 1)

            return sampled_s  # sampled_s_list

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
    def _kernel_sa(self, seq, knn_ind = None):
        indices = [i for i in range(self.state_dim)]

        if knn_ind is not None:
            action = seq[self.action_ind]
            knn_prot = self.prot_array[knn_ind]
            k_s = np.zeros(len(knn_ind))

            same_a = np.where(knn_prot[:, self.action_ind] == action)[0]
            seq = np.outer(np.ones(len(same_a)), seq)
            seq_s = seq[:, indices]
            proto_s = knn_prot[same_a][:, indices]
            diffs = seq_s - proto_s
            _, covmat_inv = self._cal_covmat_inv(indices)
            k_s[same_a] = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            pos = knn_ind

        else:

            action = seq[self.action_ind]

            if action not in self.same_a_ind.keys():
                return [], []

            else:
                same_a = self.same_a_ind[action]

            if len(same_a) <= self.num_near:
                seq = np.outer(np.ones(len(same_a)), seq)
                seq_s = seq[:, indices]
                proto_s = self.prot_array[same_a][:, indices]

                diffs = seq_s - proto_s
                _, covmat_inv = self._cal_covmat_inv(indices)
                k_s = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

                pos = np.array(same_a)

            else:
                seq = np.outer(np.ones(self.num_near), seq)
                seq_s = seq[:, indices]
                _, covmat_inv = self._cal_covmat_inv(indices)

                start = time.time()
                _, [index] = self.s_tree[action].query([seq_s[0]], k=self.num_near)
                self.running_time["search"] += time.time() - start

                proto_s = self.prot_array[same_a][index][:,indices]

                # # changed -> added
                # print(proto_s)
                # print(seq_s)
                # print("==================")

                diffs = seq_s - proto_s
                k_s = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

                pos = np.array(np.array(same_a)[index])


                # print(seq_s, proto_s)
                # print(diffs)
                # print(k_s)
                # print("--")
        # if(np.sum(k_s)) == 0:
        #     k_s = np.ones(len(k_s)) / float(len(k_s))
        # # else:
        # #     k_s /= np.sum(k_s)

        return k_s, pos

    """similarity between 2 (a, s') pairs"""
    def _kernel_asp(self, seq):
        indices = [i for i in range(self.state_dim + 1, self.state_dim * 2 + 1)]
        action = seq[self.action_ind]

        if action not in self.same_a_ind.keys():
            return [], []
        else:
            same_a = self.same_a_ind[action]

        if len(same_a) < self.num_near:
            seq = np.outer(np.ones(len(same_a)), seq)
            seq_sp = seq[:, indices]
            proto_sp = self.prot_array[same_a][:, indices]
            diffs = seq_sp - proto_sp
            _, covmat_inv = self._cal_covmat_inv(indices)
            k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

            pos = np.array(same_a)

        else:

            seq = np.outer(np.ones(self.num_near), seq)
            seq_sp = seq[:, indices]
            _, covmat_inv = self._cal_covmat_inv(indices)

            start = time.time()
            _, [index] = self.sp_tree[action].query([seq_sp[0]], k=self.num_near)
            self.running_time["search"] += time.time() - start

            proto_sp = self.prot_array[same_a][index][:,indices]
            diffs = seq_sp - proto_sp
            k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

            pos = np.array(np.array(same_a)[index])

        # if(np.sum(k_sp)) == 0:
        #     k_sp = np.ones(len(k_sp)) / float(len(k_sp))
        # # else:
        # #     k_sp /= np.sum(k_sp)

        return k_sp, pos


    """similarity between 2 (s', r, g) pairs"""
    def _kernel_sprg(self, seq, knn_ind = None):
        indices = [i for i in range(self.state_dim + 1, self.seq_dim)]

        # if knn_ind is not None:
        seq = np.outer(np.ones(len(knn_ind)), seq)
        seq_sprg = seq[:, indices]
        protos_sprg = self.prot_array[knn_ind][:, indices]
        diffs = seq_sprg - protos_sprg
        _, covmat_inv = self._cal_covmat_inv(indices)
        k_sprg = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

        # if(np.sum(k_sprg)) == 0:
        #     k_sprg = np.ones(len(k_sprg)) / float(len(k_sprg))
        # # else:
        # #     k_sprg /= np.sum(k_sprg)

        return k_sprg, knn_ind

    """similarity between 2 s'"""
    def _kernel_sp(self, seq):
        indices = [i for i in range(self.state_dim + 1, self.state_dim * 2 + 1)]
        if self.b <= self.num_near:
            seq = np.outer(np.ones(self.b), seq)
            seq_sp = seq[:, indices]
            protos_sp = self.prot_array[:self.b, indices]
            diffs = seq_sp - protos_sp
            _, covmat_inv = self._cal_covmat_inv(indices)
            k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

            pos = np.array([i for i in range(self.b)])

        else:
            seq = np.outer(np.ones(self.num_near), seq)
            seq_sp = seq[:, indices]
            _, covmat_inv = self._cal_covmat_inv(indices)

            start = time.time()
            _, [index] = self.sp_allA_tree.query([seq_sp[0]], k=self.num_near)
            self.running_time["search"] += time.time() - start

            protos_sp = self.prot_array[index][:, indices]
            diffs = seq_sp - protos_sp
            k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

            pos = index

        return k_sp, index

    """
    Get covariance matrix and its inverse
    """
    def _cal_covmat_inv(self, indices):
        covmat = self.sig_prot[indices][:, indices]
        covmat_inv = self.sig_prot_inv[indices][:, indices]
        return covmat, covmat_inv

    """
    Find k nearest neighbors of (s,a)
    """
    def _sa_n_near(self, seq):
        kernel, nz_ind = self._kernel_sa(seq)
        return kernel, nz_ind

    """
    Find k nearest neighbors of (a, s')
    """
    def _asp_n_near(self, seq):
        kernel, nz_ind = self._kernel_asp(seq)
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
    def get_protos(self):
        return self.prot_array[:self.b]

    """
    Get number of prototypes
    """
    def get_len_protos(self):
        return self.b

    """
    Check running time
    Not used for learning process
    """
    def _check_time(self):
        res = self.running_time.copy()
        for key in self.running_time.keys():
            self.running_time[key] = 0.0
        return res


    def get_added_prototype(self):
        return self.added_prototype
