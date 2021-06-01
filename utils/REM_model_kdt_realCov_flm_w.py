import numpy as np
import random
import time
import sklearn.neighbors as skln
import sklearn.preprocessing as sklp
from utils.get_learned_representation import *

import matplotlib.pylab as plt
import utils.get_learned_state as gls

np.set_printoptions(precision=3)

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

from utils.recover_state import RecvState
temp_decoder = RecvState(32, [128, 256, 512], 2, 0.9, 0.9)
temp_decoder.loading("./feature_model_fixed_env/",
                     "feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_gamma[0.998, 0.8]_epoch1000_nfeature32_beta0.1_seperateRcvs")


class Forward_model:
    def __init__(self, state_dim, reward_dim, gamma_dim, seq):
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.gamma_dim = gamma_dim
        self.sprg_dim = self.state_dim + self.reward_dim + self.gamma_dim
        self.state = seq[: self.state_dim]
        self.a = seq[self.state_dim]

        self.avg = seq[self.sprg_dim * -1 :]
        self.w = 1.0

        self.s = np.zeros((self.sprg_dim, self.sprg_dim))
        self.cov = np.zeros((self.sprg_dim, self.sprg_dim))

    def update_model(self, new_seq, similarity):
        sprg = new_seq[self.sprg_dim * -1 :]
        self.w += similarity

        self.avg += similarity / self.w * (sprg - self.avg)

        diff = sprg - self.avg
        self.s += similarity * np.outer(diff, diff)
        self.cov = self.s / self.w

        # print("======")
        # print("s   ", temp_decoder.test2(self.state))
        # print("a   ", self.a)
        # print("sprg", temp_decoder.test2(self.avg[:self.state_dim]))
        # input("Interrupt")

    def predict(self, given_s):
        pred_sp = given_s + self.avg[: self.state_dim]
        pred_r = self.avg[self.state_dim: self.state_dim + self.reward_dim]
        pred_g = self.avg[self.state_dim + self.reward_dim: ]

        # print("======")
        # print("proto s ", temp_decoder.test2(self.state))
        # print("pred  sp", temp_decoder.test2(pred_sp))
        # input("Interrupt")

        return pred_sp, pred_r, pred_g

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
        self.prot_array = np.empty((self.prot_len, self.seq_dim))
        self.prot_array_model_forward = {}

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

        self.sample_single_neighbour = False
        self.sample_weighted_mean = False

        # from utils.recover_state import RecvState
        # self.temp_decoder = RecvState(state_dim, [128, 256, 512], 2, 0.9, 0.9)
        # self.temp_decoder.loading("./feature_model/", "feature_embedding_continuous_input[0.0, 1]_envSucProb1.0_seperateRcvs")

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
        if self.learning_mode == SINGLE_REP or \
                self.learning_mode == SINGLE_NORMREP or \
                self.learning_mode == SINGLE_NORMREP_FIXCOV or \
                self.learning_mode == TCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL_FIXCOV or \
                self.learning_mode == BIASTCREPVF_REPMODEL or \
                self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST or \
                self.learning_mode == SINGLE_REP_CHECKDIST or \
                self.learning_mode == TCREPVF_NORMREPMODEL or \
                self.learning_mode == BIASTCREPVF_NORMREPMODEL or \
                self.learning_mode == TCRAWVF_NORMREPMODEL or \
                self.learning_mode == TCRAWVF_NORMREPMODEL_FIXCOV:
            assert len(last_state) == self.state_dim
            assert self.state_dim == 32
        else:
            assert len(last_state) == 2

        seq = self._seq2array(last_state, last_action, state, reward, gamma)
        self._update_t_mu_cov(seq)

        if last_action not in self.b_a:
            self._update_prototype(seq)
            self.b_a.add(last_action)
        else:
            self._update_prototype(seq)

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
        # covmat_inv = 0.001*np.eye(len(indices))
        # Used
        self.s_tree[action] = skln.BallTree(X[:, indices], leaf_size=leaf_size,
                                                   metric=skln.DistanceMetric.get_metric('mahalanobis', VI=covmat_inv))

        indices = [i for i in range(self.state_dim + 1, self.state_dim * 2 + 1)]
        _, covmat_inv = self._cal_covmat_inv(indices)
        # covmat_inv = 0.001*np.eye(len(indices))
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

            #1-NN
            _, [index] = self.ssprg_tree[action].query([seq[self.index_no_a]], k=1)

            prototypes = self.prot_array[np.array(same_a)][index][:,self.index_no_a]
            seq_mat_diff = prototypes - np.outer(np.ones(len(index)), seq[self.index_no_a])
            prototypes_transition = -1*np.diag((seq_mat_diff).dot(covmat_inv).dot(seq_mat_diff.T))
            # print(prototypes_transition)
            if np.max(prototypes_transition) < self.add_prot_limit:
                add_prot = True

        # k-NN
        kernel, nz_ind = self._kernel_sa(seq)

        for i in range(len(kernel)):
            self.prot_array_model_forward[nz_ind[i]].update_model(seq, kernel[i])

        self.added_prototype = False
        if add_prot:
            self.added_prototype = True
            if self.b < self.prot_len:
                self.prot_array[self.b] = seq
                self.prot_array_model_forward[self.b] = Forward_model(self.state_dim, 1, 1, seq)
            else:
                self.prot_array = np.concatenate((self.prot_array, np.array([seq])), axis=0)
                self.prot_array_model_forward[self.b] = Forward_model(self.state_dim, 1, 1, seq)

            if action in self.same_a_ind.keys():
                self.same_a_ind[action].append(self.b)
            else:
                self.same_a_ind[action] = [self.b]

            self.b += 1

        self._kdtree_const(action)
        return

    # Algorithm 5
    """
    sample (s', r, \gamma) given current state and action
    """
    def sample_sprg(self, last_state, last_action, depth=0):
        if depth >= self.depth_limit:
            return None

        if self.b == 0:
            return None

        else:
            occupied = [i for i in range(self.state_dim + 1)]
            seq = self._refill_seq(np.concatenate((last_state, np.array([last_action]))), occupied)

            kernel, nz_ind = self._kernel_sa(seq)

            if(np.sum(kernel)) == 0:
                kernel = np.ones(len(kernel)) / float(len(kernel))
            else:
                kernel /= np.sum(kernel)

            predictions = {}
            # target_mu = np.zeros((self.state_dim+2))
            pos = 0
            for i in range(len(nz_ind)):
                # print("ind", nz_ind[i])
                predictions[i] = self.prot_array_model_forward[nz_ind[i]].predict(last_state)
                # target_mu += (kernel[pos]*predictions[i])
                pos += 1

            state, [reward], [gamma] = predictions[0]

            # if self.sample_weighted_mean:
            #     sampled_sprg = target_mu
            # elif self.sample_single_neighbour:
            #     s_ind = np.random.choice(nz_ind, 1)[0]
            #     sampled_sprg = predictions[s_ind]
            # else:
            #     # s_ind = np.random.choice(nz_ind, 1, p=kernel)[0]
            #     # target_mu = predictions[s_ind]
            #
            #     cov_mu = np.zeros((self.state_dim+2,self.state_dim+2))
            #     #conditional covariance
            #     pos = 0
            #     for i in nz_ind:
            #         diff = predictions[i]-target_mu
            #         cov_mu += ((kernel[pos]*diff).dot(diff.T)) # if self.sample_weighted_mean:
            #     sampled_sprg = target_mu
            # elif self.sample_single_neighbour:
            #     s_ind = np.random.choice(nz_ind, 1)[0]
            #     sampled_sprg = predictions[s_ind]
            # else:
            #     # s_ind = np.random.choice(nz_ind, 1, p=kernel)[0]
            #     # target_mu = predictions[s_ind]
            #
            #     cov_mu = np.zeros((self.state_dim+2,self.state_dim+2))
            #     #conditional covariance
            #     pos = 0
            #     for i in nz_ind:
            #         diff = predictions[i]-target_mu
            #         cov_mu += ((kernel[pos]*diff).dot(diff.T))
            #         pos += 1
            #
            #     if self.fix_cov == 0:
            #         sampled_sprg = np.random.multivariate_normal(target_mu, cov_mu + np.eye(cov_mu.shape[0]) * self.kscale, 1)[0]
            #         # sampled_sprg = np.zeros((self.state_dim+2))
            #         # sampled_sprg[:self.state_dim] = np.random.multivariate_normal(target_mu[:self.state_dim], cov_mu[:self.state_dim,:self.state_dim] + np.eye(self.state_dim) * self.kscale, 1)[0]
            #         # sampled_sprg[self.state_dim] = np.random.multivariate_normal(np.array([target_mu[self.state_dim]]), cov_mu[self.state_dim,self.state_dim] + np.eye(1) * self.kscale, 1)[0]
            #         # sampled_sprg[self.state_dim+1] = np.random.multivariate_normal(np.array([target_mu[self.state_dim+1]]), cov_mu[self.state_dim+1,self.state_dim+1] + np.eye(1) * self.kscale, 1)[0]
            #     else:
            #         sampled_sprg = np.random.multivariate_normal(target_mu, np.eye(cov_mu.shape[0]) * self.fix_cov, 1)[0]

            # state, reward, gamma = sampled_sprg[:self.state_dim], \
            #                        sampled_sprg[self.state_dim], \
            #                        sampled_sprg[self.state_dim + 1]
            #         pos += 1
            #
            #     if self.fix_cov == 0:
            #         sampled_sprg = np.random.multivariate_normal(target_mu, cov_mu + np.eye(cov_mu.shape[0]) * self.kscale, 1)[0]
            #         # sampled_sprg = np.zeros((self.state_dim+2))
            #         # sampled_sprg[:self.state_dim] = np.random.multivariate_normal(target_mu[:self.state_dim], cov_mu[:self.state_dim,:self.state_dim] + np.eye(self.state_dim) * self.kscale, 1)[0]
            #         # sampled_sprg[self.state_dim] = np.random.multivariate_normal(np.array([target_mu[self.state_dim]]), cov_mu[self.state_dim,self.state_dim] + np.eye(1) * self.kscale, 1)[0]
            #         # sampled_sprg[self.state_dim+1] = np.random.multivariate_normal(np.array([target_mu[self.state_dim+1]]), cov_mu[self.state_dim+1,self.state_dim+1] + np.eye(1) * self.kscale, 1)[0]
            #     else:
            #         sampled_sprg = np.random.multivariate_normal(target_mu, np.eye(cov_mu.shape[0]) * self.fix_cov, 1)[0]

            # state, reward, gamma = sampled_sprg[:self.state_dim], \
            #                        sampled_sprg[self.state_dim], \
            #                        sampled_sprg[self.state_dim + 1]

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
                    sprg = self.sample_sprg(last_state, last_action, depth=depth+1)
                    if sprg is not None:
                        (last_state, last_action, state, reward, gamma) = sprg

            elif self.learning_mode == BIASTCREPVF_REPMODEL_CHECKDIST or \
                    self.learning_mode == SINGLE_REP_CHECKDIST:
                satisfied = self._check_distance(state, last_state, rep=True)
                if not satisfied:
                    sprg = self.sample_sprg(last_state, last_action, depth=depth+1)
                    if sprg is not None:
                        (last_state, last_action, state, reward, gamma) = sprg

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
                predecessor_list.append(self.sample_sprg(pred_s, last_action))
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
                predictions[i] = self.prot_array_model_reverse[i].predict(state_copy)
                target_mu += (kernel[pos]*predictions[i])
                pos += 1

            if self.sample_weighted_mean:
                sampled_s = target_mu
            elif self.sample_single_neighbour:
                s_ind = np.random.choice(nz_ind, 1)[0]
                sampled_s = predictions[s_ind]
            else:
                # s_ind = np.random.choice(nz_ind, 1, p=kernel)[0]
                # target_mu = predictions[s_ind]

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

            # print("Predecessor:",self.temp_decoder.test2(state),last_action,self.temp_decoder.test2(sampled_s))

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

            # k_s = np.copy(np.diag(seq_s.dot(proto_s.T)))

            pos = knn_ind

        else:
            # k = np.zeros(self.b)
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

                # k_s = np.copy(np.diag(seq_s.dot(proto_s.T)))

                pos = same_a

            else:
                seq = np.outer(np.ones(self.num_near), seq)
                seq_s = seq[:, indices]

                start = time.time()
                _, [index] = self.s_tree[action].query([seq_s[0]], k=self.num_near)
                self.running_time["search"] += time.time() - start

                proto_s = self.prot_array[same_a][index][:,indices]

                diffs = seq_s - proto_s
                _, covmat_inv = self._cal_covmat_inv(indices)
                k_s = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

                # k_s = np.copy(np.diag(seq_s.dot(proto_s.T)))

                pos = np.array(same_a)[index]

                # return k_s, np.array(same_a)[index]

        # if(np.sum(k_s)) == 0:
        #     k_s = np.ones(len(k_s)) / float(len(k_s))
        # else:
        #     k_s /= np.sum(k_s)

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

            # k_sp = np.copy(np.diag(seq_sp.dot(proto_sp.T)))

            pos = same_a

        else:

            seq = np.outer(np.ones(self.num_near), seq)
            seq_sp = seq[:, indices]

            start = time.time()
            _, [index] = self.sp_tree[action].query([seq_sp[0]], k=self.num_near)
            self.running_time["search"] += time.time() - start

            proto_sp = self.prot_array[same_a][index][:,indices]

            diffs = seq_sp - proto_sp
            _, covmat_inv = self._cal_covmat_inv(indices)
            k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

            # k_sp = np.copy(np.diag(seq_sp.dot(proto_sp.T)))

            pos = np.array(same_a)[index]

        # if(np.sum(k_sp)) == 0:
            # k_sp = np.ones(len(k_sp)) / float(len(k_sp))
        # else:
        #     k_sp /= np.sum(k_sp)

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

        # k_sprg = np.copy(np.diag(seq_sprg.dot(proto_sprg.T)))

        # if(np.sum(k_sprg)) == 0:
            # k_sprg = np.ones(len(k_sprg)) / float(len(k_sprg))
        # else:
        #     k_sprg /= np.sum(k_sprg)

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
