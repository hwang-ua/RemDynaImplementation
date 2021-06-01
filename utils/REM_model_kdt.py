import numpy as np
import random
import time
import sklearn.neighbors as skln

class REM_Model:
    def __init__(self, state_dim, num_near, add_prot_limit, model_params):
        self.state_dim = state_dim
        self.seq_dim = state_dim * 2 + 3  # s, a, s', r, gamma
        self.action_ind = state_dim
        self.index_no_ag = [i for i in range(self.seq_dim - 1)]
        self.index_no_ag.remove(self.action_ind)

        # the action and gamma are not included
        self.sum_array = np.zeros((self.seq_dim - 2))
        self.sig_prot = np.zeros((self.seq_dim - 2, self.seq_dim - 2))
        self.sig_prot_first = np.zeros((self.seq_dim - 2, self.seq_dim - 2))
        self.sig_prot_inv = np.zeros((self.seq_dim, self.seq_dim))

        self.prot_len = 10000
        self.c = np.ones((self.prot_len))
        self.cr = np.ones((self.prot_len))
        self.prot_array = np.empty((self.prot_len, self.seq_dim))

        self.b = 0  # number of prototype in model
        self.t = 0  # number of sequence sent to model (added & not added)
        self.kscale = model_params["kscale"]

        self.add_prot_limit = add_prot_limit
        self.num_near = num_near
        self.kernel_near = 250
        return

    def add2Model(self, last_state, last_action, state, reward, gamma):
        self.update_rem(last_state, last_action, state, reward, gamma)
        return

    def KDE_sampleSpRG(self, last_state, last_action):
        sample = self.sample_sprg(last_state, last_action)
        if sample is not None:
            last_state, last_action, state, reward, gamma = sample
            return (last_state, state, reward, gamma, last_action)
        else:
            return None

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


        seq = self._seq2array(last_state, last_action, state, reward, gamma)
        self._update_t_mu_cov(seq)
        self._update_prototype(seq, self.num_near)

        random_prot = np.array([i for i in range(self.b)])
        prot_a = np.copy(self.prot_array[:self.b])
        sa_n_near, rho, nnear_ind = self._sa_n_near(seq, prot_a)

        if self.b > 1:
            self.c[random_prot][nnear_ind] = np.multiply((1.0 - rho), self.c[random_prot][nnear_ind]) + \
                                             np.multiply(rho, self._kernel_sprg(seq, sa_n_near))

            asp_n_near, _, nnear_ind = self._asp_n_near(seq, prot_a)
            rhor = self._kernel_asp(seq, asp_n_near)
            self.cr[random_prot][nnear_ind] = np.multiply(1.0 - rhor, self.cr[random_prot][nnear_ind]) + \
                                              np.multiply(rhor, self._kernel_s(seq, asp_n_near))
        else:
            self.c[nnear_ind] = np.multiply((1.0 - rho), self.c[nnear_ind]) + \
                                np.multiply(rho, self._kernel_sprg(seq, sa_n_near))

            asp_n_near, _, nnear_ind = self._asp_n_near(seq, prot_a)
            rhor = self._kernel_asp(seq, asp_n_near)
            self.cr[nnear_ind] = np.multiply(1.0 - rhor, self.cr[nnear_ind]) + \
                                 np.multiply(rhor, self._kernel_s(seq, asp_n_near))
        return

    def _update_t_mu_cov(self, seq):
        self.t += 1
        # seq = seq[self.index_no_ag]
        # self.sum_array += seq
        # self.mu_prot = self.sum_array / float(self.t)
        # self.sig_prot_first = ((self.t - 1.0) * self.sig_prot_first + np.outer(seq, seq)) / self.t
        # self.sig_prot = self.sig_prot_first - np.outer(self.mu_prot, self.mu_prot)
        #
        # sig_prot_inv_first = np.linalg.inv((self.sig_prot + (np.eye(self.sig_prot.shape[0]) * self.kscale)))
        # # divided by self.b before get the inverse
        # # sig_prot_inv_first = sig_prot_inv_first / float(self.b) if self.b != 0 else sig_prot_inv_first
        #
        # self.sig_prot = np.insert(self.sig_prot, self.action_ind, np.zeros((self.seq_dim - 2)), axis = 1)
        # self.sig_prot = np.insert(self.sig_prot, self.action_ind, np.zeros((self.seq_dim - 1)), axis=0)
        # self.sig_prot = np.concatenate((self.sig_prot, np.zeros((self.seq_dim - 1, 1))), axis=1)
        # self.sig_prot = np.concatenate((self.sig_prot, np.zeros((1, self.seq_dim))), axis=0)
        #
        # self.sig_prot_inv = np.insert(sig_prot_inv_first, self.action_ind, np.zeros((self.seq_dim - 2)), axis = 1)
        # self.sig_prot_inv = np.insert(self.sig_prot_inv, self.action_ind, np.zeros(self.seq_dim - 1), axis = 0)
        #
        # self.sig_prot_inv = np.concatenate((self.sig_prot_inv, np.zeros((self.seq_dim - 1, 1))), axis=1)
        # self.sig_prot_inv = np.concatenate((self.sig_prot_inv, np.zeros((1, self.seq_dim))), axis=0)
        # self.sig_prot_inv = self.sig_prot_inv / float(self.b) if self.b != 0 else self.sig_prot_inv
        return

    def _k_near_neighbor(self, X, seq, lsize, vi, ksize):
        tree = skln.BallTree(X, leaf_size=lsize,
                             metric=skln.DistanceMetric.get_metric('mahalanobis', VI=vi))
        # tree = skln.KDTree(prot_a[same_a][:, self.index_no_ag], leaf_size=int(np.ceil(len(same_a) / 2.0)),
        #                    metric='euclidean')
        dist, index = tree.query([seq], k=ksize)
        return dist[0], index[0]

    # Algorithm 8
    def _update_prototype(self, seq, num_near):
        prot_a = np.copy(self.prot_array[:self.b])

        add_prot = False
        if self.b == 0:
            add_prot = True
        else:
            same_a = np.where(prot_a[:,self.action_ind] == seq[self.action_ind])[0]
            if len(same_a) >= self.num_near:
                _, vi = self._cal_covmat_inv(self.index_no_ag)
                dist, _ = self._k_near_neighbor(prot_a[same_a][:, self.index_no_ag],
                                                seq[self.index_no_ag],
                                                int(np.ceil(len(same_a) / 2.0)),
                                                vi,
                                                self.num_near)
                if np.max(dist) > self.add_prot_limit:
                    add_prot = True
            else:
                add_prot = True

        if add_prot:
            if self.b < self.prot_len:
                self.prot_array[self.b] = seq
            else:
                self.prot_array = np.concatenate((self.prot_array, np.array([seq])), axis=0)
            self.b += 1
        return


    # Algorithm 5
    def sample_sprg(self, last_state, last_action):
        if self.b == 0:
            return None

        else:
            prot_a = np.copy(self.prot_array[:self.b])
            occupied = [i for i in range(self.state_dim + 1)]
            seq = self._refill_seq(np.concatenate((last_state, np.array([last_action]))), occupied)

            betas = np.multiply(self.c[:self.b], self._kernel_sa(seq, prot_a))
            n_sa = np.sum(betas)
            if n_sa != 0 and not np.isnan(n_sa):
                betas /= float(n_sa)
                mu = betas.dot(prot_a[:, self.state_dim + 1: -1])
                diff = prot_a[:, self.state_dim + 1: -1] - mu

                cov = (diff * betas[:, None]).T.dot(diff)

                s_ind = np.random.choice(range(self.b), 1, p=betas)[0]
                target_mu = prot_a[s_ind, self.state_dim + 1: -1]

                sampled_sprg = np.random.multivariate_normal(target_mu, cov + np.eye(cov.shape[0]) * self.kscale, 1)[0]
                state, reward, gamma = sampled_sprg[:self.state_dim], sampled_sprg[self.state_dim], None
                # reward = np.clip(reward, 0., 1.)
                state = np.clip(state, 0., 1.)
                # return is a tuple
                return (last_state, last_action, state, reward, gamma)
            else:
                return None

    def _refill_seq(self, existing, occupied):
        seq = np.zeros((self.seq_dim))
        seq[occupied] = existing
        return seq

    # Algorithm 4
    def _sample_predecessor(self, num_action, state, f):
        predecessor_list = []
        for last_action in range(num_action):
            if self._prob_a_sp(last_action, state) != 0:
                pred_s = self._sample_s(last_action, state)
                if pred_s is not None:
                    predecessor_list.append(self.sample_sprg(pred_s, last_action))
        return predecessor_list

    # Algorithm 6
    # sample f precessors
    def _sample_s(self, last_action, state):
        if self.b == 0:
            return None

        else:
            prot_a = np.copy(self.prot_array[:self.b])
            occupied = [i for i in range(self.action_ind, self.state_dim * 2 + 1)]
            seq = self._refill_seq(np.concatenate((np.array([last_action]), state)), occupied)
            betas = np.multiply(self.cr[:self.b], self._kernel_asp(seq, prot_a))
            n_spa = np.sum(betas)

            if n_spa != 0 and not np.isnan(n_spa):
                betas = betas / float(n_spa)
            else:
                return None

            mu = betas.dot(prot_a[:, :self.state_dim])
            shape = self.state_dim  # s
            diff = prot_a[:, :self.state_dim] - mu
            cov = (diff * betas[:, None]).T.dot(diff) / float(self.b)  # n_spa
            s_ind = np.random.choice(range(len(prot_a)), 1, p=betas)[0]
            target_mu = prot_a[s_ind, :self.state_dim]
            sampled_s = np.clip(np.random.multivariate_normal(target_mu, cov + np.eye(cov.shape[0]) * self.kscale),
                                0., 1.)
            return sampled_s  # sampled_s_list

    def _seq2array(self, last_state, last_action, state, reward, gamma):
        return np.concatenate((last_state, np.array([last_action]), state, np.array([reward]), np.array([gamma])),
                              axis=0)

    def _kernel_seq(self, seq, protos):
        indices = [i for i in range(self.seq_dim)]
        indices.remove(self.action_ind)
        k = np.zeros((len(protos)))

        same_a = np.where(seq[self.action_ind] == protos[:, self.action_ind])[0]
        if len(same_a) == 0:
            return k

        elif len(same_a) <= self.kernel_near:
            seq = np.outer(np.ones(len(same_a)), seq)
            seq_ssprg = seq[:, indices]
            proto_ssprg = protos[same_a][:, indices]
            diffs = seq_ssprg - proto_ssprg
            _, covmat_inv = self._cal_covmat_inv(indices)
            k_ssprg = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            k[same_a] = k_ssprg
            return k

        else:
            seq = np.outer(np.ones(self.kernel_near), seq)
            seq_ssprg = seq[:, indices]

            _, covmat_inv = self._cal_covmat_inv(indices)
            dist, index = self._k_near_neighbor(protos[same_a][:, indices],
                                                seq[0, indices],
                                                int(np.ceil(len(same_a) / 2.0)),
                                                covmat_inv,
                                                self.kernel_near)
            proto_ssprg = protos[same_a[index]][:, indices]
            diffs = seq_ssprg - proto_ssprg
            k_ssprg = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            k[same_a[index]] = k_ssprg
            return k


    def _kernel_sa(self, seq, protos):
        indices = [i for i in range(self.state_dim)]
        k = np.zeros((len(protos)))

        same_a = np.where(seq[self.action_ind] == protos[:, self.action_ind])[0]
        if len(same_a) == 0:
            return k

        elif len(same_a) <= self.kernel_near:
            seq = np.outer(np.ones(len(same_a)), seq)
            seq_s = seq[:, indices]
            proto_s = protos[same_a][:, indices]
            diffs = seq_s - proto_s
            _, covmat_inv = self._cal_covmat_inv(indices)
            k_s = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

            k[same_a] = k_s
            if np.where(k_s > 1)[0].shape[0] != 0:
                print("large kernel", k)
                print("            ", diffs)
                print("            ", covmat_inv)
            return k

        else:
            seq = np.outer(np.ones(self.kernel_near), seq)
            seq_s = seq[:, indices]
            _, covmat_inv = self._cal_covmat_inv(indices)
            dist, index = self._k_near_neighbor(protos[same_a][:, indices],
                                                seq[0, indices],
                                                int(np.ceil(len(same_a) / 2.0)),
                                                covmat_inv,
                                                self.kernel_near)
            proto_s = protos[same_a[index]][:, indices]
            diffs = seq_s - proto_s
            k_s = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            k[same_a[index]] = k_s
            return k

    def _kernel_asp(self, seq, protos):
        indices = [i for i in range(self.state_dim + 1, self.state_dim * 2 + 1)]
        k = np.zeros((len(protos)))

        same_a = np.where(seq[self.action_ind] == protos[:, self.action_ind])[0]
        if len(same_a) == 0:
            return k

        elif len(same_a) <= self.kernel_near:
            seq = np.outer(np.ones(len(same_a)), seq)
            seq_sp = seq[:, indices]
            proto_sp = protos[same_a][:, indices]
            diffs = seq_sp - proto_sp
            _, covmat_inv = self._cal_covmat_inv(indices)
            k_asp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            k[same_a] = k_asp
            return k

        else:
            seq = np.outer(np.ones(self.kernel_near), seq)
            seq_sp = seq[:, indices]
            _, covmat_inv = self._cal_covmat_inv(indices)
            dist, index = self._k_near_neighbor(protos[same_a][:, indices],
                                                seq[0, indices],
                                                int(np.ceil(len(same_a) / 2.0)),
                                                covmat_inv,
                                                self.kernel_near)
            proto_sp = protos[same_a[index]][:, indices]
            diffs = seq_sp - proto_sp
            k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            k[same_a[index]] = k_sp
            return k

    def _kernel_sprg(self, seq, protos):
        indices = [i for i in range(self.state_dim + 1, self.seq_dim - 1)]
        if len(protos) <= self.kernel_near:
            seq = np.outer(np.ones(len(protos)), seq)
            seq_spr = seq[:, indices]
            protos_spr = protos[:, indices]
            diffs = seq_spr - protos_spr
            _, covmat_inv = self._cal_covmat_inv(indices)
            k_sprg = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            return k_sprg
        else:
            k = np.zeros(len(protos))
            seq = np.outer(np.ones(self.kernel_near), seq)
            seq_spr = seq[:, indices]
            _, covmat_inv = self._cal_covmat_inv(indices)
            dist, index = self._k_near_neighbor(protos[:, indices],
                                                seq[0, indices],
                                                int(np.ceil(len(protos) / 2.0)),
                                                covmat_inv,
                                                self.kernel_near)
            protos_spr = protos[index][:, indices]
            diffs = seq_spr - protos_spr
            k_sprg = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            k[index] = k_sprg
            return k

    def _kernel_s(self, seq, protos):
        indices = [i for i in range(self.state_dim)]
        if len(protos) <= self.kernel_near:
            seq = np.outer(np.ones(len(protos)), seq)
            seq_s = seq[:, indices]
            protos_s = protos[:, indices]
            diffs = seq_s - protos_s
            _, covmat_inv = self._cal_covmat_inv(indices)
            k_s = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            return k_s
        else:
            k = np.zeros(len(protos))
            seq = np.outer(np.ones(self.kernel_near), seq)
            seq_s = seq[:, indices]
            _, covmat_inv = self._cal_covmat_inv(indices)
            dist, index = self._k_near_neighbor(protos[:, indices],
                                                seq[0, indices],
                                                int(np.ceil(len(protos) / 2.0)),
                                                covmat_inv,
                                                self.kernel_near)
            protos_s = protos[index][:, indices]
            diffs = seq_s - protos_s
            k_s = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            k[index] = k_s
            return k

    def _kernel_sp(self, seq, protos):
        indices = [i for i in range(self.state_dim + 1, self.state_dim * 2 + 1)]
        if len(protos) <= self.kernel_near:
            seq = np.outer(np.ones(len(protos)), seq)
            seq_sp = seq[:, indices]
            protos_sp = protos[:, indices]
            diffs = seq_sp - protos_sp
            _, covmat_inv = self._cal_covmat_inv(indices)
            k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            return k_sp
        else:
            k = np.zeros(len(protos))
            seq = np.outer(np.ones(self.kernel_near), seq)
            seq_sp = seq[:, indices]
            _, covmat_inv = self._cal_covmat_inv(indices)
            dist, index = self._k_near_neighbor(protos[:, indices],
                                                seq[0, indices],
                                                int(np.ceil(len(protos) / 2.0)),
                                                covmat_inv,
                                                self.kernel_near)
            protos_sp = protos[index][:, indices]
            diffs = seq_sp - protos_sp
            k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            k[index] = k_sp
            return k

    # def _kernel_a(self, seq, protos):
    #     seq = np.outer(np.ones(len(protos)), seq)
    #     seq_a = seq[:, self.action_ind]
    #     protos_a = protos[:, self.action_ind]
    #     return seq_a == protos_a

    def _cal_covmat_inv(self, indices):
        covmat_inv = 1.0 / self.kscale * np.eye(len(indices))
        # covmat = self.sig_prot[indices][:, indices]
        # covmat_inv = self.sig_prot_inv[indices][:, indices]
        return None, covmat_inv

    def _sa_n_near(self, seq, prot_a):
        indices = [i for i in range(self.state_dim)]
        same_a = np.where(prot_a[:, self.action_ind] == seq[self.action_ind])[0]

        if len(same_a) == 0:
            return prot_a[:self.num_near], np.zeros(self.num_near), [i for i in range(self.num_near)]

        elif len(same_a) < self.num_near:
            ker_list = self._kernel_sa(seq, prot_a)
            dist_list = 1.0 - ker_list
            nnear_ind = np.argsort(dist_list)[:min(len(dist_list), self.num_near)]
            return prot_a[nnear_ind], ker_list[nnear_ind], nnear_ind

        else:
            _, covmat_inv = self._cal_covmat_inv(indices)
            dist, index = self._k_near_neighbor(prot_a[same_a][:, indices],
                                                seq[indices],
                                                int(np.ceil(len(same_a) / 2.0)),
                                                covmat_inv,
                                                self.num_near)
            nnear_ind = same_a[index]
            return prot_a[nnear_ind], 1.0 - dist, nnear_ind

    def _asp_n_near(self, seq, prot_a):
        indices = [i for i in range(self.action_ind, self.state_dim*2+1)]
        same_a = np.where(prot_a[:, self.action_ind] == seq[self.action_ind])[0]

        if len(same_a) == 0:
            return prot_a[:self.num_near], np.zeros(self.num_near), [i for i in range(self.num_near)]

        elif len(same_a) < self.num_near:
            ker_list = self._kernel_asp(seq, prot_a)
            dist_list = 1.0 - ker_list
            nnear_ind = np.argsort(dist_list)[:min(len(dist_list), self.num_near)]
            return prot_a[nnear_ind], ker_list[nnear_ind], nnear_ind

        else:
            _, covmat_inv = self._cal_covmat_inv(indices)
            dist, index = self._k_near_neighbor(prot_a[same_a][:, indices],
                                                seq[indices],
                                                int(np.ceil(len(same_a) / 2.0)),
                                                covmat_inv,
                                                self.num_near)
            nnear_ind = same_a[index]
            return prot_a[nnear_ind], 1.0 - dist, nnear_ind

    def _prob_a_sp(self, last_action, state):
        random_prot = np.array([i for i in range(self.b)])
        # np.random.shuffle(random_prot)
        # random_prot = random_prot[:min(self.b, self.num_rand_proto)]

        prot_a = self.prot_array[random_prot]

        occupied = [i for i in range(self.action_ind, self.state_dim * 2 + 1)]
        seq = self._refill_seq(np.concatenate((np.array([last_action]), state)), occupied)
        sum_prob_asp = np.sum(self._kernel_asp(seq, prot_a))
        sum_prob_sp = np.sum(self._kernel_sp(seq, prot_a))
        if sum_prob_sp == 0:
            return 1
        else:
            return sum_prob_asp / sum_prob_sp


    def get_protos(self):
        return self.prot_array[:self.b]

    def get_len_protos(self):
        return self.b