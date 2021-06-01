import numpy as np
import random
import time


class REM_Model:
    def __init__(self, state_dim, num_near, add_prot_limit, model_params):
        self.num_rand_proto = 2000

        self.state_dim = state_dim
        self.seq_dim = state_dim * 2 + 3 # s, a, s', r, gamma
        self.action_ind = state_dim
        self.index_no_ag = [i for i in range(self.seq_dim - 1)]
        self.index_no_ag.remove(self.action_ind)

        # the action and gamma are not included
        self.sum_array = np.empty((self.seq_dim - 2))
        self.sig_prot = np.zeros((self.seq_dim - 2, self.seq_dim - 2))
        self.sig_prot_first = np.zeros((self.seq_dim - 2, self.seq_dim - 2))
        self.sig_prot_inv = np.zeros((self.seq_dim, self.seq_dim))

        self.prot_len = 10000
        self.c = np.ones((self.prot_len))
        self.cr = np.ones((self.prot_len))
        self.prot_array = np.empty((self.prot_len, self.seq_dim))

        self.b = 0  # number of prototype
        self.t = 0  # number of sequence
        self.kscale = model_params["kscale"]

        self.add_prot_limit = add_prot_limit
        self.num_near = num_near

        return

    def add2Model(self, last_state, last_action, state, reward, gamma):
        self.update_rem(last_state, last_action, state, reward, gamma)
        return
    def KDE_sampleSpRG(self, last_state, last_action):
        sample = self.sample_sprg(last_state, last_action)
        if sample is not None:
            last_state, last_action, state, reward, gamma = sample
            # print(last_state, last_action, state, reward, gamma)
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
        random_prot = np.array([i for i in range(self.b)])
        np.random.shuffle(random_prot)
        random_prot = random_prot if self.b < self.num_rand_proto else random_prot[:self.num_rand_proto]

        prot_a = np.copy(self.prot_array) if self.b == 0 else np.copy(self.prot_array[random_prot])

        seq = self._seq2array(last_state, last_action, state, reward, gamma)
        self._update_t_mu_cov(seq)
        self._update_prototype(seq, self.num_near)

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

    # Algorithm 8
    def _update_prototype(self, seq, num_near):
        random_prot = np.array([i for i in range(self.b)])
        np.random.shuffle(random_prot)
        random_prot = random_prot if self.b < self.num_rand_proto else random_prot[:self.num_rand_proto]

        prot_a = np.copy(self.prot_array) if self.b == 0 else np.copy(self.prot_array[random_prot])

        limit = self.add_prot_limit
        add_prot = False
        if self.b == 0:
            add_prot = True
        else:
            dist_array = 1.0 - self._kernel_seq(seq, prot_a)
            nnear_ind = np.argsort(dist_array)[:min(num_near, len(dist_array))]
            add_prot = True if dist_array[nnear_ind[-1]] > limit else False
        # add prototype if the dist is larger than some limit
        if add_prot:
            if self.b < self.prot_len:
                self.prot_array[self.b] = seq
            else:
                self.prot_array = np.concatenate((self.prot_array, np.array([seq])), axis=0)
            self.b += 1
        return
    # def _update_prototype(self, seq, num_near):
    #     limit = self.add_prot_limit
    #     add_prot = False
    #     if self.b == 0:
    #         add_prot = True
    #     else:
    #         dist_array = np.ones((self.b)) - self._kernel_seq(seq, self.prot_array[:self.b])
    #         nnear_ind = np.argsort(dist_array)[:min(num_near, len(dist_array))]
    #         add_prot = True if dist_array[nnear_ind[-1]] > limit else False
    #     # add prototype if the dist is larger than some limit
    #     if add_prot:
    #         if self.b < self.prot_len:
    #             self.prot_array[self.b] = seq
    #         else:
    #             self.prot_array = np.concatenate((self.prot_array, np.array([seq])), axis=0)
    #         self.b += 1
    #     return

    # Algorithm 5
    def sample_sprg(self, last_state, last_action):
        if self.b == 0:
            return None
        else:
            random_prot = np.array([i for i in range(self.b)])
            np.random.shuffle(random_prot)
            random_prot = random_prot[:min(self.b, self.num_rand_proto)]

            prot_a = self.prot_array[random_prot]

            occupied = [i for i in range(self.state_dim + 1)]
            seq = self._refill_seq(np.concatenate((last_state, np.array([last_action]))), occupied)

            temp = self._kernel_sa(seq, prot_a)
            betas = np.multiply(self.c[random_prot], self._kernel_sa(seq, prot_a))
            n_sa = np.sum(betas)
            if n_sa != 0 and not np.isnan(n_sa):
                betas /= float(n_sa)
                mu = betas.dot(prot_a[:, self.state_dim + 1: -1])
                diff = prot_a[:, self.state_dim + 1: -1] - mu

                cov = (diff * betas[:, None]).T.dot(diff)

                s_ind = np.random.choice(range(len(random_prot)), 1, p=betas)[0]
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
            random_prot = np.array([i for i in range(self.b)])
            np.random.shuffle(random_prot)
            random_prot = random_prot[:min(self.b, self.num_rand_proto)]

            prot_a = self.prot_array[random_prot]

            occupied = [i for i in range(self.action_ind, self.state_dim * 2 + 1)]
            seq = self._refill_seq(np.concatenate((np.array([last_action]), state)), occupied)
            betas = np.multiply(self.cr[random_prot], self._kernel_asp(seq, prot_a))
            n_spa = np.sum(betas)

            if n_spa != 0 and not np.isnan(n_spa):
                betas = betas / float(n_spa)
            else:
                return None

            mu = betas.dot(prot_a[:, :self.state_dim])
            shape = self.state_dim # s
            cov = np.zeros((shape, shape))
            not_zero = np.where(betas != 0)[0]
            diff = prot_a[:, :self.state_dim] - mu

            cov = (diff * betas[:, None]).T.dot(diff) / float(self.b)#n_spa

            s_ind = np.random.choice(range(len(random_prot)), 1, p=betas)[0]
            target_mu = prot_a[s_ind, :self.state_dim]
            sampled_s = np.clip(np.random.multivariate_normal(target_mu, cov + np.eye(cov.shape[0]) * self.kscale), 0., 1.)

            return sampled_s #sampled_s_list
    # def _sample_s(self, last_action, state):
    #     if self.b == 0:
    #         return None
    #     else:
    #         occupied = [i for i in range(self.action_ind, self.state_dim * 2 + 1)]
    #         seq = self._refill_seq(np.concatenate((np.array([last_action]), state)), occupied)
    #         betas = np.multiply(self.cr[:self.b], self._kernel_asp(seq, self.prot_array[:self.b]))
    #         n_spa = np.sum(betas)
    #
    #         if n_spa != 0 and not np.isnan(n_spa):
    #             betas = betas / float(n_spa)
    #         else:
    #             return None
    #
    #         mu = betas.dot(self.prot_array[:self.b, :self.state_dim])
    #         shape = self.state_dim # s
    #         cov = np.zeros((shape, shape))
    #         not_zero = np.where(betas != 0)[0]
    #         diff = self.prot_array[:self.b, :self.state_dim] - mu
    #
    #         cov = (diff * betas[:, None]).T.dot(diff) / float(self.b)#n_spa
    #
    #         s_ind = np.random.choice(range(self.b), 1, p=betas)[0]
    #         target_mu = self.prot_array[s_ind, :self.state_dim]
    #         sampled_s = np.clip(np.random.multivariate_normal(target_mu, cov + np.eye(cov.shape[0]) * self.kscale), 0., 1.)
    #
    #         return sampled_s #sampled_s_list

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
        seq = np.outer(np.ones(len(same_a)), seq)
        seq_ssprg = seq[:, indices]
        proto_ssprg = protos[same_a][:, indices]
        diffs = seq_ssprg - proto_ssprg
        _, covmat_inv = self._cal_covmat_inv(indices)
        k_ssprg = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
        k[same_a] = k_ssprg
        return k

    def _kernel_sa(self, seq, protos):
        indices = [i for i in range(self.state_dim)]
        k = np.zeros((len(protos)))

        same_a = np.where(seq[self.action_ind] == protos[:, self.action_ind])[0]
        if len(same_a) == 0:
            return k

        seq = np.outer(np.ones(len(same_a)), seq)
        seq_s = seq[:, indices]
        protos_s = protos[same_a][:, indices]
        diffs = seq_s - protos_s
        _, covmat_inv = self._cal_covmat_inv(indices)
        k_s = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))

        k[same_a] = k_s
        if np.where(k_s > 1)[0].shape[0] != 0:
            print("large kernel", k)
            print("            ", diffs)
            print("            ", covmat_inv)
        return k

    def _kernel_asp(self, seq, protos):
        indices = [i for i in range(self.state_dim + 1, self.state_dim * 2 + 1)]
        k = np.zeros((len(protos)))

        same_a = np.where(seq[self.action_ind] == protos[:, self.action_ind])[0]
        if len(same_a) == 0:
            return k

        seq = np.outer(np.ones(len(same_a)), seq)
        seq_sp = seq[:, indices]
        protos_sp = protos[same_a][:,  indices]
        diffs = seq_sp - protos_sp
        _, covmat_inv = self._cal_covmat_inv(indices)
        k_asp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
        k[same_a] = k_asp
        return k

    def _kernel_sprg(self, seq, protos):
        indices = [i for i in range(self.state_dim + 1, self.seq_dim - 1)]
        seq = np.outer(np.ones(len(protos)), seq)
        seq_spr = seq[:, indices]
        protos_spr = protos[:, indices]
        diffs = seq_spr - protos_spr
        _, covmat_inv = self._cal_covmat_inv(indices)
        k_sprg = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
        return k_sprg

    def _kernel_s(self, seq, protos):
        indices = [i for i in range(self.state_dim)]
        seq = np.outer(np.ones(len(protos)), seq)
        seq_s = seq[:, indices]
        protos_s = protos[:, indices]
        diffs = seq_s - protos_s
        _, covmat_inv = self._cal_covmat_inv(indices)
        k_s = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
        return k_s

    def _kernel_sp(self, seq, protos):
        indices = [i for i in range(self.state_dim + 1, self.state_dim * 2 + 1)]
        seq = np.outer(np.ones(len(protos)), seq)
        seq_sp = seq[:, indices]
        protos_sp = protos[:, indices]
        diffs = seq_sp - protos_sp
        _, covmat_inv = self._cal_covmat_inv(indices)
        k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
        return k_sp

    def _kernel_a(self, seq, protos):
        seq = np.outer(np.ones(len(protos)), seq)
        seq_a = seq[:, self.action_ind]
        protos_a = protos[:, self.action_ind]
        return seq_a == protos_a

    def _cal_covmat_inv(self, indices):
        covmat_inv = 1.0 / self.kscale * np.eye(len(indices))
        # covmat = self.sig_prot[indices][:, indices]
        # covmat_inv = self.sig_prot_inv[indices][:, indices]
        return None, covmat_inv

    def _sa_n_near(self, seq, prot_a):
        ker_list = self._kernel_sa(seq, prot_a)
        dist_list = 1.0 - ker_list
        nnear_ind = np.argsort(dist_list)[:min(len(dist_list), self.num_near)]
        return prot_a[nnear_ind], ker_list[nnear_ind], nnear_ind

    def _asp_n_near(self, seq, prot_a):
        ker_list = self._kernel_asp(seq, prot_a)
        dist_list = 1.0 - ker_list
        nnear_ind = np.argsort(dist_list)[:min(len(dist_list), self.num_near)]
        return prot_a[nnear_ind], ker_list[nnear_ind], nnear_ind
    # def _asp_n_near(self, seq):
    #     ker_list = self._kernel_asp(seq, self.prot_array[:self.b])
    #     dist_list = 1.0 - ker_list
    #     nnear_ind = np.argsort(dist_list)[:min(len(dist_list), self.num_near)]
    #     return self.prot_array[nnear_ind], ker_list[nnear_ind], nnear_ind

    def _prob_a_sp(self, last_action, state):
        random_prot = np.array([i for i in range(self.b)])
        np.random.shuffle(random_prot)
        random_prot = random_prot[:min(self.b, self.num_rand_proto)]

        prot_a = self.prot_array[random_prot]
        
        occupied = [i for i in range(self.action_ind, self.state_dim * 2 + 1)]
        seq = self._refill_seq(np.concatenate((np.array([last_action]), state)), occupied)
        sum_prob_asp = np.sum(self._kernel_asp(seq, prot_a))
        sum_prob_sp = np.sum(self._kernel_sp(seq, prot_a))
        if sum_prob_sp == 0:
            return 1
        else:
            return sum_prob_asp / sum_prob_sp
    # def _prob_a_sp(self, last_action, state):
    #     occupied = [i for i in range(self.action_ind, self.state_dim * 2 + 1)]
    #     seq = self._refill_seq(np.concatenate((np.array([last_action]), state)), occupied)
    #     sum_prob_asp = np.sum(self._kernel_asp(seq, self.prot_array[:self.b]))
    #     sum_prob_sp = np.sum(self._kernel_sp(seq, self.prot_array[:self.b]))
    #     if sum_prob_sp == 0:
    #         return 1
    #     else:
    #         return sum_prob_asp / sum_prob_sp

    def get_protos(self):
        return self.prot_array[:self.b]

    def get_len_protos(self):
        return self.b