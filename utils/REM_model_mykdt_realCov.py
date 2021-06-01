import numpy as np
import random
import time
# import sklearn.neighbors as skln
import KDTree_Cpp.pieKDTree as kdt

'''
when insert data / rebuild tree, combine seq and index together
'''


class REM_Model:
    def __init__(self, state_dim, num_near, add_prot_limit, model_params):
        self.state_dim = state_dim
        self.seq_dim = state_dim * 2 + 3  # s, a, s', r, gamma
        self.num_action = 4
        self.action_ind = state_dim
        self.index_no_ag = [i for i in range(self.seq_dim)]
        self.index_no_ag.remove(self.action_ind)

        self.s_and_idx = [i for i in range(self.state_dim)] + [self.seq_dim]
        self.sp_and_idx = [i for i in range(self.state_dim+1, self.state_dim*2+1)] + [self.seq_dim]
        self.sprg_and_idx = [i for i in range(self.state_dim+1, self.seq_dim)] + [self.seq_dim]
        self.ssprg_and_idx = self.index_no_ag + [self.seq_dim]

        # the action and gamma are not included
        self.sum_array = np.zeros((self.seq_dim))
        self.sig_prot = np.zeros((self.seq_dim, self.seq_dim))
        self.sig_prot_first = np.zeros((self.seq_dim, self.seq_dim))
        self.sig_prot_inv = np.zeros((self.seq_dim, self.seq_dim))

        self.prot_len = 10000
        self.c = np.ones((self.prot_len))
        self.cr = np.ones((self.prot_len))
        self.prot_array = np.empty((self.prot_len, self.seq_dim))
        self.prot_array_action = []
        for _ in range(self.num_action):
            self.prot_array_action.append(np.empty((self.prot_len, self.seq_dim+1)))

        self.b = 0  # number of prototype in model
        self.b_action = np.zeros(self.num_action)
        self.t = 0  # number of sequence sent to model (added & not added)
        self.kscale = model_params["kscale"]

        self.add_prot_limit = add_prot_limit
        self.num_near = num_near
        self.kernel_near = 250

        self.s_tree = []
        self.sp_tree = []
        self.sprg_tree = []
        self.ssprg_tree = []

        for _ in range(self.num_action):
            self.s_tree.append(kdt.KDTree())
            self.sp_tree.append(kdt.KDTree())
            self.sprg_tree.append(kdt.KDTree())
            self.ssprg_tree.append(kdt.KDTree())
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
        self.sum_array += seq
        mu_prot = self.sum_array / float(self.t)
        self.sig_prot_first = ((self.t - 1.0) * self.sig_prot_first + np.outer(seq, seq)) / float(self.t)
        self.sig_prot = self.sig_prot_first - np.outer(mu_prot, mu_prot)
        self.sig_prot_inv = np.linalg.inv(self.sig_prot + np.eye(len(seq)) * self.kscale)
        return

    def _k_near_neighbor(self, tree, target, covmat_inv, ksize):
        res = tree.kNN(target, ksize, covmat_inv)
        seqs = res[:, :len(target)]
        dist = res[:, len(target)]
        index = res[:, -1].astype(int)
        return seqs, index, dist


    # Algorithm 8
    def _update_prototype(self, seq, num_near):
        add_prot = False
        crt_a = int(seq[self.action_ind])
        if self.b == 0:
            add_prot = True
        else:
            tree = self.ssprg_tree[crt_a]
            same_a = self.b_action[crt_a]
            if same_a >= num_near:
                _, covmat_inv = self._cal_covmat_inv(self.index_no_ag)
                _, _, dist = self._k_near_neighbor(tree, seq[self.index_no_ag], covmat_inv, num_near)
                if dist[0] > self.add_prot_limit:
                    add_prot = True
            else:
                add_prot = True

        if add_prot:
            if self.b < self.prot_len:
                self.prot_array[self.b] = seq
            else:
                self.prot_array = np.concatenate((self.prot_array, np.array([seq])), axis=0)

            tree_size = int(self.b_action[crt_a])
            seq_and_idx = np.concatenate((seq, np.array([self.b])), axis=0)
            if tree_size < self.prot_len:
                self.prot_array_action[crt_a][tree_size] = seq_and_idx
            else:
                self.prot_array_action[crt_a] = \
                    np.concatenate((self.prot_array_action[seq[int(self.action_ind)]], np.array([seq_and_idx])), axis=0)

            self.b += 1
            self.b_action[crt_a] += 1

            if self.b_action[crt_a] % 300 == 1:
                self._rebuild_kdt(crt_a)
                # self._check_running_time()
            else:
                self.s_tree[crt_a].Insert(seq_and_idx[self.s_and_idx])
                self.sp_tree[crt_a].Insert(seq_and_idx[self.sp_and_idx])
                self.sprg_tree[crt_a].Insert(seq_and_idx[self.sprg_and_idx])
                self.ssprg_tree[crt_a].Insert(seq_and_idx[self.ssprg_and_idx])

        return

    def _rebuild_kdt(self, crt_a):
        # init an empty tree
        self.s_tree[crt_a] = kdt.KDTree()
        self.sp_tree[crt_a] = kdt.KDTree()
        self.sprg_tree[crt_a] = kdt.KDTree()
        self.ssprg_tree[crt_a] = kdt.KDTree()

        next_empty_pos = int(self.b_action[crt_a])
        self.s_tree[crt_a].BuildTree(self.prot_array_action[crt_a][:next_empty_pos, self.s_and_idx])
        self.sp_tree[crt_a].BuildTree(self.prot_array_action[crt_a][:next_empty_pos, self.sp_and_idx])
        self.sprg_tree[crt_a].BuildTree(self.prot_array_action[crt_a][:next_empty_pos, self.sprg_and_idx])
        self.ssprg_tree[crt_a].BuildTree(self.prot_array_action[crt_a][:next_empty_pos, self.ssprg_and_idx])

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
            # shape = self.state_dim  # s
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

            return k

        else:
            seq_s = np.outer(np.ones(self.kernel_near), seq)[:, indices]
            _, covmat_inv = self._cal_covmat_inv(indices)
            tree = self.s_tree[int(seq[self.action_ind])]
            neighbor, index, dist = self._k_near_neighbor(tree, seq[indices], covmat_inv, self.kernel_near)
            diffs = seq_s - neighbor
            k_s = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            k[index] = k_s
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
            seq_sp = np.outer(np.ones(self.kernel_near), seq)[:, indices]
            _, covmat_inv = self._cal_covmat_inv(indices)
            tree = self.sp_tree[int(self.action_ind)]
            neighbor, index, dist = self._k_near_neighbor(tree, seq[indices], covmat_inv, self.kernel_near)
            diffs = seq_sp - neighbor
            k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            k[index] = k_sp
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
            seq_spr = np.outer(np.ones(self.kernel_near), seq)[:, indices]
            _, covmat_inv = self._cal_covmat_inv(indices)
            tree = self.sprg_tree[int(self.action_ind)]
            neighbor, index, dist = self._k_near_neighbor(tree, seq[indices], covmat_inv, self.kernel_near)
            diffs = seq_spr - neighbor
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
            seq_s = np.outer(np.ones(self.kernel_near), seq)[:, indices]
            _, covmat_inv = self._cal_covmat_inv(indices)
            tree = self.s_tree[int(self.action_ind)]
            neighbor, index, dist = self._k_near_neighbor(tree, seq[indices], covmat_inv, self.kernel_near)
            diffs = seq_s - neighbor
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
            seq_sp = np.outer(np.ones(self.kernel_near), seq)[:, indices]
            _, covmat_inv = self._cal_covmat_inv(indices)
            tree = self.sp_tree[int(self.action_ind)]
            neighbor, index, dist = self._k_near_neighbor(tree, seq[indices], covmat_inv, self.kernel_near)
            diffs = seq_sp - neighbor
            k_sp = np.exp(-1 * np.diag(diffs.dot(covmat_inv).dot(diffs.T)))
            k[index] = k_sp
            return k

    def _cal_covmat_inv(self, indices):
        covmat = self.sig_prot[indices][:, indices]
        covmat_inv = self.sig_prot_inv[indices][:, indices]
        return covmat, covmat_inv

    def _sa_n_near(self, seq, prot_a):
        indices = [i for i in range(self.state_dim)]
        same_a = np.where(prot_a[:, self.action_ind] == seq[self.action_ind])[0]

        if len(same_a) == 0:
            return prot_a[:self.kernel_near], np.zeros(self.kernel_near), [i for i in range(self.kernel_near)]

        elif len(same_a) < self.kernel_near:
            ker_list = self._kernel_sa(seq, prot_a)
            dist_list = 1.0 - ker_list
            nnear_ind = np.argsort(dist_list)[:min(len(dist_list), self.kernel_near)]
            return prot_a[nnear_ind], ker_list[nnear_ind], nnear_ind

        else:
            _, covmat_inv = self._cal_covmat_inv(indices)
            tree = self.s_tree[int(self.action_ind)]
            neighbor, index, dist = self._k_near_neighbor(tree, seq[indices], covmat_inv, self.kernel_near)
            return prot_a[index], 1.0 - dist, index

    def _asp_n_near(self, seq, prot_a):
        indices = [i for i in range(self.action_ind+1, self.state_dim*2+1)]
        same_a = np.where(prot_a[:, self.action_ind] == seq[self.action_ind])[0]

        if len(same_a) == 0:
            return prot_a[:self.kernel_near], np.zeros(self.kernel_near), [i for i in range(self.kernel_near)]

        elif len(same_a) < self.kernel_near:
            ker_list = self._kernel_asp(seq, prot_a)
            dist_list = 1.0 - ker_list
            nnear_ind = np.argsort(dist_list)[:min(len(dist_list), self.kernel_near)]
            return prot_a[nnear_ind], ker_list[nnear_ind], nnear_ind

        else:
            _, covmat_inv = self._cal_covmat_inv(indices)
            tree = self.sp_tree[int(self.action_ind)]
            neighbor, index, dist = self._k_near_neighbor(tree, seq[indices], covmat_inv, self.kernel_near)
            return prot_a[index], 1.0 - dist, index

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

    def _check_running_time(self):
        for i in range(self.num_action):
            print("action =", i)
            print("running time: s -", self.s_tree[i].check_time())
            print("running time: sp -", self.sp_tree[i].check_time())
            print("running time: sprg -", self.sprg_tree[i].check_time())
            print("running time: ssprg -", self.ssprg_tree[i].check_time())
