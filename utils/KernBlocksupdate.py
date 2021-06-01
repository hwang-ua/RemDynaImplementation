import numpy as np
import math
import sys
import utils.KMeans as km
#from sklearn.cluster import KMeans
from sklearn import cluster
#from scipy.cluster.hierarchy import fclusterdata
import utils.logdetutils as ut

'''Protocal: all samples are column vector'''
KernValConst = 0.000001

class KernBlocks:
    #pass in set of centers and kmeans result
    def __init__(self, n_clusters, stateDim, sampleDim, threshold):
        self.n_clusters = n_clusters
        #maintain three mappings
        self.kmats_inv = []
        '''NOTE: kmats do not use regularizer self.lam'''
        self.kmats = []
        self.kmats_score = np.zeros(n_clusters)
        self.min_scores = []
        self.cluster_ids = {}
        self.cluster_samples = {}
        self.threshold = threshold
        self.labels = None
        self.data = None
        # submodular score regularizer
        self.lam = 1.0
        # the larger the kscale, the larger the bandwidth
        self.kscale = 5.0/n_clusters
        # covariance matrix
        self.covmat_inv = None
        self.stateDim = stateDim
        self.sampleDim = sampleDim
        self.SSRGIndex = [i for i in range(sampleDim)]
        self.SSRGIndex.remove(stateDim)
        self.cluster_model = km.KMeans(self.n_clusters, self.sampleDim, self.stateDim)

    def Build_blocks(self, data, covmat):
        #print 'clustering performed!!!!!!!!!!!------------- - -- - -- - - - -'
        self.kmats_inv = []
        self.kmats = []
        self.min_scores = []
        self.cluster_ids = {}
        self.cluster_samples = {}
        self.data = data
        #self.covmat_inv = np.linalg.inv((covmat + np.eye(covmat.shape[0])*0.0001)*self.kscale)
        # remove the last row & column of covmat ????
        tempSSRGSig = ut.remove_rc(covmat, self.stateDim)
        #self.covmat_inv = np.diag(np.linalg.inv((tempSSRSig + np.eye(tempSSRSig.shape[0])*0.0001)*self.kscale))
        self.covmat_inv = np.linalg.inv((tempSSRGSig + np.eye(tempSSRGSig.shape[0])*0.0001)*self.kscale)
        #print covmat
        #do clustering
        #self.cluster_model = km.KMeans(self.n_clusters, self.sampleDim, self.stateDim)
        self.cluster_model.fit(data, self.covmat_inv)
        #self.cluster_model = cluster.SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed').fit(self.construct_K(data))
        labels = self.cluster_model.labels_
        #print labels
        for i in range(self.n_clusters):
            #print len(self.cluster_indexes(labels, i))
            mat, mat_inv = self.Build_KMatInv(data[labels==i,:])
            self.kmats_inv.append(mat_inv)
            self.kmats.append(mat)
            self.cluster_ids[str(i)] = list(self.cluster_indexes(labels, i))
        for i in range(len(labels)):
            if str(labels[i]) not in self.cluster_samples:
                self.cluster_samples[str(labels[i])] = []
            self.cluster_samples[str(labels[i])].append(data[i,:].T.copy())
        self.init_scores(labels)
        #self.build_K()

    def cluster_indexes(self, labels, cluster_id):
        return np.where(labels == cluster_id)[0]

    #init min score of each cluster
    def init_scores(self, labels):
        for i in range(self.n_clusters):
            if len(labels[labels == i]) == 1:
                #if there is only one sample, set the loss huge so it will not be replaced
                self.min_scores.append([-1, sys.maxsize])#sys.maxint])
                #self.kmats_score[i] = -math.log(np.linalg.det(self.kmats_inv[i]))
                continue
            min_ind, min_loss = self.find_min_score(i)
            self.min_scores.append([min_ind, min_loss])
            #self.kmats_score[i] = -math.log(np.linalg.det(self.kmats_inv[i]))

    def find_min_score(self, block_id):
        blockmat_inv = self.kmats_inv[block_id]
        cluster_ids = self.cluster_ids[str(block_id)]
        min_ind, min_loss = -1, sys.maxsize#sys.maxint
        if len(cluster_ids) <= 1:
            return (min_ind, min_loss)
        for j in range(len(cluster_ids)):
            old_kernsample = self.kmats[block_id][j,:]
            #old_kernsample = self.kernSample(centers[j], np.array(centers))
            loss = -ut.logdet_delete(blockmat_inv, old_kernsample, j, self.lam)
            (min_ind, min_loss) = (cluster_ids[j], loss) if loss < min_loss else (min_ind, min_loss)
        return (min_ind, min_loss)

    def Build_KMatInv(self, samples):
        kernmat = np.zeros((samples.shape[0], samples.shape[0]))
        for i in range(samples.shape[0]):
            kernmat[i, i] = 1.0 + self.lam
            for j in range(i+1, samples.shape[0], 1):
                kernmat[i, j] = self.kernFunc(samples[i,:].T, samples[j,:].T)
                kernmat[j, i] = kernmat[i, j]
        #print kernmat
        return (kernmat - np.eye(kernmat.shape[0])*self.lam, np.linalg.inv(kernmat))

    '''return the best gain by replacing a center; and the gain of simply adding the sample'''
    def replace_add_gain(self, index, sample):
        kernsample = self.kernSample(sample, np.array(self.cluster_samples[str(index)]))
        add_gain = ut.logdet_attach(self.kmats_inv[index], kernsample, self.lam)
        id, replace_gain = self.find_to_replace(index, sample)
        replace_id = self.cluster_ids[str(index)][id] if id>=0 else id
        return (replace_id, replace_gain, add_gain)

    '''find the best one to replace in the closest block'''
    def find_to_replace(self, index, sample):
        id, max_gain = -1, -sys.maxsize#-sys.maxint
        if len(self.cluster_ids[str(index)]) <= 1:
            return (id, max_gain)
        newkern_sample = self.kernSample(sample, np.array(self.cluster_samples[str(index)]))
        for i in range(len(self.cluster_ids[str(index)])):
            oldkern_sample = self.kmats[index][i,:]
            newcopy = newkern_sample.copy()
            newcopy[i] = oldkern_sample[i]
            gain = ut.logdet_replace(self.kmats_inv[index], oldkern_sample, newcopy, i)
            if gain > max_gain:
                id, max_gain = i, gain
        return (id, max_gain) if max_gain > self.threshold else (-1, -sys.maxsize)#-sys.maxint)

    # add sample to the block_id block
    def block_add(self, block_id, data_id, sample):
        new_kernsample = self.kernSample(sample, np.array(self.cluster_samples[str(block_id)]))
        '''note'''
        #self.kmats_score[block_id] += ut.logdet_attach(self.kmats_inv[block_id], new_kernsample, self.lam)
        self.kmats_inv[block_id] = ut.kmat_inv_attach(self.kmats_inv[block_id], new_kernsample, self.lam)
        self.kmats[block_id] = ut.kmat_attach(self.kmats[block_id], new_kernsample)
        self.cluster_samples[str(block_id)].append(sample)
        self.cluster_ids[str(block_id)].append(data_id)
        min_ind, min_loss = self.find_min_score(block_id)
        self.min_scores[block_id] = [min_ind, min_loss]

    def block_delete(self, block_id, data_id):
        #print 'data id is ----------------- ' + str(data_id)
        id_in_block = self.cluster_ids[str(block_id)].index(data_id)
        self.cluster_ids[str(block_id)].remove(data_id)
        #old_sample = self.cluster_samples[str(block_id)][id_in_block].copy()
        #old_kernsample = self.kernSample(old_sample, np.array(self.cluster_samples[str(block_id)]))
        old_kernsample = self.kmats[block_id][id_in_block,:]
        '''note'''
        #self.kmats_score[block_id] += ut.logdet_delete(self.kmats[block_id], old_kernsample, id_in_block, self.lam)
        del self.cluster_samples[str(block_id)][id_in_block]
        self.kmats_inv[block_id] = ut.kmat_inv_delete(self.kmats_inv[block_id], old_kernsample, id_in_block)
        self.kmats[block_id] = ut.remove_rc(self.kmats[block_id], id_in_block)
        min_ind, min_loss = self.find_min_score(block_id)
        self.min_scores[block_id] = [min_ind, min_loss]

    def block_replace(self, block_id, data_id, sample):
        #print self.cluster_ids[str(block_id)]
        id_in_block = self.cluster_ids[str(block_id)].index(data_id)
        #old_sample = self.cluster_samples[str(block_id)][id_in_block]
        #old_kernsample = self.kernSample(old_sample, np.array(self.cluster_samples[str(block_id)]))
        old_kernsample = self.kmats[block_id][id_in_block,:]
        self.cluster_samples[str(block_id)][id_in_block] = sample
        new_kernsample = self.kernSample(sample, np.array(self.cluster_samples[str(block_id)]))
        '''note'''
        #self.kmats_score[block_id] += ut.logdet_replace(self.kmats[block_id], old_kernsample, new_kernsample, id_in_block)
        self.kmats_inv[block_id] = ut.kmat_inv_replace(self.kmats_inv[block_id], old_kernsample, new_kernsample, id_in_block)
        self.kmats[block_id] = ut.kmat_replace(self.kmats[block_id], new_kernsample, id_in_block)
        min_ind, min_loss = self.find_min_score(block_id)
        self.min_scores[block_id] = [min_ind, min_loss]

    '''return kern repres of a sample, diagonal covariance'''
    #def kernSample(self, sample, centers):
    #    return np.exp(-0.5*np.sum(((sample[self.SSRIndex] - centers[:,self.SSRIndex])**2)*self.covmat_inv, axis=1))*(sample[self.stateDim] == centers[:,self.stateDim])

    '''Note in use, map is slower'''
    #def kernSample(self, sample, centers):
    #    return np.array(map(lambda center: self.kernFunc(sample, center), centers))

    '''non-diagonal covariance case'''
    def kernSample(self, sample, centers):
        diffs = sample - centers
        diffs = diffs[:,self.SSRGIndex]
        return np.exp(-0.5 * np.sum(diffs.dot(self.covmat_inv) * diffs, axis=1))*(sample[self.stateDim]==centers[:,self.stateDim])

    '''
    def kernFunc(self, sx, sy):
        diff = sx - sy
        kv = math.exp(-0.5*diff.T.dot(self.covmat_inv).dot(diff))
        return kv if kv > 0.000001 else 0
    '''

    def kernFunc(self, x, y):
        ax = x[self.stateDim]
        ay = y[self.stateDim]
        diff = x - y
        diff = diff[self.SSRGIndex]
        #kv = np.exp(-0.5*(diff*self.covmat_inv).T.dot(diff))*int(ax == ay)
        kv = np.exp(-0.5 * diff.T.dot(self.covmat_inv).dot(diff)) * int(ax == ay)
        return kv

    def construct_K(self, samples):
        kernmat = np.zeros((samples.shape[0], samples.shape[0]))
        for i in range(samples.shape[0]):
            #no regularizer here
            kernmat[i, i:] = self.kernSample(samples[i,:], samples[i:,:])
            #kernmat[i, i] = 1.0
            for j in range(i + 1, samples.shape[0], 1):
            #    kernmat[i, j] = self.kernFunc(samples[i, :].T, samples[j, :].T)
                kernmat[j, i] = kernmat[i, j]
        #print kernmat
        return kernmat

    def spectral_predict(self, sample):
        similarities = np.array(map(lambda blockid: np.mean(self.kernSample(sample, np.array(self.cluster_samples[str(blockid)]))), range(self.n_clusters)))
        closestInd = np.argmax(similarities)
        return closestInd

    '''higher level interface: pass in the sample and decide whether or what to replace'''
    '''replace id will be negative if do not accept the sample'''
    def maximize(self, sample):
        #print '\n'
        # find out what cluster our new sample belongs to
        cluster_id = self.cluster_model.predict(sample)
        #print 'predicted cluster index is ' + str(cluster_id)
        #cluster_id = self.spectral_predict(sample)
        delete_block_id = cluster_id
        # find out gain if replacement or add in closest cluster
        replace_id, replace_gain, add_gain = self.replace_add_gain(cluster_id, sample)
        #print replace_id, replace_gain, add_gain
        #print self.min_scores
        # find out the replace_id th center to do replacement
        for i in range(self.n_clusters):
            if i != cluster_id:
                [min_id, min_loss] = self.min_scores[i]
                gain = add_gain - min_loss
                if gain > replace_gain and gain > self.threshold:
                    replace_id, replace_gain, delete_block_id = min_id, gain, i
        #print replace_id, replace_gain, delete_block_id
        # check whether the amount of gain reaches the threshold
        if replace_gain > self.threshold:
            #self.data[replace_id, :] = sample
            #self.update_K(replace_id)
            #print self.fullscore
            if cluster_id == delete_block_id:
                self.block_replace(cluster_id, replace_id, sample)
            else:
                # replace id will be assigned to the added sample
                self.block_add(cluster_id, replace_id, sample)
                # auto replace the most useless sample in the block_id block
                self.block_delete(delete_block_id, replace_id)
            #print 'the total score is ' + str(self.direct_score_sum())         
        #if replace_id >= 0 and replace_gain < self.threshold:
        #    print 'error!!!!!!!!!!!!!!!!!!!-------------------------********'
        #    print replace_id, replace_gain
        return replace_id

    def build_K(self):
        self.K = np.zeros((self.data.shape[0], self.data.shape[0]))
        for i in range(self.data.shape[0]):
            self.K[i, i] = 1.0 + self.lam
            for j in range(i + 1, self.data.shape[0], 1):
                self.K[i, j] = self.kernFunc(self.data[i, :], self.data[j, :])
                self.K[j, i] = self.K[i, j]
        #self.fullscore = math.log(np.linalg.det(self.K))

    def update_K(self, replace_id):
        kernsample = self.kernSample(self.data[replace_id,:], self.data)
        self.K[replace_id, :] = kernsample.T
        self.K[:, replace_id] = kernsample
        self.K[replace_id, replace_id] += self.lam
        self.fullscore = math.log(np.linalg.det(self.K))

    def blocks_score_sum(self):
        return np.sum(self.kmats_score)

    def direct_score_sum(self):
        mysum = 0
        for i in range(self.n_clusters):
            mysum += -math.log(np.linalg.det(self.kmats_inv[i]))
        return mysum
