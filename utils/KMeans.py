import numpy as np
import random
import sys

class KMeans:
    def __init__(self, k, sampleDim, stateDim):
        self.centroids = None
        self.lastcentroids = None
        self.newfit = False
        self.labels_ = None
        self.acts_centroids = None
        self.ssr_centroids = None
        self.n_clusters = k
        self.act_i = stateDim
        self.ssr_i = [i for i in range(sampleDim)]
        self.ssr_i.remove(stateDim)
        self.covmat_inv = None
        self.max_iters = 500
        self.num_kmeans = 2
        self.num_max_kmeans = 10 

    '''distance is defined as 1.0 - kernel value'''
    def kern_distance(self, x, y):
        diff = x - y
        diff = diff[self.ssr_i]
        #return 1.0 - a_indicator * np.exp(-0.5*(diff*self.covmat_inv).T.dot(diff))
        return 1.0 - (x[self.act_i] == y[self.act_i])*np.exp(-0.5*diff.T.dot(self.covmat_inv).dot(diff))

    def initialize_centroids(self, points):
        """returns k centroids from the initial points"""
        if self.lastcentroids is not None and self.newfit:
            self.centroids = self.lastcentroids
            self.lastcentroids = self.centroids + 10
            return
        centroids = points.copy()
        np.random.shuffle(centroids)
        self.centroids = centroids[:self.n_clusters]
        self.lastcentroids = self.centroids + 10

    def closest_centroid(self, points):
        '''returns an array containing the index to the nearest centroid for each point'''
        ssr_points, ssr_centroids = points[:, self.ssr_i], self.centroids[:, self.ssr_i]
        act_points, act_centroids = points[:, self.act_i], self.centroids[:, self.act_i]
        '''Possible bug here'''
        SSpR_diff = ssr_points - ssr_centroids[:, np.newaxis]
        #SSpR_distances = np.exp(-0.5*(((ssr_points - ssr_centroids[:, np.newaxis])**2)*self.covmat_inv).sum(axis=2))
        SSpR_distances = np.exp(-0.5 * (SSpR_diff.dot(self.covmat_inv)*SSpR_diff).sum(axis=2))
        act_distances = (act_points != act_centroids[:, np.newaxis])
        SSpR_distances[act_distances] = 0.0
        distances = 1.0 - SSpR_distances
        self.labels_ = np.argmin(distances, axis=0)
        #print distances
        #print np.std(distances.reshape(-1)), np.min(distances), np.max(distances)
        #print 'The statistics are: '
        '''
        unique, counts = np.unique(self.labels_, return_counts=True)
        checkarr = np.zeros(self.n_clusters)
        checkarr[unique] = 1.0
        if np.any(checkarr == 0.0):
            nonind = np.argmin(checkarr)
            mininds = np.argsort(distances[nonind, :])
            self.labels_[mininds[0]] = nonind
        #print dict(zip(unique, counts))
        #print self.labels_
        '''
    """ ??? """
    def move_centroids(self, points):
        """returns the new centroids assigned from the points closest to them"""
        self.lastcentroids = self.centroids
        #self.centroids = np.array([points[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)])
        centroids = []
        for k in range(self.n_clusters):
            if len(points[self.labels_ == k]) != 0:
                temp = points[self.labels_ == k]
                centroids.append(points[self.labels_ == k].mean(axis=0))
            else:
                centroids.append([0 for _ in temp.mean(axis=0)])
                print("DIVIDED BY 0")
        self.centroids = np.array(centroids)

    def naive_clustering(self, points):
        self.labels_ = np.random.randint(self.n_clusters, size=points.shape[0]) 
        self.labels_[:self.n_clusters] = np.arange(0, self.n_clusters, dtype=int)
        self.centroids = np.array([points[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)])

    def fit(self, data, covmatinv):
        self.covmat_inv = covmatinv
        best_var = sys.maxsize#sys.maxint
        bestlabels = None
        bestcentroids = None
        self.newfit = True
        counts = [0]*self.n_clusters
        i = 0
        while i < self.num_kmeans or len(counts)<self.n_clusters:
            if i >= self.num_max_kmeans:
                self.naive_clustering(data)
                bestlabels = self.labels_
                bestcentroids = self.centroids
                break
            self.initialize_centroids(data)
            n_iter = 0
            while np.linalg.norm(self.lastcentroids-self.centroids) > 0.01 and n_iter<self.max_iters:
                self.closest_centroid(data)
                self.move_centroids(data)
                n_iter += 1
            _, counts = np.unique(self.labels_, return_counts=True)
            curstd = np.std(counts)
            if curstd < best_var and len(counts) == self.n_clusters:
                best_var = curstd
                bestlabels = self.labels_
                bestcentroids = self.centroids
            i += 1
            self.newfit = False
        self.labels_ = bestlabels
        self.centroids = bestcentroids
        self.acts_centroids = self.centroids[:,self.act_i]
        self.ssr_centroids = self.centroids[:,self.ssr_i]
        self.lastcentroids = self.centroids.copy()

    '''Note can only predict one sample currently'''
    def predict(self, sample):
        sample_ssr = sample[self.ssr_i]
        sample_a = sample[self.act_i]
        diff_ssr = sample_ssr - self.ssr_centroids
        similarities = np.exp(-0.5*np.sum(diff_ssr.dot(self.covmat_inv)*diff_ssr, axis=1))*(self.acts_centroids == sample_a)
        #if sample[4]>1:
             #print diff_ssr
        #     print np.exp(-0.5*np.sum(diff_ssr.dot(self.covmat_inv)*diff_ssr, axis=1))
        #     print 'similarity is '+str(np.max(similarities))
        #zero_is = np.where(similarities == 0)
        #if len(zero_is) > 1:
            #print zero_is
        #    return random.choice(zero_is)
        return np.argmax(similarities)
