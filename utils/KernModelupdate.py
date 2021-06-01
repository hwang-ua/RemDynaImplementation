import math
import random
import numpy as np
from numpy import linalg as LA
import utils.KernBlocksupdate as KB

'''global parameters for KDE model'''
smoothingparam = 0.001 #0.001 #need automatic way of tuning
threshold = 0.001 # for accepting new prototypes. need automatic way of adaping
KDESmallConst = 10**(-4)

def tupleInList(SARS, l):
    if len(l) == 0:
        return False
    S, a, r, Sp,gamma = SARS
    b = True
    for sars in l:
        Sb, ab, rb, Spb,gammab = sars
        b = b and np.all(S == Sb) and a == ab and r == rb and np.all(Spb == Sp) and gamma==gammab

    return b

############## KDE model code #################################################################
#sig_scale is used as scalor for computing kernel values#
class KernModel:
    def __init__(self, stateDim, memSize=100, blockSize=10, sig_scale = 0.01, scorethreshold = 0.01, usegamma = 1):
        self.stateDim = stateDim
        self.sampleDim = self.stateDim * 2 + 2 + usegamma
        self.sigSq = 0.1
        self.memSize = memSize

        self.sigmaS = np.eye(stateDim)
        self.scaled_sigmaS_inv = np.eye(stateDim)
        self.scaled_sigmaSp_inv = np.eye(stateDim)
        self.scaled_sigmaRG_inv = np.eye(2)

        self.sigmaSA = np.eye(stateDim)
        self.sigmaSR = np.eye(stateDim + 1)
        self.nonCondSigmaSR = np.eye(stateDim + 1)
        self.sigmaSSR_inv = np.eye(self.sampleDim-1)
        self.sigmaAllSq = np.eye(self.sampleDim)
        self.sigmaAll_first = np.zeros((self.sampleDim, self.sampleDim))
        self.sigmaAll = np.eye(self.sampleDim)
        self.muAll = np.zeros(self.sampleDim)

        self.SSRGInds = [i for i in range(self.sampleDim)]
        self.SSRGInds.remove(self.stateDim)
        self.SpRGInds = np.arange(self.stateDim+1, self.sampleDim)
        #print self.SpRGInds
        '''NOTE: scalor for scaling covariance matrix'''
        #self.kscale = 1.0/math.pow(self.memSize, 2.0/(self.stateDim+4.0))
        self.kscale = 1.0/float(self.memSize/(self.sampleDim-1))
        self.kscale = 1.0/1000.
        '''below variables are for submodular maximization'''
        self.blockSize = blockSize
        self.numClusters = int((math.ceil(memSize / self.blockSize)))
        self.SigProto = np.zeros((self.sampleDim, self.sampleDim))
        self.SigProto_first = np.zeros((self.sampleDim, self.sampleDim))
        self.muProto = np.zeros(self.sampleDim)
        self.initial = 'k-means++'
        # count number of centers
        self.n = 0
        self.t = 0 #num of steps will be increasing all the time
        self.clusterGap = self.numClusters
        self.clusterClock = self.clusterGap
        self.clustered = False
        self.threshold = scorethreshold
        self.data = np.zeros((memSize, self.sampleDim))

        #use for sample p(sp, r,g|s,a)
        self.piqi = np.ones(memSize) 
        #use for sample p(sp|s,a)
        self.piqi_sp = np.ones(memSize)
        #use for sample p(s|sp,a)
        self.pre_piqi = np.ones(memSize)
        #use for sample p(a|sp)
        self.apiqi = np.ones(memSize)
        #use for sample p(r,g|s,a,sp)
        self.piqi_rg = np.ones(memSize)
        #use for sample p(s|sp)
        self.piqi_s = np.ones(memSize)
        self.episode = 1.
        # This is used to sample (s, a)
        self.fixedRecentBuffer = []
        # init obj used to perform block operations
        self.blocks = KB.KernBlocks(self.numClusters, self.stateDim, self.sampleDim, self.threshold)
        #self.blocks = None

        self.kernel_gamma = sig_scale #0.012
        print('kernel gamma is :: ', self.kernel_gamma)

    '''pointer t will be increasing all the time'''
    def add2Model(self, S, A, Sp, R, gamma, episodeEnd = False):
        # update number of steps
        self.t += 1
        # update number of episode when one episode ends
        self.episode = self.episode + 1. if episodeEnd else self.episode
        # 
        replaceInd, oldsample = self.addCenter(S, A, Sp, R, gamma)
        if self.t > self.memSize:
            self.updatePiQi(S, A, Sp, R, gamma, replaceInd, oldsample)
        self.updateSigProto(S, A, Sp, R, gamma)
    
    def updatePiQi(self, S, A, Sp, R, gamma, replaceInd, oldsample):
        sDim = self.stateDim
        sample = self.sample2Array(S, A, Sp, R, gamma)
        curdata = self.data
        sprgk_new = self.kFuncCentersSSRG(curdata[:,self.SSRGInds], sample[self.SSRGInds]) 
        ak_new = 1.0*(curdata[:,sDim]==sample[sDim])
        sk_new = self.kFuncCentersS(curdata[:,:sDim], sample[:sDim])
        #compute self.piqi is for p(sp,r,g|s,a)
        sak_new = ak_new*sk_new
        rho = sak_new
        self.piqi = (1.0 - rho)*self.piqi + rho*sprgk_new
        #compute weights self.pre_piqi, compute p(s|sp,a)
        spk_new = self.kFuncCentersSp(curdata[:,sDim+1:2*sDim+1], sample[:sDim+1:2*sDim+1])
        spak_new = spk_new*ak_new
        rho = spak_new
        self.pre_piqi = (1.0 - rho)*self.pre_piqi + rho*sk_new
        #compute weights for self.apiqi is for p(a|sp)
        rho = spk_new
        self.apiqi = (1.0 - rho)*self.apiqi + rho*ak_new
        #compute weights self.piqi_sp for p(sp|s,a)
        rho = sak_new
        self.piqi_sp = (1.0 - rho)*self.piqi_sp + rho*spk_new
        #compute weights self.piqi_rg for p(r,g|s,a,sp)
        rho = sak_new*spk_new
        rgk_new = self.kFuncCentersRG(curdata[:,-2:], sample[-2:])
        self.piqi_rg = (1.0 - rho)*self.piqi_rg + rho*rgk_new
        #compute weights self.piqi_s for p(s|sp)
        rho = spk_new
        self.piqi_s = (1.0 - rho)*self.piqi_s + rho*sk_new
    '''
    def updatePiQi(self, S, A, Sp, R, gamma, replaceInd, oldsample):
        sDim = self.stateDim
        sample = self.sample2Array(S, A, Sp, R, gamma)
        curdata = self.data
        sprgk_new = self.kFuncCentersSSRG(curdata[:,self.SSRGInds], sample[self.SSRGInds]) 
        ak_new = 1.0*(curdata[:,sDim]==sample[sDim])
        sk_new = self.kFuncCentersS(curdata[:,:sDim], sample[:sDim])
        #compute self.piqi is for p(sp,r,g|s,a)
        sak_new = ak_new*sk_new
        rho = sak_new
        self.piqi = (1.0 - rho)*self.piqi + rho*sprgk_new*ak_new
        #compute weights self.pre_piqi, compute p(s|sp,a)
        spk_new = self.kFuncCentersSp(curdata[:,sDim+1:2*sDim+1], sample[:sDim+1:2*sDim+1])
        spak_new = spk_new*ak_new
        rho = spak_new
        self.pre_piqi = (1.0 - rho)*self.pre_piqi + rho*sk_new*rho
        #compute weights for self.apiqi is for p(a|sp)
        rho = spk_new
        self.apiqi = (1.0 - rho)*self.apiqi + rho*ak_new*rho
        #compute weights self.piqi_sp for p(sp|s,a)
        rho = sak_new
        self.piqi_sp = (1.0 - rho)*self.piqi_sp + rho*spk_new*rho
        #compute weights self.piqi_rg for p(r,g|s,a,sp)
        rho = sak_new*spk_new
        rgk_new = self.kFuncCentersRG(curdata[:,-2:], sample[-2:])
        self.piqi_rg = (1.0 - rho)*self.piqi_rg + rho*rgk_new*rho
        #compute weights self.piqi_s for p(s|sp)
        rho = spk_new
        self.piqi_s = (1.0 - rho)*self.piqi_s + rho*sk_new
    '''

    # called by self.addCenter()
    def update(self, old_sample = None, new_sample = None):
        if new_sample is None:
            return
        sDim = self.stateDim
        # when add center (not replace)
        if old_sample is None:
            # self.n - 1 ???
            self.sigmaAll_first = ((self.n - 1.0) * self.sigmaAll_first + np.outer(new_sample, new_sample)) / self.n
            self.muAll = ((self.n - 1.0) * self.muAll + new_sample) / self.n
        # when replace center
        else:
            self.sigmaAll_first = (self.n * self.sigmaAll_first - np.outer(old_sample, old_sample) + np.outer(new_sample, new_sample)) / self.n
            self.muAll = (self.n * self.muAll - old_sample + new_sample) / self.n
        self.sigmaAll = self.sigmaAll_first - np.outer(self.muAll, self.muAll)

        sigmaSA = self.sigmaAll[0:sDim + 1, 0:sDim + 1]
        sigmaSASR = self.sigmaAll[0:sDim + 1, sDim + 1:self.sampleDim]
        sigmaSRSA = self.sigmaAll[sDim + 1:len(new_sample), 0:sDim + 1]
        sigmaSR = self.sigmaAll[sDim + 1:len(new_sample), sDim + 1:len(new_sample)]
        self.nonCondSigmaSR = sigmaSR
        sigmaSSRG = np.delete(self.sigmaAll, sDim, axis=0)
        sigmaSSRG = np.delete(sigmaSSRG, sDim, axis=1)
        #self.scaled_sigmaSSRG_inv = np.linalg.inv((sigmaSSRG+np.eye(self.sampleDim-1)*0.0001)*self.kscale)
        #compute inverse of cov_S
        self.sigmaS = self.sigmaAll[0:sDim, 0:sDim]
        self.scaled_sigmaS = (self.sigmaS+np.eye(self.stateDim)*0.0001)*self.kscale
        #self.scaled_sigmaS_inv = np.linalg.inv(self.scaled_sigmaS)
        #compute inverse of cov_Sp
        self.sigmaSp = self.sigmaAll[sDim+1:2*sDim+1, sDim+1:2*sDim+1]
        #self.scaled_sigmaSp_inv = np.linalg.inv((self.sigmaSp+np.eye(self.stateDim)*0.0001)*self.kscale)
        #self.sigmaSR = sigmaSR - np.dot(sigmaSRSA, np.dot(np.linalg.inv((sigmaSA + np.eye(sigmaSA.shape[0]))*self.sig_scale), sigmaSASR))
        #if self.n >= self.memSize-1:
        #     self.sigmaSR = sigmaSR - np.dot(sigmaSRSA, np.dot(np.linalg.inv(sigmaSA), sigmaSASR))
        self.scaled_sigmaS_inv = 1.0/self.kernel_gamma*np.eye(self.stateDim)
        self.scaled_sigmaSp_inv = 1.0/self.kernel_gamma*np.eye(self.stateDim)
        self.scaled_sigmaSSRG_inv = 1.0/self.kernel_gamma*np.eye(self.sampleDim-1)
        self.scaled_sigmaSpRG_inv = 1.0/self.kernel_gamma*np.eye(self.sampleDim-self.stateDim-1)
        self.scaled_sigmaRG_inv = 1.0/self.kernel_gamma*np.eye(2)

    def updateSigProto(self, S, A, Sp, R, gamma):
        SAarray = self.sample2Array(S, A, Sp, R, gamma)
        self.SigProto_first = ((self.t - 1.0) * self.SigProto_first + np.outer(SAarray, SAarray)) / self.t
        self.muProto = ((self.t - 1.0) * self.muProto + SAarray) / self.t
        self.SigProto = self.SigProto_first - np.outer(self.muProto, self.muProto)
    
    # called by self.add2Model()
    def addCenter(self, S, A, Sp, R, gamma):
        # convert <S, A, Sp, R> sample into an array
        SAarray = self.sample2Array(S, A, Sp, R, gamma)
        # wait until memory fills up before adjusting prototypes
        if self.n < self.memSize:
            # save sample in model
            self.add(S, A, Sp, R, gamma, self.n)
            # update index (of next empty position)
            self.n += 1
            # update covariance matricies and mus
            self.update(None, SAarray)
            return (-1, None)
        # at this point memory is full, check if we have yet clustered our prototypes
        if not self.clustered or self.clusterClock == 0:
            self.blocks.Build_blocks(self.data, self.SigProto)
            self.clustered = True
            self.clusterClock = self.clusterGap
        #decide whether accept a new center
        replace_id = self.blocks.maximize(SAarray)
        #if self.t > 800:
        #    replace_id = -1
        #else:
        #    replace_id = self.blocks.maximize(SAarray)
        #replace_id = random.randint(0, self.memSize - 1) if random.uniform(0., 1.)>0.5 else -1
        if replace_id >= 0:
            #if np.all(Sp>1):
            #    print 'happened for the terminal states'
            #print 'replace id is ---------------' + str(replace_id)
            oldsample = self.data[replace_id, :]
            self.add(S, A, Sp, R, gamma, replace_id)  # inplace replacement
            self.update(oldsample, SAarray)  # update covariance matricies and mus
            self.clusterClock -= 1
            return (replace_id, oldsample)
        return (-1, None)
    
    # called by self.addCenter()
    # save sample in model
    def add(self, S, A, Sp, R, gamma, index):
        self.data[index] = self.sample2Array(S, A, Sp, R, gamma)

    def sample2Array(self, S, A, Sp, R, gamma):
        return np.concatenate((S, np.array([A]), Sp, np.array([R]), np.array([gamma])), axis=0)

    def array2Sample(self, ar):
        S = ar[0:self.stateDim]
        A = ar[self.stateDim]
        Sp = ar[self.stateDim + 1:self.stateDim + 1 + self.stateDim]
        R = ar[self.stateDim * 2 + 1]
        gamma = ar[-1]
        return (S, A, Sp, R, gamma)

    def sampleFromNext_pan(self, Sp, f, acts):
        out = []
        sDim = self.stateDim
        endind = min(self.memSize-1, self.t-1)
        # compute k(s', s')
        kspsp = self.kFuncCentersSp(self.data[:endind,sDim + 1:sDim + 1 + sDim], Sp)
        #print kspsp[0:50]
        #if self.t % 100 == 0:
        #    print self.inv_pre_piqi
        kspspaw = kspsp*self.apiqi[:endind]
        kspspaw = kspspaw
        Nsp = np.sum(kspspaw)
        if Nsp < KDESmallConst:
            return out
        # use indicator with k(s',s') to create P(a|s')
        Pa = map(lambda a: np.sum(kspspaw[self.data[:endind,sDim]==a]), range(acts))
        
        # This line is added in python3
        Pa = list(Pa)
        
        feasible_act = []
        for i in range(acts):
            if Pa[i] > KDESmallConst:
                feasible_act.append(i)
        for a in feasible_act:
            # sample a from P(a|s')
            #print np.array(Pa)/Nsp, np.sum(np.array(Pa)) == Nsp
            #a = np.random.choice(range(acts), 1, p=np.array(Pa)/Nsp)[0] if Nsp > KDESmallConst else np.random.choice(range(acts))
            # compute conditional mean and covariance matrix
            kspa = kspsp*(self.data[:endind,sDim]==a)
            # use kspa times weights
            kspa *= self.pre_piqi[:endind]
            #if self.t % 100 == 0:
            #    print self.inv_pre_piqi[:10]
            Nspa = np.sum(kspa)
            condi_mu_S = np.sum(kspa[:,None]*self.data[:endind,:sDim],axis=0)/Nspa if Nspa>KDESmallConst else np.sum(self.data[:endind,:sDim],axis=0)/self.memSize
            if Nspa < KDESmallConst:
                continue
            diff_Si_mu = self.data[:endind, :sDim] - condi_mu_S
            condi_sigmaS = (diff_Si_mu*kspa[:,None]).T.dot(diff_Si_mu)/Nspa if Nspa > 0 else 0
            #print condi_sigmaS
            # get sample index
            prob = kspa/Nspa if Nspa>KDESmallConst else np.ones(kspa.shape[0])*1.0/self.memSize
            SIndex = np.random.choice(range(endind), 1, p=prob)[0]
            mu_i = self.data[SIndex, :sDim]
            S = np.random.multivariate_normal(mu_i, condi_sigmaS+np.eye(condi_sigmaS.shape[0])*0.0001, 1)[0] if np.sum(abs(condi_sigmaS)) > KDESmallConst else mu_i
            #if np.any(S > 1.0) or np.any(S < 0.0):
            #    continue
            S = np.clip(S, 0., 1.)
            # given S,a; sample Sp,R,gamma
            kss = self.kFuncCentersS(self.data[:endind,:sDim], S)
            kssa = kss*(self.data[:endind,sDim]==a)
            kssa *= self.piqi[:endind]
            Nssa = np.sum(kssa)
            if Nssa < KDESmallConst:
                continue
            condi_mu_SpRG = np.sum(kssa[:,None]*self.data[:endind,sDim+1:],axis=0)/Nssa
            diff_SpRGi_mu = self.data[:endind,sDim+1:] - condi_mu_SpRG
            condi_sigmaSpRG = (diff_SpRGi_mu*kssa[:,None]).T.dot(diff_SpRGi_mu)/Nssa if Nssa > 0 else 0
            prob = kssa/Nssa if Nssa>KDESmallConst else np.ones(kssa.shape[0])*1.0/self.memSize
            SpRGIndex = np.random.choice(range(endind), 1, p=prob)[0]
            SpRG_mu_i = self.data[SpRGIndex, sDim+1:]
            SpRG = np.random.multivariate_normal(SpRG_mu_i, condi_sigmaSpRG+np.eye(condi_sigmaSpRG.shape[0])*0.0001, 1)[0] if np.sum(abs(condi_sigmaSpRG)) > KDESmallConst else SpRG_mu_i
            Sp, R, gamma = SpRG[:sDim], SpRG[sDim], SpRG[-1]
            Sp = np.clip(Sp, 0., 1.)
            gamma = np.clip(gamma, 0., 1.)
            #print Sp, R, gamma
            if not tupleInList((S, a, R, Sp, gamma), out):
                out.append((S, a, R, Sp, gamma))
        return out

    '''
    def sampleFromNext_pan(self, Sp, f, acts):
        out = []
        sDim = self.stateDim
        # compute k(s', s')
        kspsp = self.kFuncCentersSp(self.data[:,sDim + 1:sDim + 1 + sDim], Sp)
        #print kspsp[0:50]
        #if self.t % 100 == 0:
        #    print self.inv_pre_piqi
        kspspaw = kspsp*self.apiqi
        Nsp = np.sum(kspspaw)
        if Nsp < KDESmallConst:
            return out
        # use indicator with k(s',s') to create P(a|s')
        Pa = map(lambda a: np.sum(kspspaw[self.data[:,sDim]==a]), range(acts))
        feasible_act = []
        for i in range(acts):
            if Pa[i] > KDESmallConst:
                feasible_act.append(i)
        for a in feasible_act:
            # sample a from P(a|s')
            #print np.array(Pa)/Nsp, np.sum(np.array(Pa)) == Nsp
            #a = np.random.choice(range(acts), 1, p=np.array(Pa)/Nsp)[0] if Nsp > KDESmallConst else np.random.choice(range(acts))
            # compute conditional mean and covariance matrix
            kspa = kspsp*(self.data[:,sDim]==a)
            # use kspa times weights
            kspa *= self.pre_piqi
            #if self.t % 100 == 0:
            #    print self.inv_pre_piqi[:10]
            Nspa = np.sum(kspa)
            condi_mu_S = np.sum(kspa[:,None]*self.data[:,:sDim],axis=0)/Nspa if Nspa>KDESmallConst else np.sum(self.data[:,:sDim],axis=0)/self.memSize
            if Nspa < KDESmallConst:
                continue
            diff_Si_mu = self.data[:, :sDim] - condi_mu_S
            condi_sigmaS = (diff_Si_mu*kspa[:,None]).T.dot(diff_Si_mu)/Nspa if Nspa > 0 else 0
            #print condi_sigmaS
            # get sample index
            prob = kspa/Nspa if Nspa>KDESmallConst else np.ones(kspa.shape[0])*1.0/self.memSize
            SIndex = np.random.choice(range(self.memSize), 1, p=prob)[0]
            mu_i = self.data[SIndex, :sDim]
            S = np.random.multivariate_normal(mu_i, condi_sigmaS+np.eye(condi_sigmaS.shape[0])*0.0001, 1)[0] if np.sum(abs(condi_sigmaS)) > KDESmallConst else mu_i
            #if np.any(S > 1.0) or np.any(S < 0.0):
            #    continue
            S = np.clip(S, 0., 1.)
            # given S,a; sample Sp,R,gamma
            kss = self.kFuncCentersS(self.data[:,:sDim], S)
            kssa = kss*(self.data[:,sDim]==a)
            kssa *= self.piqi
            Nssa = np.sum(kssa)
            if Nssa < KDESmallConst:
                continue
            condi_mu_SpRG = np.sum(kssa[:,None]*self.data[:,sDim+1:],axis=0)/Nssa if Nssa>KDESmallConst else np.sum(self.data[:,sDim+1:],axis=0)/self.memSize
            diff_SpRGi_mu = self.data[:,sDim+1:] - condi_mu_SpRG
            condi_sigmaSpRG = (diff_SpRGi_mu*kssa[:,None]).T.dot(diff_SpRGi_mu)/Nssa if Nssa > 0 else 0
            prob = kssa/Nssa if Nssa>KDESmallConst else np.ones(kssa.shape[0])*1.0/self.memSize
            SpRGIndex = np.random.choice(range(self.memSize), 1, p=prob)[0]
            SpRG_mu_i = self.data[SpRGIndex, sDim+1:]
            SpRG = np.random.multivariate_normal(SpRG_mu_i, condi_sigmaSpRG+np.eye(condi_sigmaSpRG.shape[0])*0.0001, 1)[0] if np.sum(abs(condi_sigmaSpRG)) > KDESmallConst else SpRG_mu_i
            Sp, R, gamma = SpRG[:sDim], SpRG[sDim], SpRG[-1]
            Sp = np.clip(Sp, 0., 1.)
            gamma = np.clip(gamma, 0., 1.) 
            #print Sp, R, gamma
            if not tupleInList((S, a, R, Sp, gamma), out):
                out.append((S, a, R, Sp, gamma))
        return out
    '''

    def sampleS_FromNext(self, Sp):
        sDim = self.stateDim
        # compute k(s', s')
        kspsp = self.kFuncCentersSp(self.data[:, sDim + 1:sDim + 1 + sDim], Sp)
        kspsp *= self.piqi_s
        Ns = np.sum(kspsp)
        if Ns < KDESmallConst:
            return None
        condi_mu_S = np.sum(kspsp[:,None]*self.data[:,:sDim],axis=0)/Ns
        diff_Si_mu = self.data[:, :sDim] - condi_mu_S
        condi_sigmaS = (diff_Si_mu*kspsp[:,None]).T.dot(diff_Si_mu)/Ns
        prob = kspsp/Ns 
        SIndex = np.random.choice(range(self.memSize), 1, p=prob)[0]
        mu_i = self.data[SIndex, :sDim]
        S = np.random.multivariate_normal(mu_i, condi_sigmaS+np.eye(condi_sigmaS.shape[0])*0.0001, 1)[0] if np.sum(abs(condi_sigmaS)) > KDESmallConst else mu_i
        S = np.clip(S, 0., 1.)
        return S 

    def sampleER(self):
        s_id = random.randint(0, min(self.memSize - 1, self.n - 1))
        sample = self.data[s_id, :]
        (S, A, Sp, R, gamma) = self.array2Sample(sample)
        return (S, Sp, R, gamma, int(A))
   
    def Proto_sample(self, numSample):
        ids = np.random.randint(0, self.memSize - 1, numSample)
        samples = []
        for i in range(len(ids)):
            (S, A, Sp, R, gamma) = self.array2Sample(self.data[i,:])
            samples.append((S, Sp, R, gamma, int(A)))
        return samples
    ''' 
    def KDE_sampleSpRG(self, S = None, a = None):
        sDim = self.stateDim
        s_id = random.randint(0, self.memSize - 1)
        S = self.data[s_id, 0:sDim] if S is None else S
        a = self.data[s_id, sDim] if a is None else a
        #S, a = self.fixedRecentBuffer[s_id][0], self.fixedRecentBuffer[s_id][1]
        #print S, a
        #sample Sp, R, gamma based on this s,a
        kss = self.kFuncCentersS(self.data[:,:sDim], S)
        kssa = kss*(self.data[:,sDim]==a)
        #piqi = np.power(self.piqi, self.t/200000.)
        kssa *= self.piqi
        #if self.t % 100 ==0:
        #    print self.piqi[:20]
        Nssa = np.sum(kssa)
        if Nssa < KDESmallConst:
           return None
        #print qidivpi[:10]
        condi_mu_SpRG = np.sum(kssa[:,None]*self.data[:,sDim+1:],axis=0)/Nssa if Nssa>KDESmallConst else np.sum(self.data[:,sDim+1:],axis=0)/float(self.memSize)
        diff_SpRGi_mu = self.data[:,sDim+1:] - condi_mu_SpRG
        condi_sigmaSpRG = (diff_SpRGi_mu*kssa[:,None]).T.dot(diff_SpRGi_mu)/Nssa if Nssa > KDESmallConst else 0.
        prob = kssa/Nssa if Nssa>KDESmallConst else np.ones(kssa.shape[0])*1.0/self.memSize
        #print prob[0:10]
        SpRGIndex = np.random.choice(range(self.memSize), 1, p=prob)[0]
        SpRG_mu_i = self.data[SpRGIndex, sDim+1:]
        SpRG = np.random.multivariate_normal(SpRG_mu_i, condi_sigmaSpRG+np.eye(condi_sigmaSpRG.shape[0])*0.0001, 1)[0] if np.sum(abs(condi_sigmaSpRG)) > KDESmallConst else SpRG_mu_i
        #SpRG = SpRG_mu_i
        Sp, R, gamma = SpRG[:sDim], SpRG[sDim], SpRG[-1]
        gamma = np.clip(gamma, 0., 1.)
        Sp = np.clip(Sp, 0., 1.)
        #if self.t % 50 ==0 :
        #    print S, Sp, R, gamma, int(a)
        return (S, Sp, R, gamma, int(a))
    '''

      
    '''Uniformly sample (s,a) from the model, then sample (s', r, gamma) by KDE'''
    def KDE_sampleSpRG(self, S = None, a = None):
        sDim = self.stateDim
        endind = min(self.memSize-1, self.t-1)
        s_id = random.randint(0, self.memSize - 1)
        S = self.data[s_id, 0:sDim] if S is None else S
        a = self.data[s_id, sDim] if a is None else a
        #S, a = self.fixedRecentBuffer[s_id][0], self.fixedRecentBuffer[s_id][1]
        #print S, a
        #sample Sp, R, gamma based on this s,a
        kss = self.kFuncCentersS(self.data[:endind,:sDim], S)
        kssa = kss*(self.data[:endind,sDim]==a)
        #piqi = np.power(self.piqi, self.t/200000.)
        kssa *= self.piqi[:endind]
        #if self.t % 100 ==0:
        #    print self.piqi[:20]
        Nssa = np.sum(kssa)
        if Nssa < KDESmallConst:
           return None
        #print qidivpi[:10]
        condi_mu_SpRG = np.sum(kssa[:,None]*self.data[:endind,sDim+1:],axis=0)/Nssa
        diff_SpRGi_mu = self.data[:endind,sDim+1:] - condi_mu_SpRG
        condi_sigmaSpRG = (diff_SpRGi_mu*kssa[:,None]).T.dot(diff_SpRGi_mu)/Nssa if Nssa > KDESmallConst else 0.
        prob = kssa/Nssa if Nssa>KDESmallConst else np.ones(kssa.shape[0])*1.0/self.memSize
        #print prob[0:10]
        SpRGIndex = np.random.choice(range(endind), 1, p=prob)[0]
        SpRG_mu_i = self.data[SpRGIndex, sDim+1:]
        SpRG = np.random.multivariate_normal(SpRG_mu_i, condi_sigmaSpRG+np.eye(condi_sigmaSpRG.shape[0])*0.0001, 1)[0] if np.sum(abs(condi_sigmaSpRG)) > KDESmallConst else SpRG_mu_i
        #SpRG = SpRG_mu_i
        Sp, R, gamma = SpRG[:sDim], SpRG[sDim], SpRG[-1]
        gamma = np.clip(gamma, 0., 1.)
        Sp = np.clip(Sp, 0., 1.)
        #if self.t % 50 ==0 :
        #    print S, Sp, R, gamma, int(a)
        return (S, Sp, R, gamma, int(a))

    # directly sample a batch of samples without using a for loop, should be more efficient
    def KDE_sample(self, numSample):
        sDim = self.stateDim
        sa_ids = np.random.randint(0, self.memSize - 1, numSample)
        S = self.data[sa_ids, 0:sDim]
        a = self.data[sa_ids, sDim]
        #sample Sp, R, gamma based on this s,a, kss = (numSample, memSize)
        kss = self.kFuncCentersSS(self.data[:,:sDim], S)
        # kssa = (numSample, memSize)*(numSample, memSize)
        kssa = kss*(self.data[:,sDim]==np.expand_dims(a, axis = 1))
         
        #piqi = self.pi/self.qi
        piqi = self.piqi
        #if np.isnan(np.sum(piqi)):
        #    piqi = np.ones(self.pi.shape)/float(self.memSize)
        #    print 'it is nan'
            #print self.pi[:5]
            #print self.qi[:5]
        #if self.t % 100 == 0:
        #    print '------------------------------------------------'
        #    print piqi
        #    print self.pi[piqi>1.]
        #    print self.qi[piqi>1.]
        #print self.qi[:20]
        #    print 'sum    mean    median     std'
        #    print np.sum(piqi), np.mean(piqi), np.median(piqi), np.std(piqi), np.min(piqi), np.max(piqi)
        #piqi = np.power(piqi, 2./(1.+200./self.episode))
        kssa *= np.expand_dims(piqi, axis=0)
        #print piqi[:10] 
        Nssa = np.sum(kssa, axis = 1)
        Nssa[Nssa < KDESmallConst] = KDESmallConst/2.0 
        #condi_mu_SpRG = (numSample, SpRG_dim)
        condi_mu_SpRG = kssa.dot(self.data[:,sDim+1:])/Nssa[:,None]
        condi_mu_SpRG[Nssa < KDESmallConst] = np.sum(self.data[:,sDim+1:],axis=0)/float(self.memSize)        
        #diff_SpRGi_mu = (numSample, memSize, SpRG_dim)
        diff_SpRGi_mu = self.data[:,sDim+1:] - np.expand_dims(condi_mu_SpRG, axis = 1)
        condi_sigmaSpRG = np.matmul(np.transpose(diff_SpRGi_mu,(0,2,1)),diff_SpRGi_mu*np.expand_dims(kssa, axis=2))/Nssa[:,None,None]
        condi_sigmaSpRG[Nssa < KDESmallConst] = 0.0
        prob = kssa/Nssa[:,None]
        #prob[Nssa < KDESmallConst] = 1.0/self.memSize 
        samples = []
        for i in range(numSample):
            if np.sum(prob[i])<KDESmallConst:
                continue
            SpRGIndex = np.random.choice(range(self.memSize), 1, p=prob[i])[0]
            SpRG_mu_i = self.data[SpRGIndex, sDim+1:]
            SpRG = np.random.multivariate_normal(SpRG_mu_i, condi_sigmaSpRG[i]+np.eye(condi_sigmaSpRG[i].shape[0])*0.0001, 1)[0] if np.sum(abs(condi_sigmaSpRG[i])) > KDESmallConst else SpRG_mu_i
            Ss, aa, Sp, R, gamma = S[i], a[i], SpRG[:sDim], SpRG[sDim], SpRG[-1]
            gamma = np.clip(gamma, 0., 1.)
            Sp = np.clip(Sp, 0., 1.)
            samples.append([Ss, Sp, R, gamma, int(aa)])
        return samples

    def KDE_sampleSA(self, numSamples, numActs):
        samples = []
        sDim = self.stateDim
        s_ids = np.random.randint(0, self.memSize - 1, numSamples)
        s_mus = self.data[s_ids, 0:sDim]
        for i in range(numSamples):
            S = np.random.multivariate_normal(s_mus[i], self.scaled_sigmaS, 1)[0] if np.sum(abs(self.scaled_sigmaS)) > KDESmallConst else s_mus[i] 
            kss = self.kFuncCentersS(self.data[:,:sDim], S)
            Pa = np.array([np.sum(kss[self.data[:,sDim]==a]) for a in range(numActs)])
            Pa = Pa/np.sum(Pa)
            a = np.random.choice(range(numActs), 1, p=Pa)[0]
            samples.append([S, int(a)])
        return samples
 
    # sample the whole transitions from KDE model
    def KDE_sampleAll(self, numSamples, numActs):
        samples = []
        sDim = self.stateDim
        s_ids = np.random.randint(0, self.memSize - 1, numSamples)
        s_mus = self.data[s_ids, 0:sDim]
        for i in range(numSamples):
            S = np.random.multivariate_normal(s_mus[i], self.scaled_sigmaS, 1)[0] if np.sum(abs(self.scaled_sigmaS)) > KDESmallConst else s_mus[i]
            kss = self.kFuncCentersS(self.data[:,:sDim], S)
            Pa = np.array([np.sum(kss[self.data[:,sDim]==a]) for a in range(numActs)])
            Pa = Pa/np.sum(Pa)
            a = np.random.choice(range(numActs), 1, p=Pa)[0]
            S, Sp, R, gamma, a = self.KDE_sampleSpRG(S, a)
            samples.append([S, Sp, R, gamma, a])
        return samples

    def proto_sampleExpSpR(self, terminalS, gamma):
        sDim = self.stateDim
        datalist = list(self.data)
        s_id = random.randint(0, self.memSize - 1)
        s = self.data[s_id, 0:sDim]
        a = self.data[s_id, sDim]
        #compute an array of kernel values
        Pvec = self.kFuncCentersS(self.data[:, 0:sDim], s) * (self.data[:, sDim] == a)
        GammaVec = map(lambda si: int(np.all(si[sDim+1:2*sDim+1]==terminalS))*gamma, datalist)
        return (Pvec, GammaVec, s, a, list(self.data[:, sDim+1:2*sDim+1]), list(self.data[:, -1]))

    '''Uniformly sample s from the model, then sample the rest by KDE'''
    def sampleSASpR(self, numActions):
        sDim = self.stateDim
        #sample SpR, compute N = Pa[a]
        Nsa = Pa[a]
        #compute alpha_i
        Alpha = kss*(1*self.data[:,sDim]==a)
        SpRIndex = np.random.choice(range(self.memSize), 1, p=Alpha/Nsa)[0]
        #compute conditional covariance matrix
        mu = np.sum(self.data[:, sDim + 1:]*Alpha[:,None], axis=0) / Nsa
        diff_SiR_mu = self.data[:, sDim + 1:] - mu
        condi_sigmaSR = (diff_SiR_mu*Alpha[:,None]).T.dot(diff_SiR_mu)/Nsa
        mu_i = self.data[SpRIndex, sDim + 1:]
        SpR = np.random.multivariate_normal(mu_i, self.sig_scale*condi_sigmaSR, 1)[0]
        return (s, SpR[0:sDim], SpR[-1], int(a))

    #only get s then one can use policy to get action
    def sampleS(self):
        s_id = random.randint(0, self.memSize - 1)
        s = self.data[s_id, 0:self.stateDim]
        return s

    def kSigmaS(self, a, b):
        diff = a - b
        sig_inv = np.linalg.inv((self.sigmaS/float(self.memSize)) + (np.eye(self.stateDim) * 0.00001))
        sim = np.exp(-0.5 * diff.T.dot(sig_inv).dot(diff))
        sim = 0 if np.isinf(sim) else sim
        return sim

    '''for diagonal case'''
    #def kFunc(self, a, b):
    #    diff = a - b
    #    return math.exp(-0.5 * (diff * self.scaled_sigmaS_inv).T.dot(diff))

    '''kernel values of a STATE corresponding to the centers, diagonal sigma'''
    #def kFuncCentersS(self, centers, s):
    #    return np.exp(-0.5*np.sum(((s - centers)**2)*self.scaled_sigmaS_inv, axis=1))

    '''for non-diagonal case'''
    def kFunc(self, a, b, covmatInv):
        diff = a - b
        kv = np.exp(-0.5 * diff.T.dot(covmatInv).dot(diff))
        kv = 0.0 if kv < KDESmallConst else kv
        return 0.0

    def kFuncCentersS(self, centers, s):
        diff = s - centers
        kv = np.exp(-0.5 * np.sum(diff.dot(self.scaled_sigmaS_inv)*diff, axis=1))
        return kv

    def kFuncCentersSS(self, centers, s):
        diff = np.expand_dims(s, axis = 1) - centers
        #print s.shape, diff.shape
        return np.exp(-0.5 * np.sum(diff.dot(self.scaled_sigmaS_inv)*diff, axis=2))

    def kFuncCentersSp(self, centers, s):
        diff = s - centers
        return np.exp(-0.5 * np.sum(diff.dot(self.scaled_sigmaSp_inv)*diff, axis=1))
    
    def kFuncCentersSpS(self, centers, s):
        diff = np.expand_dims(s, axis = 1) - centers
        return np.exp(-0.5 * np.sum(diff.dot(self.scaled_sigmaSp_inv)*diff, axis=2))

    def kFuncCentersSSRG(self, centers, ssrg):
        diff = ssrg - centers
        kv = np.exp(-0.5 * np.sum(diff.dot(self.scaled_sigmaSSRG_inv)*diff, axis=1))
        return kv

    def kFuncCentersSpRG(self, centers, sprg):
        diff = sprg - centers
        kv = np.exp(-0.5 * np.sum(diff.dot(self.scaled_sigmaSpRG_inv)*diff, axis=1))
        return kv

    def kFuncCentersSSRGS(self, centers, ssrg):
        diff = np.expand_dims(ssrg, axis = 1) - centers
        return np.exp(-0.5 * np.sum(diff.dot(self.scaled_sigmaSSRG_inv)*diff, axis=2))

    def kFuncCentersRG(self, centers, rg):
        diff = rg - centers
        kv = np.exp(-0.5 * np.sum(diff.dot(self.scaled_sigmaRG_inv)*diff, axis=1))
        return kv	

    def _SimpleGaussian(self, a, b):
        d = LA.norm(np.array(a) - np.array(b))
        return math.exp(-d / (2.0 * self.sigSq))

