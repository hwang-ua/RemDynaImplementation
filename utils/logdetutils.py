import numpy as np
import math
import sys

'''all input matrix in this file should be kernel matrix'''
'''all vectors are column vectors'''
'''returns the utility change: utility after deletion - before deletion'''

'''Below are utility functions for computing log det'''
'''given inverse of K, compute logdet(K + uv^T + vu^T) - logdetK,
u is sparse with nonzero index being ind'''
def logdet_rank_two(mat_inv, u, v, ind):
    a = u[ind]*np.inner(mat_inv[ind,:], v)
    b = mat_inv[ind, ind]*u[ind]*u[ind]
    c = v.T.dot(mat_inv).dot(v)
    return math.log((1.0+a)*(1.0+a) - b*c)

#eta is the regularizer, k is the ind th col in kernel matrix K
def logdet_delete(mat_inv, k, ind, eta):
    kcopy = k.copy()
    kcopy[ind] = 0
    x = np.zeros(kcopy.shape[0])
    x[ind] = -1.0
    return logdet_rank_two(mat_inv, x, kcopy, ind) - math.log(eta + 1.0)

#NOTE: k.shape[0] = mat_inv.shape[0]
def logdet_attach(mat_inv, k, eta):
    dotprod = k.T.dot(mat_inv).dot(k)
    return math.log(eta + 1.0 - dotprod)

#passing in the inverse of K, original kernel reps of k, new ... of k, and the index
#return the change of replacing
def logdet_replace(mat_inv, oldk, newk, ind):
    y = newk - oldk
    x = np.zeros(y.shape[0])
    x[ind] = 1.0
    return logdet_rank_two(mat_inv, x, y, ind)

'''Below are functions for updating kernel matrix'''

'''u is sparse and nonzero position is ind'''
def sherman_update_spu(Kinv, u, v, ind):
    Kinvdotu = u[ind]*Kinv[:, ind]
    denominator = 1.0 + np.inner(v.T, Kinvdotu)
    vdotKinv = v.T.dot(Kinv)
    newKinv = Kinv - np.outer(1.0/denominator*Kinvdotu, vdotKinv)
    return newKinv

'''v is sparse and nonzero position is ind'''
def sherman_update_spv(Kinv, u, v, ind):
    vdotKinv = v[ind]*Kinv[ind,:]
    denominator = 1.0 + np.inner(vdotKinv, u)
    Kinvdotu = Kinv.dot(u)
    newKinv = Kinv - np.outer(1.0/denominator*Kinvdotu, vdotKinv)
    return newKinv

'''remove ind th row and column'''
def remove_rc(K, ind):
    K1 = np.delete(K, ind, axis=0)
    K2 = np.delete(K1, ind, axis=1)
    return K2

'''attach k to row and col: k is the same dimension with mat_inv rows'''
def kmat_inv_attach(Kinv, k, eta):
    kdotKinv = k.T.dot(Kinv)
    v = 1.0/(1.0 + eta - kdotKinv.dot(k))
    updated_Kinv = np.zeros((Kinv.shape[0]+1,Kinv.shape[1]+1))
    updated_Kinv[:-1,:-1] = Kinv + np.outer(v*kdotKinv.T, kdotKinv)
    updated_Kinv[-1,:-1] = -v*kdotKinv
    updated_Kinv[:-1,-1] = -v*kdotKinv.T
    updated_Kinv[-1,-1] = v
    return updated_Kinv

'''delete the ind th row and column in the K matrix, update Kinv mat'''
'''k is the ind th column'''
def kmat_inv_delete(Kinv, k, ind):
    x = np.zeros(k.shape[0])
    x[ind] = -1.0
    kcopy = k.copy()
    kcopy[ind] = 0
    updatedKinv = sherman_update_spu(Kinv, x, kcopy, ind)
    updatedKinv = sherman_update_spv(updatedKinv, kcopy, x, ind)
    return remove_rc(updatedKinv, ind)

'''replace ith row and column k by ktilde, update Kinv mat'''
def kmat_inv_replace(Kinv, k, ktilde, ind):
    x = np.zeros(k.shape[0])
    x[ind] = 1.0
    y = ktilde - k
    updatedKinv = sherman_update_spu(Kinv, x, y, ind)
    updatedKinv = sherman_update_spv(updatedKinv, y, x, ind)
    return updatedKinv

def kmat_replace(Kmat, k, ind):
    newK = Kmat.copy()
    kxx = Kmat[ind, ind]
    newK[ind, :] = k
    newK[:, ind] = k
    newK[ind, ind] = kxx
    return newK

def kmat_attach(Kmat, k):
    newK = np.zeros((Kmat.shape[0]+1, Kmat.shape[0]+1))
    newK[:-1,:-1] = Kmat.copy()
    newK[-1,:-1] = k
    newK[:-1,-1] = k
    newK[-1, -1] = Kmat[-1, -1]
    return newK
