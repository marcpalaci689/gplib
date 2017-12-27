import numpy as np
from numba import jit,f8,i8,b1
from numba import types



@jit(f8[:,:](i8,f8[:,:],f8[:,:],f8,f8[:],b1),nopython=True,cache=True)
def rbfard(D,x1,x2,sigma,lengthscale,training):
    N1 = x1.shape[0]
    if not training:
        N2 = x2.shape[0]
        K  = np.zeros((N1,N2))
        for i in xrange(N1):
            for j in xrange(N2):
                for d in xrange(D):
                    K[i,j] -= ((x1[i,d]-x2[j,d])/lengthscale[d])**2
    else:
        K  = np.zeros((N1,N1))
        for i in xrange(1,N1):
            for j in xrange(i):
                for d in xrange(D):
                    K[i,j] -= ((x1[i,d]-x1[j,d])/lengthscale[d])**2        
                K[j,i] += K[i,j]
    return sigma**2.0*np.exp(0.5*K)

@jit(f8[:,:](i8,f8[:,:],f8[:,:],f8,f8,b1),nopython=True,cache=True)
def rbf(D,x1,x2,sigma,lengthscale,training):
    N1 = x1.shape[0]
    if not training:
        N2 = x2.shape[0]
        K  = np.zeros((N1,N2))
        for i in xrange(N1):
            for j in xrange(N2):
                for d in xrange(D):
                    K[i,j] -= ((x1[i,d]-x2[j,d])/lengthscale)**2
    else:
        K  = np.zeros((N1,N1))
        for i in xrange(1,N1):
            for j in xrange(i):
                for d in xrange(D):
                    K[i,j] -= ((x1[i,d]-x1[j,d])/lengthscale)**2
                K[j,i] += K[i,j]                
    return sigma**2.0*np.exp(0.5*K)
    
        

@jit(f8(f8[:,:],f8[:,:],i8),nopython=True,cache=True)    
def trace(M,G,N):
    trace = 0
    for i in xrange(N):
        for j in xrange(i):
            trace += 2*M[i,j]*G[j,i]
        trace+= M[i,i]*G[i,i]
    return -0.5*trace

@jit(f8[:](f8[:,:],f8[:,:],f8[:,:],f8[:],i8,i8),nopython=True,cache=True)
def drbfard_dlengthscale(dML,K,x,l,N,D):
    grad = np.zeros(D)
    for i in xrange(N):
        for j in xrange(i):
            for d in xrange(D):
                grad[d] -= (K[i,j]*((x[i,d] - x[j,d])**2/l[d]**3))*dML[i,j]
    return grad

@jit(f8(f8[:,:],f8[:,:],f8[:,:],f8,i8,i8),nopython=True,cache=True)
def drbf_dlengthscale(dML,K,x,l,N,D):
    grad = 0
    for i in xrange(N):
        for j in xrange(i):
            for d in xrange(D):
                grad -= K[i,j]*((x[i,d] - x[j,d])**2/l**3)*dML[i,j]            
    return grad 


    