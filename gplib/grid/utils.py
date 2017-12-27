import numpy as np
from numba import jit,f8,i8,u4
import numba
import copy
import time 
from scipy.sparse import csr_matrix
from scipy.sparse import kron

@jit(numba.types.UniTuple(f8[:],2)(f8[:,:]),nopython=True)
def minmax(x):
    N,D = x.shape
    maximum = -np.inf*np.ones(D)
    minimum = np.inf*np.ones(D)
    for i in xrange(N):
        for j in xrange(D):
            if x[i,j]>maximum[j]:
                maximum[j]=x[i,j]              
            if x[i,j]<minimum[j]:
                minimum[j]=x[i,j]      
    return maximum,minimum

@jit(numba.types.List(f8)(f8,f8,f8,f8),nopython=True)
def CubicInterpolation(d1,d2,d3,d4):
    ''' 
    Given the distances to the bounding inducing points, get cubic interpolation weights
        
        Inputs:
            d1 --> Distance to furthest lower bound
            d2 --> Distance to closest lower bound
            d3 --> Distance to closest upper bound
            d4 --> Distance to furthest upper bound
    
        Outputs:
            [W1,W2,W3,W4] --> List of weights
    '''
    x1,x2,x3,x4 = 0,d1-d2,d1+d3,d1+d4
    W1 = (d2*-d3*-d4)/(-x4*-x3*-x2)
    W2 = (d1*-d3*-d4)/(x2*(x2-x3)*(x2-x4))
    W3 = (d1*d2*-d4)/(x3*(x3-x2)*(x3-x4))
    W4 = (d1*d2*-d3)/(x4*(x4-x2)*(x4-x3))
    return [W1,W2,W3,W4]

@jit(numba.types.List(i8)(i8),nopython=True)
def adding(l):
    a=[]
    a.append(l)
    return a

@jit(numba.types.Tuple((numba.types.List(f8),numba.types.List(i8),numba.types.List(i8)))(f8[:],f8[:]),nopython=True)    
def interpolate(x,grid):
    grid_size = len(grid)
    N         = len(x)
    right_index = np.searchsorted(grid,x)
    #W = csr_matrix((N,grid_size))
    data=[]
    col=[]
    row=[]
    for i in xrange(N):
        r = right_index[i]
        d1 = x[i]-grid[r-2]
        d2 = x[i]-grid[r-1]
        if d2 <=1e-6:
            #W[i,r-1] = 1.0
            data.append(1.0)
            col.append(r-1)
            row.append(i)
            continue
        d3 = grid[r]- x[i]
        if d3 <= 1e-6:
            #W[i,r] = 1.0
            data.append(1.0)
            col.append(r)
            row.append(i)
            continue
        d4 = grid[r+1]-x[i]
        data = data+CubicInterpolation(d1,d2,d3,d4)
        col  = col + list(xrange(r-2,r+2))
        row  = row + [i]*4
    return data,col,row


def MV_kronprod(krons,b):
    '''
    This function will perform a vector matrix product between a packed kronecker matrix
    and a column vector
    
    Inputs:
        krons --> list of tensor matrices
        b     --> Column vector
    
    Outputs:
        x  --> column vector resulting from product between the unpacked kronecker product and
               vector b 
    '''
    x = b
    N = len(b)
    D = len(krons)
    
    for d in reversed(xrange(D)):
        ld = len(krons[d])
        X = x.reshape((ld,N/ld),order='f') 
        Z = np.dot(krons[d],X).T
        x = Z.reshape((-1,1),order='f')
    return x
 


def KhatriRao_row(KR,b):
    '''
    This function will perform a matrix-vector product between a packed row partitioned khatri-rao matrix
    and a column vector
    
    Inputs:
        KR    --> list of khatri-rao sub-matrices
        b     --> Column vector
    
    Outputs:
        x  --> column vector resulting from matrix-vector multiplication between KR and b
    ''' 
    D = len(KR)
    N = KR[0].shape[0]
    x = np.zeros((N,1))
    for i in xrange(N):
        row = KR[0][i,:]
        for d in xrange(1,D):
            row = kron(row,KR[d][i,:])
        x[i] = x[i]+row.dot(b)[0]
    return x

def KhatriRao_col(KR,b):
    '''
    This function will perform a matrix-vector product between the transpose of a packed row 
    partitioned khatri-rao matrix and a column vector (KR'b).
    
    Inputs:
        KR    --> list of khatri-rao sub-matrices (in row-partitioned format)
        b     --> Column vector
    
    Outputs:
        x  --> column vector resulting from matrix-vector multiplication between KR' and b
    ''' 
    D = len(KR)
    N = b.shape[0]
    M = KR[0].shape[1]
    for i in xrange(1,D):
        M = M*KR[i].shape[1] 
    x = np.zeros((M,1))
    for i in xrange(N):
        if b[i,0] == 0:
            continue
        else:
            col = KR[0][i,:]*b[i,0]
            for d in xrange(1,D):
                col = kron(col,KR[d][i,:])
               
            x[:,0] = x[:,0] + col
    return x
    
def MMMV_kr_kron(W,K,y,noise):
    ''' 
    This function performs a Matrix-Matrix-Matrix-Vector multiplication
    K_SKI*y = (WKW' + spherical_noise)*y
    efficiently in O(N+m^2) time and memory
    
    Inputs:
        W     --> Interpolation weights matrix in compressed row format
        K     --> mxm grid kernel matrix
        y     --> Nx1 target value vector
        noise --> scalar value of the noise
    
    Outputs:
        (WKW' + spherical_noise)*y --> for use in Conjugate Gradient method
    '''
    b = KhatriRao_col(W,y)
    b = MV_kronprod(K,b)      
    b = KhatriRao_row(W,b)
    return b+(noise**2)*y    


def unpack_Kski(W,K):
    N = W[0].shape[0]
    Kski = np.zeros((N,N))
    mask = np.zeros((N,1))
    for i in xrange(N):   
        mask[i] = 1.0
        Kski[:,i] = MMMV_kr_kron(W,K,mask,0).flatten()
        mask[i] = 0
    return Kski

def unpack_W(W):
    D = len(W)
    N,M = W[0].shape
    for i in xrange(1,D):
        M = M*W[i].shape[1] 

    new_W = csr_matrix((N,M))
    for i in xrange(N):
        row = W[0][i,:] 
        for d in xrange(1,D):
            row = kron(row,W[d][i,:])
        #print row.shape    
        new_W[i,:] = row
    return new_W
if __name__ == '__main__':
    x = np.random.uniform(-20,20,size=10000)
    grid = np.linspace(-21,21,100)
    
    s = time.time()
    for i in xrange(10):
        W,C,R = interpolate(x,grid)
        W = csr_matrix((W,(R,C)),shape=(10000,100))
    
    e = time.time()
    
    print((e-s)/10)
    
    s = time.time()
    for i in xrange(10):
        W1 = interpolate1(x,grid)
    e = time.time()
    
    print((e-s)/10)