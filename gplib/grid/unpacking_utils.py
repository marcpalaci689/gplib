import numpy as np
from scipy.sparse import csc_matrix,csr_matrix,coo_matrix
from scipy.sparse import kron


### NOT GOOD

def transpose_reshape(a, shape):
    """Reshape the sparse matrix `a`.

    Returns a coo_matrix with shape `shape`.
    """
    if not hasattr(shape, '__len__') or len(shape) != 2:
        raise ValueError('`shape` must be a sequence of two integers')
    
    c = a.tocoo()
    nrows, ncols = c.shape
    size = nrows * ncols

    new_size =  shape[0] * shape[1]
    if new_size != size:
        raise ValueError('total size of new array must be unchanged')

    flat_indices = ncols * c.row + c.col
    new_row, new_col = divmod(flat_indices, shape[1])

    b = coo_matrix((c.data, (new_row, new_col)), shape=shape)
    b = csr_matrix(b)
    return b

def sparse_reshape(a, shape):
    """Reshape the sparse matrix `a`.

    Returns a coo_matrix with shape `shape`.
    """
    if not hasattr(shape, '__len__') or len(shape) != 2:
        raise ValueError('`shape` must be a sequence of two integers')
    
    c = (a.transpose()).tocoo()
    nrows, ncols = c.shape
    size = nrows * ncols

    new_size =  shape[0] * shape[1]
    if new_size != size:
        raise ValueError('total size of new array must be unchanged')

    flat_indices = ncols * c.row + c.col
    new_row, new_col = divmod(flat_indices, shape[1])

    b = coo_matrix((c.data, (new_row, new_col)), shape=shape)
    return b.transpose()


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
    N = b.shape[0]
    D = len(krons)
    
    ld = len(krons[D-1])
    X = transpose_reshape(x,(N/ld,ld))
    Z = X.dot(krons[D-1].T)
    x = Z.reshape((-1,1),order='f')    
    
    for d in reversed(xrange(D-1)):
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
    x = csc_matrix((M,1))
    for i in xrange(N):
        if b[i,0] == 0:
            continue
        else:
            col = KR[0][i,:]*b[i,0]
            for d in xrange(1,D):
                col = kron(col,KR[d][i,:])
            x = x + col.transpose()
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

