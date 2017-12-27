import numpy as np


def optimize(model):
    model.optimize()
    return {'objective':model.negative_log_likelihood(),'model':model}


def marginal_likelihood(y,alpha,chol,N):
    if N<1000:
        return 0.5*np.sum(np.dot(y.reshape(1,-1),alpha)) + np.sum(np.log(np.diag(chol))) + 0.5*N*np.log(2*np.pi)
    else:
        return 0.5*np.sum(y*alpha) + np.sum(np.log(np.diag(chol))) + 0.5*N*np.log(2*np.pi)
    