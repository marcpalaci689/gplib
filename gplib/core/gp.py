import numpy as np

class GP(object):
    def __init__(self,x,y,kernel,noise_var):  
        self.x_train    = x
        self.N,self.D   = x.shape
        self.y_train    = y
        self.kernel     = kernel
        self._noise_var = noise_var
        self._params    = np.hstack((kernel.hyp,noise_var))
        self.training   = False
        self.likelihood = None
        return
