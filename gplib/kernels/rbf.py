import numpy as np
import kernel_utils
from scipy.linalg import solve_triangular

class RBF(object):
    def __init__(self,D=1,sigma=None,lengthscale=None,ARD=False):
        self.state = {'train':False,'ARD':ARD}
        self.D  = D
        self._sigma = 1.0 if sigma is None else sigma
        if ARD:
            if lengthscale is None:
                self._lengthscale = np.ones(D)
            else:
                if isinstance(lengthscale,list):
                    self._lengthscale = 1.0*np.array(lengthscale)
                elif isinstance(lengthscale,np.ndarray):
                    self._lengthscale = 1.0*lengthscale
                else:
                    raise ValueError('Lengthscale must be a list or array')
                assert(self.D == self._lengthscale.shape[0]), 'lengthscale dimensionality inconsistent with specified D'
        else:
            if lengthscale is None:
                self._lengthscale = 1.0
            else:
                if isinstance(lengthscale,float):
                    self._lengthscale = lengthscale
                elif isinstance(lengthscale,int):
                    self._lengthscale = 1.0*lengthscale
                else:
                    raise ValueError('Lengthscale must be a float or int')

        if sigma is None:
            self._sigma = 1.0
        else:
            if isinstance(sigma,float):
                self._sigma = sigma
            elif isinstance(sigma,int):
                self._sigma = 1.0*sigma
            else:
                raise ValueError('Sigma must be a float or int')

        self._hyp = np.hstack((self._sigma,self._lengthscale))
        self.chol   = None
        self.K      = None
        self.a      = None
        self.dML_dK = None   

    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self,new_sigma):
        self._sigma = float(new_sigma)
        self._hyp[0] = self._sigma
        
    @property
    def lengthscale(self):
        return self._lengthscale
    
    @lengthscale.setter
    def lengthscale(self,new_lengthscale):
        self._lengthscale = 1.0*new_lengthscale
        self._hyp[1:]     = self._lengthscale
        
    @property
    def hyp(self):
        return self._hyp
    
    @hyp.setter
    def hyp(self,new_hyp):
        new_hyp = 1.0*new_hyp
        self._sigma = new_hyp[0]
        self._lengthscale = new_hyp[1:]
        if not self.state['ARD']:
            self._lengthscale = self._lengthscale[0]
        self._hyp = new_hyp
        
    def evaluate(self,x1,x2):
        if self.state['ARD']:
            return kernel_utils.rbfard(self.D,x1,x2,self.sigma,self.lengthscale,self.state['train'])
        else:
            return kernel_utils.rbf(self.D,x1,x2,self.sigma,self.lengthscale,self.state['train'])

    def clear(self):
        self.chol   = None
        self.K      = None
        self.a      = None
        self.dML_dK = None
        
    def alpha(self,x,y,noise_var):
        self.K = self.evaluate(x,x)
        Ky     = self.K.copy()
        diag_ind = np.diag_indices_from(Ky)
        Ky[diag_ind] += noise_var**2
        self.chol = np.linalg.cholesky(Ky) 
        if self.state['train']:
            self.dML_dK = solve_triangular(self.chol,np.eye(Ky.shape[0]),lower=True,check_finite=False)
            self.dML_dK = np.dot(self.dML_dK.T,self.dML_dK)
            self.a      = np.dot(self.dML_dK,y)
            self.dML_dK = np.dot(self.a,self.a.T) - self.dML_dK
            return {'param_change':False,'alpha':True,'inv_K':True,'K':True}
        else:
            self.a = solve_triangular(self.chol,y,lower=True,check_finite=False)
            self.a = solve_triangular(self.chol.T,self.a,check_finite=False)
            self.dML_dK = None
            return {'param_change':False,'alpha':True,'inv_K':False,'K':True} 
    
    def dML_dhyp(self,x,noise_var):
        N = x.shape[0]
        grad = np.zeros(self.hyp.shape[0]+1)
        grad[0]  = (2.0/self.sigma)*kernel_utils.trace(self.dML_dK,self.K,N)    
        if self.state['ARD']:
            grad[1:-1] = kernel_utils.drbfard_dlengthscale(self.dML_dK,self.K,x,self.lengthscale,N,self.D)
        else:
            grad[1] = kernel_utils.drbf_dlengthscale(self.dML_dK,self.K,x,self.lengthscale,N,self.D)
               
        grad[-1] = -noise_var*np.trace(self.dML_dK) 
        return grad

if __name__ == '__main__':
    x = np.random.uniform(-5,5,size=(1000,1))
    y = np.sin(x) + np.random.normal(scale=0.1,size=(1000,1))
    
    rbf = RBF(sigma=3,lengthscale=[2.0])
    
            
        

