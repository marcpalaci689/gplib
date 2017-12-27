from .. import kernels
from ..core import GP
from .. import plotting
import numpy as np
import scipy
from scipy.optimize import minimize
import copy
import matplotlib.pyplot as plt
import models_utils
import dill
dill.settings['recurse'] = True
import pathos.multiprocessing

class GPRegression(GP):
    def __init__(self,x,y,kernel=kernels.RBF(),noise_var=1.0):
        super(GPRegression,self).__init__(x,y,kernel=kernel,noise_var=noise_var)
        self._cache     = self.kernel.alpha(self.x_train,self.y_train,self.noise_var)
        self.likelihood = models_utils.marginal_likelihood(self.y_train,self.kernel.a,self.kernel.chol,self.N)
        
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self,new_params):
        if not self.training:
            assert isinstance(new_params,np.ndarray), 'parameters must be in numpy array'
            assert new_params.shape==self._params.shape , 'shape of parameter array is inconsistent'
            new_params = 1.0*new_params
        if not np.array_equal(self._params,new_params):
            self._cache['param_change']  = True
            self._cache['alpha']         = False
            self._cache['inv_k']         = False
            self._cache['K']        = False
        self._params    = new_params
        self.noise_var = new_params[-1]
        self.kernel.hyp = new_params[0:-1]
        
    @property
    def noise_var(self):
        return self._noise_var
    
    @noise_var.setter
    def noise_var(self,new_noise):
        if not self.training:
            assert isinstance(new_noise,float) or isinstance(new_noise,int), 'noise_var must be a float or int'
            if self._noise_var != new_noise:
                self._cache['param_change']  = True
                self._cache['alpha']         = False
                self._cache['inv_k']         = False                
            new_noise= 1.0*new_noise
            self._noise_var  = new_noise
            self.params[-1]  = new_noise          
        else:
            self._noise_var  = new_noise
        
    def train(self):
        self.training=True
        self.kernel.state['train'] = True
    
    def eval(self):
        self.training =False
        self.kernel.state['train']=False

    
    def negative_log_likelihood(self,*params): 
        # if the module was called with a params argument, do the following:
        if params:
            current_cache  = self._cache.copy()       # store current cache 
            current_params = self._params.copy()      # store current parameters
            self.params = params[0]                   # update model parameters
            
            # if we are in training mode, determine whether we need to recompute the likelihood or simply use the cached value 
            if self.training and self.check_cache():
                #print 'training...computing ML' 
                self._cache = self.kernel.alpha(self.x_train,self.y_train,self.noise_var)
                ML = models_utils.marginal_likelihood(self.y_train,self.kernel.a,self.kernel.chol,self.N)                
                self.likelihood = ML
            
            # if not in training mode and we need to recompute the ML, then we must assure to revert the model back to its original parameters    
            elif not self.training and self.check_cache():
                stored = self.store_state()                 # store old kernel parameters if necessary 
                self._cache = self.kernel.alpha(self.x_train,self.y_train,self.noise_var)
                ML = models_utils.marginal_likelihood(self.y_train,self.kernel.a,self.kernel.chol,self.N)                     
                # if we have stored parameters, we must return them to the model
                if stored is not None:
                    #print 'computing ML and reverting to original state'
                    self.kernel.K = stored['K']
                    self.kernel.a = stored['alpha']
                    self.kernel.chol = stored['chol']
                    self.kernel.dML_dK = stored['dML_dK']
                    self.params = current_params
                    self._cache = current_cache
            
            # if we already have the desired ML in cache, just return the cached value        
            else:
                #print 'using cached ML'
                ML = self.likelihood
                
        # if no params argument was passed, simply check cache to see whether we need to compute ML or not        
        else:
             if self.check_cache():
                 #print 'computing ML' 
                 self._cache = self.kernel.alpha(self.x_train,self.y_train,self.noise_var)
                 ML = models_utils.marginal_likelihood(self.y_train,self.kernel.a,self.kernel.chol,self.N)
                 self.likelihood = ML
             else:
                 #print 'Using cached ML'
                 ML = self.likelihood
        return ML    
    
    def nll_grad(self,*params):
        # if model is not in training, force to training mode and record the change
        not_training = False
        if not self.training:
            not_training = True
            self.train()
          
        if params:
            current_cache  = self._cache.copy()       # store current cache 
            current_params = self._params.copy()      # store current parameters
            self.params = params[0]                   # update model parameters
            if self.check_cache():
                stored = self.store_state()                 # store old kernel parameters if necessary 
                self._cache = self.kernel.alpha(self.x_train,self.y_train,self.noise_var)
                # compute gradient
                grad = self.kernel.dML_dhyp(self.x_train,self.noise_var)
                # if we have stored parameters, we must return them to the model
                if stored is not None:
                    print 'computing inv_K and reverting to original state'
                    self.kernel.K = stored['K']
                    self.kernel.a = stored['alpha']
                    self.kernel.chol = stored['chol']
                    self.kernel.dML_dK = stored['dML_dK']
                    self.params = current_params
                    self._cache = current_cache 
            else:
                # compute gradient
                #print 'using cached inv_K'
                grad = self.kernel.dML_dhyp(self.x_train,self.noise_var)
                self.params = current_params
                self._cache = current_cache         
        # determine whether we need to recompute the the inverse of the Gram matrix 
        else:
            if self.check_cache():
                #print 'computing inv_K' 
                self._cache = self.kernel.alpha(self.x_train,self.y_train,self.noise_var)
            
            # compute gradient
            grad = self.kernel.dML_dhyp(self.x_train,self.noise_var)
        
        # return the model to its original state
        if not_training:
            self.eval()
            
        return grad
    
    def check_cache(self):
        if self._cache['param_change']:
            return True
        elif self.training and not self._cache['inv_K']:
                return True
        elif not self.training and not self._cache['alpha']:
                return True
        else:
            return False
        
    def store_state(self):
        if self.kernel.a is not None:
            store = {'K':self.kernel.K,'alpha':self.kernel.a,'chol':self.kernel.chol, 'dML_dK':self.kernel.dML_dK}            
            return store
        else:
            return None

                
    def optimize(self):
        #self.kernel.clear()
        self.train()
        opt = minimize(self.negative_log_likelihood,self.params,jac=self.nll_grad,method='L-BFGS-B')
        print('optimum found with negative log likelihood = %.5f' %(opt['fun']))
        self.params = opt['x']
        self.eval()
        return
    

    
    def optimize_restarts(self,num_restarts=3,multicore=True):
        if multicore:
            # create different models
            tasks=[self]
            for res in xrange(num_restarts-1):
                params = np.random.normal(scale=1,size=self.params.shape[0])
                new_model = copy.deepcopy(self)
                new_model.params = params
                tasks.append(new_model)       
            
            pool = pathos.multiprocessing.Pool()
            res = pool.map(models_utils.optimize,tasks)
            #print res
            opt = res[0]['objective']
            opt_ind = 0
            
            for i in xrange(1,num_restarts):
                if res[i]['objective']< opt:
                    opt = res[i]['objective']
                    opt_ind = i
                    
            self.params = res[opt_ind]['model'].params
            pool.terminate()
            pool.close()
        
        else:
            res = models_utils.optimize(self)
            opt = res['objective']
            opt_params = res['model'].params
            for i in xrange(1,num_restarts):
                params = np.random.normal(scale=3,size=self.params.shape[0])
                self.params = params
                res = models_utils.optimize(self)
                if res['objective']<opt:
                    opt = res['objective']
                    opt_params = res['model'].params
            self.params = opt_params      
        return 
    
    
    def predict(self,x):
        if self._cache['K']:
            K = self.kernel.K
        else:
            K = self.kernel.evaluate(self.x_train,self.x_train)
            
        diag_ind = np.diag_indices_from(K)
        K[diag_ind] += self.noise_var**2
        chol = np.linalg.cholesky(K)
        
        ks = self.kernel.evaluate(self.x_train,x)
        
        # compute the mean at our test points.
        Lk  = scipy.linalg.solve_triangular(chol,ks,lower=True,check_finite=False)
        Lk2 = scipy.linalg.solve_triangular(chol,self.y_train,lower=True,check_finite=False)
        mu = np.dot(Lk.T,Lk2)

        # compute the variance at our test points.
        Kss = self.kernel.evaluate(x,x)
        var = np.sqrt(np.diag(Kss) - np.sum(Lk**2, axis=0) + self.noise_var**2) 
        return mu.flatten(),var              
       
    def plot(self):
        plotting.plot(self)
        return
        