import matplotlib.pyplot as plt
import numpy as np

def plot(model):
    
    assert model.D == 1, 'Plotting only works for 1D input data'
    
    min = np.min(model.x_train)
    max = np.max(model.x_train)
    padding = (max-min)*0.15
    
    x_test = np.linspace(min-padding,max+padding,1000).reshape(-1,1)
    mu,var = model.predict(x_test)
    
    plt.scatter(model.x_train,model.y_train,marker='x',c='k',label='Data')
    plt.plot(x_test.flatten(),mu,'darkslateblue', linewidth=2, label='Mean')
    plt.fill_between(x_test.flatten(),mu+2*var,mu-2*var,interpolate=True,alpha=0.25,color='cornflowerblue',label='Confidence')
    
    axes=plt.gca()
    axes.set_xlim(left=min-padding,right=max+padding)
    plt.title('GP Model',fontsize=30)
    plt.xlabel('X',fontsize=20)
    plt.ylabel('Y',fontsize=20)
    plt.legend()
    plt.show()
    return