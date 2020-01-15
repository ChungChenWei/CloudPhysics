import numpy as np
import matplotlib.pyplot as plt

def initial_function(x): #initial distribution
    #return -3.*x+2.5
    #return 3.*x-0.5
    #return np.sin(x)+1.
    n=1
    return 4.*x**(n)*np.exp(-2.*x)


xmin = 0
xmax = 10

plt.plot(np.linspace(xmin,xmax,1000),initial_function(np.linspace(xmin,xmax,1000)),'--k',linewidth=5)
plt.show()