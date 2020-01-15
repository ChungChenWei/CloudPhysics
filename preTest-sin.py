import numpy as np
import matplotlib.pyplot as plt
from bin_node import Bin_Node as BN

def initial_function(x): #initial distribution
    #return -3.*x+2.5
    #return 3.*x-0.5
    return np.sin(x)+1.

ndf_normal = 0 #n0+k(x-x0)
ndf_modified = 1 #k_star(x-x_star)
num_of_bin = 13 #number of bins
xmax = 13

x1arr = np.linspace(0,xmax,num_of_bin+1)[:-1] #array of x1
x2arr = np.linspace(0,xmax,num_of_bin+1)[1:]

plt.figure(figsize=(11.5,8))
plt.title("Pre-test of fitting $sin(x)+1$",fontsize=20)
plt.plot(np.linspace(0,13,100),initial_function(np.linspace(0,13,100)),'--k')
for x1,x2 in zip(x1arr,x2arr):
    bin1 = BN(initial_function,x1,x2)
    x0 = bin1.x0
    k  = bin1.k
    #bin1.NDF(x0)
    kf  = bin1.k
    x0  = bin1.x0
    state = bin1.NDF_state
    """
    if(state):
        if(k>0):
            n1 = bin1.NDF(x0)
            n2 = bin1.NDF(x2)
            plt.plot([x0,x2],[n1,n2],'-b')
        else:
            n1 = bin1.NDF(x1)
            n2 = bin1.NDF(x0)
            plt.plot([x1,x0],[n1,n2],'-b')
    else:
        n1 = bin1.NDF(x1)
        n2 = bin1.NDF(x2)
        plt.plot([x1,x2],[n1,n2],'-b')
    """
    n1 = bin1.Normal_NDF(x1)
    n2 = bin1.Normal_NDF(x2)
    plt.plot([x1,x2],[n1,n2],'-b',linewidth=1)
    print("s=%d x1=%5.2f x2=%5.2f x0=%5.2f\nk=%f k=%f\nn1=%f n2=%f\n" %(state,x1,x2,x0,k,kf,n1,n2))
    plt.vlines(x1,-0.6,n1,color='b')
    plt.vlines(x2,-0.6,n2,color='b')
    #plt.hlines(0,0,30,linestyle="dashed",linewidth=0.5)
plt.yticks([0,1,2],["0","1","2",""])
plt.ylabel("$n(x)$",fontsize=16)
plt.xlabel("X",fontsize=16)
plt.xlim(0,xmax)
plt.ylim(-0.6,3)
plt.savefig("r.png",dpi=100)
plt.show()