import numpy as np
import matplotlib.pyplot as plt
from bin_node import Bin_Node as BN

def initial_function(x): #initial distribution
    #return -3.*x+2.5
    #return 3.*x-0.5
    #return np.sin(x)+1.
    #x /= 2.8e-4
    return 4.*x*np.exp(-2.*x)

B  = 5e-4
s  = -0.3
dt = 1.
k  = 5

def x_growth_function(x,N,M):
    return x+B*s*(M/N)**(1/3)*dt*(x/(M/N))**(1/3)

def M_growth_function(N,M):
    return M+N*B*s*(M/N)**(1/3)*dt

num_of_bin = 30 #number of bins
xmax = 10

power = 2.0
#x1arr = np.linspace(0,xmax,num_of_bin+1)[:-1]
x1arr = (power**(np.arange(0,num_of_bin)[:-1]-12))/power**(13)
#x2arr = np.linspace(0,xmax,num_of_bin+1)[1:]
x2arr = (power**(np.arange(0,num_of_bin)[1:]-12))/power**(13)
dxarr = np.zeros(num_of_bin)
dNarr = np.zeros(num_of_bin)
dMarr = np.zeros(num_of_bin)
Bindic = {}


for i,x1,x2 in zip(range(num_of_bin),x1arr,x2arr):
    Bin = BN(initial_function,x1,x2)
    Bindic[i] = Bin

for t in range(100):

    N = np.zeros(num_of_bin)

    for i,x1,x2 in zip(range(num_of_bin),x1arr,x2arr):
        #print(i)
        bin1 = Bindic[i]
        bin1.state_compute(bin1.x1,bin1.x2)
        bin1.NDF()

        #print("t=%d, i=%d, s=%d, x1=%4.3f, x2=%4.3f, x0=%4.3f" %(t,i,bin1.NDF_state,bin1.x1,bin1.x2,bin1.x0))
        
        if(bin1.NDF_state):
            if(bin1.k>0):
                n1 = bin1.NDF(bin1.x0)
                n2 = bin1.NDF(bin1.x2)
                #plt.plot([bin1.x0,bin1.x2],[n1,n2],'-b')
            else:
                n1 = bin1.NDF(bin1.x1)
                n2 = bin1.NDF(bin1.x0)
                #plt.plot([bin1.x1,bin1.x0],[n1,n2],'-b')
        else:
            n1 = bin1.NDF(bin1.x1)
            n2 = bin1.NDF(bin1.x2)
            #plt.plot([bin1.x1,bin1.x2],[n1,n2],'-b')
        N[i] = bin1.N

        #print("\tn1=%4.3f, n2=%4.3f\n" %(n1,n2))

        #"""
        dx,dN,dM = bin1.Bin_Shift(x_growth_function,M_growth_function)
        dxarr[i] = dx
        dNarr[i] = dN
        dMarr[i] = dM
        #"""

        Bindic[i] = bin1

        #print(n1,n2,bin1.x1,bin1.x2)
        #plt.vlines(bin1.x1,0,n1,color='b')
        #plt.vlines(bin1.x2,0,n2,color='b')
        #plt.hlines(0,0,xmax,linestyle="dashed",linewidth=0.5)
    if(t%k == 0):
        plt.figure(figsize=(11.5,8))
        plt.title("Pre-test of fitting $sin(x)+1$",fontsize=20)
        plt.title("t="+str(t),loc='right',fontsize=14)
        plt.plot(N,'o-b')
        plt.ylabel("$n(x)$",fontsize=16)
        plt.xlabel("X",fontsize=16)
        plt.xlim(0,num_of_bin)
        plt.ylim(0,0.35)
        plt.savefig("./pic/"+str(t)+".png",dpi=100)
        plt.show()
        plt.clf()

    for i in range(num_of_bin):
        #"""
        #print("i=%d, N=%4.3f , M=%4.3f" %(i,Bindic[i].N,Bindic[i].M))
        if(i > 0 ):
            if(dxarr[i-1]>0):
                Bindic[i].M += dMarr[i-1]
                Bindic[i].N += dNarr[i-1]
        if(i != num_of_bin-1):
            if(dxarr[i+1]<0):
                Bindic[i].M += dMarr[i+1]
                Bindic[i].N += dNarr[i+1]
        #print("i=%d, N=%4.3f , M=%4.3f" %(i,Bindic[i].N,Bindic[i].M))
        #"""