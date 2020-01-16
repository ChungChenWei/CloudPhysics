import numpy as np
import matplotlib.pyplot as plt
from bin_node import Bin_Node as BN
import scipy.integrate as integrate

normal_constant = 2.8e-4

B  = 4.15e-6
s  = -0.3
dt = 1

print_lapse  = 100
iteration    = 5001

num_of_bin = 30 #number of bins
xmax = 10


def initial_function(x): #initial distribution
    #return -3.*x+2.5
    #return 3.*x-0.5
    #return np.sin(x)+1.
    x /= normal_constant
    return 4.*x*np.exp(-2.*x)
    #return x+1


def x_growth_function(x,N,M):
    if(N==0 or M==0):
        return x
    return x+B*s*(M/N)**(1/3)*dt*(x/(M/N))**(1/3)
    #return x+0.5

def M_growth_function(N,M):
    if(N==0):
        return M
    return M+N*B*s*(M/N)**(1/3)*dt
    #return M+N*0.5




power = 2.0
#x1arr = np.linspace(0,xmax,num_of_bin+1)[:-1]
x1arr = (power**(np.arange(0,num_of_bin+1)[:-1]-25))*normal_constant/1.47
#x2arr = np.linspace(0,xmax,num_of_bin+1)[1:]
x2arr = (power**(np.arange(0,num_of_bin+1)[1:]-25))*normal_constant/1.47
dxarr = np.zeros(num_of_bin)
dNarr = np.zeros(num_of_bin)
dMarr = np.zeros(num_of_bin)
Bindic = {}

Nanalytical = np.zeros(num_of_bin)


for i,x1,x2 in zip(range(num_of_bin),x1arr,x2arr):
    Bin = BN(initial_function,x1,x2)
    Bindic[i] = Bin

plt.figure(figsize=(11.5,8))
for t in range(iteration):

    N = np.zeros(num_of_bin)
    
    #plt.figure(figsize=(11.5,8))

    for i,x1,x2 in zip(range(num_of_bin),x1arr,x2arr):
        #print(i)
        bin1 = Bindic[i]
        bin1.state_compute(bin1.lbound,bin1.rbound)
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
        #print(n1,n2,bin1.x1,bin1.x2)
        #plt.vlines(bin1.x1,0,n1,color='b')
        #plt.vlines(bin1.x2,0,n2,color='b')
        #plt.hlines(0,0,xmax,linestyle="dashed",linewidth=0.5)

        #"""
        dx,dN,dM = bin1.Bin_Shift(x_growth_function,M_growth_function)
        dxarr[i] = dx
        dNarr[i] = dN
        dMarr[i] = dM
        #"""

        Bindic[i] = bin1
    
    if(t%print_lapse == 0):
        def analytical_sol(x):
            
            xt = (x**(2./3.)-2./3.*B*s*t*dt)**(3./2.)
            xt /= normal_constant
            return (x**(-1./3.)*(x**(2./3.)-(2./3.)*B*s*t*dt)**(1./2.))*4.*xt*np.exp(-2.*xt)
        for j in range(num_of_bin):
            Nanalytical[j] = integrate.quad(analytical_sol,x1arr[j], x2arr[j])[0]
        if(not t):
            N_init = np.array(N)
            #print(N_init)
        plt.title("Pre-test of fitting $Gamma Function$",fontsize=20)
        plt.title("t="+str(t),loc='right',fontsize=14)
        plt.plot(N/normal_constant,'o-b',label="dt = 1 s")
        plt.plot(N_init/normal_constant,'o-k',label="Initial")
        plt.plot(Nanalytical/normal_constant,'+--r',label="analytical")
        plt.ylabel("NORMALIZED NUMBER IN THE BIN",fontsize=16)
        plt.xlabel("BIN NUMBER",fontsize=16)
        plt.yticks(np.arange(0,0.41,0.05),["0","","0.1","","0.2","","0.3","","0.4"])
        plt.xticks(np.arange(14,30),["15","","","","","20","","","","","25","","","","","30"])
        plt.xlim(14,num_of_bin-1)
        plt.ylim(0,0.4)
        plt.legend(fontsize=16)
        #plt.grid()
        plt.savefig("./pic/"+str(t)+".png",dpi=100)
        #plt.show()
        plt.clf()
        print(t,"Done")

    for i in range(num_of_bin):
        #"""
        #print("Before M,N shift from others: bin_number=%d, N=%8.10f , M=%8.10f" %(i,Bindic[i].N,Bindic[i].M))
        if(i > 0 ):
            if(dxarr[i-1]>0):
                Bindic[i].M += dMarr[i-1]
                Bindic[i].N += dNarr[i-1]
                #print("Left push dx=%8.10f, dN=%8.10f, dM=%8.10f" %(dxarr[i-1],dNarr[i-1],dMarr[i-1]))
        if(i != num_of_bin-1):
            if(dxarr[i+1]<0):
                Bindic[i].M += dMarr[i+1]
                Bindic[i].N += dNarr[i+1]
                #print("Right push dx=%8.10f, dN=%8.10f, dM=%8.10f" %(dxarr[i-1],dNarr[i-1],dMarr[i-1]))
        #print("After M,N shift from others: bin_number=%d, N=%8.10f , M=%8.10f" %(i,Bindic[i].N,Bindic[i].M))
        #"""
    #print('')