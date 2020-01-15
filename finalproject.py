
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

ndf_normal = 0 #n0+k(x-x0)
ndf_modified = 1 #k_star(x-x_star)
num_of_bin = 30 #number of bins
xmax = 30

x1arr = np.linspace(0,xmax,num_of_bin+1)[:-1] #array of x1
x2arr = np.linspace(0,xmax,num_of_bin+1)[1:]
ndftarr = np.zeros(num_of_bin,dtype=int) #number density function type
karr = np.zeros(num_of_bin)
n0arr = np.zeros(num_of_bin)
x_stararr = np.zeros(num_of_bin)
Narr = np.zeros(num_of_bin)
Marr = np.zeros(num_of_bin)

def linearndf(M,N,x1,x2): #number density function
    x0 = (x1+x2)/2.0
    n0 = N/(x2-x1)
    k = 12.0*(M-x0*N)/(x2-x1)**3.0
    nx1 = n0 + k*(x1-x0)
    nx2 = n0 + k*(x2-x0)
    if nx1 < 0: #n(x1) < 0
        x_star = 3.0*M/N - 2.0*x2
        k_star = 2.0*N/(x2-x_star)**2.0
        return ndf_modified, x_star, k_star
    elif nx2 < 0: #n(x2) < 0
        x_star = 3.0*M/N - 2.0*x1
        k_star = (-2.0)*N/(x1-x_star)**2.0
        return ndf_modified, x_star, k_star
    return ndf_normal, n0, k

def f_N(n0,x0,k,a,b): #bin shift Number
    return (b-a)*(n0-k*(x0-(a+b)/2.0))

def f_M(n0,x0,k,a,b): #bin shift Mass
    a_b2 = (a**2-b**2)/2.0
    a_b3 = (a**3-b**3)/3.0
    return n0*a_b2+k*(a_b3-x0*a_b2)

def initial_function(x): #initial distribution
    return np.sin(x)+1

def initial_bins(x1ar, x2ar, n): #initiate the bin with given initial funcion
    def integrand(x):
        return x*initial_function(x)
    Nar = np.zeros(n)
    Mar = np.zeros(n)
    for i in range(n):
        Nar[i] = integrate.quad(initial_function, x1ar[i], x2ar[i])[0]
        Mar[i] = integrate.quad(integrand, x1ar[i], x2ar[i])[0]
    return Nar, Mar

Narr, Marr = initial_bins(x1arr, x2arr, num_of_bin)
for i in range(num_of_bin):
    ndftarr[i], temp, karr[i] = linearndf(Marr[i],Narr[i],x1arr[i],x2arr[i])
    if ndftarr[i] == ndf_normal:
        n0arr[i] = temp
    else:
        x_stararr[i] = temp

plt.figure()
for i in range(num_of_bin):
    if ndftarr[i] == ndf_normal:
        x0 = (x1arr[i]+x2arr[i])/2.0
        n1 = n0arr[i]+karr[i]*(x1arr[i]-x0)
        n2 = n0arr[i]+karr[i]*(x2arr[i]-x0)
    else:
        n1 = karr[i]*(x1arr[i]-x_stararr[i])
        n2 = karr[i]*(x2arr[i]-x_stararr[i])
    plt.plot([x1arr[i],x2arr[i]],[n1,n2],'-b')
plt.show()