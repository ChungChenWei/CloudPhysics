import scipy.integrate as integrate

class Bin_Node:
    # INDF Initial Number Density Function
    def __init__(self,INDF,x1,x2):
        self.INDF   = INDF
        self.lbound = x1
        self.rbound = x2
        self.N      = integrate.quad(self.INDF,         self.lbound, self.rbound)[0]
        self.M      = integrate.quad(self.First_Moment, self.lbound, self.rbound)[0]
        self.state_compute(x1,x2)

    def state_compute(self,x1,x2):
        delx = (x2-x1)
        self.x1    = x1
        self.x2    = x2
        self.x0    = (x1+x2)/2.
        self.n0    = self.N/delx
        self.k     = 12.*(self.M-self.x0*self.N)/delx**3
        self.NDF_state = 0 
        # 0 for Normal, 1 for Modyfied (n(x2)<0), 2 for Modyfied (n(x1)<0)

    def First_Moment(self,x):
        return x*self.INDF(x)

    def Normal_NDF(self,x):
        return self.n0 + self.k * (x - self.x0)

    def Modified_NDF(self,x,x_input,state):
        self.x0  = 3.*self.M/self.N-2.*x_input
        self.k   = (-1)**(state)*2.*self.N/(x_input-self.x0)**2.
        self.n0  = 0.
        self.NDF_state = state
        return self.k * (x - self.x0)

    def NDF(self,x=0):
        #n(x1) < 0
        if (self.Normal_NDF(self.x1) < 0): 
            return self.Modified_NDF(x,self.x2,2)
        #n(x2) < 0
        elif self.Normal_NDF(self.x2) < 0:
            return self.Modified_NDF(x,self.x1,1)
        else:
            return self.Normal_NDF(x)

    # bin shift Number
    def del_N_function(self,n0,x0,k,a,b):
        return (b-a)*(n0-k*(x0-(a+b)/2.0))
    # bin shift Mass
    def del_M_function(self,n0,x0,k,a,b):
        b_a2 = (b**2-a**2)/2.0
        b_a3 = (b**3-a**3)/3.0
        return n0*b_a2+k*(b_a3-x0*b_a2)

    def Bin_Shift(self,x_growth_function,M_growth_function):
        # Assume x1 and x2 will grouth same direction
        self.x1 = x_growth_function(self.lbound,self.N,self.M)
        self.x2 = x_growth_function(self.rbound,self.N,self.M)
        x0_n = x_growth_function(self.x0,self.N,self.M)
        delx = self.x1-self.lbound
        self.M = M_growth_function(self.N,self.M)
        #print("State %d N=%8.10f M=%8.10f" %(self.NDF_state,self.N,self.M))
        #print("After Shift delx=%8.10f, x1=%8.10f,x2=%8.10f" %(delx,self.x1,self.x2))

        self.state_compute(self.x1,self.x2)

        #print("x1=%8.10f x2=%8.10f x0=%8.10f n0=%8.10f k=%8.10f" %(self.x1,self.x2,self.x0,self.n0,self.k))

        self.NDF()

        #print("State %d" %(self.NDF_state))
        #print("After Shift delx=%8.10f, x1=%8.10f,x2=%8.10f" %(delx,self.x1,self.x2))


        if(delx>0):
            if(self.NDF_state==1):
                a = self.rbound
                if(self.x0<=self.rbound):
                    b = self.rbound
                else:
                    b = self.x0
            else:
                a = self.rbound
                b = self.x2
        else:
            if(self.NDF_state==2):
                a = self.x0
                if(self.x0>=self.lbound):
                    b = self.x0
                else:
                    b = self.lbound
            else:
                a = self.x1
                b = self.lbound

        #print("Before del M N n0=%8.10f x0=%8.10f k=%8.10f a=%8.10f b=%8.10f" %(self.n0,self.x0,self.k,a,b))
        delN = self.del_N_function(self.n0,self.x0,self.k,a,b)
        delM = self.del_M_function(self.n0,self.x0,self.k,a,b)

        #print("In the bin before modify\n\tN=%8.10f,M=%8.10f,dN=%8.10f,dM=%8.10f" %(self.N,self.M,delN,delM))

        # numerical fixed
        if self.N-delN < 0:
            delN = self.N
            self.N = 0.
        else:
            self.N -= delN
        if self.M-delM < 0:
            delM = self.M
            self.M = 0.
        else:
            self.M -= delM

        #print("In the bin after modify\n\tN=%8.10f,M=%8.10f,dN=%8.10f,dM=%8.10f" %(self.N,self.M,delN,delM))
        #print("")

        return delx,delN,delM