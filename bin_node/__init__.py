import scipy.integrate as integrate

class Bin_Node:
    # INDF Initial Number Density Function
    def __init__(self,INDF,x1,x2):
        self.INDF = INDF
        self.x1   = x1
        self.x2   = x2
        self.N    = integrate.quad(self.INDF,         self.x1, self.x2)[0]
        self.M    = integrate.quad(self.First_Moment, self.x1, self.x2)[0]
        self.state_compute(x1,x2)

    def state_compute(self,x1,x2):
        delx = (x2-x1)
        self.xs    = x1
        self.xe    = x2
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
        a_b2 = (a**2-b**2)/2.0
        a_b3 = (a**3-b**3)/3.0
        return n0*a_b2+k*(a_b3-x0*a_b2)

    def Bin_Shift(self,x_growth_function,M_growth_function):
        # Assume x1 and x2 will grouth same direction
        x1_n = x_growth_function(self.x1)
        x0_n = x_growth_function(self.x0)
        x2_n = x_growth_function(self.x2)
        delx = x1_n-self.x1
        self.M = M_growth_function(self.N,self.M)

        self.state_compute(x1_n,x2_n)
        self.NDF()

        if(delx>0):
            if(self.NDF_state==1):
                a = self.x2
                if(self.x0<=self.x2):
                    b = self.x2
                else:
                    b = self.x0
            else:
                a = self.x2
                b = x2_n
        else:
            if(self.NDF_state==2):
                a = self.x0
                if(self.x0>=self.x1):
                    b = self.x0
                else:
                    b = self.x1
            else:
                a = x1_n
                b = self.x1

        delN = self.del_N_function(self.n0,self.x0,self.k,a,b)
        delM = self.del_M_function(self.n0,self.x0,self.k,a,b)

        self.N -= delN
        self.M -= delM

        return delx,delN,delM