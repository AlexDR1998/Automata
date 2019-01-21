import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal



"""
    1 dimensional cellular automata, with multiple colour options

"""


class Grid(object):


    def __init__(self,size,density,states,nsize,iterations):
        self.size = size
        #selection of initialising methods for state
        self.init_density = density
        self.nsize = nsize
        #nsize should be odd number
        self.states = states
        #self.current_state = np.zeros(size)
        self.rule = np.random.randint(states,size=states**nsize)
        self.rule[0]=0
        for x in range(states**nsize):
            if np.random.rand()>0.6:
                self.rule[x]=0
        if self.nsize%2==0:
            for x in range(states):
                self.rule[x]=x
        #self.current_state = np.random.randint(self.states,size=self.size)

        self.previous_state = np.zeros(size)
        self.next_state = np.zeros(size)
        #store data here for image output
        self.image = np.zeros((iterations,size))
        self.max_iters = iterations



    def update_cell(self,pos):
        """
        updates cell at position pos to next state, based on rule
        and neighbouring cells
        """

        n = int((self.nsize-1)/2)
        #print(len(self.rule))
        index = 0
        
        if self.nsize%2==0:
            #if even number of neighbouring cells count, use middle cell of older state
            index = index+self.previous_state[pos]#*self.states**(self.nsize-1)
            
            for x in range(1,self.nsize-1):
                #loop for each neighbouring cell
                index = index+self.current_state[(pos-n+x)%self.size]*self.states**x
                #print(x)
            

        else:
            for x in range(self.nsize):

                #loop for each neighbouring cell
                index = index+self.current_state[(pos-n+x)%self.size]*self.states**x
                #print(x)



        self.next_state[pos]=self.rule[int(index)]

    def run(self):

        """
        runs update_cell on every cell in a given state, then updates the state,
        then repeats. Saves data to self.image for output
        """
        self.current_state = np.random.randint(self.states,size=self.size)

        for x in range(self.size):
            if np.random.rand()>self.init_density:
                self.current_state[x]=0

        for i in range(self.max_iters):
            for x in range(self.size):
                self.update_cell(x)
            self.previous_state = self.current_state
            self.current_state = self.next_state
            self.image[i] = self.current_state


    def im_out(self):
        return self.image


    #def save(self):


class Grid2D(object):

    """
    2 dimensional grid cellular automata - like game of life but more colours

           [3]
        [2][1][2]
     [3][1][0][1][3]
        [2][1][2]
           [3]

    """
    def __init__(self,size,density,states,nsize,iterations):
        
        #Size of square grid
        self.size = size
        #Density of random initial state - between 0 and 1
        self.init_density = density
        #Number of states
        self.states = states
        #Size of neighbourhood
        #        [3]
        #     [2][1][2]
        #  [3][1][0][1][3]
        #     [2][1][2]
        #        [3]
        #
        self.nsize = nsize
        s = self.states
        if self.nsize==2:
            self.k = np.array([[s,s**2,s],[s**2,1,s**2],[s,s**2,s]])
        elif self.nsize==3:
            self.k = np.array([[0,s,s**2,s,0],[s,s**2,s**3,s**2,s],
                [s**2,s**3,1,s**3,s**2],
                [s,s**2,s**3,s**2,s],[0,s,s**2,s,0]])

        elif self.nsize==1:
            self.k = np.array([[0,s,0],[s,1,s],[0,s,0]])
        #How many iterations to run the code for
        self.max_iters = iterations
        #Call rule generator
        self.rule_gen()
        #Define variable for storing next state
        self.next_state = np.zeros((size,size))
        #Store all data for passing to visualiser program
        self.image = np.zeros((iterations,size,size))
        self.starttype = 0



    def change_start_type(self):
        self.starttype = 1 - self.starttype

    def update_cell(self,xpos,ypos):
        #---Disused

        index = 0
        if self.nsize==1:

            for x in [0,1,2]:
                index=index+self.current_state[(xpos+x-1)%self.size,ypos]*self.states**abs(x-1)
            for y in [0,2]:
                index=index+self.current_state[xpos,(ypos+y-1)%self.size]*self.states**abs(y-1)
        if self.nsize==2:
            for x in [0,1,2]:
                for y in [0,1,2]:
                    index=index+self.current_state[(xpos+x-1)%self.size,(ypos+y-1)%self.size]*self.states**(abs(x-1)+abs(y-1))
        if self.nsize==3:
            for x in [0,1,2,3,4]:
                for y in [0,1,2,3,4]:
                    if abs(x-2)+abs(y-2)<4:
                        index = index + self.current_state[(xpos+x-2)%self.size,(ypos+y-2)%self.size]*self.states**(abs(x-2)+abs(y-2))

        self.next_state[xpos,ypos]=self.rule[int(index)%len(self.rule)]


    def init_grid(self):
        if self.starttype==0:
            self.current_state = np.random.randint(self.states,size=(self.size,self.size))
        elif self.starttype==1:
            condition = np.mean(((self.size/2.0-np.mgrid[0:self.size:1, 0:self.size:1])**2),axis=0)<self.size/2
            #print(condition)
            r = np.random.randint(self.states,size=(self.size,self.size))
            #print(r)
            z = np.zeros((self.size,self.size)).astype(int)
            self.current_state = np.where(condition,r,z)
            #print(self.current_state)

    def run(self):
        #initialises random starting state
        
        self.init_grid()


        p = len(self.rule)

        for i in range(self.max_iters):
            self.next_state=signal.convolve2d(self.current_state,self.k,boundary='wrap',mode='same')

            v = np.vectorize(lambda y:self.rule[y%p])
            self.current_state = v(self.next_state)
            self.image[i] = self.current_state
        """
        for x in range(self.size):
            for y in range(self.size):
                #if ((x-self.size/2)**2+(y-self.size/2)**2)>32:
                #    self.current_state[x,y]=0
                if np.random.rand()>self.init_density:
                    self.current_state[x,y]=0
        for i in range(self.max_iters):
            for x in range(self.size):
                for y in range(self.size):

                    self.update_cell(x,y)
            self.current_state = self.next_state
            self.image[i] = self.current_state
        """   

    def im_out(self):
        return self.image

    def rule_gen(self):
        n = self.states-1
        #print(self.nsize)
        for x in range(0,self.nsize):
            n = n + 4*(self.states-1)*self.states**(x+1)

        #print(n+1)

        self.rule = np.random.randint(self.states,size=(n+1))
        """
        self.rule = np.random.poisson(1,size=(n+1))
        self.rule = (self.rule/float(np.max(self.rule))*self.states).astype(int)
        #print(self.rule)

        self.rule[0]=0
        #print(self.rule)
        """
        xs = np.linspace(0,n,n+1)
        g = np.vectorize(lambda y:gaussian(y,xs[int(np.floor(n/2))],(n/5)))
        gxs = g(xs)
        rs = np.random.rand(n+1)
        zs = np.zeros(n+1)
        #print(len(gxs))
        #print(len(rs))
        #print(len(zs))
        #print(len(self.rule))


        self.rule = np.where(rs<gxs,self.rule,zs).astype(int)
        #print(self.rule)
        """
        
        for x in range(n):
            #gs[x] = gaussian(xs[x],xs[int(np.floor(n/6))],(n/10))   #6, 10        
            if np.random.rand()>gaussian(xs[x],xs[int(np.floor(n/2))],(n/5)):#float(x)/n:
                self.rule[x]=0
        """

    def rule_save(self,name):

        """
        method to save a rule to csv format
        """
        nlist = open('2D_rules/n'+str(self.nsize)+'/s'+str(self.states)+'/namelist.txt','a')
        nlist.write('\n'+name)
        nlist.close()



        f = open('2D_rules/n'+str(self.nsize)+'/s'+str(self.states)+'/'+name+'.csv','w+')

        np.save(f,self.rule)
        f.close()


    def rule_load(self,name):
        """
        loads a previously saved rule
        """

        f = open('2D_rules/n'+str(self.nsize)+'/s'+str(self.states)+'/'+name+'.csv','r')#
        self.rule = np.load(f)
        f.close()


    def rule_input(self,rule_in):
        """
        Allows another program to feed in a rule
        """
        self.rule = rule_in

    def rule_perm(self):
        """
        method to slightly modify a rule
        """
        for x in range(len(self.rule)):
            r = np.random.rand()
            if r>0.95:
                self.rule[x]=(self.rule[x]+1)%self.states
            elif r<0.05:
                self.rule[x]=(self.rule[x]-1)%self.states

        self.rule[0]=0

    def rule_smooth(self,amount):
        """
        smoothes a rule, element by element
        """
        s_rule = self.rule
        for x in range(len(self.rule)):
            n_mean=0
            for y in range(2*amount+1):
                n_mean = n_mean+self.rule[(x+y-amount)%len(self.rule)]
            n_mean = (n_mean//(2*amount+1))%self.states
            s_rule[x] = n_mean
        self.rule = s_rule

    def rule_fold(self,amount):
        """
        a bit like the opposit of the smooth method
        """
        s_rule = self.rule
        for x in range(len(self.rule)):
            n_sum=0
            for y in range(2*amount+1):
                n_sum = n_sum+self.rule[(x+y-amount)%len(self.rule)]
            n_sum = n_sum%self.states
            s_rule[x] = n_sum
        self.rule = s_rule


    def rule_comb(self,rule_name_1,rule_name_2,mode):
        """
        combines 2 rules together, element by element
        """
        
        if rule_name_1=="":
            rule1=self.rule
        else:
            f = open('2D_rules/n'+str(self.nsize)+'/s'+str(self.states)+'/'+rule_name_1+'.csv','r')
            rule1 = np.load(f)
            f.close()
        
        if rule_name_2=="":
            rule2=self.rule
        else:
            f = open('2D_rules/n'+str(self.nsize)+'/s'+str(self.states)+'/'+rule_name_2+'.csv','r')
            rule2 = np.load(f)
            f.close()
        

        if mode=="+":
            self.rule = np.remainder((rule1 + rule2),self.states)
        if mode=="*":
            self.rule = np.remainder((rule1 * rule2),self.states)
        if mode=="-":
            self.rule = np.remainder((rule1 - rule2),self.states)
        if mode=="m":
            self.rule = np.remainder((rule1 + rule2)//2,self.states)
        if mode=="z":
            for x in range(len(self.rule)):
                if x%2==0:
                    self.rule[x]=rule1[x]
                else:
                    self.rule[x]=rule2[x]
        if mode=="c":
            #print(len(self.rule)//2)
            for x in range(len(self.rule)):
                if x < (len(self.rule)//2):
                    
                    self.rule[x]=rule1[x]
                else:
                    
                    self.rule[x]=rule2[x]



    def rule_inv(self):
        """
        inverts a rule, element by element
        """
        self.rule = self.states - self.rule

    def store(self,t,name):
        if t=="d":
            f = open('2D_rules/n1/ml_data/training/dead/'+name+'.csv','w+')
        elif t=="i":
            f = open('2D_rules/n1/ml_data/training/interesting/'+name+'.csv','w+')
        elif t=="s":
            f = open('2D_rules/n1/ml_data/training/s_noise/'+name+'.csv','w+')
        elif t=="n":
            f = open('2D_rules/n1/ml_data/training/noise/'+name+'.csv','w+')
        np.save(f,self.rule)
        f.close()
        


def gaussian(x,mu,sig):
    return np.exp(-np.power(x-mu,2)/(2*np.power(sig,2)))

def multi_subset(x,y):
    """
    how many possible arangements of y items, of x choices, where duplicates are alowed.
    (1,2,2)=(2,1,2)!=(1,1,2)
    """
    return math.factorial(x+y-1)/(math.factorial(y)*math.factorial(x-1))
    