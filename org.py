import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal
import sys


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
        self.init_state = np.zeros(size)
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

    def initialise_random(self):
        self.current_state = np.random.randint(self.states,size=self.size)
        for x in range(self.size):
            if np.random.rand()>self.init_density:
                self.current_state[x]=0
        self.init_state = self.current_state[:]

    def initialise_mutate(self,n):
        r = np.random.randint(0,self.size,n)
        self.init_state[r] = np.random.randint(0,self.states,n)

    def run(self):

        """
        runs update_cell on every cell in a given state, then updates the state,
        then repeats. Saves data to self.image for output
        """
        
        self.current_state = self.init_state[:]
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
    def __init__(self,size,density,states,nsize,iterations,symmetries=1):
        
        #Size of square grid
        self.size = size
        #Density of random initial state - between 0 and 1
        self.init_density = density
        #Number of states
        self.states = states
        #Define density matrix
        self.d = np.zeros((self.states,self.states))
        #Size of neighbourhood
        #        [3]
        #     [2][1][2]
        #  [3][1][0][1][3]
        #     [2][1][2]
        #        [3]
        #
        self.nsize = nsize
        self.symmetries = symmetries
        s = self.states
        if symmetries==1:
            self.n_struct=np.array([4,4,4])
            _k = 1+(s-1)*self.n_struct
            if self.nsize==2:
                self.k = np.array([[    1,      _k[1],1   ],
                                   [_k[1],_k[0]*_k[1],_k[1]],
                                   [    1,      _k[1],1   ]])
            elif self.nsize==3:
                self.k = np.array([[    0,          1,            _k[2],          1,0   ],
                                   [    1,      _k[2],      _k[2]*_k[1],      _k[2],1   ],
                                   [_k[2],_k[2]*_k[1],_k[2]*_k[1]*_k[0],_k[2]*_k[1],_k[2]],
                                   [    1,      _k[2],      _k[2]*_k[1],      _k[2],1   ],
                                   [    0,          1,            _k[2],          1,0   ]])

            elif self.nsize==1:
                self.k = np.array([[0,    1,0],
                                   [1,_k[0],1],
                                   [0,    1,0]])
        elif symmetries==2:
            self.n_struct=np.array([8,16])
            _k = 1+(s-1)*self.n_struct
            if self.nsize==1:
                self.k=np.array([[1,    1,1],
                                 [1,_k[0],1],
                                 [1,    1,1]])
            elif self.nsize==2:
                self.k=np.array([[1,    1,          1,    1,1],
                                 [1,_k[1],      _k[1],_k[1],1],
                                 [1,_k[1],_k[0]*_k[1],_k[1],1],
                                 [1,_k[1],      _k[1],_k[1],1],
                                 [1,    1,          1,    1,1]])
        elif symmetries==0:
            if self.nsize==1:
                self.n_struct=np.array([8,0])
                _k = 1+(s-1)*self.n_struct
                self.k=np.array([[1,1,1],
                                 [1,_k[0],1],
                                 [1,1,1]])
            elif self.nsize==2:
                self.n_struct=np.array([20,0])
                _k = 1+(s-1)*self.n_struct
                self.k=np.array([[0,1,1,1,0],
                                 [1,1,1,1,1],
                                 [1,1,_k[0],1,1],
                                 [1,1,1,1,1],
                                 [0,1,1,1,0]])

            elif self.nsize==3:
                self.n_struct=np.array([24,0])
                _k = 1+(s-1)*self.n_struct
                self.k=np.array([[1,1,1,1,1],
                                 [1,1,1,1,1],
                                 [1,1,_k[0],1,1],
                                 [1,1,1,1,1],
                                 [1,1,1,1,1]])
        #How many iterations to run the code for
        self.max_iters = iterations
        #Call rule generator
        self.rule_gen()
        #Define array for storing next state
        self.next_state = np.zeros((size,size))
        #Store all data for passing to visualiser program
        self.image = np.zeros((iterations,size,size))
        self.starttype = 0
        #food modes force small sections of the grid to remain at the highest cell value
        self.food = 0



    def print_info(self):
        #Prints info about current rule parameters
        print("Number of states: "+str(self.states))
        print("Rule length: "+str(len(self.rule)))
        print("Number of possible rules (log2): "+str(np.log2(float(self.states))*float(len(self.rule))))

    def print_rule(self):
        print(self.rule)
    def change_start_type(self):
        self.starttype = 1 - self.starttype

    def food_mode(self,mode):
        self.food = mode

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
            self.init_state = np.copy(self.current_state)
        elif self.starttype==1:
            condition = np.mean(((self.size/2.0-np.mgrid[0:self.size:1, 0:self.size:1])**2),axis=0)<self.size/2
            #print(condition)
            r = np.random.randint(self.states,size=(self.size,self.size))
            #print(r)
            z = np.zeros((self.size,self.size)).astype(int)
            self.current_state = np.where(condition,r,z)
            self.init_state = np.copy(self.current_state)
            #print(self.current_state)

    def run_mutate(self,am=0.1):
        #Runs 4 rules in parallel, 1 on each corner.
        #Rules are mutated from initial rule
        self.init_grid()
        length = len(self.rule)

        rule1 = np.zeros(length)
        rule2 = np.zeros(length)
        rule3 = np.zeros(length)
        rule4 = np.zeros(length)
        #print(len(rule1))
        rule1 = np.array(self.rule,copy=True)
        rule2 = np.array(self.rule,copy=True)
        rule3 = np.array(self.rule,copy=True)
        rule4 = np.array(self.rule,copy=True)


        #print(len(rule1))
        #print(self.rule.shape)
        #print(self.rule)
        for x in range(0,int(length/4)):
            #print(x)
            r = np.random.rand()
            if r>1-am:
                rule1[x]=(rule1[x]+1)%self.states
            elif r<am:
                rule1[x]=(rule1[x]-1)%self.states
        
        for x in range(int(length/4),int(length/2)):
            r = np.random.rand()
            if r>1-am:
                rule2[x]=(rule2[x]+1)%self.states
            elif r<am:
                rule2[x]=(rule2[x]-1)%self.states
        
        for x in range(int(length/2),int(3*length/4)):
            r = np.random.rand()
            rule3[x]=0
            if r>1-am:
                rule3[x]=(rule3[x]+1)%self.states
            elif r<am:
                rule3[x]=(rule3[x]-1)%self.states
        
        for x in range(int(3*length/4),int(length)):
            r = np.random.rand()
            #rule4[x] = self.states-1
            if r>1-am:
                rule4[x]=(rule4[x]+1)%self.states
            elif r<am:
                rule4[x]=(rule4[x]-1)%self.states

        #print(self.rule-rule1)
        #print(self.rule-rule2)
        #print(self.rule-rule3)
        #print(self.rule-rule4)


        p = len(self.rule)
        zs = np.zeros((self.size/2,self.size/2))
        for i in range(self.max_iters):
            self.next_state=signal.convolve2d(self.current_state,self.k,boundary='wrap',mode='same')
            v1 = np.vectorize(lambda y:rule1[y%p])
            v2 = np.vectorize(lambda y:rule2[y%p])
            v3 = np.vectorize(lambda y:rule3[y%p])
            v4 = np.vectorize(lambda y:rule4[y%p])


            self.current_state[0:self.size/2,0:self.size/2] = v1(self.next_state[0:self.size/2,0:self.size/2])
            self.current_state[0:self.size/2,self.size/2:self.size] = v2(self.next_state[0:self.size/2,self.size/2:self.size])
            self.current_state[self.size/2:self.size,0:self.size/2] = v3(self.next_state[self.size/2:self.size,0:self.size/2])
            self.current_state[self.size/2:self.size,self.size/2:self.size] = v4(self.next_state[self.size/2:self.size,self.size/2:self.size])
            self.current_state[:][self.size/2]=0
            self.current_state[self.size/2][:]=0

            self.image[i] = self.current_state

        self.child1 = rule1
        self.child2 = rule2
        self.child3 = rule3
        self.child4 = rule4

    def choose_offspring(self,best):

        if best==1:
            self.rule=np.array(self.child1,copy=True)
        elif best==2:
            self.rule=np.array(self.child2,copy=True)
        elif best==3:
            self.rule=np.array(self.child3,copy=True)
        elif best==4:
            self.rule=np.array(self.child4,copy=True)

    def run(self,random_init=True,compute_t_matrix=False):
        #initialises random starting state
        if random_init:
            self.init_grid()
        self.d = np.zeros((self.states,self.states))
        s = self.states
        ss = np.arange(s)
        p = len(self.rule)

        for i in range(self.max_iters):
            #Convolve to calculate next global state
            v = np.vectorize(lambda y:self.rule[y%p])
            self.next_state=v(signal.convolve2d(self.current_state,self.k,boundary='wrap',mode='same'))
            
            #Update transition density matrix - gets slow for large number of states
            if compute_t_matrix:
                g_i = np.tile(self.current_state,(s,1,1))
                g_j = np.tile(self.next_state,(s,1,1))
                g_i_eq = np.equal(g_i,ss[:,None,None]).astype(int)
                g_j_eq = np.equal(g_j,ss[:,None,None]).astype(int)
                self.d+=np.einsum('ixy,jxy->ij',g_i_eq,g_j_eq)
            

            #for a in range(self.states):
            #    for b in range(self.states):
            #        self.d[a,b]+=np.count_nonzero(np.logical_and(self.current_state==a,self.next_state==b))
            
            #Set current state to new state
            self.current_state = self.next_state
            if self.food==1:
                self.current_state[:,0] = self.states-1
            if self.food==2:
                self.current_state[(self.size/2-10):(self.size/2+10),(self.size/2-10):(self.size/2+10)] = self.states-1
            self.image[i] = self.current_state
        #print(self.iterations)
        """
        for x in range(self.size):
            for y in range(self.size):
                #if ((x-self.size/2)**2+(y-self.size/2)**2)>32:
                #    self.current_state[x,y]=0
                if np.random.rand()>self.init_density:
                    self.current_state[x,y]=0
        for i in range(self.max_iters):
            for x in range(self.size):african
                for y in range(self.size):

                    self.update_cell(x,y)
            self.current_state = self.next_state
            self.image[i] = self.current_state
        """   

    def im_out(self):
        return self.image


    def density_matrix(self,random_init=True):

        self.run(random_init,True)
        self.d = (self.d.T/np.sum(self.d,axis=1)).T 
        #print(np.linalg.eig(self.d)[0])
        #print(self.d)
        #self.d/=np.sum(self.d,axis=1)
        return self.d#/(l*10)


    def fft(self):
        print(np.mean(self.image,axis=(1,2)).shape)
        self.fft_data = np.abs(np.fft.fftshift(np.fft.fft2(self.image-np.mean(self.image,axis=(1,2))[:,None,None])))

    def lyap(self,N):
        #Run N simulations that differ by 1 cell in initial configuration from a given start state. Returns difference
        self.init_grid()
        #print("_"*N)
        lyap_data = np.zeros((N+2,self.max_iters))
        for i in range(N):
            sys.stdout.write("#")
            sys.stdout.flush()
            x = np.random.randint(self.size)
            y = np.random.randint(self.size)
            
            #for i in range(1):
            #print(np.sum((self.current_state-grid_2)**2))
            
            self.current_state = np.copy(self.init_state)
            self.run(False)
            data1 = np.copy(self.image)

            grid_2 = np.copy(self.init_state)
            grid_2[x,y]= (grid_2[x,y]+ np.random.randint(1,self.states))%self.states
            
            self.current_state = np.copy(grid_2)
            self.run(False)
            data2 = np.copy(self.image)
            
            lyap_data[i+2] = np.sum((data1-data2)**2,axis=(1,2))/(self.size**2) 
        #plt.plot()
        #plt.show()
        sys.stdout.write("|")
        sys.stdout.flush()
        self.image = data1-data2
        return lyap_data

    def entropy(self):
        #Calculate information entropy at each timestep
        p = np.zeros((self.states,self.max_iters))
        print(self.image.shape)
        ss = np.arange(self.states)
        s = np.equal(np.tile(self.image,(self.states,1,1,1)),ss[:,None,None,None])
        p = np.sum(s,axis=(2,3))/float(self.size*self.size)
        en = -np.sum(p*np.log(p),axis=0)
        return en
        #print(en)
        #print(s.shape)

    def rule_gen(self,mu=0.5,sig=0.2):
        #n = self.states-1
        #print(self.nsize)
        #for x in range(0,self.nsize):
        #    n = n + self.n_struct[x]*(self.states-1)*self.states**(x+1)
        n = self.states
        for x in range(0,self.nsize):
            n*=(1+(self.states-1)*self.n_struct[x])
        #print(n+1)


        self.rule = np.random.randint(self.states,size=n)
        """
        self.rule = np.random.poisson(1,size=(n+1))
        self.rule = (self.rule/float(np.max(self.rule))*self.states).astype(int)
        #print(self.rule)

        self.rule[0]=0
        #print(self.rule)
        """
        xs = np.linspace(0,n,n)
        g = np.vectorize(lambda y:gaussian(y,xs[int(np.floor(n*mu))],(n*sig)))
        gxs = g(xs)
        rs = np.random.rand(n)
        zs = np.zeros(n)
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
        nlist = open('2D_rules/sym'+str(self.symmetries)+'/n'+str(self.nsize)+'/s'+str(self.states)+'/namelist.txt','a')
        nlist.write('\n'+name)
        nlist.close()



        f = open('2D_rules/sym'+str(self.symmetries)+'/n'+str(self.nsize)+'/s'+str(self.states)+'/'+name+'.csv','w+')

        np.save(f,self.rule)
        f.close()


    def rule_load(self,name):
        """
        loads a previously saved rule
        """

        f = open('2D_rules/sym'+str(self.symmetries)+'/n'+str(self.nsize)+'/s'+str(self.states)+'/'+name+'.csv','r')#
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
    