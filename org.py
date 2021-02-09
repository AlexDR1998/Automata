import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp 
from scipy import signal
from scipy import ndimage
import sys
#from sklearn import neighbors
#import h5py
#import tensorflow as tf 
#from tensorflow import keras


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
        self.image = np.zeros((self.max_iters,self.size,self.size))
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
        zs = np.zeros((self.size//2,self.size//2))
        for i in range(self.max_iters):
            self.next_state=signal.convolve2d(self.current_state,self.k,boundary='wrap',mode='same')
            v1 = np.vectorize(lambda y:rule1[y%p])
            v2 = np.vectorize(lambda y:rule2[y%p])
            v3 = np.vectorize(lambda y:rule3[y%p])
            v4 = np.vectorize(lambda y:rule4[y%p])


            self.current_state[0:self.size//2,0:self.size//2] = v1(self.next_state[0:self.size//2,0:self.size//2])
            self.current_state[0:self.size//2,self.size//2:self.size] = v2(self.next_state[0:self.size//2,self.size//2:self.size])
            self.current_state[self.size//2:self.size,0:self.size//2] = v3(self.next_state[self.size//2:self.size,0:self.size//2])
            self.current_state[self.size//2:self.size,self.size//2:self.size] = v4(self.next_state[self.size//2:self.size,self.size//2:self.size])
            self.current_state[:][self.size//2]=0
            self.current_state[self.size//2][:]=0

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
  

    def im_out(self):
        return self.image


    def n_hot(self,ns):
        #Lets user filter out certain states
        ss = np.arange(self.states)
        g = np.tile(self.image,(self.states,1,1,1))
        im_one_hot = np.equal(g,ss[:,None,None,None]).astype(int)
        self.image = np.zeros(self.image.shape).astype(int)
        for n in ns:
            self.image+=n*im_one_hot[n]

#--- Automated analysis


    def density_matrix(self,random_init=True):

        self.run(random_init,True)
        self.d = (self.d.T/np.sum(self.d,axis=1)).T 
        return self.d

    def fft(self):
        #Performs fft of each state channel seperately and returns (symmetry, [spacial peaks], [temporal peaks])

        #//16 or //32 work best
        filter_resolution=self.size//32

        
        #'background' signal removal parameters
        t0 = self.max_iters//2
        q0 = self.size//2
        
        # One hot decode real space trajectory and fft the indivual binary representations of each state
        im_one_hot = one_hot_encode(self.image,self.states)
        fd = (np.fft.fftshift(np.fft.fftn(im_one_hot,axes=(1,2,3))))
        #sys.stdout.write("#")
        #sys.stdout.flush()
        #Data for visualising purposes
        self.fft_data = np.abs(one_hot_decode(fd))
         
        #Remove background signal (q=0,f=0)
        
        ts,xs,ys=np.meshgrid(np.arange(self.max_iters),np.arange(self.size),np.arange(self.size),indexing="ij")
        gs = gaus3d(ts,xs,ys,t0,q0,q0,4,4,4)
        fd_filtered = np.zeros(fd.shape,dtype=fd.dtype)
        for i in range(self.states):
            fd_filtered[i] = (1-gs)*fd[i]
            #fd_filtered[i] = fd[i]
        #sys.stdout.write("#")
        #sys.stdout.flush()
        #Apply local max filter to enhance peaks and suppress noise
        n_peaks = self.size*self.size//8
        for i in range(self.states):
            for t in range(self.max_iters):
                slice = ndimage.maximum_filter(np.abs(fd_filtered[i,t]),filter_resolution,mode="wrap")
                flat = np.abs((slice)).flatten()
                k = np.partition(flat,-n_peaks)[-n_peaks]
                slice[np.abs(slice)<k]=0
                fd_filtered[i,t] = slice 
        
        #sys.stdout.write("#")
        #sys.stdout.flush()
        #Peak detection along time axis
        #print(fd_filtered[0,:,q0,q0].shape)
        max_t_peaks = np.zeros((4,self.states)).astype(int)
        a,b = sp.signal.butter(3,0.3,analog=False)
        for i in range(self.states):
            tdata = np.pad(np.abs(np.sum(fd_filtered[i,t0:],axis=(1,2))),(0,filter_resolution),mode='reflect')
            
            tdata = sp.signal.filtfilt(a,b,tdata)
            #tdata = np.pad(np.abs(fd_filtered[i,t0:,q0,q0]),(0,filter_resolution),mode='reflect')
            peaks,dicts = signal.find_peaks(tdata,prominence=256)
            if len(peaks)==0:
                #No peaks found
                ind_max=0
                ind_2nd=0
                max_t_peaks[0,i] = 0#peaks[ind_max]
                max_t_peaks[1,i] = 0#peaks[ind_2nd]
            elif len(peaks)==1:
                #only one peak found
                ind_max = np.argmax(dicts['prominences'])
                ind_2nd=0
                max_t_peaks[0,i] = peaks[ind_max]
                max_t_peaks[1,i] = 0#peaks[ind_2nd]

            else:
                #at least 2 peaks found
                ind_max = np.argmax(dicts['prominences'])
                ind_2nd = np.argpartition(dicts['prominences'],-2)[-2]
                max_t_peaks[0,i] = peaks[ind_max]
                max_t_peaks[1,i] = peaks[ind_2nd]

            max_t_peaks[2,i] = tdata[max_t_peaks[0,i]]
            max_t_peaks[3,i] = tdata[max_t_peaks[1,i]]
            #plt.plot(tdata)
            #plt.scatter(peaks,tdata[peaks])
            #plt.scatter(max_t_peaks[0,i],max_t_peaks[2,i],color="red")
            #plt.scatter(max_t_peaks[1,i],max_t_peaks[3,i],color="green")
            #plt.show()

        max_t_peak_mean = (2/self.max_iters)*safe_average(max_t_peaks[0,max_t_peaks[0]!=0],max_t_peaks[2,max_t_peaks[0]!=0])
        max_t_peak_var = (2/self.max_iters)*np.std(max_t_peaks[0,max_t_peaks[0]!=0])
        harmonic_t_mean = (2/self.max_iters)*safe_average(max_t_peaks[1,max_t_peaks[1]!=0],max_t_peaks[3,max_t_peaks[1]!=0])
        harmonic_t_var = (2/self.max_iters)*np.std(max_t_peaks[1,max_t_peaks[1]!=0])

        temporal_peaks = np.array([[max_t_peak_mean,max_t_peak_var],[harmonic_t_mean,harmonic_t_var]])
        #print(temporal_peaks)
        #sys.stdout.write("#")
        #sys.stdout.flush()










        #Check for 4-fold symmetry
        fd_filtered_rot = np.rot90(fd_filtered,axes=(2,3))
        rot_coeff = np.einsum('ijkl,ijkl',fd_filtered_rot,fd_filtered)
        norm = np.einsum('ijkl,ijkl',fd_filtered,fd_filtered)
        symmetry_coeff = np.abs(rot_coeff/norm)
        #sys.stdout.write("#")
        #sys.stdout.flush()
        #print(symmetry_coeff)
        
        #Calculuate radial distribution functions of fft data
        #print("Calculating rdf")
        fd_rdf = np.zeros((self.states,self.max_iters,self.size//2)).astype(fd.dtype)
        fd_rdf_filtered = np.zeros((self.states,self.max_iters,self.size//2)).astype(fd.dtype)
        for x in range(self.size//2):
            sq = square_ring(self.size,x+1)
            fd_rdf_filtered[:,:,x] = np.einsum('ijkl,kl->ij',fd_filtered,sq)/np.sum(sq)
            fd_rdf[:,:,x] = np.einsum('ijkl,kl->ij',fd,sq)/np.sum(sq)

        #sys.stdout.write("#")
        #sys.stdout.flush()

        #peak detection on rdf
        #print("Finding peaks")
        max_peaks = np.zeros((4,self.states)).astype(int)
        for i in range(self.states):
            xdata = np.pad(np.abs(fd_rdf_filtered[i,t0]),(0,filter_resolution),mode='reflect')

            #peaks,dicts = signal.find_peaks(xdata,height=256,prominence=None,width=filter_resolution)
            peaks,dicts = signal.find_peaks(xdata,prominence=None,width=filter_resolution)
            if len(peaks)==0:
                #No peaks found
                ind_max=0
                ind_2nd=0
                max_peaks[0,i] = 0#peaks[ind_max]
                max_peaks[1,i] = 0#peaks[ind_2nd]
            elif len(peaks)==1:
                #only one peak found
                ind_max = np.argmax(dicts['prominences'])
                ind_2nd=0
                max_peaks[0,i] = peaks[ind_max]
                max_peaks[1,i] = 0#peaks[ind_2nd]

            else:
                #at least 2 peaks found
                ind_max = np.argmax(dicts['prominences'])
                ind_2nd = np.argpartition(dicts['prominences'],-2)[-2]
                max_peaks[0,i] = peaks[ind_max]
                max_peaks[1,i] = peaks[ind_2nd]
            #Magnitude at peaks
            max_peaks[2,i] = xdata[max_peaks[0,i]]
            max_peaks[3,i] = xdata[max_peaks[1,i]]
            #plt.plot(xdata)
            #plt.scatter(peaks,xdata[peaks])
            #plt.show()
            
        max_peak_mean = (2/self.size)*safe_average(max_peaks[0,max_peaks[0]!=0],max_peaks[2,max_peaks[0]!=0])
        max_peak_var = (2/self.size)*np.std(max_peaks[0,max_peaks[0]!=0])
        harmonic_mean = (2/self.size)*safe_average(max_peaks[1,max_peaks[1]!=0],max_peaks[3,max_peaks[1]!=0])
        harmonic_var = (2/self.size)*np.std(max_peaks[1,max_peaks[1]!=0])
        #print(max_peak_var)
        
        spacial_peaks = np.array([[max_peak_mean,max_peak_var],[harmonic_mean,harmonic_var]])

        #print(spacial_peaks)
        #sys.stdout.write("#")
        #sys.stdout.flush()






        #waves[waves<thresh]=0
        #stat_struct_filtered = np.zeros(stat_struct.shape,dtype='complex')

        #stat_struct_filtered[np.abs(stat_struct_filtered)<thresh]=0
        #waves_filtered = signal.convolve(waves,f_gs_3,mode='same')
        #plt.matshow(gs)
        #plt.show()
        self.image = np.abs(np.fft.ifftn(np.fft.ifftshift(one_hot_decode(fd_filtered))))
        

        """    
        #Split up sections of 3d fft that correspond to certain features - disused

        stat_struct = fd[:,t0]
        stat_struct[:,q0,q0]=0
        stat_struct_filtered = fd_filtered[:,t0]#ndimage.maximum_filter(np.abs(stat_struct),self.size//16,mode="wrap")
        stat_struct_filtered_rot = fd_filtered_rot[:,t0]
        blinkers = fd_filtered[:,:,q0,q0]
        #blinkers[:,t0]=0
        waves = np.copy(fd)
        waves[:,t0] = 0
        waves[:,:,q0,q0]=0
        for i in range(self.states):
            epsilon=1E-20
            xdata = (np.abs(fd_rdf[i,t0]))
            peaks,_ = signal.find_peaks(xdata,prominence=None,width=filter_resolution)

            plt.plot(xdata)
            plt.scatter(peaks,xdata[peaks])
            plt.matshow((np.abs(stat_struct[i])))
            plt.show()

            #plt.plot(np.sum(np.abs(stat_struct[i]),axis=0))
            #plt.plot(np.sum(np.abs(stat_struct[i]),axis=1))
            #plt.show()
            xdata = (np.abs(fd_rdf_filtered[i,t0]))
            peaks,dicts = signal.find_peaks(xdata,prominence=None,width=filter_resolution)
            
            #maximum peak height
            #max_peaks[0,i] = peaks[np.argmax(xdata[peaks])]
            #most prominant peak
            max_peaks[0,i] = peaks[np.argmax(dicts['prominences'])]
            max_peaks[1,i] = xdata[max_peaks[0,i]]
            plt.plot(xdata)
            plt.scatter(peaks,xdata[peaks])
            plt.scatter(max_peaks[0,i],max_peaks[1,i],color="red")
            plt.matshow((np.abs(stat_struct_filtered[i])))
            #plt.matshow(laplace(np.abs(stat_struct[i])))
            plt.show()
            #plt.show()
            #plt.matshow((np.abs(stat_struct_filtered_rot[i])))
            #plt.show()

            #plt.plot(np.sum(np.abs(stat_struct_filtered[i]),axis=0))
            #plt.plot(np.sum(np.abs(stat_struct_filtered[i]),axis=1))
            #plt.show()
        
        
        

        for j in range(self.states):
            #j=1
            plt.plot(np.abs(blinkers[j]),label=str(j))
        #plt.plot(np.abs(fd_ref[:,q0,q0]),label="reference")
        #plt.plot(blinkers[j].real,alpha=0.2)
        #plt.plot(blinkers[j].imag,alpha=0.2)
        plt.legend()
        plt.show()

        for j in range(self.states):
            plt.plot(np.abs(np.sum(fd_filtered[j],axis=(1,2))),label=str(j))
        plt.legend()
        plt.show()

        for j in range(self.states):
            plt.plot(np.sum(np.abs(fd_filtered[j]),axis=(1,2)),label=str(j))
        plt.legend()
        plt.show()
        self.fft_data=(np.abs(one_hot_decode(fd_filtered)))
        """

        #sys.stdout.write("|")
        #sys.stdout.flush()
        return (symmetry_coeff,spacial_peaks,temporal_peaks)



    def lyap(self,N,norm=False):
        #Run N simulations that differ by 1 cell in initial configuration from a given start state. Returns difference
        #If norm, then use transition matrix to weight differences between states
        self.init_grid()
        #print("_"*N)
        lyap_data = np.zeros((N,self.max_iters))
        for i in range(N):
            #sys.stdout.write("#")
            #sys.stdout.flush()
         

            #Do the first run
            self.current_state = np.copy(self.init_state)
            self.run(False)
            data1 = np.copy(self.image).astype(int)


            #Setup second run, differing by 1 cell state from first run initial condition
            x = np.random.randint(self.size)
            y = np.random.randint(self.size)
            grid_2 = np.copy(self.init_state)
            grid_2[x,y]= (grid_2[x,y]+ np.random.randint(1,self.states))%self.states

            
            #Do second run
            self.current_state = np.copy(grid_2)
            self.run(False)
            data2 = np.copy(self.image).astype(int)

            
            #Calculate difference
            if norm:
                where_ne = np.not_equal(data1,data2)
                def f(a,b):
                    return np.abs(1-np.sqrt(self.d[a,b]*self.d[b,a]))
                lyap_data[i] = np.sum(np.where(where_ne,f(data1,data2),0),axis=(1,2))/(self.size**2)
            else:
                lyap_data[i] = np.sum((np.not_equal(data1,data2).astype(int)),axis=(1,2))/(self.size**2) 
        
        #plt.plot()
        #plt.show()
        #sys.stdout.write("|")
        #sys.stdout.flush()
        self.image = data1-data2
        return lyap_data

    def entropy(self,N=1):
        #Calculate information entropy at each timestep, repeats N times with different random initial conditions
        p = np.zeros((self.states,self.max_iters))
        ent_data = np.zeros((N,self.max_iters))
        ss = np.arange(self.states)
        #Small epsilon to avoid problem p=0 causing log(0) to crash
        epsilon=1E-20
        for i in range(N):
            #sys.stdout.write("#")
            #sys.stdout.flush()
            self.run()
            s = np.equal(np.tile(self.image,(self.states,1,1,1)),ss[:,None,None,None])
            
            p = np.sum(s,axis=(2,3))/float(self.size*self.size)
            ent_data[i] = -np.sum(p*np.log(p+epsilon),axis=0)/np.log(self.states)
            #return en
        #sys.stdout.write("|")
        #sys.stdout.flush()
        return ent_data
        #print(en)
        #print(s.shape)

    def rule_entropy(self):
        #Returns the entropy of the rule array
        #p = np.zeros(self.states)
        epsilon=1E-20
        ss = np.arange(self.states)
        p = np.sum(np.equal(np.tile(self.rule,(self.states,1)),ss[:,None]),axis=1)/float(self.rule.shape[0])
        ent = -np.sum(p*np.log(p+epsilon),axis=0)/np.log(self.states)
        return ent
        #print(p)
        #print(self.rule)

    def get_metrics(self,N):
        #Calculates and returns all the useful metrics for a rule (rule entropy, variation of simulation entropy, divergence powerlaw)
        #N denotes number of repeated simulations to average over

        #sys.stdout.write("||")
        #sys.stdout.flush()
        #---  Divergence
        
        #set simulation size - smaller runtime avoids PBC effects
        temp_iters = self.max_iters
        self.max_iters = self.size//2
        self.image = np.zeros((self.max_iters,self.size,self.size))
        
        #Run divergence data simulations and fit to power law
        l_data = np.mean(self.lyap(4*N,norm=True),axis=0)#[:self.size//2]
        ts = np.arange(l_data.shape[0])
        l_params,_ = sp.optimize.curve_fit(power_law,ts,l_data)


        #---  Entropy
        
        #reset simulation size
        self.max_iters = 512
        self.image = np.zeros((self.max_iters,self.size,self.size))

        #Run entropy simulations and calculate smoothed variations
        e_data = np.mean(self.entropy(N),axis=0)
        a,b = sp.signal.butter(5,0.1,analog=False)
        e_smooth_var = np.mean(np.abs(np.diff(sp.signal.filtfilt(a,b,e_data),prepend=0)[100:self.max_iters-50]))
        e_mean = np.mean(e_data)
        e_var = np.std(e_data)


        r_entropy = self.rule_entropy()




        #--- FFT
        self.max_iters = 256
        self.image = np.zeros((self.max_iters,self.size,self.size))
        self.run()
        symmetry,spacial,temporal = self.fft()




        #--- Transition matrix

        mat = self.density_matrix()


        metrics = np.array([l_params[0],l_params[1],l_params[2],
                            e_smooth_var,e_mean,e_var,
                            r_entropy,
                            symmetry,
                            spacial[0,0],spacial[0,1],
                            spacial[1,0],spacial[1,1],
                            temporal[0,0],temporal[0,1],
                            temporal[1,0],temporal[1,1]])

        #Sometimes std for small number of states makes NaN - set these to 0
        metrics[np.isnan(metrics)]=0
        return metrics,mat


        #plt.plot(l_data)
        #plt.plot(ts,power_law(ts,l_params[0],l_params[1],l_params[2]))
        #plt.show()



    """ Commented out because cplab doesn't have tensorflow

    def predict_interesting(self,N=1):
        #Runs get_metrics on current rule, then feeds output to trained neural network
        model = tf.keras.models.load_model('interesting_predictor.h5',compile=False)
        metrics = self.get_metrics(N)
        metrics = metrics.reshape((1,metrics.shape[0]))
        print(metrics)
        return model.predict(metrics)
    """



#--- Rule generation, manipulation, saving and loading

    def rule_gen(self,mu=0.5,sig=0.25,mode=0):
        #n = self.states-1
        #print(self.nsize)
        #for x in range(0,self.nsize):
        #    n = n + self.n_struct[x]*(self.states-1)*self.states**(x+1)
        n = self.states
        for x in range(0,self.nsize):
            n*=(1+(self.states-1)*self.n_struct[x])
        self.rule = np.random.randint(self.states,size=n)
        
        if mode==0:
            xs = np.linspace(0,n,n)
            g = np.vectorize(lambda y:gaussian(y,xs[int(np.floor(n*mu))],(n*sig)))
            gxs = g(xs)
            rs = np.random.rand(n)
            zs = np.zeros(n)
            self.rule = np.where(rs<gxs,self.rule,zs).astype(int)

        

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



        f = '2D_rules/sym'+str(self.symmetries)+'/n'+str(self.nsize)+'/s'+str(self.states)+'/'+name

        np.save(f,self.rule)



    def rule_load(self,name):
        """
        loads a previously saved rule
        """
        try:
            f = ('2D_rules/sym'+str(self.symmetries)+'/n'+str(self.nsize)+'/s'+str(self.states)+'/'+name+'.csv')#,'r')#
            self.rule = np.load(f)
            
        except Exception as e:
            f = ('2D_rules/sym'+str(self.symmetries)+'/n'+str(self.nsize)+'/s'+str(self.states)+'/'+name+'.npy')#,'r')#
            self.rule = np.load(f)
            #raise e
        #f.close()

    def rule_input(self,rule_in):
        """
        Allows another program to feed in a rule
        """
        self.rule = rule_in

    def rule_perm(self):
        """
        method to slightly modify a rule
        """
        #for x in range(len(self.rule)):
        #    r = np.random.rand()
        #    if r>0.95:
        #        self.rule[x]=(self.rule[x]+1)%self.states
        #    elif r<0.05:
        #        self.rule[x]=(self.rule[x]-1)%self.states

        #self.rule[0]=0
        L = self.rule.shape[0]
        k=L//20
        ii = np.random.choice(L,size=k,replace=False)
        self.rule[ii] = (self.rule[ii]+np.random.randint(1,self.states,size=k))%self.states


    def rule_smooth(self,amount):
        """
        smoothes a rule, element by element
        """
        
        rule = self.rule
        for a in range(amount):
            w = np.random.randint(1,len(rule)//4)
            i = np.random.randint(w,len(rule)-w)
            #print(i)
            #print(w)
            st = rule[i]
            rule[i-w:i+w]=st
        self.rule = rule
        #s_rule = self.rule
        #for x in range(len(self.rule)):
        #    n_mean=0
        #    for y in range(2*amount+1):
        #        n_mean = n_mean+self.rule[(x+y-amount)%len(self.rule)]
        #    n_mean = (n_mean//(2*amount+1))%self.states
        #    s_rule[x] = n_mean
        #self.rule = s_rule


    def rule_offset(self,amount):
        rule = self.rule
        for x in range(amount):
            w = np.random.randint(1,len(rule)//4)
            i = np.random.randint(w,len(rule)-w)
            rule[i-w:i+w]=(rule[i-w:i+w]+np.random.randint(1,self.states))%self.states
        self.rule = rule

            
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
        



# Generic helper functions

def gaussian(x,mu,sig):
    return np.exp(-np.power(x-mu,2)/(2*np.power(sig,2)))


def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return np.exp(-((x-mx)**2./(2.*sx**2.) + (y-my)**2./(2.*sy**2.)))
def gaus3d(t=0, x=0, y=0, mt=0, mx=0, my=0, st=1, sx=1, sy=1):
    return np.exp(-((x-mx)**2./(2.*sx**2.) + (y-my)**2./(2.*sy**2.) + (t-mt)**2./(2.*st**2.)))
def multi_subset(x,y):
    """
    how many possible arangements of y items, of x choices, where duplicates are alowed.
    (1,2,2)=(2,1,2)!=(1,1,2)
    """
    return math.factorial(x+y-1)/(math.factorial(y)*math.factorial(x-1))

def power_law(x,A,B,C):
    return A*(np.float_power(x,B))+C

def one_hot_encode(data,s):
    #One hot encoder for 3D array of ints
    ss = np.arange(s)
    g = np.tile(data,(s,1,1,1))
    return np.equal(g,ss[:,None,None,None])


def one_hot_decode(data):
    i=data.shape
    output = np.zeros((i[1],i[2],i[3])).astype(data.dtype)
    for n in range(1,data.shape[0]):
        output+=n*data[n]
    return output

def laplace(gr):
    #Returns the 2d laplacian of a grid
    kernal = np.array([[0,1,0],
                       [1,-4,1],
                       [0,1,0]])
    return signal.convolve2d(gr,kernal,boundary='wrap',mode='same')


def square_ring(N,i):
    #Returns a square matrix (NxN) of zeros except for a ring of 1s with 'radius' i
    a = np.zeros((N,N))
    a[N//2-(i-1):N//2+(i-1),N//2-(i-1):N//2+(i-1)]=1
    b = np.copy(a)
    b[N//2-i:N//2+i,N//2-i:N//2+i]=1
    a = np.logical_xor(a,b).astype(int)
    return a

def safe_average(data,w):
    #Returns average of 0 if weights sum to 0, rather than crashing
    if np.sum(w)==0:
        return 0
    else:
        return np.average(data,weights=w)
