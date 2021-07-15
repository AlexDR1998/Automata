import numpy as np
from scipy import signal
from scipy import ndimage
import scipy as sp 
#import matplotlib.pyplot as plt
import math
import sys
import itertools
#from sklearn import neighbors
#import h5py
#import tensorflow as tf 
#from tensorflow import keras
from tqdm import tqdm



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
        
        self.k = int(multi_subset(self.states,8))
        #self.k=np.array([[1, 1,1],
        #                 [1,_k,1],
        #                 [1, 1,1]])
        

        self.k_ns = np.array([[1,1,1],
                              [1,0,1],
                              [1,1,1]])
        _ar = np.zeros((self.k,8))
        for i,el in enumerate(itertools.combinations_with_replacement(range(self.states),8)):
            _ar[i,:] = el
        
        #Use lookup table to map result of array exponential function to interval 0-Rule size
        sorted_outputs = np.sort(np.sum((8+1)**(_ar),axis=1)/8).astype(int)
        self.lookup = np.zeros(np.max(sorted_outputs)+1).astype(int)
        for i in range(sorted_outputs.shape[0]):
            self.lookup[sorted_outputs[i]]=i
        #print(self.lookup)

        self.rule_length = int(self.states*self.k)
        """

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
        
        """
        #How many iterations to run the code for
        self.max_iters = iterations
        self.rule_mode=0
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


    def run_parallel(self,rule1,rule2,rule3,rule4):
        #Runs 4 rules, 1 in each quadrant. Used for graphing purposes
        self.init_grid()
        for i in range(self.max_iters):
            #self.next_state=signal.convolve2d(self.current_state,self.k,boundary='wrap',mode='same').astype(int)
            v1 = np.vectorize(lambda y:rule1[y%rule1.shape[0]])
            v2 = np.vectorize(lambda y:rule2[y%rule2.shape[0]])
            v3 = np.vectorize(lambda y:rule3[y%rule3.shape[0]])
            v4 = np.vectorize(lambda y:rule4[y%rule4.shape[0]])


            self.current_state[0:self.size//2,0:self.size//2] = self.update_grid(self.current_state[0:self.size//2,0:self.size//2],v1)
            self.current_state[0:self.size//2,self.size//2:self.size] = self.update_grid(self.current_state[0:self.size//2,self.size//2:self.size],v2)
            self.current_state[self.size//2:self.size,0:self.size//2] = self.update_grid(self.current_state[self.size//2:self.size,0:self.size//2],v3)
            self.current_state[self.size//2:self.size,self.size//2:self.size] = self.update_grid(self.current_state[self.size//2:self.size,self.size//2:self.size],v4)
            #self.current_state[:][self.size//2]=0
            #self.current_state[self.size//2][:]=0

            self.image[i] = self.current_state
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

        mask1 = np.zeros(length)
        mask2 = np.zeros(length)
        mask3 = np.zeros(length)
        mask4 = np.zeros(length)
    
        mask1[:length//4] = 1
        mask2[length//4:length//2] = 1
        mask3[length//2:3*length//4] = 1
        mask4[3*length//4:] = 1

        amount = np.random.choice([0,1],size=length,p=[1-am,am])
        mutation = np.random.randint(1,self.states,size=length)
        

        m1 = (mask1*amount*mutation).astype(int)
        m2 = (mask2*amount*mutation).astype(int)
        m3 = (mask3*amount*mutation).astype(int)
        m4 = (mask4*amount*mutation).astype(int)

        rule1 = (rule1+m1)%self.states
        rule2 = (rule2+m2)%self.states
        rule3 = (rule3+m3)%self.states
        rule4 = (rule4+m4)%self.states


        p = len(self.rule)
        zs = np.zeros((self.size//2,self.size//2))
        for i in range(self.max_iters):
            #self.next_state=signal.convolve2d(self.current_state,self.k,boundary='wrap',mode='same').astype(int)
            v1 = np.vectorize(lambda y:rule1[y])
            v2 = np.vectorize(lambda y:rule2[y])
            v3 = np.vectorize(lambda y:rule3[y])
            v4 = np.vectorize(lambda y:rule4[y])


            self.current_state[0:self.size//2,0:self.size//2] = self.update_grid(self.current_state[0:self.size//2,0:self.size//2],v1)
            self.current_state[0:self.size//2,self.size//2:self.size] = self.update_grid(self.current_state[0:self.size//2,self.size//2:self.size],v2)
            self.current_state[self.size//2:self.size,0:self.size//2] = self.update_grid(self.current_state[self.size//2:self.size,0:self.size//2],v3)
            self.current_state[self.size//2:self.size,self.size//2:self.size] = self.update_grid(self.current_state[self.size//2:self.size,self.size//2:self.size],v4)
            #self.current_state[:][self.size//2]=0
            #self.current_state[self.size//2][:]=0

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



    def update_grid(self,current_state,v):
        #returns the next state grid, same shape as current grid
        _exp = 9**current_state
        _ns = (sp.signal.convolve2d(_exp,self.k_ns,boundary='wrap',mode='same')/8).astype(int)
        neighbour_number = self.lookup[_ns]
        #print(neighbour_number)
        key = (self.k*current_state) + neighbour_number
        #print(key)
        return v(key)


    def run(self,random_init=True,compute_t_matrix=False):
        #initialises random starting state
        if random_init:
            self.init_grid()
        self.d = np.zeros((self.states,self.states))
        self.d_temp = np.zeros((self.max_iters,self.states,self.states))
        s = self.states
        ss = np.arange(s)
        p = len(self.rule)

        # --- is outer-totalistic, not pseudo-outer-totalistic

        v = np.vectorize(lambda y:self.rule[y])
        for i in range(self.max_iters):
            #Convolve to calculate next global state
            #v = np.vectorize(lambda y:self.rule[y%p])
            
            #self.next_state=v(signal.convolve2d(self.current_state,self.k,boundary='wrap',mode='same').astype(int))
            self.next_state = self.update_grid(self.current_state,v)
            #Update transition density matrix - gets slow for large number of states

            if compute_t_matrix:
                g_i = np.tile(self.current_state,(s,1,1))
                g_j = np.tile(self.next_state,(s,1,1))
                g_i_eq = np.equal(g_i,ss[:,None,None]).astype(int)
                g_j_eq = np.equal(g_j,ss[:,None,None]).astype(int)
                self.d_temp[i] = np.einsum('ixy,jxy->ij',g_i_eq,g_j_eq)#/float(self.size**2)
                self.d+=self.d_temp[i]
            

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


    def density_matrix(self,random_init=True,check_accuracy=True):

        self.run(random_init,True)
        self.d = (self.d.T/np.sum(self.d,axis=1)).T 
        self.tmat = np.copy(self.d)
        #self.d_temp = (self.d_temp/np.sum(self.d_temp,axis=2))
        if check_accuracy:
            #setup mean field approximation
            #mean_field_individual = np.ones((self.max_iters,self.states))/float(self.states)
            mean_field_converged = np.ones((self.max_iters,self.states))/float(self.states)
            for i in range(self.max_iters-1):
                #self.d_temp[i] = (self.d_temp[i].T/np.sum(self.d_temp[i],axis=1)).T 
                #mean_field_individual[i+1]=np.einsum("ij,i",self.d_temp[i],mean_field_individual[i])
                mean_field_converged[i+1]=np.einsum("ij,i",self.tmat,mean_field_converged[i])
            #compute actual proportions of each state at each timestep
            ss = np.arange(self.states)
            s = np.equal(np.tile(self.image,(self.states,1,1,1)),ss[:,None,None,None])
            prop = np.sum(s,axis=(2,3))/float(self.size*self.size)
            err = np.sum(np.abs(prop.T-mean_field_converged),axis=1)
            #plt.plot(err)
            #plt.plot(np.sum(mean_field_converged,axis=1))
            #plt.show()
            
            #chop off first 16 steps as they are dominated by noise of initial conditions
            err_mean = np.mean(err[16:])
            err_var = np.std(err[16:])
            return (self.d,err_mean,err_var,err)
        else:
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
            #plt.matshow(fd[i,t0])
            #plt.show()
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
            peaks,dicts = sp.signal.find_peaks(tdata,prominence=256)
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
            peaks,dicts = sp.signal.find_peaks(xdata,prominence=None,width=filter_resolution)
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
        return (symmetry_coeff,spacial_peaks,temporal_peaks,fd[:,t0])



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
                    return np.abs(1-np.sqrt(self.tmat[a,b]*self.tmat[b,a]))
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

        temp_iters = self.max_iters

        
        #--- transition matrix meanf field approximation

        #set simulation size - longer runtime gives more strongly converged tmat
        self.max_iters = 512
        self.image = np.zeros((self.max_iters,self.size,self.size))
        #do first as self.tmat is needed for lyapunov divergence
        tmat,mf_err_mean,mf_err_var,mf_err = self.density_matrix(True,True)





        #---  Divergence
        
        #set simulation size - smaller runtime avoids PBC effects
        self.max_iters = self.size//2
        self.image = np.zeros((self.max_iters,self.size,self.size))
        
        #Run divergence data simulations and fit to power law
        l_data = np.mean(self.lyap(4*N,norm=True),axis=0)#[:self.size//2]
        ts = np.arange(l_data.shape[0])
        try:
            #Include try/except as curve_fit occasionally fails to find optimal parameters and then crashes
            l_params,_ = sp.optimize.curve_fit(power_law,ts,l_data)    
        except RuntimeError:
            l_params = np.zeros(3)


        #---  Entropy
        
        #reset simulation size
        self.max_iters = 512
        self.image = np.zeros((self.max_iters,self.size,self.size))

        #Run entropy simulations and calculate smoothed variations
        e_data = np.mean(self.entropy(N),axis=0)
        a,b = sp.signal.butter(5,0.1,analog=False)
        filtered = np.r_[0,sp.signal.filtfilt(a,b,e_data)]
        e_smooth_var = np.mean(np.abs(np.diff(filtered)[100:self.max_iters-50]))
        e_mean = np.mean(e_data)
        e_var = np.std(e_data)


        r_entropy = self.rule_entropy()




        #--- FFT
        self.max_iters = 256
        self.image = np.zeros((self.max_iters,self.size,self.size))
        self.run()
        symmetry,spacial,temporal,stat_struct = self.fft()






        metrics = np.array([l_params[0],l_params[1],l_params[2],
                            e_smooth_var,e_mean,e_var,
                            r_entropy,
                            symmetry,
                            spacial[0,0],spacial[0,1],
                            spacial[1,0],spacial[1,1],
                            temporal[0,0],temporal[0,1],
                            temporal[1,0],temporal[1,1],
                            mf_err_mean,mf_err_var])

        #Sometimes std for small number of states makes NaN - set these to 0
        metrics[np.isnan(metrics)]=0
        return metrics,tmat,e_data,l_data,stat_struct,mf_err


        #plt.plot(l_data)
        #plt.plot(ts,power_law(ts,l_params[0],l_params[1],l_params[2]))
        #plt.show()



    #Commented out because cplab doesn't have tensorflow
    """
    def predict_interesting(self,N=1):
        #Runs get_metrics on current rule, then feeds output to trained neural network
        model = tf.keras.models.load_model('interesting_predictor.h5',compile=False)
        metrics = self.get_metrics(N)[0]
        metrics = metrics.reshape((1,metrics.shape[0]))
        #print(metrics)
        return model.predict(metrics),metrics
    



    def auto_evolve(self,depth=3,am=0.1,mangle=0.1):
        #Uses neural network interestingness predictor to automatically iteratatively mutate rules to get more interesting results
        self.init_grid()
        length = len(self.rule)

        rule1 = np.zeros(length)
        rule2 = np.zeros(length)
        rule3 = np.zeros(length)
        rule4 = np.zeros(length)
        #print(len(rule1))

        mask1 = np.zeros(length)
        mask2 = np.zeros(length)
        mask3 = np.zeros(length)
        mask4 = np.zeros(length)
    
        mask1[:length//4] = 1
        mask2[length//4:length//2] = 1
        mask3[length//2:3*length//4] = 1
        mask4[3*length//4:] = 1

        pred = np.zeros(4)
        pre_pred = np.zeros(4)
        r_history = np.zeros((int(depth),length)).astype(int)
        p_history = np.zeros(int(depth))
        for i in range(int(depth)):
            seed = np.random.random()
            current_rule = np.copy(self.rule)
            amount = np.random.choice([0,1],size=length,p=[1-am,am])
            mutation = np.random.randint(1,self.states,size=length)
            
            m1 = (mask1*amount*mutation).astype(int)
            m2 = (mask2*amount*mutation).astype(int)
            m3 = (mask3*amount*mutation).astype(int)
            m4 = (mask4*amount*mutation).astype(int)
            
            rule1 = np.array(self.rule,copy=True)
            rule2 = np.array(self.rule,copy=True)
            rule3 = np.array(self.rule,copy=True)
            rule4 = np.array(self.rule,copy=True)

            rule1 = (rule1+m1)%self.states
            rule2 = (rule2+m2)%self.states
            rule3 = (rule3+m3)%self.states
            rule4 = (rule4+m4)%self.states

            pre_pred = np.copy(pred)
            self.rule=rule1
            if seed<mangle:
                self.rule_random_mod()
            pred[0] = self.predict_interesting()
            
            self.rule=rule2
            if seed<mangle:
                self.rule_random_mod()
            pred[1] = self.predict_interesting()
            
            self.rule=rule3
            if seed<mangle:
                self.rule_random_mod()
            pred[2] = self.predict_interesting()
            
            self.rule=rule4
            if seed<mangle:
                self.rule_random_mod()
            pred[3] = self.predict_interesting()
            
            print(pred)
            if np.max(pred)>=np.max(pre_pred):
                p_history[i] = np.max(pred)
                if np.argmax(pred)==0:
                    self.rule = rule1
                elif np.argmax(pred)==1:
                    self.rule = rule2
                elif np.argmax(pred)==2:
                    self.rule = rule3
                elif np.argmax(pred)==3:
                    self.rule = rule4
            else:
                p_history[i]=np.max(pre_pred)
                self.rule = current_rule
            r_history[i] = self.rule
        return r_history,p_history
        #self.run()


    """
    def random_walk(self,L,am,N=1):
        #Performs a random walk of length L, where each step changes am*Rule_length array entries. Computes and stores observables of each step
        #initial_metric = self.get_metrics(N)
        obs_history = np.zeros((L,18))
        tmat_history = np.zeros((L,self.states,self.states))
        rule_history = np.zeros((L,self.rule_length))
        for l in tqdm(range(L)):
            obs_history[l],tmat_history[l],_,_,_,_ = self.get_metrics(N)
            rule_history[l] = self.rule
            self.rule_perm(am)

        #print(obs_history)
        return obs_history,rule_history,tmat_history
#--- Rule generation, manipulation, saving and loading

    def rule_gen(self,mu=0.5,sig=0.25):
        #n = self.states-1
        #print(self.nsize)
        #for x in range(0,self.nsize):
        #    n = n + self.n_struct[x]*(self.states-1)*self.states**(x+1)
        n = self.rule_length#
        """
        for x in range(0,self.nsize):
            n*=(1+(self.states-1)*self.n_struct[x])
        """

        self.rule = np.random.randint(self.states,size=n)
        
        if self.rule_mode==0:
            xs = np.linspace(0,n,n)
            g = np.vectorize(lambda y:gaussian(y,xs[int(np.floor(n*mu))],(n*sig)))
            gxs = g(xs)
            rs = np.random.rand(n)
            zs = np.zeros(n)
            self.rule = np.where(rs<gxs,self.rule,zs).astype(int)
        if self.rule_mode==1:
            self.rule = np.random.binomial(self.states-1,mu,n)
        

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
        nlist = open('2D_rules/s'+str(self.states)+'/namelist.txt','a')
        nlist.write('\n'+name)
        nlist.close()



        f = '2D_rules/s'+str(self.states)+'/'+name

        np.save(f,self.rule)



    def rule_load(self,name):
        """
        loads a previously saved rule
        """
        try:
            f = ('2D_rules/s'+str(self.states)+'/'+name+'.npy')#,'r')#
            self.rule = np.load(f)
            
        except Exception as e:
            f = ('2D_rules/s'+str(self.states)+'/'+name+'.csv')#,'r')#
            self.rule = np.load(f)
            #raise e
        #f.close()

    def rule_input(self,rule_in):
        """
        Allows another program to feed in a rule
        """
        self.rule = rule_in

    def rule_perm(self,am=0.05):
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
        k=max(int(L*am),1)
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

            


    def rule_swap(self):
        #Shuffles rule array such that the labels of central states are swapped randomly
        rule = self.rule
        L = rule.shape[0]
        rule = np.random.permutation(rule.reshape((self.states,L//self.states))).reshape(L)
        self.rule = rule
    
    def rule_roll(self,amount):
        rule = self.rule
        L = rule.shape[0]
        r_mat = rule.reshape((self.states,L//self.states))
        self.rule = np.roll(r_mat,amount,axis=1).reshape(L)
        

    def rule_random_mod(self):
        #Randomly applies one of the methods to perturb a rule
        choice = np.random.randint(6)
        if choice==0:
            self.rule_perm()
        elif choice==1:
            self.rule_smooth(np.random.randint(4))
        elif choice==2:
            self.rule_offset(np.random.randint(4))
        elif choice==3:
            self.rule_swap()
        elif choice==4:
            self.rule_roll(np.random.randint(self.rule_length//2))
        elif choice==5:
            self.rule_inv()

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
            
            try:
                f = ('2D_rules/s'+str(self.states)+'/'+rule_name_1+'.npy')#,'r')#
                rule1 = np.load(f)
            
            except Exception as e:
                f = ('2D_rules/s'+str(self.states)+'/'+rule_name_1+'.csv')#,'r')#
                rule1 = np.load(f)

            #f = open('2D_rules/n'+str(self.nsize)+'/s'+str(self.states)+'/'+rule_name_1+'.csv','r')
            #rule1 = np.load(f)
            #f.close()
        
        if rule_name_2=="":
            rule2=self.rule
        else:
            try:
                f = ('2D_rules/s'+str(self.states)+'/'+rule_name_2+'.npy')#,'r')#
                rule2 = np.load(f)
            
            except Exception as e:
                f = ('2D_rules/s'+str(self.states)+'/'+rule_name_2+'.csv')#,'r')#
                rule2 = np.load(f)

            #f = open('2D_rules/n'+str(self.nsize)+'/s'+str(self.states)+'/'+rule_name_1+'.csv','r')
            #rule1 = np.load(f)
            #f.close()
        

        if mode=="+":
            self.rule = np.remainder((rule1 + rule2),self.states)
        if mode=="*":
            self.rule = np.remainder((rule1 * rule2),self.states)
        if mode=="-":
            self.rule = np.remainder((rule1 - rule2),self.states)
        if mode=="a":
            self.rule = np.remainder((rule1 + rule2)//2,self.states)
        if mode=="z":
            for x in range(len(self.rule)):
                if x%2==0:
                    self.rule[x]=rule1[x]
                else:
                    self.rule[x]=rule2[x]
        if mode=="b":
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
        self.rule = (self.states - self.rule -1)

    
        



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
    return A*(np.power(x.astype(float),B.astype(float)))+C

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
    return sp.signal.convolve2d(gr,kernal,boundary='wrap',mode='same')


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
