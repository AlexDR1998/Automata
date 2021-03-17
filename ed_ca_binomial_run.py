from org import Grid2D
import numpy as np
import math
#import scipy as sp 
#from scipy import signal
#from scipy import ndimage
import sys
import os
#import itertools

	
N_rules=1000
N_obs_reps = 4
states = int(sys.argv[1])
instance = int(sys.argv[2])
size = 128
iterations = 128

try:
	os.mkdir(str(states)+"_state_binomial_results")
except OSError:
	print("Failed to creat directory")


g = Grid2D(size,0.5,states,1,iterations,1)
#Generate rule arrays from a range of binomial distributions
splits=100
K = N_rules//splits
ps = np.linspace(0,1,K+2)[1:-1]
_rules = np.zeros((K,splits,g.rule_length)).astype(int)
for k in range(K):
	_rules[k] = np.random.binomial(states-1,ps[k],size=(splits,g.rule_length))
rules = _rules.reshape((N_rules,g.rule_length))



observables = np.zeros((N_rules,18))
mats = np.zeros((N_rules,states,states))
e_data = np.zeros((N_rules,512))
l_data = np.zeros((N_rules,g.size//2))
mf_err = np.zeros((N_rules,512))
for i in range(N_rules):
	print(i)
	g.rule = rules[i]
	observables[i],mats[i],e_data[i],l_data[i],_,mf_err[i]=g.get_metrics(N_obs_reps)



np.save(str(states)+"_state_results/observables"+str(instance)+".npy",observables)
np.save(str(states)+"_state_results/transition_mats"+str(instance)+".npy",mats)
np.save(str(states)+"_state_results/rules"+str(instance)+".npy",rules)
np.save(str(states)+"_state_results/raw_entropy"+str(instance)+".npy",e_data)
np.save(str(states)+"_state_results/raw_lyap"+str(instance)+".npy",l_data)
np.save(str(states)+"_state_results/raw_mf_err"+str(instance)+".npy",mf_err)