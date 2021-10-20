from automata_class import Grid2D
import numpy as np
import math
#import scipy as sp 
#from scipy import signal
#from scipy import ndimage
import sys
import os
#import itertools

	
N_rules=100
N_obs_reps = 4
B = 500
states = int(sys.argv[1])
instance = int(sys.argv[2])
size = 128
iterations = 128
max_persistance=100

try:
	os.mkdir(str(states)+"_state_B"+str(B)+"_mc_results")
except OSError:
	pass
	#print("Failed to create directory")
print("Running "+str(states)+" state rules instance "+str(instance))

g = Grid2D(size,0.5,states,1,iterations,1)
g.rule_mode=1
g.rule_gen(mu=0.3)
ps,observables,rules,mats,mc_stats = g.monte_carlo(B,N_rules,0.01,N_obs_reps,max_persistance)
rules = rules.astype(int)

#print(observables)
np.save(str(states)+"_state_B"+str(B)+"_mc_results/mc_stats"+str(instance)+".npy",mc_stats)
np.save(str(states)+"_state_B"+str(B)+"_mc_results/predictions"+str(instance)+".npy",ps)
np.save(str(states)+"_state_B"+str(B)+"_mc_results/observables"+str(instance)+".npy",observables)
np.save(str(states)+"_state_B"+str(B)+"_mc_results/transition_mats"+str(instance)+".npy",mats)
np.save(str(states)+"_state_B"+str(B)+"_mc_results/rules"+str(instance)+".npy",rules)
