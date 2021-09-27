from automata_class import Grid2D
import numpy as np
import math
#import scipy as sp 
#from scipy import signal
#from scipy import ndimage
import sys
import os
#import itertools
import tensorflow as tf 
	



def main():
	#Start with population of rules
	np.seterr(all="ignore")
	global states
	global symm
	symm = 2
	neighbours = 1
	global size
	size = 128
	global iterations
	iterations = 256
	global g
	global screendata
	global matrix
	global data
	global colour
	colour = "gist_earth"
	#mu=0.5
	#sig=0.2
	
	N_obs_reps = 2
	states = int(sys.argv[1])
	instance = int(sys.argv[2])


	K=20 # number of rules from breeding pool
	I=50 # number of iterations
	N_rules = 100 # number of rules per generation
	mutation = 0.02 # chance of random mutations being applied

	
	try:
		os.mkdir(str(states)+"_state_genetic_results")
	except OSError:
		pass
		#print("Failed to create directory")
	print("Running "+str(states)+" state rules instance "+str(instance))


	g = Grid2D(size,0.5,states,neighbours,iterations,symm)
	g.rule_mode=1

	all_rules = np.zeros((I,N_rules,g.rule_length)).astype(int)
	all_preds = np.zeros((I,N_rules))
	all_obs = np.zeros((I,N_rules,18))
	all_tmats = np.zeros((I,N_rules,states,states))

	#Initialise pool of rules to breed from
	for r in range(N_rules):
		g.rule_gen()
		all_rules[0,r] = g.rule
		all_preds[0,r],all_obs[0,r],all_tmats[0,r] = g.predict_interesting()

	
	
    #Iterate across generations
	for i in range(1,I):


		#Randomly sample best rules of previous generation
		ps = all_preds[i-1]/np.sum(all_preds[i-1])
		max_k = np.random.choice(N_rules,K,p=ps,replace=False)
		best_rules = all_rules[i-1,max_k]
		best_preds = all_preds[i-1,max_k]
		best_obs = all_obs[i-1,max_k]
		best_tmats = all_tmats[i-1,max_k]

		not_best_rules = np.delete(all_rules[i-1],max_k,axis=0)
		not_best_preds = np.delete(all_preds[i-1],max_k,axis=0)
		not_best_obs = np.delete(all_obs[i-1],max_k,axis=0)
		not_best_tmats = np.delete(all_tmats[i-1],max_k,axis=0)

		#Cross breed the best K rules from previous generation
		new_rules = rule_crossover(K,best_rules)
		new_preds = np.zeros(K)
		new_obs = np.zeros((K,18))
		new_tmats = np.zeros((K,states,states))
		replace = np.random.choice(N_rules-K,K,replace=False)
		
		#Evaluate newly bred rules
		for k in range(K):
			g.rule = new_rules[k]
			#Apply mutation
			if np.random.random()<mutation:
				g.rule_random_mod()
				new_rules[k] = g.rule

			new_preds[k],new_obs[k],new_tmats[k] = g.predict_interesting()
			
			#Replace random bad rules with new generation of rules
			not_best_rules[replace[k]]=new_rules[k]
			not_best_preds[replace[k]]=new_preds[k]
			not_best_obs[replace[k]]=new_obs[k]
			not_best_tmats[replace[k]]=new_tmats[k]

		#Recombine breeder and mixed bad and bred rules
		all_rules[i]  = np.vstack((best_rules,not_best_rules))
		all_preds[i] = np.concatenate((best_preds,not_best_preds))
		all_obs[i] = np.vstack((best_obs,not_best_obs))
		all_tmats[i] = np.vstack((best_tmats,not_best_tmats))

	#print(new_rules)
	#print(min_k)
	#print(max_k)
	np.save(str(states)+"_state_genetic_results/rules"+str(instance)+".npy",all_rules)
	np.save(str(states)+"_state_genetic_results/predictions"+str(instance)+".npy",all_preds)
	np.save(str(states)+"_state_genetic_results/observables"+str(instance)+".npy",all_obs)
	np.save(str(states)+"_state_genetic_results/transition_mats"+str(instance)+".npy",all_tmats)
	print(all_preds)
	






def rule_crossover(N,rules):
	#Generates N random genetic crossover pairs from rules
	L = rules.shape[1]
	R = rules.shape[0]
	new_rules = np.zeros((N,L)).astype(int)
	xs = np.arange(L).astype(int)
	for n in range(N):
		r1 = np.random.randint(L)
		p1 = np.random.randint(R)
		p2 = (p1+np.random.randint(R-1))%R
		new_rules[n] = np.where(xs<r1,rules[p1],rules[p2])
	return new_rules



main()