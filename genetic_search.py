from automata_class import Grid2D
import numpy as np
from scipy import ndimage
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import time
import sys
import warnings
#from sklearn import neighbors
import h5py
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf 
from tensorflow import keras
from tqdm import tqdm

warnings.filterwarnings('ignore')
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
	


	states = 2
	K=20 # number of rules from breeding pool
	I=20 # number of iterations
	N_rules = 100 # number of rules per generation
	mutation = 0.02 # chance of random mutations being applied
	


	g = Grid2D(size,0.5,states,neighbours,iterations,symm)
	rules,predictions,obs = load_rules_binomial(N_rules,states)
	print(predictions)

	all_rules = np.zeros((I,N_rules,rules.shape[1])).astype(int)
	all_preds = np.zeros((I,N_rules))
	all_obs = np.zeros((I,N_rules,18))
	for i in tqdm(range(I)):

		ps = predictions/np.sum(predictions)
		max_k = np.random.choice(N_rules,K,p=ps,replace=False)
		#max_k = np.argpartition(predictions, -K)[-K:]
		#min_k = np.argpartition(predictions,K)[:K]	
		best_rules = rules[max_k]
		best_preds = predictions[max_k]
		best_obs = obs[max_k]

		not_best_rules = np.delete(rules,max_k,axis=0)
		not_best_preds = np.delete(predictions,max_k,axis=0)
		not_best_obs = np.delete(obs,max_k,axis=0)

		




		new_rules = rule_crossover(K,best_rules)
		new_preds = np.zeros(K)
		new_obs = np.zeros((K,18))
		replace = np.random.choice(N_rules-K,K,replace=False)
		for k in tqdm(range(K)):
			g.rule = new_rules[k]
			if np.random.random()<mutation:
				g.rule_random_mod()
				new_rules[k] = g.rule

			new_preds[k],new_obs[k] = g.predict_interesting()
			#Replace random bad rules with new generation of rules
			not_best_rules[replace[k]]=new_rules[k]
			not_best_preds[replace[k]]=new_preds[k]
			not_best_obs[replace[k]]=new_obs[k]

		#Recombine breeder and mixed bad and bred rules
		rules = np.vstack((best_rules,not_best_rules))
		predictions = np.concatenate((best_preds,not_best_preds))
		obs = np.vstack((best_obs,not_best_obs))

		all_rules[i] = rules
		all_preds[i] = predictions
		all_obs[i] = obs

	#print(new_rules)
	#print(min_k)
	#print(max_k)
	np.save("Data/genetic_search/binomial_start/"+str(states)+"_state/rules.npy",all_rules)
	np.save("Data/genetic_search/binomial_start/"+str(states)+"_state/predictions.npy",all_preds)
	np.save("Data/genetic_search/binomial_start/"+str(states)+"_state/observables.npy",all_obs)
	print(all_preds)
	
	g.rule = all_rules[-1,np.argmax(all_preds[-1])]
	g.run()
	ani_display()





def load_rules_uniform(N,states):
	#Selects N random rule/observables pairs from the set of all uniformly sampled rules
	rule_zero = np.load("unlabeled_data/"+str(states)+"_state/rules1.npy")[0]
	L = rule_zero.shape[0]
	rules = np.zeros((N,L)).astype(int)
	obs = np.zeros((N,18))
	r1 = np.random.choice(100,N,replace=True)
	r2 = np.random.choice(1000,N,replace=False)
	for n in range(N):
		rules[n] = np.load("unlabeled_data/"+str(states)+"_state/rules"+str(r1[n]+1)+".npy")[r2[n]]
		obs[n] = np.load("unlabeled_data/"+str(states)+"_state/observables"+str(r1[n]+1)+".npy")[r2[n]]
	model = keras.models.load_model('interesting_predictor.h5',compile=False)
	preds = model.predict(obs)
	return rules,preds.T[0],obs


def load_rules_binomial(N,states):
	#Selects N random rule/observables pairs from the set of all binomially sampled rules
	rule_zero = np.load("unlabeled_data/binomial/"+str(states)+"_state/rules1.npy")[0]
	L = rule_zero.shape[0]
	rules = np.zeros((N,L)).astype(int)
	obs = np.zeros((N,18))
	r1 = np.random.choice(10,N,replace=True)
	r2 = np.random.choice(1000,N,replace=False)
	for n in range(N):
		rules[n] = np.load("unlabeled_data/binomial/"+str(states)+"_state/rules"+str(r1[n]+1)+".npy")[r2[n]]
		obs[n] = np.load("unlabeled_data/binomial/"+str(states)+"_state/observables"+str(r1[n]+1)+".npy")[r2[n]]
	model = keras.models.load_model('interesting_predictor.h5',compile=False)
	preds = model.predict(obs)
	return rules,preds.T[0],obs


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





def ani_display(mode=0,n=1):
    if mode==0:
        data = g.im_out()
    elif mode==1:
        data = smooth(g.im_out(),3)

    elif mode==2:
        data = np.moveaxis(smooth(g.im_out(),3),0,1)

    elif mode==4:
        data = np.moveaxis(g.im_out(),0,1)
    
    elif mode==5:
        data = np.abs(np.diff(np.diff(g.im_out(),axis=0),axis=0))
    elif mode==6:
        data = g.im_out()[::n]


    def update(i):
        screendata = data[i]
        matrix.set_array(screendata)
    screendata = data[0]
    fig, ax = plt.subplots()            
    matrix = ax.matshow(screendata,cmap=colour)
    plt.colorbar(matrix)
    ani = animation.FuncAnimation(fig,update,frames=iterations,interval=100)
    plt.show()




main()