from org import Grid2D
import numpy as np
from scipy import ndimage
import scipy as sp
#from interface import Interface as i 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import time
import sys

"""
	Various methods for trialling/developing analysis
	of the results from CA simulations
"""


#Use wolfram classes of automata
	#  	1 - homogenous endpoint - (almost) everything dies out and converges to the same homogenous state
	#	2 - stable structures - fixed points or time periodic structures
	#	3 - noisy - any structures are dominated by noise, changes spread fast
	#   4 - dynamic structures - interesting stuff like GoL

def main():
	np.seterr("ignore")
	global states
	global symm
	states = 4
	symm = 2
	neighbours = 1
	global size
	size = 128
	global iterations
	iterations = 128
	global g
	global screendata
	global matrix
	global data
	global colour
	colour = "magma"
	#mu=0.5
	#sig=0.2
	g = Grid2D(size,0.5,states,neighbours,iterations,symm)



	#N=16
	#data = perm_explore(128)
	#data = hamming_explore(128,2)
	#print(data)
	
	#data = random_walk_explore(128)
	#print(data)
	#np.save("random_walk_gol_rules",data)
	

	
	#data = smooth_perm_explore(32)
	#np.save("4state_rules_sp_32_16",data)
	
	"""
	r1 = np.load("4state_rules_sp_32_1.npy")
	r2 = np.load("4state_rules_sp_32_2.npy")
	r3 = np.load("4state_rules_sp_32_3.npy")
	r4 = np.load("4state_rules_sp_32_4.npy")
	r5 = np.load("4state_rules_sp_32_5.npy")
	r6 = np.load("4state_rules_sp_32_6.npy")
	r7 = np.load("4state_rules_sp_32_7.npy")
	r8 = np.load("4state_rules_sp_32_8.npy")
	r9 = np.load("4state_rules_sp_32_9.npy")
	r10 = np.load("4state_rules_sp_32_10.npy")
	r11 = np.load("4state_rules_sp_32_11.npy")
	r12 = np.load("4state_rules_sp_32_12.npy")
	r13 = np.load("4state_rules_sp_32_13.npy")
	r14 = np.load("4state_rules_sp_32_14.npy")
	r15 = np.load("4state_rules_sp_32_15.npy")
	r16 = np.load("4state_rules_sp_32_16.npy")
	rules = np.vstack((r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16))
	print(rules.shape)
	np.save("4state_rules_sp_combined",rules)
	"""

	#data = np.load("8state_ml_data.npy")
	#print(data.shape)





	"""	
	rules = np.load("Data/8state_rules_sp_combined.npy")

	g.rule = rules[0,2:]
	states = g.states
	print(states)

	R = rules.shape[0]
	print("_"*R)
	
	mats = np.zeros((R,states,states))
	for i in range(R):
		g.rule = rules[i,2:]
	
		mats[i] = g.density_matrix()
		sys.stdout.write("#")
		sys.stdout.flush()

	#np.save(str(states)+"state_ml_data.npy",observables)
	np.save(str(states)+"state_transition_matrices.npy",mats)
	
	#observables = np.load("2state_ml_data.npy")
	print(observables[0])
	print(observables.shape)
	"""
	






	#print(rules.shape)
	#print(rules[1])
	rules = np.load("4state_rules_sp_combined.npy")
	generate_observables(8,rules)
	#plot_observables(15,13)
	"""
	g.rule_gen()
	g.rule_fold(2)
	g.run()
	data = g.im_out()
	ani_display()
	print(g.get_metrics(4))
	"""




	#plt.scatter(e_fit[rules[:,1]==0],l_fit[rules[:,1]==0,1],color="red")
	#plt.scatter(e_fit[rules[:,1]==1],l_fit[rules[:,1]==1,1],color="blue")
	#for i in range(512):
	#	if rules[i,1]==0:
	#		plt.plot(lyap_data[0,i],color="red",alpha=0.2)
	#	if rules[i,1]==1:
	#		plt.plot(lyap_data[0,i],color="blue",alpha=0.2)
	#plt.plot([0],[0],color="red",label="boring")
	#plt.plot([0],[0],color="blue",label="interesting")
	#plt.legend()
	#plt.title("Divergence with binary classifier (512 2 state rules)")
	#plt.xlabel("Timesteps")
	#plt.ylabel("Divergence (normalised between 0-1)")
	#plt.show()
	
	#lyap_fit(lyap_data,rules[:,0],rules[:,1])
	
	#ent_data = entropy_evaluate(rules[:,0],rules[:,1],rules[:,2:],32)
	#np.save("random_walk_gol_entropy_N32_I256.npy",ent_data)
	
	#np.save("random_walk_gol_lya_div_N32_I128.npy",lyap_data)


	#rules = random_explore(16)
	#np.save("3state_test_rules",rules)
	#lyap_data= lyap_evaluate(rules[:,0],rules[:,1],rules[:,2:],4,norm=True)


def old_load():
	l_data1 = np.load("random_2s_lya_div_N32_I128.npy")[:,:,:64]
	l_data2 = np.load("random_walk_2s_lya_div_N32_I128.npy")[:,:,:64]
	l_data3 = np.load("hd2_gol_lya_div_N32_I128_normalised.npy")
	l_data4 = np.load("random_walk_gol_lya_div_N32_I128.npy")[:,:,:64]
	lyap_data = np.hstack((l_data1,l_data2,l_data3,l_data4))
	
	print(lyap_data.shape)

	e_data1 = np.load("random_2s_entropy_N32_I256.npy")
	e_data2 = np.load("random_walk_2s_entropy_N32_I256.npy")
	e_data3 = np.load("hd2_gol_entropy_N32_I256.npy")
	e_data4 = np.load("random_walk_gol_entropy_N32_I256.npy")

	#print(e_data1.shape)
	#print(e_data2.shape)
	#print(e_data3.shape)
	#print(e_data4.shape)
	e_data = np.hstack((e_data1,e_data2,e_data3,e_data4))[0]
	print(e_data.shape)

	
	

	e_fit,e_mean,e_var = entropy_fit(e_data,rules[:,0],rules[:,1])
	l_fit = lyap_fit(lyap_data,rules[:,0],rules[:,1])
	print(l_fit.shape)
	print(e_fit.shape)
	print(e_mean.shape)
	print(e_var.shape)
	r_ent = np.zeros(rules.shape[0])
	r_mean = np.zeros(rules.shape[0])
	r_var = np.zeros(rules.shape[0])
	for r in range(rules.shape[0]):
		g.rule = rules[r,2:]
		r_ent[r] = g.rule_entropy()
		r_mean[r] = np.mean(g.rule)
		r_var[r] = np.std(g.rule)


	trn_data = np.concatenate((rules[:,0,None],
							   rules[:,1,None],
							   l_fit,
							   e_fit[:,None],
							   e_mean[:,None],
							   e_var[:,None],
							   r_ent[:,None]),axis=1)
	#np.save("2state_ml_data",trn_data)
	print(trn_data.shape)
	print(trn_data[10])





def generate_observables_unlabelled(N,rules):
	g.rule = rules[0]
	states = g.states


	R = rules.shape[0]
	print("_"*R)
	observables = np.zeros((R,16))
	mats = np.zeros((R,states,states))
	for i in range(R):
		g.rule = rules[i]
		observables[i],mats[i]= g.get_metrics(N)
		sys.stdout.write("#")
		sys.stdout.flush()

	np.save(str(states)+"state_unlabelled_observables.npy",observables)
	np.save(str(states)+"state_unlabelled_transition_matrices.npy")
	
	#observables = np.load("2state_ml_data.npy")
	print(observables[0])
	print(observables.shape)



def generate_observables(N,rules):
	g.rule = rules[0,2:]
	states = g.states


	R = rules.shape[0]
	print("_"*R)
	observables = np.zeros((R,18))
	mats = np.zeros((R,states,states))
	for i in range(R):
		g.rule = rules[i,2:]
		observables[i,0] = rules[i,0] #wolfram class label
		observables[i,1] = rules[i,1] #is interesting label
		observables[i,2:],mats[i]= g.get_metrics(N)
		sys.stdout.write("#")
		sys.stdout.flush()

	np.save(str(states)+"state_ml_data.npy",observables)
	np.save(str(states)+"state_transition_matrices.npy",mats)
	
	#observables = np.load("2state_ml_data.npy")
	print(observables[0])
	print(observables.shape)



def generate_observables_2state(N=4):
	#Loads and combines 2 state data sets and generates observables
	
	rules1 = np.load("random_2_state_rules.npy")
	rules2 = np.load("random_walk_2_state_rules.npy")
	rules3 = np.load("near_gol_ham_dist_2.npy")
	rules4 = np.load("random_walk_gol_rules.npy")

	rules = np.vstack((rules1,rules2,rules3,rules4))
	#print(rules[32])

	
	observables = np.zeros((rules.shape[0],18))
	for i in range(rules.shape[0]):
		g.rule = rules[i,2:]
		observables[i,0] = rules[i,0] #wolfram class label
		observables[i,1] = rules[i,1] #is interesting label
		observables[i,2:]= g.get_metrics(N)


	np.save("2state_ml_data.npy",observables)
	
	#observables = np.load("2state_ml_data.npy")
	print(observables[0])
	print(observables.shape)



def plot_observables(a,b):
	"""
	l_params[0],l_params[1],l_params[2],
                            e_smooth_var,e_mean,e_var,
                            r_entropy,
                            symmetry,
                            spacial[0,0],spacial[0,1],
                            spacial[1,0],spacial[1,1],
                            temporal[0,0],temporal[0,1],
                            temporal[1,0],temporal[1,1]])
	"""


	names = ["divergence_prefactor","divergence_exponent","divergence_offset",
			 "entropy_smooth","entropy_mean","entropy_variance","rule_entropy",
			 "fft_symmetry","spacial_first_peak_mean","spacial_first_peak_var","spacial_second_peak_mean","spacial_second_peak_var",
			 "temporal_first_peak_mean","temporal_first_peak_var","temporal_second_peak_mean","temporal_second_peak_var"]
	observables = np.load("2state_ml_data.npy")
	plt.scatter(observables[observables[:,1]==0,2+a],observables[observables[:,1]==0,2+b],color="red",label="boring")
	plt.scatter(observables[observables[:,1]==1,2+a],observables[observables[:,1]==1,2+b],color="blue",label="interesting")
	plt.legend()
	plt.xlabel(names[a])
	plt.ylabel(names[b])
	plt.show()

def hamming_explore(N,D):
	#Explores N random rules a hamming distance of D from GOL
	
	n_flips = np.zeros(18).astype(int)
	for d in range(1,D+1):
		n_flips[-d]=1
	
	
	rule_and_class = np.zeros((N,20)).astype(int)
	gol_rule = np.array([0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0]) 
	
	for n in range(N):	
		bit_flips = np.random.permutation(n_flips)
		r_new = (gol_rule - bit_flips)%2
		#print(bit_flips)
		#print(gol_rule)
		print(r_new)
		g.init_grid()
		g.rule = r_new
		g.run()
		ani_display()
		rule_and_class[n,0] = int(input("Estimate wolfram class: "))
		rule_and_class[n,1] = int(input("Is there interesting structure: "))
		rule_and_class[n,2:]=np.copy(r_new)

	return rule_and_class

def random_explore(N):
	#Explores N random rules
	
	#n_flips = np.zeros(18).astype(int)
	#for d in range(1,D+1):
	#	n_flips[-d]=1
	
	g.rule_gen()
	L = g.rule.shape[0]
	rule_and_class = np.zeros((N,L+2)).astype(int)
	#gol_rule = np.array([0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0]) 
	
	for n in range(N):	
		#bit_flips = np.random.permutation(n_flips)
		#r_new = (gol_rule - bit_flips)%2
		#print(bit_flips)
		#print(gol_rule)
		#print(r_new)
		print(n)
		g.init_grid()
		#g.rule = np.random.randint(2,size=18).astype(int)
		g.rule_gen()
		print(g.rule_entropy())
		#print(g.rule)
		g.run()
		ani_display()
		rule_and_class[n,0] = int(input("Estimate wolfram class: "))
		rule_and_class[n,1] = int(input("Is there interesting structure: "))
		rule_and_class[n,2:]= np.copy(g.rule)

	return rule_and_class

def random_walk_explore(N):
	#Explores N random rules, where each one differs by the previous by k entries
	g.rule_gen(mode=1)
	rule = np.copy(g.rule)
	L = rule.shape[0]
	k = max(L//20,1)
	print(str(k)+" flips")
	rule_and_class = np.zeros((N,L+2)).astype(int)
	
	
	for n in range(N):	
		#bit_flips = np.random.permutation(n_flips)
		#r_new = (gol_rule - bit_flips)%2
		rule = np.copy(g.rule)
		ii = np.random.choice(L,size=k,replace=False)
		rule[ii] = (rule[ii]+np.random.randint(1,g.states,size=k))%g.states
		#print(rule)
		print(g.rule_entropy())
		print(n)
		g.init_grid()
		g.rule = np.copy(rule)
		#print(g.rule)
		g.run()
		ani_display()
		rule_and_class[n,0] = int(input("Estimate wolfram class: "))
		rule_and_class[n,1] = int(input("Is there interesting structure: "))
		rule_and_class[n,2:]= np.copy(g.rule)

	return rule_and_class


def smooth_perm_explore(N):
	#Random walk but where user chooses whether to "smooth" or "permute" rule, reducing or increasing entropy respectively
	g.rule_gen(mode=0)
	rule = np.copy(g.rule)
	L = rule.shape[0]
	rule_and_class = np.zeros((N,L+2)).astype(int)
	k = 1
	
	for n in range(N):	

		print(g.rule_entropy())
		print(n)
		g.init_grid()
		#print(g.rule)
		g.run()
		ani_display()
		rule_and_class[n,0] = int(input("Estimate wolfram class: "))
		rule_and_class[n,1] = int(input("Is there interesting structure: "))
		rule_and_class[n,2:]= np.copy(g.rule)
		choice = int(input("Smooth, permute, offset or mutate? (0/1/2/3): "))
		if choice==0:
			g.rule_smooth(3)
		if choice==1:
			g.rule_perm()
		if choice==2:
			g.rule_offset(k)
		if choice==3:
			g.run_mutate()
			ani_display()
			best = int(input("Select best rule (1-4): "))
			g.choose_offspring(best)
	return rule_and_class


def mutate_explore(N):
	#Random walk but where user chooses mutations
	g.rule_gen(mode=1)
	rule = np.copy(g.rule)
	L = rule.shape[0]
	rule_and_class = np.zeros((N,L+2)).astype(int)
	#k = 1
	
	for n in range(N):	

		print(g.rule_entropy())
		print(n)
		g.init_grid()
		#print(g.rule)
		g.run()
		ani_display()
		rule_and_class[n,0] = int(input("Estimate wolfram class: "))
		rule_and_class[n,1] = int(input("Is there interesting structure: "))
		rule_and_class[n,2:]= np.copy(g.rule)
		g.run_mutate()
		ani_display()
		best = int(input("Select best rule (1-4): "))
		g.choose_offspring(best)
		
		
	return rule_and_class




def perm_explore(N):
	#1st column is wolfram classifier, rest is rule
	rule_and_class = np.zeros((N,19)).astype(int)

	gol_rule = np.array([0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0]) 
	l = gol_rule.shape[0]
	gol_rule_r = gol_rule.reshape((2,l//2))
	for n in range(N):	
		r_new = (np.random.permutation(gol_rule_r.T).T).reshape(l)
		print(r_new)
		g.init_grid()
		g.rule = r_new
		g.run()
		ani_display()
		rule_and_class[n,0] = int(input("Enter wolfram class: "))
		rule_and_class[n,1:]=np.copy(r_new)

	return rule_and_class





def entropy_evaluate(classifications,is_interesting,rules,N):
	K = classifications.shape[0]
	print(("_"*N+"|")*K)
	ent_data = np.zeros((N+2,K,iterations))
	for k in range(K):
		g.rule=rules[k]
		ent_data[2:,k] = g.entropy(N)
		ent_data[0,k] = np.mean(ent_data[2:,k,:],axis=0)
		ent_data[1,k] = np.std(ent_data[2:,k,:],axis=0)


	for k in range(K):
		if classifications[k]==1:
			plt.plot(ent_data[0,k],color="orange",alpha=0.5)
		elif classifications[k]==2:
			plt.plot(ent_data[0,k],color="green",alpha=0.5)
		elif classifications[k]==3:
			plt.plot(ent_data[0,k],color="red",alpha=0.5)
		elif classifications[k]==4:
			plt.plot(ent_data[0,k],color="blue",alpha=0.5)

	plt.plot([0],[0],color="orange",label="1",alpha=0.5)
	plt.plot([0],[0],color="green",label="2",alpha=0.5)
	plt.plot([0],[0],color="red",label="3",alpha=0.5)
	plt.plot([0],[0],color="blue",label="4",alpha=0.5)

	plt.legend(loc=1,framealpha=1)
	plt.xlabel("Timesteps")
	plt.ylabel("Entropy")
	plt.show()

	for k in range(K):
		if is_interesting[k]==1:
			plt.plot(ent_data[0,k],color="blue",alpha=0.5)
		else:
			plt.plot(ent_data[0,k],color="red",alpha=0.5)


	plt.plot([0],[0],color="red",label="Boring",alpha=0.5)
	plt.plot([0],[0],color="blue",label="Interesting",alpha=0.5)

	plt.legend(loc=1,framealpha=1)
	plt.xlabel("Timesteps")
	plt.ylabel("Entropy")
	plt.show()
	return ent_data

def lyap_evaluate(classifications,is_interesting,rules,N,norm=False):
	
	K = classifications.shape[0]
	print(("_"*N+"|")*K)
	#First 2 columns are mean and std over K repeated simulations of length "iterations"
	lyap_data = np.zeros((N+2,K,iterations))
	t_mats = np.zeros((K,states,states))
	eig_vals = np.zeros((K,states,states)) 
	for k in range(K):

		g.rule = rules[k]
		if norm:
			t_mats[k] = g.density_matrix()
		lyap_data[2:,k] = g.lyap(N,norm)
		
		lyap_data[0,k] = np.mean(lyap_data[2:,k,:],axis=0)
		lyap_data[1,k] = np.std(lyap_data[2:,k,:],axis=0)
		
		#eig_vals[k] = np.linalg.eig(t_mats[k])[1]

		#for k in range(K):
	for k in range(K):
		if classifications[k]==1:
			plt.plot(lyap_data[0,k],color="orange",alpha=0.5)
		elif classifications[k]==2:
			plt.plot(lyap_data[0,k],color="green",alpha=0.5)
		elif classifications[k]==3:
			plt.plot(lyap_data[0,k],color="red",alpha=0.5)
		elif classifications[k]==4:
			plt.plot(lyap_data[0,k],color="blue",alpha=0.5)

	plt.plot([0],[0],color="orange",label="1",alpha=0.5)
	plt.plot([0],[0],color="green",label="2",alpha=0.5)
	plt.plot([0],[0],color="red",label="3",alpha=0.5)
	plt.plot([0],[0],color="blue",label="4",alpha=0.5)

	#plt.plot(lyap_data[0,-1],color="black",label="GoL")
	plt.legend()
	plt.xlabel("Timesteps")
	plt.ylabel("Divergence")
	plt.show()


	for k in range(K):
		if is_interesting[k]==1:
			plt.plot(lyap_data[0,k],color="blue",alpha=0.5)
		else:
			plt.plot(lyap_data[0,k],color="red",alpha=0.5)


	plt.plot([0],[0],color="red",label="Boring",alpha=0.5)
	plt.plot([0],[0],color="blue",label="Interesting",alpha=0.5)

	plt.legend()
	plt.xlabel("Timesteps")
	plt.ylabel("Divergence")
	plt.show()

	return lyap_data





def lyap_fit(data,wclass,is_interesting):
	#Fit lyap divergence to a power lay L=At^B
	#Axis 0 - Which rule?
	#Axis 1 - data per rule
	mean_data = data[0,:,:]
	M = data.shape[0]-2
	N = mean_data.shape[0]
	its = mean_data.shape[1]
	ts = np.arange(its)
	def power_law(x,A,B,C):
		return A*(np.float_power(x,B))+C

	params = np.zeros((N,3))
	
	for i in range(N):
		params[i],_ = sp.optimize.curve_fit(power_law,ts,mean_data[i])
	

	
	
	
	plt.subplot(221)
	plt.scatter(params[wclass==1,1],params[wclass==1,0],color="orange",label="1")
	plt.scatter(params[wclass==2,1],params[wclass==2,0],color="green",label="2")
	plt.scatter(params[wclass==3,1],params[wclass==3,0],color="red",label="3")
	plt.scatter(params[wclass==4,1],params[wclass==4,0],color="blue",label="4")
	plt.xlabel("Exponent")
	plt.ylabel("Prefactor")
	
	#plt.legend()

	plt.subplot(223)
	plt.scatter(params[wclass==1,1],params[wclass==1,2],color="orange",label="1")
	plt.scatter(params[wclass==2,1],params[wclass==2,2],color="green",label="2")
	plt.scatter(params[wclass==3,1],params[wclass==3,2],color="red",label="3")
	plt.scatter(params[wclass==4,1],params[wclass==4,2],color="blue",label="4")
	plt.xlabel("Exponent")
	plt.ylabel("Offset")
	#plt.legend()

	plt.subplot(222)
	plt.scatter(params[wclass==1,2],params[wclass==1,0],color="orange",label="1")
	plt.scatter(params[wclass==2,2],params[wclass==2,0],color="green",label="2")
	plt.scatter(params[wclass==3,2],params[wclass==3,0],color="red",label="3")
	plt.scatter(params[wclass==4,2],params[wclass==4,0],color="blue",label="4")
	plt.xlabel("Offset")
	plt.ylabel("Prefactor")
	plt.legend()

	plt.suptitle("Power law fits to divergence of CA, with Wolfram class (512 2 state rules)")
	plt.show()
	
	plt.subplot(221)
	plt.scatter(params[is_interesting==0,1],params[is_interesting==0,0],color="red",label="boring")
	plt.scatter(params[is_interesting==1,1],params[is_interesting==1,0],color="blue",label="interesting")
	plt.xlabel("Exponent")
	plt.ylabel("Prefactor")
	#plt.legend()

	plt.subplot(223)
	plt.scatter(params[is_interesting==0,1],params[is_interesting==0,2],color="red",label="boring")
	plt.scatter(params[is_interesting==1,1],params[is_interesting==1,2],color="blue",label="interesting")
	plt.xlabel("Exponent")
	plt.ylabel("Offset")
	#plt.legend()

	plt.subplot(222)
	plt.scatter(params[is_interesting==0,2],params[is_interesting==0,0],color="red",label="boring")
	plt.scatter(params[is_interesting==1,2],params[is_interesting==1,0],color="blue",label="interesting")
	plt.xlabel("Offset")
	plt.ylabel("Prefactor")
	plt.legend()
	plt.suptitle("Power law fits to divergence of CA, with binary classifier (512 2 state rules)")
	plt.show()


	#plt.plot(params[:,1],params[:,2])
	#plt.show()

	plt.title("Distribution of power law exponent, with binary classifier")
	plt.hist(params[is_interesting==0,1],color="red",bins=100,alpha=0.5,label="boring")
	plt.hist(params[is_interesting==1,1],color="blue",bins=100,alpha=0.5,label="interesting")
	plt.legend()
	plt.xlabel("Exponent")
	plt.show()


	return params

def entropy_fit(data,wclass,is_interesting):
	#See how entropy behaviour correlates with classifiers
	
	#Plot all entropy data
	ts = np.arange(data.shape[1])
	plt.plot(ts,data[is_interesting==0].T,color="red",alpha=0.2)
	plt.plot(ts,data[is_interesting==1].T,color="blue",alpha=0.2)
	plt.plot([0],[0],color="red",label="boring")
	plt.plot([0],[0],color="blue",label="interesting")
	plt.title("Entropy with binary classifier (512 2 state rules)")
	plt.xlabel("Timesteps")
	plt.ylabel("Entropy (normalised between 0-1)")
	plt.legend()
	plt.show()
	A=100
	B=200
	
	
	#mean absolute derivative of smoothed entropy
	a,b = sp.signal.butter(5,0.1,analog=False)
	data_sm = np.zeros(data.shape)
	for i in range(data.shape[0]):
		data_sm[i] = np.diff(sp.signal.filtfilt(a,b,data[i]),prepend=0)
		if is_interesting[i]==0:
			plt.plot(ts[A:B],data_sm[i,A:B],color="red",alpha=0.5)
		elif is_interesting[i]==1:
			plt.plot(ts[A:B],data_sm[i,A:B],color="blue",alpha=0.5)
	plt.show()
	m = np.zeros(data.shape[0])
	for i in range(data.shape[0]):
		m[i] = np.mean(np.abs(data_sm[i,A:B]))
	plt.hist(m[is_interesting==0],bins=200,color="red",alpha=0.5,range=[0,0.003])
	plt.hist(m[is_interesting==1],bins=200,color="blue",alpha=0.5,range=[0,0.003])
	plt.show()
	

	mean = np.mean(data[:,A:],axis=1)
	var = np.std(data[:,A:],axis=1)




	print(mean.shape)
	print(m.shape)
	print(var.shape)
	return m,mean,var
	#Plot smoothed entropy data
	#plt.legend()






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