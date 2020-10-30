from org import Grid2D
import numpy as np
from scipy import ndimage
#from interface import Interface as i 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import time
import sys


def main():
	global states
	global symm
	states = 2
	symm = 2
	neighbours = 1
	global size
	size = 128
	global iterations
	iterations = 64
	global g
	global screendata
	global matrix
	global data
	global colour
	colour = "gist_earth"
	#mu=0.5
	#sig=0.2
	g = Grid2D(size,0.5,states,neighbours,iterations,symm)
	"""
	
	gol_rule = np.array([0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0]) 
	g.rule = gol_rule
	#Use wolfram classes of automata
	#  	1 - homogenous endpoint - (almost) everything dies out and converges to the same homogenous state
	#	2 - stable structures - fixed points or time periodic structures
	#	3 - noisy - any structures are dominated by noise, changes spread fast
	#   4 - dynamic structures - interesting stuff like GoL
	class_near_gol = np.array([3,3,3,1,4,4,4,4,4,4,4,1,2,4,3,4,4,4,4])
	#g.run()
	#plt.plot(g.lyap())
	rules = np.tile(gol_rule,(19,1))
	I = np.eye(19)[:,:18]
	print(I)
	#print((rules-I)%2)
	rules = ((rules-I)%2).astype(int)
	print(rules)
	lyap_evaluate(class_near_gol,rules,4)
	"""
	#N=16
	#data = perm_explore(128)
	data = hamming_explore(128,2)
	#print(data)
	np.save("near_gol_ham_dist_2",data)
	lyap_evaluate(data[:,0],data[:,2:],4)



def hamming_explore(N,D):
	#Explores all rules a hamming distance of 2 from GOL
	
	n_flips = np.zeros(18).astype(int)
	for d in range(1,D+1):
		n_flips[-d]=1
	#print(n_flips)
	#bit_flips = np.array(list(set(permutations(n_flips))))
	#bit_flips = multiset_permutations(n_flips)
	#for i in range(18):
	#	bit_flips = np.stack(bit_flips,)
	
	#print(N)
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




def lyap_evaluate(classifications,rules,N):
	
	K = classifications.shape[0]
	print(("_"*N+"|")*K)
	lyap_data = np.zeros((N+2,K,iterations))
	t_mats = np.zeros((K,states,states))
	eig_vals = np.zeros((K,states,states)) 
	for k in range(K):
		#r = np.copy(gol_rule)
		#if k==18:
		#	g.rule=r
		#else:

		#	r[k] = 1-r[k]
		#	g.rule = r
		#g.print_rule()
		#print(rules[k])
		g.rule = rules[k]
		lyap_data[:,k] = g.lyap(N)
		
		lyap_data[0,k] = np.mean(lyap_data[2:,k,:],axis=0)
		lyap_data[1,k] = np.std(lyap_data[2:,k,:],axis=0)
		
		t_mats[k] = g.density_matrix()
		eig_vals[k] = np.linalg.eig(t_mats[k])[1]
		#g.run()
		#ani_display()
		#g.fft()
		#sys.stdout.write("#")
		#sys.stdout.flush()
		#ani_display()
		"""
		for i in range(N):
			plt.plot((lyap_data[i+2,k]),color="blue",alpha=0.1)
		plt.plot(lyap_data[0,k],color="black")
		plt.plot(lyap_data[0,k]+lyap_data[1,k],color="red",alpha=0.4)
		plt.plot(lyap_data[0,k]-lyap_data[1,k],color="red",alpha=0.4)
		plt.show()
		"""
		#for k in range(K):
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
	



	"""
	for k in range(1,K):
		if class_near_gol[k]==1:
			plt.scatter(eig_vals[k,0,0],eig_vals[k,0,1],color = "blue")
			plt.scatter(eig_vals[k,1,0],eig_vals[k,1,1],color = "purple")
		else:
			plt.scatter(eig_vals[k,0,0],eig_vals[k,0,1],color="red")
			plt.scatter(eig_vals[k,1,0],eig_vals[k,1,1],color="orange")
	
	plt.plot([0],[0],color = "blue",label="interesting")
	plt.plot([0],[0],color = "purple",label="interesting")
	plt.plot([0],[0],color = "red",label="boring")
	plt.plot([0],[0],color = "orange",label="boring")
	plt.legend()
	plt.show()
	"""
	#t_mat_long = g.density_matrix()
	#print(t_mat_short)
	#print(t_mat_long)
	#ani_display()









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