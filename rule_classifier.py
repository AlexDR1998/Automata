import numpy as np 
import matplotlib.pyplot as plt
import math
from scipy import stats
from org import Grid2D
import itertools
import time

"""

	0 - dead
	1 - interesting
	2 - noise
	3 - structured noise
"""


def main():
	global trn_label
	global tst_label
	global n_rules
	global n_tst_rules
	n_tst_rules=82
	n_rules=277
	
	trn_label = np.zeros(n_rules)
	
	#current data set is rules of 8 states and neighbour mode 2
	size=64
	iterations = 100

	g = Grid2D(size,0.5,8,2,iterations)
	
	
	
	trn_data = np.array([])
	#rule1=load_training_data(5)
	#print(len(rule1))
	
	
	for x in range(n_rules):
		trn_data = np.append(trn_data,load_training_data(x))
	
	trn_data = np.reshape(trn_data,(n_rules,2024)).astype(int)
	

	res = np.zeros(n_rules)
	for x in range(n_rules):
		
	
		#print(trn_data[x])
		r = trn_data[x]
		g.rule_input(r)
		g.run()


		#out = g.im_out()
		f = lambda x:np.diff(x,axis=0)
		#print(trn_label[x])

		data = (f(g.im_out()))
	    
		res[x] = np.sum(abs(np.sum(data,axis=(1,2)))/(iterations*size**2))
		#xs = np.arange(len(ys))
		#plt.plot(xs,ys)
		#plt.show()
		#plt.pause(0.0001)
		
	plt.scatter(trn_label,res)
	plt.show()
	#print(len(trn_data[20]))


	
	"""
	tst_data = np.array([])
	for x in range(n_tst_rules):
		tst_data = np.append(tst_data,load_test_data(x))
	tst_data = np.reshape(tst_data,(n_tst_rules,2640))


	#print(len(trn_label))
	densities = np.sum(trn_data,axis=1)
	state_count = np.zeros((9,n_rules))
	#print(state_count[0])
	for x in range(9):
		state_count[x] = np.count_nonzero(trn_data==x,axis=1)
	#print(np.sum(state_count,axis=0))
	
	#print(state_count[0])
	#count = 0
	#for x in range(2640):
#		if trn_data[0][x]==0:
	#		count = count+1
	#print(count)
	#print(len(densities))
	#print(len(trn_label))
	#plt.scatter(trn_label,state_count[3])
	#plt.show()
	#print(np.count_nonzero(trn_label==2))
	#for x in range(n_rules):
		#loop for each rule, calculate density
	#	densities.append(sum(trn_data[x]))
	#print(sum(trn_data[2]))
	rule_consistency(8)
	"""



	#print(trn_data)
	#print(np.shape(trn_data))
	#accs = []
	#for x in range(1,30):
	#	preds = my_knn_classify(trn_data,trn_label,tst_data,[x])[0]
	#	accs.append(my_confusion(tst_label,preds)[1])



def rule_consistency(states):
	"""
	returns how often a rule returns a number that was put into it
	"""
	#all_inputs = np.zeros((2640,5))
	x = list(itertools.combinations_with_replacement(range(states),4))
	all_inputs = list(itertools.product(range(states),x))
	#sums = np.sum(x,axis=1)
	#print(sums*8)
	#print(all_inputs[201][1][2])
	test = np.zeros(2640)
	

	for x in range(2640):
		for y in range(4):
			test[x] = test[x] + all_inputs[x][1][y]*8
		test[x] = test[x] + all_inputs[x][0]
	print(list(test))

	#plt.plot(range(1,30),accs)
	#plt.show()
	#print(tst_label)

	#print()
	#print(all(data[2]==rule1))
	#print(data[2])
	#print(np.shape(data))
	#for x in range(276):
		#print(trn_label[x])	
	#print(rtype)


def load_training_data(n):
	counter = n
	while True:
		try:
			f = open('2D_rules/n1/ml_data/training/dead/rule_'+str(counter)+'.csv','r')
			rtype=0
			break
		except:
			try:
				f = open('2D_rules/n1/ml_data/training/interesting/rule_'+str(counter)+'.csv','r')
				rtype=1
				break
			except:
				try:
					f = open('2D_rules/n1/ml_data/training/noise/rule_'+str(counter)+'.csv','r')
					rtype=2
					break
				except:
					try:
						f = open('2D_rules/n1/ml_data/training/s_noise/rule_'+str(counter)+'.csv','r')
						rtype=3
						break
					except:
						counter = counter+1
	trn_label[n-1]=rtype
	#print(rtype)
	a = np.load(f)
	f.close()
	return a

def load_test_data(n):
	counter = n
	while True:
		try:
			f = open('2D_rules/n1/ml_data/test/dead/rule_'+str(counter)+'.csv','r')
			rtype=0
			break
		except:
			try:
				f = open('2D_rules/n1/ml_data/test/interesting/rule_'+str(counter)+'.csv','r')
				rtype=1
				break
			except:
				try:
					f = open('2D_rules/n1/ml_data/test/noise/rule_'+str(counter)+'.csv','r')
					rtype=2
					break
				except:
					try:
						f = open('2D_rules/n1/ml_data/test/s_noise/rule_'+str(counter)+'.csv','r')
						rtype=3
						break
					except:
						counter = counter+1
	tst_label[n-1]=rtype
	#print(rtype)
	a = np.load(f)
	f.close()
	return a


main()