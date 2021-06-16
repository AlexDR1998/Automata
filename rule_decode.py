import numpy as np
from automata_class import Grid2D
#Takes the random seed saved when generating a set of rules, and outputs those rules

def decode(instance,states,N_rules,mode):
	
	_g = Grid2D(4,0.5,states,1,4,1)
	rng = np.random.RandomState(instance)

	if mode==0:
		#Uniform sampling
		rules = rng.randint(states,size=(N_rules,_g.rule_length))

	if mode==1:
		#Binomial sampling with 100 different sub-divided distributions
		splits=100
		K = N_rules//splits
		ps = np.linspace(0,1,K+2)[1:-1]
		_rules = np.zeros((K,splits,_g.rule_length)).astype(int)
		
		for k in range(K):
			_rules[k] = rng.binomial(states-1,ps[k],size=(splits,_g.rule_length))
		rules = _rules.reshape((N_rules,_g.rule_length))
	return rules
print(decode(0,4,1000,1))