import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.animation as animation
#import scipy
"""
An implemntation of the 2D cellular automata generating code, without OOP,
optimised using only numpy when possible
"""

def gaussian(x,mu,sig):
	return np.exp(-np.power(x-mu,2)/(2*np.power(sig,2)))


def rule_init(s):
	n = s-1
	#print(s)
	for x in range(0,1):
		n = n + 4*(s-1)*(s**(x+1))
	print(n)
	
	rule = np.random.randint(s,size=n+1)
	


	rule[0]=0
	xs = np.linspace(0,n,n+1)
	for x in range(n):
		#print(x)
		

		if np.random.rand()>gaussian(xs[x],xs[int(np.floor(n/2))],(n/5)):
			rule[x]=0
	return rule

def main():
	from scipy import signal
	g_size = int(input("Enter size of grid: "))
	iterations = int(input("Enter number of iterations: "))
	s = int(input("Enter number of states: "))
	density = int(input("Enter initial density: "))
	#Initialise rule
	rule = rule_init(s)





	#Initialise grid
	grid = np.random.randint(s,size=(g_size,g_size))
	s_grid = np.zeros((g_size,g_size))
	#global image_out
	image_out = np.zeros((iterations,g_size,g_size))
	ntype=1
	if ntype==2:
		k = np.array([[s**2,s,s**2],[s,1,s],[s**2,s,s**2]])
	else:
		k = np.array([[0,s,0],[s,1,s],[0,s,0]])
	#Run rule
	for x in range(iterations):
		#print(x)
		s_grid = signal.convolve2d(grid,k,boundary='wrap',mode='same')
		v = np.vectorize(lambda y:rule[y])
		grid = v(s_grid)
		image_out[x]=grid
	screendata = image_out[0]

	fig, ax = plt.subplots()            
	#global matrix
	matrix = ax.matshow(screendata,cmap="nipy_spectral")
	def update(i):
		screendata = image_out[i]
		matrix.set_array(screendata)
	plt.colorbar(matrix)
	ani = animation.FuncAnimation(fig,update,frames=iterations,interval=100)
	plt.show()



main()