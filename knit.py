import numpy as np 
import matplotlib.pyplot as plt 
import sys

def update_rule(grid,grid_old,pos):
	n = grid.shape[0]
	#print(n)
	s=bool(grid[pos])
	s_old=bool(grid_old[pos])
	l = bool(grid[(pos-1)%n])
	r = bool(grid[(pos+1)%n])
	
	#return (l^r^s)# and not s
	#return (l ^ ((s ^ s_old) or r))
	return (l ^ (s or r ^ s_old))and not s
	
	#return ((l  or r ^ (s_old^s)) and not s) 
	

def main():
	N = int(sys.argv[1])
	L = int(sys.argv[2])
	grid = np.random.choice([0,1],size=N)
	grid_old = np.random.choice([0,1],size=N)
	grid_new = np.zeros(N)
	data = np.zeros((L,N))
	for l in range(L):
		for x in range(N):
			grid_new[N-x-1] = update_rule(grid,grid_old,N-x-1)
		grid_old = np.copy(grid)
		grid = np.copy(grid_new)
		data[l] = grid
	plt.imshow(data,cmap="magma")
	plt.show()
main()