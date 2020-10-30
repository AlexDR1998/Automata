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
	states = 2#int(input("Enter number of states: "))
	symm = 2#int(input("Enter neighbourhood type (0,1 or 2): "))
	neighbours = 1#int(input("Enter size of cell neighbourhood: "))
	global size
	size = 256#int(input("Enter size of grid: "))
	global iterations
	iterations = 16#int(input("Enter number of iterations: "))
	global g
	global screendata
	global matrix
	global data
	global colour
	colour = "gist_earth"
	mu=0.5
	sig=0.2
	g = Grid2D(size,0.5,states,neighbours,iterations,symm)
	g.init_grid()
	g.rule = np.array([0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0])
	#g.rule_gen(mu,sig)
	#g.iterations = 32
	N = 128
	t_mats = np.zeros((N,states,states))
	eig_vals = np.zeros((N,states))
	print("_"*N)

	#temp = np.array((size,size))
	for n in range(N):	
		t_mats[n] = g.density_matrix(False)	
		#ani_display()
		sys.stdout.write("#")
		sys.stdout.flush()
		eig_vals[n] = np.linalg.eig(t_mats[n])[0]
	print("")
	m_tmat = np.mean(t_mats,axis=0)
	s_tmat = np.std(t_mats,axis=0)
	print("Mean t_matrix over whole run")
	print(m_tmat)
	print(s_tmat)
	print("Mean t_matrix over latter half")
	print(np.mean(t_mats[N//2:],axis=0))
	print(np.std(t_mats[N//2:],axis=0))
	plt.plot(eig_vals[:,0],label = "eigenval_1")
	plt.plot(eig_vals[:,1],label = "eigenval_2")
	plt.xlabel("Timesteps (in chunks of "+str(iterations)+")")
	plt.legend()
	plt.show()
	#g.print_info()
	#ani_display()
	#t_mat_short = np.linalg.matrix_power(g.density_matrix(),10)
	
	#g.iterations = 1024
	#g.run()
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
