from org import Grid
from org import Grid2D
import numpy as np
from scipy import ndimage
#from interface import Interface as i 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import time

"""
Program to demonstrate cellular automate code
"""

def main():
	global states
	global size
	global iterations
	global g
	global screendata
	global matrix
	global data
	global colour
	colour = "magma"
	inp = "a"
	

	
	#demo first rule - 8 states, neighbourhood size 1
	states = 8
	neighbours = 1
	size = 100
	iterations = 200
	g = Grid2D(size,0.5,states,neighbours,iterations)
	g.rule_load("fluid")
	g.run()
	ani_display("8 state rule")
	
	colour = "nipy_spectral"
	#Demo second rule - 100 states, n=1, smoothed to show large scale structure
	states = 100
	neighbours = 1
	size = 200
	iterations = 200
	g = Grid2D(size,0.5,states,neighbours,iterations)
	g.rule_load("decay3")
	g.run()
	ani_display("100 state rule, with smoothing",mode=1)
	snapshot(0,"Stationary 2D projection")


	states = 8
	neighbours = 2
	size = 200
	iterations = 200
	g = Grid2D(size,0.5,states,neighbours,iterations)
	g.rule_load("sr2")
	g.run()
	ani_display("8 state rule with larger cell neighbourhood")
	ani_display("Same rule but with smoothing",mode=1)
	

	states = 100
	neighbours = 1
	size = 200
	iterations = 1000
	g = Grid2D(size,0.5,states,neighbours,iterations)
	g.rule_load("tartan")
	g.run()
	ani_display("100 state rule")
	ani_display("Same rule but with smoothing",mode=1)
	snapshot(0,"Stationary 2D projection")
	



def snapshot(i,title):
    if i==0:
        data = g.im_out()[10:,:,:]   
    else:
        data = g.im_out()

    ys = np.sum(data,axis=i)
    plt.matshow(ys,cmap=colour)
    plt.title(title)
    plt.show()



def smooth(d,am):
    k = np.ones((am,am,am))
    k = k/np.sum(k)
    return ndimage.convolve(d,k)




def ani_display(title,mode=0):
    if mode==0:
        data = g.im_out()
    elif mode==1:
        data = smooth(g.im_out(),3)

    elif mode==2:
        data = np.moveaxis(smooth(g.im_out(),3),0,1)

    elif mode==4:
        data = np.moveaxis(g.im_out(),0,1)
    
    elif mode==5:
        data = np.diff(np.diff(g.im_out(),axis=0),axis=0)
        


    def update(i):
        screendata = data[i]
        matrix.set_array(screendata)
    screendata = data[0]
    fig, ax = plt.subplots()            
    matrix = ax.matshow(screendata,cmap=colour)
    plt.colorbar(matrix)
    ani = animation.FuncAnimation(fig,update,frames=iterations,interval=100)
    plt.title(title)
    plt.show()



main()
