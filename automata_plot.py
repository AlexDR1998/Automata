from org import Grid
from org import Grid2D
import numpy as np
from scipy import ndimage
#from interface import Interface as i 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import time


"""
Program to interface between user and Cellular automata code in org.py
"""



def main():
    global states
    states = int(input("Enter number of states: "))
    neighbours = int(input("Enter size of cell neighbourhood: "))
    global size
    size = int(input("Enter size of grid: "))
    global iterations
    iterations = int(input("Enter number of iterations: "))
    global g
    global screendata
    global matrix
    global data
    global colour
    colour = "magma"#"gist_earth"
    inp = "a"
    g = Grid2D(size,0.5,states,neighbours,iterations)
    counter = 277
    while inp!="q":
        counter = counter+1
        inp = str(raw_input(":")) 
        if inp=="h":
            #Print help.txt to terminal
            f = open("text_resources/help.txt","r")
            print(f.read())
            f.close()
        if inp=="r":
            #Rerun the same rule on different initial state
            g.run()
            ani_display()            
        if inp=="":
            #Generate new rule and run          
            g.rule_gen()
            g.run()
            ani_display()
        if inp=="t":
            #generate new rule and tranpose output
            g.rule_gen()
            g.run()
            ani_display(mode=4)
        
        if inp=="smooth":
            #Apply image smoothing to output
            g.run()
            ani_display(mode=1)


        if inp=="tsmooth":
            #Apply image smoothing and tranpose output
            g.run()
            ani_display(mode=2)

        if inp=="rt":
            #Rerun same rule, transposed
            g.run()
            ani_display(mode=4)
        if inp=="s": 
            #save current rule        
            filename = str(raw_input("Enter rule name: "))
            g.rule_save(filename)
        if inp=="l":
            #load from saved rules
            read_saved_rules(neighbours,states)
            filename = str(raw_input("Enter rule name: "))
            g.rule_load(filename)
            g.run()
            ani_display()
        if inp=="n":
            #display saved rules
            read_saved_rules(neighbours,states)
        if inp=="p":
            #permute current rule and rerun           
            g.rule_perm()
            g.run()
            ani_display()
        if inp=="d":
            #Change initial state density  
            d = float(raw_input("Enter initial density: "))
            g = Grid2D(size,d,states,neighbours,iterations)
        if (inp=="+" or inp=="*" or inp=="-" or inp=="m" or inp=="z" or inp=="c"):
            #Combine 2 rules in some way
            read_saved_rules(neighbours,states)
            rule1 = str(raw_input("Enter first rule: "))
            rule2 = str(raw_input("Enter second rule: "))
            g.rule_comb(rule1,rule2,inp)
            g.run()
            ani_display()
        if inp=="i":
            #Invert rule
            g.rule_inv()
            g.run()
            ani_display()
        if inp=="w":
            #Smooth rule
            am = int(raw_input("Enter smoothing amount: "))
            g.rule_smooth(am)
            g.run()
            ani_display()
        if inp=="f":
            #fold rule
            am = int(raw_input("Enter folding amount: "))
            g.rule_fold(am)
            g.run()
            ani_display()
        if inp=="ch":
            #change initial conditon type
            g.change_start_type()
        if inp=="e":
            #Plot entropy of CA simulation
            #g.run()
            #ani_display(mode=5)
            entropy_plot()
        if inp=="2d":
            #Project output onto 2d surface
            axis = int(raw_input("Enter axis: "))
            snapshot(axis)


        #if inp!="q":
        #    name = "rule_"+str(counter)
        #    t = str(raw_input("Enter rule type(i,d,n,s): "))
        #    g.store(t,name)


def ani_display(mode=0):
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
    plt.show()



def snapshot(i):
    if i==0:
        data = g.im_out()[10:,:,:]   
    else:
        data = g.im_out()

    ys = np.sum(data,axis=i)
    plt.matshow(ys,cmap=colour)
    plt.show()
    

def entropy_plot():
    f = lambda x:np.diff(x,axis=0)

    data = f(g.im_out())
    ys = abs(np.sum(data,axis=(1,2)))/size**2
    xs = np.arange(len(ys))
    plt.plot(xs,ys)
    plt.show()
    #print(xs)


def smooth(d,am):
    k = np.ones((am,am,am))
    k = k/np.sum(k)
    return ndimage.convolve(d,k)




def read_saved_rules(n,s):
    f = open("text_resources/name_header.txt","r")
    print(f.read())
    f.close()
    f = open('2D_rules/n'+str(n)+'/s'+str(s)+'/namelist.txt','r')
    print(f.read())
    f.close()
    f = open("text_resources/name_foot.txt","r")
    print(f.read())
    f.close()

if __name__ == '__main__':
    main()



