#from org import Grid
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from automata_class import Grid2D
import numpy as np
from scipy import ndimage
#from interface import Interface as i 
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
#import time
#import cv2

"""
Program to interface between user and Cellular automata code in org.py
"""



def main_loop():
    global states
    global symm
    states = int(input("Enter number of states: "))
    symm = 2#int(input("Enter neighbourhood type (0,1 or 2): "))
    neighbours = 1#int(input("Enter size of cell neighbourhood: "))
    global size
    size = int(input("Enter size of grid: "))
    global iterations
    iterations = int(input("Enter number of iterations: "))
    global g
    global screendata
    global matrix
    global data
    global colour
    colour = "gnuplot2"
    inp = "a"
    mu=0.7
    sig=0.3
    g = Grid2D(size,0.5,states,neighbours,iterations,symm)

    while inp!="q":
        #inp = str(input(":")) 
        inp = str(input(":"))
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
            g.rule_gen(mu,sig)
            g.run()
            ani_display()
        if inp=="fft":
            _,_,_,stat = g.fft()
            """
            s = g.size//2
            stat[:,s,s]=0
            for i in range(g.states):
                plt.imshow(np.abs(stat[i]),extent=[-s,s,-s,s],cmap=colour)
                plt.colorbar()
                plt.show()
            #print(g.fft())
            """
            ani_display(7)
            #ani_display()
        """
        if inp=="auto":
            d = int(input("Iteration depth: "))
            rs,ps = g.auto_evolve(d)
            print(ps)
            for i in range(d):
                g.rule=rs[i]
                g.run()
                ani_display()
        if inp=="metric":
            a = g.get_metrics(4)
            print(a)
            print(a.shape)
        if inp=="gol":
            g.rule = np.array([0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0])
            g.run()
            ani_display()
        """
        if inp=="info":
            g.print_info()

        if inp=="print":
            g.print_rule()

        """
        if inp=="dmat":
            mat,err_mean,err_var,err = g.density_matrix()
            print(err_mean)
            print(err_var)
            plt.matshow(mat)
            plt.ylabel("Inputs")
            plt.xlabel("Outputs")
            plt.show()
            plt.plot(err)
            plt.show()
        """    
        if inp=="multi":
            read_saved_rules(neighbours,states,symm)
            file1 = str(input("Enter rule 1: "))
            g.rule_load(file1)
            rule1 = g.rule

            file2 = str(input("Enter rule 2: "))
            g.rule_load(file2)
            rule2 = g.rule

            file3 = str(input("Enter rule 3: "))
            g.rule_load(file3)
            rule3 = g.rule

            file4 = str(input("Enter rule 4: "))
            g.rule_load(file4)
            rule4 = g.rule

            g.run_parallel(rule1,rule2,rule3,rule4)
            ani_display()
        if inp=="c":
            g.run(False)
            ani_display()
        if inp=="ms":
            mu = float(input("Enter mu: "))
            sig = float(input("Enter sigma: "))
            g.rule_gen(mu,sig)
            g.run()
            ani_display()

        if inp=="t":
            #tranpose output
            #g.rule_gen(mu,sig)
            g.run()
            ani_display(mode=4)
        
        if inp=="sm":
            #Apply image smoothing to output
            g.run()
            ani_display(mode=1)
        if inp=="g":
            #Change rule generation mode
            g.rule_mode = 1-g.rule_mode

        """
        #if inp=="tsmooth":
            #Apply image smoothing and tranpose output
        #    g.run()
        #    ani_display(mode=2)

        #if inp=="diff":
        #    #double derivative
        #    g.run()
        #    ani_display(mode=5)
        if inp=="rt":
            #Rerun same rule, transposed
            g.run()
            ani_display(mode=4)
        #if inp=="lyap":
        #    g.lyap()
        #    ani_display()
        """
        if inp=="s": 
            #save current rule        
            filename = str(input("Enter rule name: "))
            g.rule_save(filename)
        if inp=="l":
            #load from saved rules
            try:
                read_saved_rules(neighbours,states,symm)
                filename = str(input("Enter rule name: "))
                g.rule_load(filename)
                
            except Exception as e:
                print("No saved rules yet")
            g.run()
            ani_display()
        if inp=="p":
            #permute current rule and rerun           
            g.rule_perm()
            g.run()
            ani_display()
        """
        if inp=="d":
            #Change initial state density  
            d = float(input("Enter initial density: "))
            g = Grid2D(size,d,states,neighbours,iterations)
        """
        if (inp=="+" or inp=="*" or inp=="-" or inp=="a" or inp=="z" or inp=="b"):
            #Combine 2 rules in some way
            read_saved_rules(neighbours,states,symm)
            rule1 = str(input("Enter first rule: "))
            rule2 = str(input("Enter second rule: "))
            g.rule_comb(rule1,rule2,inp)
            g.run()
            ani_display()
        if inp=="i":
            #Invert rule
            g.rule_inv()
            g.run()
            ani_display()
        if inp=="x":
            g.rule_swap()
            g.run()
            ani_display()
        if inp=="v":
            g.rule_roll(int(input("Enter roll amount: ")))
            g.run()
            ani_display()
        if inp=="w":
            #Smooth rule
            am = int(input("Enter smoothing amount: "))
            g.rule_smooth(am)
            g.run()
            ani_display()
        #if inp=="f":
            #fold rule
        #    am = int(input("Enter folding amount: "))
        #    g.rule_fold(am)
        #    g.run()
        #    ani_display()
        #if inp=="ch":
            #change initial conditon type
        #    g.change_start_type()
        #if inp=="e":
            #return rule entropy
        #    print(g.rule_entropy())
            #en = g.entropy()
            #plt.plot(en)
            #plt.show()
        if inp=="2d":
            #Project output onto 2d surface
            axis = int(input("Enter axis: "))
            snapshot(axis)

        if inp=="stat":
            
            stationary_summary()
            #axis = int(input("Enter axis: "))
            #if axis==0:
            #    plt.matshow(g.im_out()[iterations//2],cmap=colour)
            #if axis==1:
            #    plt.matshow(g.im_out()[:,size//2],cmap=colour)
            #if axis==2:
            #    plt.matshow(g.im_out()[:,:,size//2],cmap=colour)
            #plt.show()

        if inp=="m":
            am = float(input("Enter mutation amount (0-1): "))
            g.run_mutate(am)
            ani_display()
            best = int(input("Select best rule (1-4): "))
            g.choose_offspring(best)
            g.run()
            ani_display()
        """
        if inp=="vid":
            name = input("Enter filename: ")
            save_video(g.im_out(),name)
        if inp=="sample":
            n=int(input("Enter sample rate: "))
            ani_display(mode=6,n=n)
        if inp=="pred":
            print(g.predict_interesting()[0])
        """
        #if inp!="q":
        #    name = "rule_"+str(counter)
        #    t = str(input("Enter rule type(i,d,n,s): "))
        #    g.store(t,name)


def ani_display(mode=0,n=1):
    if mode==0:
        data = g.im_out()
    elif mode==1:
        data = smooth(g.im_out(),1)

    elif mode==2:
        data = np.moveaxis(smooth(g.im_out(),3),0,1)

    elif mode==4:
        data = np.moveaxis(g.im_out(),0,1)
    
    elif mode==5:
        data = np.abs(np.diff(np.diff(g.im_out(),axis=0),axis=0))
    elif mode==6:
        data = g.im_out()[::n]
    elif mode==7:
        data = g.fft_data


    def update(i):
        screendata = data[i]
        matrix.set_array(screendata)
    screendata = data[0]
    fig, ax = plt.subplots()            
    matrix = ax.matshow(screendata,cmap=colour)
    plt.colorbar(matrix)
    ani = animation.FuncAnimation(fig,update,frames=iterations,interval=100)
    plt.show()


"""

def save_video(data,name):
    its = data.shape[0]
    am=3
    frameSize = (am*data.shape[1],am*data.shape[2])
    out = cv2.VideoWriter(str(name)+'.avi',cv2.VideoWriter_fourcc(*'FMP4'), 8, frameSize)
    images = num_to_rgb(data,am)
    for i in range(its):
        img = np.uint8(images[i])
        out.write(img)

    out.release()

def num_to_rgb(data,am=2):
    image = np.zeros((data.shape[0],am*data.shape[1],am*data.shape[2],3)).astype(int)

    r = np.mod(ndimage.zoom(data,(1,am,am),order=0).astype(int)*117,255).astype(int)
    g = np.mod(ndimage.zoom(data,(1,am,am),order=0).astype(int)*183,255).astype(int)
    b = np.mod(ndimage.zoom(data,(1,am,am),order=0).astype(int)*211,255).astype(int)
    image[:,:,:,0] = r
    image[:,:,:,1] = g
    image[:,:,:,2] = b
    return image


"""

def stationary_summary():
    data = g.im_out()
    fig,ax = plt.subplots(1,3)
    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    ax[0].imshow(data[0],origin="lower",cmap=colour)
    ax[0].set_xlabel("1 step")
    ax[1].imshow(data[10],origin="lower",cmap=colour)
    ax[1].set_xlabel("10 steps")
    ax[2].imshow(data[100],origin="lower",cmap=colour)
    ax[2].set_xlabel("100 steps")
    
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
    #k = np.ones((am,am,am))
    #k = k/np.sum(k)
    #return ndimage.convolve(d,k)
    return ndimage.gaussian_filter(d,am)



def read_saved_rules(n,s,sym):
    f = open("text_resources/name_header.txt","r")
    print(f.read())
    f.close()
    f = open('2D_rules/s'+str(s)+'/namelist.txt','r')
    print(f.read())
    f.close()
    f = open("text_resources/name_foot.txt","r")
    print(f.read())
    f.close()

if __name__ == '__main__':
    main_loop()



