from org import Grid
#from org import Grid2D
#from interface import Interface as i 
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import matplotlib.animation as animation
#import matplotlib.cm as cm
#import time



def main():
    #atype = int(input("Enter dimensions: "))
    states = int(input("Enter number of states: "))
    neighbours = int(input("Enter size of cell neighbourhood: "))
    size = int(input("Enter size of grid: "))
    global iterations
    iterations = int(input("Enter number of iterations: "))
    global g
    
    
    inp = "a"
     
    g = Grid(size,0.5,states,neighbours,iterations)
    
    #g.run()
    #ani_display()


    while inp!="q":
        inp = str(raw_input(":")) 
        if inp=="h":
            f = open("text_resources/help.txt","r")
            print(f.read())
            f.close()
        if inp=="r":
            g.initialise_random()
            g.run()
            plt.matshow(g.im_out(),cmap="nipy_spectral")
            plt.show()
        if inp=="m":
            ani_display()
            #g.initialise_mutate(1)
            #g.run()
            #plt.matshow(g.im_out(),cmap="nipy_spectral")
            #plt.show()
        if inp=="":
             
            g = Grid(size,0.5,states,neighbours,iterations)
            g.initialise_random()
            g.run()
            plt.matshow(g.im_out(),cmap="nipy_spectral")
            plt.show()
            


def ani_display(mode=0):


    def update(i):
        g.initialise_mutate(1)
        g.run()
        matrix.set_array(g.im_out())
    g.initialise_mutate(1)
    g.run()
    fig, ax = plt.subplots()            
    matrix = ax.matshow(g.im_out(),cmap="nipy_spectral")
    #plt.colorbar(matrix)
    ani = animation.FuncAnimation(fig,update,frames=100,interval=100)
    plt.show()

main()
