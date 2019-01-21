from org import Grid2D
from rgbmatrix import RGBMatrix, RGBMatrixOptions


def main():
	states = int(input("Enter number of states: "))
    neighbours = int(input("Enter size of cell neighbourhood: "))
    size = int(input("Enter size of grid: "))
    global iterations
    iterations = int(input("Enter number of iterations: "))
    global g
    #global screendata
    global matrix
    matrix = RGBMatrix()
    
    inp = "a"
    g = Grid2D(size,0.5,states,neighbours,iterations)
    
    


    while inp!="q":
    	#Text interface for running automata
        counter = counter+1
        inp = str(raw_input(":")) 
        if inp=="h":
            f = open("text_resources/help.txt","r")
            print(f.read())
            f.close()
        if inp=="r":
            g.run()
            ani_display()            
        if inp=="":            
            g.rule_gen()
            g.run()
            ani_display()          
        if inp=="s":         
            filename = str(raw_input("Enter rule name: "))
            g.rule_save(filename)
        if inp=="l":
            read_saved_rules(neighbours,states)
            filename = str(raw_input("Enter rule name: "))
            g.rule_load(filename)
            g.run()
            ani_display()
        if inp=="n":
            read_saved_rules(neighbours,states)
        if inp=="p":           
            g.rule_perm()
            g.run()
            ani_display()
        if inp=="d":           
            d = float(raw_input("Enter initial density: "))
            g = Grid2D(size,d,states,neighbours,iterations)
        if (inp=="+" or inp=="*" or inp=="-" or inp=="m" or inp=="z" or inp=="c"):
            read_saved_rules(neighbours,states)
            rule1 = str(raw_input("Enter first rule: "))
            rule2 = str(raw_input("Enter second rule: "))
            g.rule_comb(rule1,rule2,inp)
            g.run()
            ani_display()
        if inp=="i":
            g.rule_inv()
            g.run()
            ani_display()
        if inp=="w":
            am = int(raw_input("Enter smoothing amount: "))
            g.rule_smooth(am)
            g.run()
            ani_display()
        if inp=="f":
            am = int(raw_input("Enter fold amount: "))
            g.rule_fold(am)
            g.run()
            ani_display()

def ani_display():
	out_data = g.im_out()

	for i in range(iterations):
        screendata = out_data[i]
        image = np.zeros((64,64))
        for x in range(64):
            for y in range(64):
                