from org import Grid2D
import numpy as np
from scipy import ndimage
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import time
import sys
from sklearn import neighbors
import tensorflow as tf 
from tensorflow import keras








def main():
    global states
    global symm
    states = 2
    symm = 2
    neighbours = 1
    global size
    size = 128
    global iterations
    iterations = 256
    global g
    global screendata
    global matrix
    global data
    global colour
    colour = "gist_earth"
    #mu=0.5
    #sig=0.2
    g = Grid2D(size,0.5,states,neighbours,iterations,symm)

    #Load and slice up data to training and validation sets
    ml_data = np.random.permutation(np.load("2state_ml_data.npy"))#[:,:9]
    
    N = ml_data.shape[0]
    trn = ml_data[:(80*N//100)]
    val = ml_data[(80*N//100):]

    y_trn = trn[:,1]
    X_trn = trn[:,2:]
    y_val = val[:,1]
    X_val = val[:,2:]


    #For 2 state system, variances of fft peak locations is always 0 - not enough degrees of freedom
    
    X_trn[:,15]+=np.random.normal(0,1,X_trn.shape[0])
    X_trn[:,13]+=np.random.normal(0,1,X_trn.shape[0])
    X_trn[:,11]+=np.random.normal(0,1,X_trn.shape[0])
    X_trn[:,9]+=np.random.normal(0,1,X_trn.shape[0])

    X_val[:,15]+=np.random.normal(0,1,X_val.shape[0])
    X_val[:,13]+=np.random.normal(0,1,X_val.shape[0])
    X_val[:,11]+=np.random.normal(0,1,X_val.shape[0])
    X_val[:,9]+=np.random.normal(0,1,X_val.shape[0])
    
    #Count how many of each class (0 or 1) are in each dataset - for plotting later
    trn_p = np.count_nonzero(y_trn==1)
    trn_n = np.count_nonzero(y_trn==0)
    val_p = np.count_nonzero(y_val==1)
    val_n = np.count_nonzero(y_val==0)


    #Data normalisation layer - NN wants data with 0 mean, unit variance. 
    #Do as part of keras model so it can be saved together 
    preprocess_layer = keras.layers.experimental.preprocessing.Normalization()
    preprocess_layer.adapt(X_trn)

    #Keras neural network model
    def build_model():
        inputs = keras.Input(X_trn.shape[1])
        x = preprocess_layer(inputs)
        #x = keras.layers.Dense(units=64,activation="swish",kernel_regularizer=keras.regularizers.L2(0.01))(x)
        x = keras.layers.Dense(units=32,activation="swish",kernel_regularizer=keras.regularizers.L2(0.01))(x)
        x = keras.layers.Dense(units=32,activation="swish",kernel_regularizer=keras.regularizers.L2(0.01))(x)
        x = keras.layers.Dense(units=16,activation="swish",kernel_regularizer=keras.regularizers.L2(0.01))(x)
        x = keras.layers.Dense(units=8,activation="swish",kernel_regularizer=keras.regularizers.L2(0.01))(x)
        #x = keras.layers.Dense(units=4,activation="relu",kernel_regularizer=keras.regularizers.L2(0.01))(x)
        outputs = keras.layers.Dense(units=1,activation="sigmoid")(x)
        return keras.Model(inputs=inputs,outputs=outputs)

    model = build_model()
    model.summary()
    batch_size = X_trn.shape[0]
    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.losses.BinaryCrossentropy(name='binary_crossentropy'),
                           keras.metrics.TruePositives(),
                           keras.metrics.FalsePositives(),
                           keras.metrics.TrueNegatives(),
                           keras.metrics.FalseNegatives()],)

    history = model.fit(X_trn,
                        y_trn,
                        batch_size=batch_size,
                        epochs=5000,
                        validation_data=(X_val,y_val))
                        #callbacks=[keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=300)])

    def plot_metrics(history):
        trn_loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        plt.plot(trn_loss,label="Training")
        plt.plot(val_loss,label="Validation")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Binary Crossentropy")
        plt.show()


        trn_tp = history.history["true_positives"]
        trn_fp = history.history["false_positives"]
        trn_tn = history.history["true_negatives"]
        trn_fn = history.history["false_negatives"]


        val_tp = history.history["val_true_positives"]
        val_fp = history.history["val_false_positives"]
        val_tn = history.history["val_true_negatives"]
        val_fn = history.history["val_false_negatives"]

        #val_acc = history.history["val_binary_accuracy"]
        
        plt.title("Training accuracy")
        plt.plot(np.array(trn_tp)/float(trn_p),label="True positives",color="blue")
        plt.plot(np.array(trn_fp)/float(trn_p),label="False positives",color="cyan")
        plt.plot(np.array(trn_tn)/float(trn_n),label="True negatives",color="red")
        plt.plot(np.array(trn_fn)/float(trn_n),label="False negatives",color="orange")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show()


        plt.title("Validation accuracy")
        plt.plot(np.array(val_tp)/float(val_p),label="True positives",color="blue")
        plt.plot(np.array(val_fp)/float(val_p),label="False positives",color="cyan")
        plt.plot(np.array(val_tn)/float(val_n),label="True negatives",color="red")
        plt.plot(np.array(val_fn)/float(val_n),label="False negatives",color="orange")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show()
    


    plot_metrics(history)
    #results = model.evaluate(X_val,y_val,batch_size=X_val.shape[1]//10)
    #print("NN validation error: "+str(results))
    

    #print("KNN validation error: "+str(rmse_val))
    #model = keras.Sequential
    #g.rule = np.array([0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0]) 
    #for i in range(10):
    #	g.rule_gen()
    #	g.run()
    #	data = g.im_out()
    #	ani_display()
    #	xx = g.get_metrics(4)
    #	print(clf.predict(xx))
    #print(ff.shape)
    #print(y_val.shape)

    #print(trn.shape)
    #print(val.shape)
    #print(ml_data[0])














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