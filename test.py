import numpy as np
from numpy import linalg as LA
import time
import matplotlib.pyplot as plt
import math
from scipy import stats



print(LA.eig(np.array([[1,2],[-3,1]]))[1].real)

"""
a = np.abs(np.arange(1,100)-50)
xy = np.mean((np.mgrid[-100:100:1, -100:100:1]**2),axis=0)<10
plt.matshow(xy)
plt.show()
print(xy)
"""





