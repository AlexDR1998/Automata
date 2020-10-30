import numpy as np
import math

from sympy.utilities.iterables import multiset_permutations

def multi_subset(x,y):
    """
    how many possible arangements of y items, of x choices, where duplicates are alowed.
    (1,2,2)=(2,1,2)!=(1,1,2)
    """
    return math.factorial(x+y-1)/(math.factorial(y)*math.factorial(x-1))

def main():
	
	
	
	a = [1,1,0,0]
	perms = multiset_permutations(a)
	#perms = np.fromiter(c,int,len(c))
	print(perms)




	"""

	a = np.array([[0,0,0,0],
				  [4,4,1,1]])
	
	b = np.array([[0,0,0,1],
				  [2,2,3,3]])
	
	c = np.zeros((5,5))
	diff = np.stack((a,b))
	#print(diff.shape)
	#print(np.count_nonzero(np.logical_and(a==1,b==1)))
	#print(np.einsum('ijk->jk',diff))
	#np.take(c,a)=1
	for i in range(c.shape[0]):
		for j in range(c.shape[1]):
			c[i,j] = np.count_nonzero(np.logical_and(a==i,b==j))
	#numbers 0 to n
	ii = np.arange(c.shape[0])
	jj = np.arange(c.shape[1])
	
	a_tile = np.tile(a,(c.shape[0],1,1))
	b_tile = np.tile(b,(c.shape[1],1,1))

	aa = np.equal(a_tile,ii[:,None,None]).astype(int)
	bb = np.equal(b_tile,jj[:,None,None]).astype(int)
	print(aa.shape)
	print(np.einsum('ixy,jxy->ij',aa,bb))
	#print(np.equal(ii,np.einsum('ijk->kji',np.tile(a,(c.shape[0],1,1)))))
	
	#print(np.count_nonzero(np.logical_and(a==ii,b==jj)))
	print(c)
	#for i in range(a.shape[0]):
	#	for j in range(a.shape[1]):
	#		c[a[i,j],b[i,j]]+=1
	#print(c)
	"""
main()