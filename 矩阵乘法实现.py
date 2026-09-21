import numpy as np
A = np.array([[1,-2,1],
              [2,-2,2]])
AT = A.T
W = np.array([[2,4,3,5],
              [1,2,3,4]])
B = np.array([1,2,3,4])
Z = AT@W 
print(Z)
Z = Z+B
print(Z)