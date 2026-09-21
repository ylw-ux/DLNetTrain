import numpy as np
X = np.array([[200,17]])
W = np.array([[1,-3,7],
              [-2,3,2]])
B = np.array([[1,2,-1]])

def dense1(A_in,W,B):
    Z = np.matmul(A_in,W) + B
    A_out = g1(Z)
    return A_out

def dense2(A_in,W,B):
    Z = A_in@W + B
    A_out = g2(Z)
    return A_out

def g1(Z):
    G = 1/(1+np.exp(-Z))
    return G

def g2(Z):
    G = 1/(1+np.exp(-Z))
    return G

rul1 = dense1(X,W,B)
print("rul1=", rul1)

rul2 = dense2(X,W,B)
print("rul2=", rul2) 