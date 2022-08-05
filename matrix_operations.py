# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:36:48 2020
@author: matth
"""

import autograd.numpy as np
#%% Kernel operations

# Returns the norm of the pairwise difference
"""
def norm_matrix_1(matrix_1, matrix_2):
    norm_square_1 = np.sum(np.square(matrix_1), axis = 1)
    norm_square_1 = np.reshape(norm_square_1, (-1,1))
    
    norm_square_2 = np.sum(np.square(matrix_2), axis = 1)
    norm_square_2 = np.reshape(norm_square_2, (-1,1))
    
    d1=matrix_1.shape
    d2=matrix_2.shape
#    print(d1)
#    print(d2)
    if d1[1]!=d2[1]:
        matrix_1=np.transpose(matrix_1)
    
    inner_matrix = np.matmul(matrix_1, np.transpose(matrix_2))
    
    norm_diff = -2 * inner_matrix + norm_square_1 + np.transpose(norm_square_2)
#    print(norm_diff.shape)
    
    return norm_diff
"""
def norm_matrix(x,y):
   
    diff = (x[:, None, :] - y[None, :, :])
    norm  = np.linalg.norm(diff, axis = -1)
   
    return norm

# Returns the pairwise inner product
def inner_matrix(matrix_1, matrix_2):
    d1=matrix_1.shape
    d2=matrix_2.shape
    if d1[1]!=d2[1]:
        matrix_1=np.transpose(matrix_1)
    return np.matmul(matrix_1, np.transpose(matrix_2))


"""
def cov_GP_centered(matrix1,ind):
    r=len(matrix1)
    c=len(matrix1[0])
    for j in range(c):
        a=(sum(matrix1[:,j])-r*matrix1[ind,j])/r
        matrix1[:,j]=matrix1[:,j]-(np.zeros(r)+a)
    print("cent",matrix1)    
    return np.cov(np.transpose(matrix1))

def cov_GP_centered(matrix1):
    r=len(matrix1)
    c=len(matrix1[0])
    matrix_c=np.zeros([r,c])
    res=[]
    for i in range(r):
        for j in range(c):
            a=(sum(matrix1[:,j])-r*matrix1[i,j])/r
            matrix_c[:,j]=(matrix1[:,j]-(np.zeros(r)+a))
        print("matr c",matrix_c)
        res.append(np.cov(np.transpose(matrix_c)))#res[i,1]=np.cov(np.transpose(matrix_c))
    
    return res    

"""
if __name__ == '__main__':
    print('This is the matrix operations file')