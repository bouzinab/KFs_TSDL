# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:32:37 2020
@author: matth
"""
#import numpy as np
import autograd.numpy as np
from matrix_operations import norm_matrix, inner_matrix

#%%

""" In this section we define various kernels. Warning, not all of them work 
at the moment, the most reliable one is the RBF kernel. Note that currently the 
laplacian kernel does not work"""
        

# Define the RBF Kernel. Takes an array of parameters, returns a value
def kernel_RBF(matrix_1, matrix_2, parameters):
    matrix = norm_matrix(matrix_1, matrix_2)
    sigma = parameters[0]
    K =  parameters[1]**2 * np.exp(-np.square(matrix)/ (2* sigma**2))
    
    return K

def kernel_RBF_RQ(matrix_1, matrix_2, parameters):
    matrix = norm_matrix(matrix_1, matrix_2)
    
    sigma = parameters[1]
    K =  (parameters[0]**2) *np.exp(-np.square(matrix)/ (2* sigma**2))

    alpha = parameters[2]
    beta = parameters[3]
    gamma = parameters[4]
    K= K + (parameters[5]**2)*(beta**2 + gamma*matrix)**(-(alpha))

    return K

# do not use right now
def kernel_laplacian(matrix_1, matrix_2, parameters):
    gamma = parameters[0]
    matrix = norm_matrix(matrix_1, matrix_2)
    K =  np.exp(-matrix * gamma)
    return K

def kernel_sigmoid(matrix_1, matrix_2, parameters):
    alpha = parameters[0]
    beta = parameters[1]
    matrix = inner_matrix(matrix_1, matrix_2)
    K = np.tanh(alpha *matrix + beta)
    return K

def kernel_rational_quadratic(matrix_1, matrix_2, parameters):
    alpha = parameters[0]
    beta = parameters[1]
    epsilon = 0.0001
    matrix = norm_matrix(matrix_1, matrix_2)
    return (beta**2 + matrix)**(-(alpha+ epsilon))

def kernel_inverse_power_alpha(matrix_1, matrix_2, parameters):
    alpha = parameters[0]
    beta = 1.0
    epsilon = 0.0001
    matrix = norm_matrix(matrix_1, matrix_2)
    return (beta**2 + matrix)**(-(alpha+ epsilon))

def kernel_inverse_multiquad(matrix_1, matrix_2, parameters):
    beta = parameters[0]
    gamma = parameters[1]
    matrix = norm_matrix(matrix_1, matrix_2)
    return (beta**2 + gamma*matrix)**(-1/2)

def kernel_cauchy(matrix_1, matrix_2, parameters):
    sigma = parameters[0]
    matrix = norm_matrix(matrix_1, matrix_2)
    return 1/(1 + matrix/sigma**2)

def kernel_quad(matrix_1, matrix_2, parameters):
    c = parameters[0]
    matrix = inner_matrix(matrix_1, matrix_2)
    K = (matrix+c) ** 2
    return K 

def kernel_poly(matrix_1, matrix_2, parameters):
    a = parameters[0]
    b = parameters[1]
    d = parameters[2]
    matrix = inner_matrix(matrix_1, matrix_2)
    K = (a * matrix + b) ** d
    return K 


def kernel_gaussian_linear(matrix_1, matrix_2, parameters):
    K = 0
    matrix = norm_matrix(matrix_1, matrix_2)
    for i in range(parameters.shape[1]):
        # print("beta", parameters[1, i])
        # print("sigma", parameters[0, i])
        K = K + parameters[1, i]**2*np.exp(-matrix / (2* parameters[0, i]**2))
    return K


def kernel_arma(matrix_1, matrix_2, parameters):
    alpha = parameters[0]
    matrix = inner_matrix(matrix_1, matrix_2)
    K = (alpha *matrix + 1)**2
    return K

def kernel_matern_32(matrix_1, matrix_2, param):
    
    l = param[0]    
    sigma = param[1]
    r = norm_matrix(matrix_1, matrix_2)
    exp = np.exp(-np.sqrt(3)*r/l)
    return sigma**2*(1 + np.sqrt(3)*r/l)*exp

def kernel_matern_52(matrix_1, matrix_2, param):
    
    l = param[0]
    sigma = param[1]
    r = norm_matrix(matrix_1, matrix_2)
    exp = np.exp(-np.sqrt(5)*r/l)
    return sigma**2*(1 + np.sqrt(5)*r/l + 5/3*(r/l)**2)*exp

def matern_52(matrix_1,matrix_2,param):
    sigma=param[0]
    l=param[1]
    sigma_1=cov_GP_centered(matrix_1)
    sigma_2=cov_GP_centered(matrix_2)
    r=np.zeros([len(matrix_1),len(matrix_1)]) 
    k=np.zeros([len(matrix_1),len(matrix_1)]) 
    for i in range(len(matrix_1)):
        for j in range(len(matrix_1)):
            diff=matrix_1[i,:]-matrix_2[j,:]
            a=(sigma_1[i]+sigma_2[j])/2+ 0.5 * np.identity(matrix_1.shape[1])
            inv=np.linalg.inv(a)
            r[i,j]=np.sqrt((diff.dot(inv)).dot(diff))
            k[i,j]=(np.linalg.det(sigma_1[i]))**(1/4) * (np.linalg.det(sigma_2[j]))**(1/4) / np.sqrt(np.linalg.det(a))
    #return r
    exp = np.exp(-np.sqrt(5)*r/l)
    mat=(1 + np.sqrt(5)*r/l + 5/3*(r/l)**2)*exp
    print(sigma**2 * k * mat)
    return sigma**2 * k * mat 

def matern_32(matrix_1,matrix_2,param):
    sigma=param[0]
    l=param[1]
    sigma_1=cov_GP_centered(matrix_1)
    sigma_2=cov_GP_centered(matrix_2)
    r=np.zeros([len(matrix_1),len(matrix_1)]) 
    k=np.zeros([len(matrix_1),len(matrix_1)]) 
    for i in range(len(matrix_1)):
        for j in range(len(matrix_1)):
            diff=matrix_1[i,:]-matrix_2[j,:]
            a=(sigma_1[i]+sigma_2[j])/2+ 0.5 * np.identity(matrix_1.shape[1])
            inv=np.linalg.inv(a)
            r[i,j]=np.sqrt((diff.dot(inv)).dot(diff))
            k[i,j]=(np.linalg.det(sigma_1[i]))**(1/4) * (np.linalg.det(sigma_2[j]))**(1/4) / np.sqrt(np.linalg.det(a))
    #return r
    exp = np.exp(-np.sqrt(3)*r/l)
    mat=(1 + np.sqrt(3)*r/l)*exp
    print(sigma**2 * k * mat)
    return sigma**2 * k * mat 


def kernel_anl(matrix_1, matrix_2, parameters):
    matrix = norm_matrix(matrix_1, matrix_2)
    sigma = parameters[0]
    K =  np.exp(-matrix/ (2* sigma**2))
    K=K*(parameters[1])**2
    
    c = (parameters[2])**2
    matrix = inner_matrix(matrix_1, matrix_2)
    K = K+ (parameters[3])**2 *(matrix+c) ** 2
    
    beta = parameters[4]
    gamma = (parameters[5])**2
    matrix = norm_matrix(matrix_1, matrix_2)
    K=K+ (parameters[6])**2 *(beta**2 + gamma*matrix)**(-1/2)
    
    alpha = parameters[7]
    beta = parameters[8]
    matrix = norm_matrix(matrix_1, matrix_2)
    K=K+ (parameters[9])**2 *(beta**2 + matrix)**(-alpha)
    
    return K

def kernel_anl2(matrix_1, matrix_2, parameters):
    i=0
    
    matrix = norm_matrix(matrix_1, matrix_2)
    sigma = parameters[i+0]
    K =  np.exp(-matrix/ (2* sigma**2))
    K=K*(parameters[i+1])**2
    i=i+2
    
    
    c = (parameters[i])**2
    imatrix = inner_matrix(matrix_1, matrix_2)
    K = K+ (parameters[i+1])**2 *(imatrix+c) ** 2
    i=i+2
    
    beta = parameters[i]
    gamma = (parameters[i+1])**2
    K=K+ (parameters[i+2])**2 *(beta**2 + gamma*matrix)**(-1/2)
    i=i+3
    
    alpha = parameters[i]
    beta = parameters[i+1]
    K=K+ (parameters[i+2])**2 *(beta**2 + matrix)**(-alpha)
    i=i+3
    
    return K

"""
def kernel_anl3(matrix_1, matrix_2, parameters):
    i=0
    
    #matrix = np.square(norm_matrix(matrix_1, matrix_2))
    matrix = norm_matrix(matrix_1, matrix_2)
    sigma = parameters[i+0]
    K =  np.exp(-matrix/ (2* sigma**2))
    K=K*(parameters[i+1])**2
    i=i+2 #2
    
    c = (parameters[i])**2
    imatrix = inner_matrix(matrix_1, matrix_2)
    K = K+ (parameters[i+1])**2 *(imatrix+c) ** 2
    i=i+2 #4
    
    beta = parameters[i]
    gamma = (parameters[i+1])**2
    K=K+ (parameters[i+2])**2 *(beta**2 + gamma*matrix)**(-1/2)
    i=i+3 #7
    
    alpha = parameters[i]
    beta = parameters[i+1]
    K=K+ (parameters[i+2])**2 *(beta**2 + matrix)**(-alpha)
    i=i+3 #10
    
    sigma = parameters[i]
    K=K+ (parameters[i+1])**2 * 1/(1 + matrix/sigma**2)
    i=i+2 #12
    
    sigma_0 = parameters[i]
    K =  K+ (parameters[i+1])**2 *np.maximum(0, 1-matrix/(sigma_0))
    i=i+2 #14
    
    p = parameters[i]
    l = parameters[i+1]
    sigma = parameters[i+2]
    K =K+ (parameters[i+3])**2 * np.exp(-np.sin(matrix*np.pi/p)**2/l**2)*np.exp(-matrix/sigma**2)
    i=i+4 #18
    
    p = parameters[i]
    l = parameters[i+1]
    K = K+ (parameters[i+2])**2 *np.exp(-np.sin(matrix*np.pi/p)/l**2)
    i=i+3 #21
    

    return K
"""
"""
def kernel_anl3(matrix_1, matrix_2, parameters):
    i=0
    
    matrix = np.square(norm_matrix(matrix_1, matrix_2))
    sigma = parameters[i+0]
    K =  np.exp(-matrix/ (2* sigma**2))
    K=K*(parameters[i+1])**2
    i=i+2 #2
    
    c = (parameters[i])**2
    imatrix = inner_matrix(matrix_1, matrix_2)
    K = K+ (parameters[i+1])**2 *(imatrix+c) ** 2
    i=i+2 #4
    
    beta = parameters[i]
    gamma = (parameters[i+1])**2
    K=K+ (parameters[i+2])**2 *(beta**2 + gamma*matrix)**(-1/2)
    i=i+3 #7
    
    alpha = parameters[i]
    beta = parameters[i+1]
    K=K+ (parameters[i+2])**2 *(beta**2 + matrix)**(-alpha)
    i=i+3 #10
    
    sigma = parameters[i]
    K=K+ (parameters[i+1])**2 * 1/(1 + matrix/sigma**2)
    i=i+2 #12
    
    alpha_0 = parameters[i]
    sigma_0 = parameters[i+1]
    alpha_1 = parameters[i+2]
    sigma_1 = parameters[i+3]
    K =  K+ (parameters[i+4])**2 *alpha_0*np.maximum(0, 1-matrix/(sigma_0))+ alpha_1 * np.exp(-matrix/ (2* sigma_1**2))
    i=i+5 #17
    
    p = parameters[i]
    l = parameters[i+1]
    sigma = parameters[i+2]
    K =K+ (parameters[i+3])**2 * np.exp(-np.sin(matrix*np.pi/p)**2/l**2)*np.exp(-matrix/sigma**2)
    i=i+4 #21
    
    p = parameters[i]
    l = parameters[i+1]
    K = K+ (parameters[i+2])**2 *np.exp(-np.sin(matrix*np.pi/p)/l**2)
    i=i+3 #24
    

    return K
"""
def kernel_anl3(matrix_1, matrix_2, parameters):
    i=0
    
    matrix = np.square(norm_matrix(matrix_1, matrix_2))
    #matrix = norm_matrix(matrix_1, matrix_2)
    sigma = parameters[i+0]
    K =  np.exp(-matrix/ (2* sigma**2))
    K=K*(parameters[i+1])**2
    i=i+2 #2
    
    c = (parameters[i])**2
    imatrix = inner_matrix(matrix_1, matrix_2)
    K = K+ (parameters[i+1])**2 *(imatrix*c+1) ** 2
    i=i+2 #4
    
    gamma = (parameters[i])**2
    K=K+ (parameters[i+1])**2 *(1 + gamma*matrix)**(-1/2)
    i=i+2 #6
    
    alpha = parameters[i]
    beta = parameters[i+1]
    K=K+ (parameters[i+2])**2 *(1+matrix*beta**2  )**(alpha)
    i=i+3 #9
    
    gamma2 = (parameters[i])**2
    K=K+ (parameters[i+1])**2 *(1 + gamma2*matrix)**(-1)
    i=i+2 #11

    sigma_0 = parameters[i]
    K =  K+ (parameters[i+1])**2 *np.maximum(0, 1-matrix*(sigma_0))
    i=i+2 #13
    
    
    p = parameters[i]
    l = parameters[i+1]
    sigma = parameters[i+2]
    K =K+ (parameters[i+3])**2 * np.exp(-np.sin(norm_matrix(matrix_1, matrix_2)*np.pi/p)**2/l**2)*np.exp(-matrix/sigma**2)
    i=i+4 #17
    
    K= K + kernel_matern_32(matrix_1,matrix_2,[parameters[i],parameters[i+1]])
    
    i=i+2 # 19

    K= K + kernel_matern_52(matrix_1,matrix_2,[parameters[i],parameters[i+1]])
    
    i=i+2 # 21
    
    return K

def kernel_anl4(matrix_1, matrix_2, parameters):
    i=0
    matrix = norm_matrix(matrix_1, matrix_2)
    sigma = parameters[i+0]
    K =  np.exp(-matrix/ (2* sigma**2))
    K=K*(parameters[i+1])**2
    i=i+2 #2
    
    
    c = (parameters[i])**2
    imatrix = inner_matrix(matrix_1, matrix_2)
    K = K+ (parameters[i+1])**2 *(imatrix+c) ** 2
    i=i+2 #4
    
    beta = parameters[i]
    gamma = (parameters[i+1])**2
    K=K+ (parameters[i+2])**2 *(beta**2 + gamma*matrix)**(-1/2)
    i=i+3 #7
    
    alpha = parameters[i]
    beta = parameters[i+1]
    K=K+ (parameters[i+2])**2 *(beta**2 + matrix)**(-alpha)
    i=i+3 #10
    
    sigma = parameters[i]
    K=K+ (parameters[i+1])**2 * 1/(1 + matrix/sigma**2)
    i=i+2 #12
    
    alpha_0 = parameters[i]
    sigma_0 = parameters[i+1]
    alpha_1 = parameters[i+2]
    sigma_1 = parameters[i+3]
    K =  K+ (parameters[i+4])**2 *alpha_0*np.maximum(0, 1-matrix/(sigma_0))+ alpha_1 * np.exp(-matrix/ (2* sigma_1**2))
    i=i+5 #17
     
    p = parameters[i]
    l = parameters[i+1]
    sigma = parameters[i+2]
    K =K+ (parameters[i+3])**2 * np.exp(-np.sin(matrix*np.pi/p)**2/l**2)*np.exp(-matrix/sigma**2)
    i=i+4 #21

    p = parameters[i]
    l = parameters[i+1]
    K = K+ (parameters[i+2])**2 *np.exp(-np.sin(matrix*np.pi/p)/l**2)
    i=i+3 #24
    
    K= K + kernel_matern_32(matrix_1,matrix_2,[parameters[i],parameters[i+1]])
    
    i=i+2 # 26

    K= K + kernel_matern_52(matrix_1,matrix_2,[parameters[i],parameters[i+1]])
    
    i=i+2 # 28
    return K

"""A dictionnary containing the different kernels. If you wish to build a custom 
 kernel, add the function to the dictionnary.
"""
kernels_dic = {"RBF" : kernel_RBF,"poly": kernel_poly, "Laplacian": kernel_laplacian, 
               "sigmoid": kernel_sigmoid, "rational quadratic": kernel_rational_quadratic,
               "inverse_multiquad": kernel_inverse_multiquad, "quadratic" : kernel_quad,
               "poly": kernel_poly, "inverse_power_alpha": kernel_inverse_power_alpha,
               "gaussian multi": kernel_gaussian_linear,"arma":kernel_arma ,"anl": kernel_anl, "anl2": kernel_anl2,
               "anl3": kernel_anl3,"anl4": kernel_anl4,"RBF_RQ":kernel_RBF_RQ}

# link to some kernels
#https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=f15b9f93c57cac59f0c577284dc23577f236e00d&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6a6b6669747a73696d6f6e732f4950794e6f7465626f6f6b5f4d616368696e654c6561726e696e672f663135623966393363353763616335396630633537373238346463323335373766323336653030642f4a757374253230416e6f746865722532304b65726e656c253230436f6f6b626f6f6b2e2e2e2e6970796e62&logged_in=false&nwo=jkfitzsimons%2FIPyNotebook_MachineLearning&path=Just+Another+Kernel+Cookbook....ipynb&platform=android&repository_id=25307773&repository_type=Repository&version=99
if __name__ == '__main__':
    print('This is the kernel file')

