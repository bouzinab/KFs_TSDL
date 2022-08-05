# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:07:58 2020

@author: matth
"""
from sklearn.metrics import mean_squared_error

import autograd.numpy as np
np.random.seed(1)
from autograd import value_and_grad 
import math
import matplotlib.pyplot as plt

from kernel_functions import kernels_dic

#%%
    
"""We define several useful functions"""
    
# Returns a random sample of the data, as a numpy array
def sample_selection(data, size):
    indices = np.arange(data.shape[0])
    sample_indices = np.sort(np.random.choice(indices, size, replace= False))
    
    return sample_indices

# This function creates a batch and associated sample
def batch_creation(data, batch_size, sample_proportion = 0.5):
    # If False, the whole data set is the mini-batch, otherwise either a 
    # percentage or explicit quantity.
    if batch_size == False:
        data_batch = data
        batch_indices = np.arange(data.shape[0])
    elif 0 < batch_size <= 1:
        batch_size = int(data.shape[0] * batch_size)
        batch_indices = sample_selection(data, batch_size)
        data_batch = data[batch_indices]
    else:
        batch_indices = sample_selection(data, batch_size)
        data_batch = data[batch_indices]
        

    # Sample from the mini-batch
    sample_size = math.ceil(data_batch.shape[0]*sample_proportion)
    sample_indices = sample_selection(data_batch, sample_size)
    
    return sample_indices, batch_indices

def replace_nan(array):
    for i in range(array.shape[0]):
        if math.isnan(array[i]) == True:
            print("Found nan value, replacing by 0")
            array[i] = 0
    return array

def replace_nan_last(array):
    #array=array.reshape(1,len(array))
    if (array[0]=='NA'):
        array[0]=0
    for i in range(1,len(array)):
        if (array[i]=='NA'):
            array[i] = array[i-1]
    return array

# Generate a prediction
def kernel_regression(X_train, X_test, Y_train, param, kernel_keyword = "RBF", regu_lambda = 0.000001):
    kernel = kernels_dic[kernel_keyword]
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    t_matrix = kernel(X_test, X_train, param) 
    prediction = np.matmul(t_matrix, np.matmul(np.linalg.inv(k_matrix), Y_train)) 
    return prediction

# redicttimeseries

def kernel_extrapolate(X_train, X_test, Y_train, param, nsteps=1, kernel_keyword = "RBF", regu_lambda = 0.000001):
    kernel = kernels_dic[kernel_keyword]
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    A=np.matmul(np.linalg.inv(k_matrix), Y_train)
    arr = np.array([])
    
    X_test0=X_test
    isteps=int(nsteps/(X_test.shape[1]))+1
    for i in range(isteps):
        X_test1=X_test0
        t_matrix = kernel(X_test1, X_train, param) 
        prediction = np.matmul(t_matrix, A) 
        X_test0=prediction
        arr = np.append(arr, np.array(prediction[0,:]))
    arr=arr[0:nsteps]
    return arr

def sample_size_linear(iterations, range_tuple):
    
    return np.linspace(range_tuple[0], range_tuple[1], num = iterations)[::-1]

#%% Rho function

# The pi or selection matrix
def pi_matrix(sample_indices, dimension):
    pi = np.zeros(dimension)
    
    for i in range(dimension[0]):
        pi[i][sample_indices[i]] = 1
    
    return pi

######### choose one of the following loss functions:
# 1st: rkhs loss function
# 2nd: new loss function for 1 dimension dynamical systems (ex: bernoulli, logistic map, univariate time series)
# 3th: new loss function for 2 dimension dynamical systems (ex: Hénon map)
# 4th: new loss function for 3 dimension dynamical systems (ex: Lorenz system)

# for the new loss function, set the delay value.


"""
####### rkhs loss function #######################################################################################
def rho(parameters, matrix_data, Y_data, sample_indices,  kernel_keyword= "RBF", regu_lambda = 0.000001):
    kernel = kernels_dic[kernel_keyword]
    
    kernel_matrix = kernel(matrix_data, matrix_data, parameters)
    pi = pi_matrix(sample_indices, (sample_indices.shape[0], matrix_data.shape[0]))   
    
    sample_matrix = np.matmul(pi, np.matmul(kernel_matrix, np.transpose(pi)))
    
    Y_sample = Y_data[sample_indices]
    
    lambda_term = regu_lambda
    #inverse_data = np.linalg.inv(kernel_matrix + lambda_term * np.identity(kernel_matrix.shape[0]))
    #inverse_sample = np.linalg.inv(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]))

    X_data=np.linalg.solve(kernel_matrix + lambda_term * np.identity(kernel_matrix.shape[0]),Y_data)    
    X_sample=np.linalg.solve(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]),Y_sample)
    
    #top = np.tensordot(Y_sample, np.matmul(inverse_sample, Y_sample))
    #bottom = np.tensordot(Y_data, np.matmul(inverse_data, Y_data))
    
    top = np.tensordot(Y_sample, X_sample)
    bottom = np.tensordot(Y_data, X_data)

    return 1 - top/bottom

"""
"""
####### new loss function for 1 dimension dynamical systems #######################################################################################
def rho(parameters, matrix_data, Y_data, sample_indices,  kernel_keyword= "RBF", regu_lambda = 0.000001):
    delay=3
    
    test_data=np.concatenate((matrix_data[0,:],np.concatenate((Y_data[0,:],Y_data[1:,len(Y_data[0])-1]),axis=0)),axis=0)
    test_data = test_data.reshape(1,len(test_data))

    lenXt=len(test_data[0,:])
    num_modes = test_data.shape[0]

    nsteps=lenXt
    X_test=np.zeros((1,delay*num_modes))
    X_test[0,:] = test_data[:,:delay].reshape(1,-1)
    X_test=X_test

    arr = np.zeros((delay,num_modes))
    
    arr[:delay,:] = test_data[:,:delay].T
    
    kernel = kernels_dic[kernel_keyword]
    k_matrix = kernel(matrix_data, matrix_data, parameters)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    #A=np.matmul(np.linalg.inv(k_matrix), Y_data)
    A=np.linalg.solve(k_matrix ,Y_data)
    
    X_test0=X_test
    isteps=int(nsteps/delay)+1
    
    for i in range(1,isteps):
        X_test1=X_test0
        t_matrix = kernel(X_test1, matrix_data, parameters) 
        prediction = np.matmul(t_matrix, A)
  
        # Just using true data
        X_test0 = (test_data[:,i*delay:(i+1)*delay]).reshape(1,-1)
        for mode in range(num_modes):
            imin=i*delay
            imax=np.minimum((i+1)*delay,nsteps)
            delaymin=imax-imin
            #arr[imin:imax,mode]=np.array(prediction[0,(mode*delay):(mode*delay+delaymin)])
            arr=np.append(arr,np.array(prediction[0,(mode*delay):(mode*delay+delaymin)]))

    
    predicted_train2=arr
    predicted_train2 =predicted_train2.reshape(1,len(predicted_train2))

    train_data2=test_data
    
    SMAPE_train_data = Calc_SMAPE(train_data2[0,:],predicted_train2[0,:])
    train_r2_score = R2_SCORE(train_data2[0,:],predicted_train2[0,:])
    MSE_score = MSE(train_data2[0,:],predicted_train2[0,:])
    
    return SMAPE_train_data+(train_r2_score-1)**2
"""

####### new loss function for 2 dimensions dynamical systems #######################################################################################
def rho(parameters, matrix_data, Y_data, sample_indices,  kernel_keyword= "RBF", regu_lambda = 0.000001):
    
    delay=3
    recons1=np.concatenate((matrix_data[0,:delay],np.concatenate((Y_data[0,:delay],Y_data[1:,len(Y_data[0])-delay-1]),axis=0)),axis=0)
    recons2=np.concatenate((matrix_data[0,delay:],np.concatenate((Y_data[0,delay:],Y_data[1:,len(Y_data[0])-1]),axis=0)),axis=0)
    test_data=np.array([recons1,recons2])
    
    lenXt=len(test_data[0,:])
    num_modes = test_data.shape[0]

    nsteps=lenXt
    X_test=np.zeros((1,delay*num_modes))
    X_test[0,:] = test_data[:,:delay].reshape(1,-1)
    X_test=X_test

    arr0 = np.zeros((delay,1))
    arr1 = np.zeros((delay,1))
    
    arr0[:delay,0] = test_data[0,:delay].T
    arr1[:delay,0] = test_data[1,:delay].T
    

    kernel = kernels_dic[kernel_keyword]
    k_matrix = kernel(matrix_data, matrix_data, parameters)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    #A=np.matmul(np.linalg.inv(k_matrix), Y_data)
    A=np.linalg.solve(k_matrix ,Y_data)
    X_test0=X_test
    isteps=int(nsteps/delay)+1
    
    for i in range(1,isteps):
        X_test1=X_test0
        t_matrix = kernel(X_test1, matrix_data, parameters) 
        prediction = np.matmul(t_matrix, A)
  
        X_test0 = (test_data[:,i*delay:(i+1)*delay]).reshape(1,-1)
        for mode in range(num_modes):
            imin=i*delay
            imax=np.minimum((i+1)*delay,nsteps)
            delaymin=imax-imin
            if mode==0:
                arr0=np.append(arr0,np.array(prediction[0,(mode*delay):(mode*delay+delaymin)]))
            else:
                arr1=np.append(arr1,np.array(prediction[0,(mode*delay):(mode*delay+delaymin)]))
    
    arr=np.array([arr0,arr1])        
    
    predicted_train2=arr
    train_data2=test_data
    
    SMAPE_train_data = Calc_SMAPE(train_data2[1,:],predicted_train2[1,:])
    train_r2_score = R2_SCORE(train_data2[1,:],predicted_train2[1,:])
    MSE_score = MSE(train_data2[1,:],predicted_train2[1,:])

    return (SMAPE_train_data+(train_r2_score-1)**2)


"""
####### new loss function for 3 dimension dynamical systems #######################################################################################
def rho(parameters, matrix_data, Y_data, sample_indices,  kernel_keyword= "RBF", regu_lambda = 0.000001):

    delay=3
    recons1=np.concatenate((matrix_data[0,:delay],np.concatenate((Y_data[0,:delay],Y_data[1:,len(Y_data[0])-2*delay-1]),axis=0)),axis=0)
    recons2=np.concatenate((matrix_data[0,delay:2*delay],np.concatenate((Y_data[0,delay:2*delay],Y_data[1:,len(Y_data[0])-delay-1]),axis=0)),axis=0)
    recons3=np.concatenate((matrix_data[0,2*delay:],np.concatenate((Y_data[0,2*delay:],Y_data[1:,len(Y_data[0])-1]),axis=0)),axis=0)

    test_data=np.array([recons1,recons2,recons3])
    
    lenXt=len(test_data[0,:])
    num_modes = test_data.shape[0]

    nsteps=lenXt
    X_test=np.zeros((1,delay*num_modes))
    X_test[0,:] = test_data[:,:delay].reshape(1,-1)
    X_test=X_test

    arr0 = np.zeros((delay,1))
    arr1 = np.zeros((delay,1))
    arr2 = np.zeros((delay,1))
    
    arr0[:delay,0] = test_data[0,:delay].T
    arr1[:delay,0] = test_data[1,:delay].T
    arr2[:delay,0] = test_data[2,:delay].T
    

    kernel = kernels_dic[kernel_keyword]
    k_matrix = kernel(matrix_data, matrix_data, parameters)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    #A=np.matmul(np.linalg.inv(k_matrix), Y_data)
    A=np.linalg.solve(k_matrix ,Y_data)
    
    X_test0=X_test
    isteps=int(nsteps/delay)+1
    
    for i in range(1,isteps):
        X_test1=X_test0
        t_matrix = kernel(X_test1, matrix_data, parameters) 
        prediction = np.matmul(t_matrix, A)
  
        X_test0 = (test_data[:,i*delay:(i+1)*delay]).reshape(1,-1)
        for mode in range(num_modes):
            imin=i*delay
            imax=np.minimum((i+1)*delay,nsteps)
            delaymin=imax-imin
            if mode==0:
                arr0=np.append(arr0,np.array(prediction[0,(mode*delay):(mode*delay+delaymin)]))
            elif mode==1:
                arr1=np.append(arr1,np.array(prediction[0,(mode*delay):(mode*delay+delaymin)]))
            else:
                arr2=np.append(arr2,np.array(prediction[0,(mode*delay):(mode*delay+delaymin)]))
    arr=np.array([arr0,arr1,arr2])        

    predicted_train2=arr
    train_data2=test_data
    
    SMAPE_train_data = Calc_SMAPE(train_data2[2,:],predicted_train2[2,:])
    train_r2_score = R2_SCORE(train_data2[2,:],predicted_train2[2,:])
    MSE_score = MSE(train_data2[2,:],predicted_train2[2,:])

    return (SMAPE_train_data+(train_r2_score-1)**2)
"""

def l2(parameters, matrix_data, Y, batch_indices, sample_indices, kernel_keyword = "RBF"):
    X_sample = matrix_data[sample_indices]
    Y_sample = Y[sample_indices]
    
    not_sample = [x for x in batch_indices not in sample_indices]
    X_not_sample = matrix_data[not_sample]
    Y_not_sample = Y[not_sample]
    prediction = kernel_regression(X_sample, X_not_sample, Y_sample, kernel_keyword)
    
    return np.dot(Y_not_sample - prediction, Y_not_sample- prediction)

#%% Grad functions

""" We define the gradieànt calculator function.Like rho, the gradient 
calculator function accesses the gradfunctions via a keyword"""

# Gradient calculator function. Returns an array
def grad_kernel(parameters, X_data, Y_data, sample_indices, kernel_keyword= "RBF", regu_lambda = 0.000001):
    grad_K = value_and_grad(rho)
    rho_value, gradient = grad_K(parameters, X_data, Y_data, sample_indices, kernel_keyword, regu_lambda = regu_lambda)
    
    return rho_value, gradient

#%% The class version of KF ############################
class KernelFlowsP():
    
    def __init__(self, kernel_keyword, parameters):
        self.kernel_keyword = kernel_keyword
        self.parameters = np.copy(parameters)
        
        # Lists that keep track of the history of the algorithm
        self.rho_values = []
        self.grad_hist = []
        self.para_hist = []
        
        self.LR = 0.1
        self.beta = 0.9
        self.regu_lambda = 0.0001
    
    def get_hist(self):
        return self.param_hist, self.gradients, self.rho_values
        
    
    def save_model(self):
        np.save("param_hist", self.param_hist)
        np.save("gradients", self.gradients)
        np.save("rho_values", self.rho_values)
        
    def get_parameters(self):
        return self.parameters
    
    def set_LR(self, value):
        self.LR = value
        
    def set_beta(self, value):
        self.beta = value
    def set_train(self, train):
        self.train = train
        
    
    def fit(self, X, Y, iterations, batch_size = False, optimizer = "SGD", 
            learning_rate = 0.01, beta = 0.9, show_it = 100, regu_lambda = 0.000001,                           # it was learning_rate = 0.1
            adaptive_size = False, adaptive_range = (), proportion = 0.5, reduction_constant = 0.0):            

        self.set_LR(learning_rate)
        self.set_beta(beta)
        self.regu_lambda = regu_lambda
        
        self.X_train = np.copy(X)
        self.Y_train = np.copy(Y)
        
        momentum = np.zeros(self.parameters.shape, dtype = "float")
        
        # This is used for the adaptive sample decay
        rho_100 = []
        adaptive_mean = 0
        adaptive_counter = 0
        
        if adaptive_size == False or adaptive_size == "Dynamic":
            sample_size = proportion
        elif adaptive_size == "Linear":
            sample_size_array = sample_size_linear(iterations, adaptive_range) 
        else:
            print("Sample size not recognized")
            
        for i in range(iterations):
            if i % show_it == 0:
                #print("parameters ", self.parameters)# used to print out
                continue
            
            if adaptive_size == "Linear":
                sample_size = sample_size_array[i]
                
            elif adaptive_size == "Dynamic" and adaptive_counter == 100:
                if adaptive_mean != 0:
                    change = np.mean(rho_100) - adaptive_mean 
                else:
                    change = 0
                adaptive_mean = np.mean(rho_100)
                rho_100 = []
                sample_size += change - reduction_constant
                adaptive_counter= 0
                
            # Create a batch and a sample
            sample_indices, batch_indices = batch_creation(X, batch_size, sample_proportion = sample_size)
            X_data = X[batch_indices]
            Y_data = Y[batch_indices]

            # Changes parameters according to SGD rules
            if optimizer == "SGD":
                rho, grad_mu = grad_kernel(self.parameters, X_data, Y_data, 
                                           sample_indices, self.kernel_keyword, regu_lambda = regu_lambda)
                """
                if  rho > 1 or rho < 0:
                    #print("Warning, rho outside [0,1]: ", rho)
                    continue
                else:
                    self.parameters -= learning_rate * grad_mu
                """
                self.parameters -= learning_rate * grad_mu
                    
            
            # Changes parameters according to Nesterov Momentum rules     
            elif optimizer == "Nesterov":
                rho, grad_mu = grad_kernel(self.parameters - learning_rate * beta * momentum, 
                                               X_data, Y_data, sample_indices, self.kernel_keyword, regu_lambda = regu_lambda)
                if  rho > 1 or rho < 0:
                    #print("Warning, rho outside [0,1]: ", rho)
                    continue
                else:
                    momentum = beta * momentum + grad_mu
                    self.parameters -= learning_rate * momentum
                
            else:
                print("Error optimizer, name not recognized")
            
            # Update history 
            self.para_hist.append(np.copy(self.parameters))
            self.rho_values.append(rho)
            self.grad_hist.append(np.copy(grad_mu))
            
            rho_100.append(rho)
            adaptive_counter +=1
                
            
        # Convert all the lists to np arrays
        self.para_hist = np.array(self.para_hist) 
        self.rho_values = np.array(self.rho_values)
        self.grad_hist = np.array(self.grad_hist)
                
        return self.parameters
    
    def predict(self,test, regu_lambda = 0.0000001):
         
        X_train = self.X_train
        Y_train = self.Y_train
        prediction = kernel_regression(X_train, test, Y_train, self.parameters, self.kernel_keyword, regu_lambda = regu_lambda) 

        return prediction

    def extrapolate(self,test, nsteps=1,regu_lambda = 0.000001):
         
        X_train = self.X_train
        Y_train = self.Y_train
        prediction = kernel_extrapolate(X_train, test, Y_train, self.parameters, nsteps,self.kernel_keyword, regu_lambda = regu_lambda) 

        return prediction

# set the kernel type and the number of hyperparameters
def fit_data_anl3(train_data,delay,regu_lambda,learning_rate,noptsteps=100): # I added the learning rate
    
    lenX=len(train_data[0,:])
    num_modes = train_data.shape[0]

    # Some constants
    nparameters=2 #it was 21 for anl3
    vdelay=delay*np.ones((num_modes,), dtype=int)
    vregu_lambda=regu_lambda*np.ones((num_modes,))

    # Get scaling factor    
    normalize=np.amax(train_data[:,:])
    # Prepare training data
    X=np.zeros((1+lenX-2*delay,delay*num_modes))

    Y=np.zeros((1+lenX-2*delay,delay*num_modes))
    for mode in range(train_data.shape[0]):
        for i in range(1+lenX-2*delay):
              X[i,(mode*delay):(mode*delay+delay)]=train_data[mode,i:(i+delay)]
              Y[i,(mode*delay):(mode*delay+delay)]=train_data[mode,(i+delay):(i+2*delay)]

    # Normalize
    X=X/normalize
    Y=Y/normalize     
    
    # Fit data
    c=np.zeros(nparameters)+1 
    #c[5]=0
    #c[1]=0    
    mu_1 = c
    # select the kernel
    kerneltype="RBF"
    K = KernelFlowsP(kerneltype, mu_1)

    if (len(X)>100):
        mu_pred = K.fit(X, Y, noptsteps, optimizer = "SGD",learning_rate=learning_rate,  batch_size = 100, show_it = 500, regu_lambda=regu_lambda) #added learning_rate
        mu_1=mu_pred
        c=mu_1
    else:
        mu_pred = K.fit(X, Y, noptsteps, optimizer = "SGD",learning_rate=learning_rate,  batch_size = False, show_it = 500, regu_lambda=regu_lambda)
        mu_1=mu_pred
        c=mu_1
    kernel = kernels_dic[kerneltype]
    k_matrix = kernel(X, X, mu_1)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    #A=np.matmul(np.linalg.inv(k_matrix), Y)
    
    A=np.linalg.solve(k_matrix,Y)
      
    return k_matrix, A, mu_1, normalize, X,Y, K.rho_values,K.para_hist #I added K.rho_values and para_hist

#set the kernel type
def test_fit_anl3(test_data,train_X,delay,k_matrix,A,param,normalize):
    
    lenXt=len(test_data[0,:])
    num_modes = test_data.shape[0]

    nsteps=lenXt
    X_test=np.zeros((1,delay*num_modes))
    X_test[0,:] = test_data[:,:delay].reshape(1,-1)
    X_test=X_test/normalize
    
    arr = np.zeros((nsteps,num_modes))
    arr[:2*delay,:] = test_data[:,:2*delay].T/normalize

    #select the kernel
    kerneltype = "RBF"
    kernel = kernels_dic[kerneltype]

    X_test0=X_test
    isteps=int(nsteps/delay)+1
    for i in range(1,isteps):
        X_test1=X_test0
        t_matrix = kernel(X_test1, train_X, param) 
        prediction = np.matmul(t_matrix, A) 
        
        # Just using true data
        X_test0 = (test_data[:,i*delay:(i+1)*delay]/normalize).reshape(1,-1)
        
        for mode in range(num_modes):
            imin=i*delay
            imax=np.minimum((i+1)*delay,nsteps)
            delaymin=imax-imin
            arr[imin:imax,mode]=np.array(prediction[0,(mode*delay):(mode*delay+delaymin)])

    # Rescale
    predall=arr*normalize

    return predall.T

####### here we define the error metrics and some other functions

# sMAPE error and smMAPE: the function uses sMAPE modified if zeros in denominator otherwise sMAPE
def Calc_SMAPE(y_true,y_pred):
    #https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    assert(len(y_true)==len(y_pred))
    n = len(y_true)
    denom = np.maximum((np.abs(y_true)+np.abs(y_pred))+np.full(n,0.2),np.full(n,0.5+0.2))/2.0

    smape = 100*(1/n)*np.sum( np.abs(y_true - y_pred) / denom )
    return smape

def R2_SCORE(y_true,y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    assert(len(y_true)==len(y_pred))
    n = len(y_true)
    y_mean=np.mean(y_true)
    top=np.sum((y_true-y_pred)**2)
    bottom=np.sum((y_true-y_mean)**2)
    return 1 - top/bottom

def MSE(y_true,y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    assert(len(y_true)==len(y_pred))
    n = len(y_true)
    top=np.sum((y_true-y_pred)**2)/n
    return top

# tranform the x(t) with ( x(t)+x(t-1)+x(t-2) )/3 and the first 3 values are the same
def Smooth(xx,n=3):   
    import pandas as pd
    ts=xx
    #strt = np.copy(ts[0])
    #stp = np.copy(ts[::-1][0])
    #ts_smoothed = [strt] +list(pd.Series(ts).rolling(window=n).mean().iloc[n-1:].values) + [stp]
    z = pd.Series(ts).rolling(window=n).mean().values
    z[:n] = ts[:n]
    return z

from scipy.stats import boxcox
# invert a boxcox transform for one value
def invert_boxcox(value, lam):
    # log case
    if lam == 0:
        return np.exp(value)
    # all other cases
    return np.exp(np.log(lam * value + 1) / lam)

def bernoulli_map(x,n):
    a=x
    lis= np.zeros(n)

    for i in range(n):
        lis[i]=a
        if 0 <= a < 0.5:
            a=2*a
        elif 0.5 <= a < 1:
            a=2*a-1
    return lis            

if __name__ == '__main__':
    print('This is fit_rkhs file where KFs is implemented')

    """
    In this file you can choose your loss function, for the moment there is 4 options:
    1st: rkhs loss function
    2nd: new loss function for 1 dimension dynamical systems (ex: bernoulli, logistic map, univariate time series)
    3th: new loss function for 2 dimension dynamical systems (ex: Hénon map)
    4th: new loss function for 3 dimension dynamical systems (ex: Lorenz system)
    

    Things that you need to change manually in this code:
    1/// the loss function
    -> choose the loss function by deleting the comment symbols  
    -> delay          : in case you are working with the new metric, you need to set the delay value 
                        (which is also the forecast horizon) in the first line of the function.
    
    2/// in the function: fit_data_anl3
    -> nparameters    :   the number of hyperparameters of the kernel, example nparameters=2 for RBF 
    -> kerneltype     :   Set the kernel type, example "RBF"  

    3/// in the function: test_fit_anl3
    -> kerneltype     :   Set the kernel type, it should be the same kernel as in the previous function  
    """
