from fit_rkhs_stdl import*
import numpy as np
import multiprocess as mp
import matplotlib.pyplot as plt
import time
from time import time
import concurrent.futures
import json
from sklearn.metrics import r2_score

np.random.seed(1)

####### henon map #########################################################################################
def henon_attractor(x, y, a=1.4, b=0.3):
    x_next = 1 - a * x ** 2 + y
    y_next = b * x
    return x_next, y_next
def henon_map(x, y, n):
    X = np.zeros(n+1)
    Y = np.zeros(n+1)
    #starting point
    X[0], Y[0] = x, y

    # add points to array
    for i in range(n):
        x_next, y_next = henon_attractor(X[i], Y[i])
        X[i+1] = x_next
        Y[i+1] = y_next
    return X, Y
    
X1=henon_map(0.9,-0.9,100)[0]
Y1=henon_map(0.9,-0.9,100)[1]

X2=henon_map(-0.1,0.1,4900)[0]
Y2=henon_map(-0.1,0.1,4900)[1]


X=np.concatenate((X1,X2))
Y=np.concatenate((Y1,Y2))

###############################################    
def henon_attractor(kernel_used,approach):
    
    # y variable  ########################################
    error_metrics = []
    ts=np.array([X,Y])

    #spliting data ###############################################
    ts_len = len(ts[0])
    train_fract = 0.02
    n_train_samps = int(train_fract*ts_len)
    train_data = np.array(ts[:,0:n_train_samps])
    n_test_samps = ts_len - n_train_samps
    test_data = np.array(ts[:,n_train_samps:])

    # Set parameters #################################################
    delay_opt = 3 
    regu_lambda_opt = 0.00001 
    learning_opt = 0.01
    
    if approach == "learning":
        noptsteps_opt = 1000 #number of iterations
    elif approach == "no_learning":
        noptsteps_opt = 2
    else:
        print("error: approach should only be learning or no_learning")    

    metric=[]
    start = time()
    #fitting training data ##################
    k_matrix, A, param, normalize, train_X,train_Y,rho_values,para_hist = fit_data_anl3(train_data,delay_opt,regu_lambda_opt,learning_opt,noptsteps_opt)
    kerneltype = kernel_used        
    kernel = kernels_dic[kerneltype]
    print(f'Time taken to fit data iterations 25000: {time() - start} seconds')

    #the optimal rho is the min of rho_values ######################### instead of using the last one because the rho is decreasing 
    ind_opt=np.nanargmin(rho_values)
    rho_opt=rho_values[ind_opt]

    if approach == "learning":
        para_opt=para_hist[ind_opt]  # the hyperparameters are computed from the KF method
    elif approach == "no_learning":
        para_opt=[1,1]     # set the hyperparameters of the kernel without KFs 
    print("opt para",para_opt)

    k_matrix_opt = kernel(train_X, train_X, para_opt)
    k_matrix_opt += regu_lambda_opt * np.identity(k_matrix_opt.shape[0])
    #A_opt=np.matmul(np.linalg.inv(k_matrix_opt), train_Y)
    A_opt=np.linalg.solve(k_matrix_opt, train_Y)
    
    
    # Predict and get error on training data ####################################################
    predicted_train = test_fit_anl3(train_data, train_X, delay_opt, k_matrix_opt, A_opt, para_opt, normalize)
    SMAPE_train_data = Calc_SMAPE(train_data[len(ts)-1,:],predicted_train[len(ts)-1,:])
    mse_train=np.round(mean_squared_error(train_data[len(ts)-1,:],predicted_train[len(ts)-1,:]),4)
    train_r2_score = np.round(r2_score(train_data[len(ts)-1],predicted_train[len(ts)-1]),4)

    # Predict and get error on testing data #######################################################
    test_data = np.array(ts[:,n_train_samps-delay_opt:])
    predicted_test = test_fit_anl3(test_data, train_X, delay_opt, k_matrix_opt, A_opt, para_opt, normalize)
    predicted_test = predicted_test[:,delay_opt:] # ignore the first delay points, they are history
    # redefine test data
    test_data = np.array(ts[:,n_train_samps:])
    
    # Visualize the error metrics
    SMAPE_test_data = Calc_SMAPE(test_data[len(ts)-1,:],predicted_test[len(ts)-1,:]) #only calc error from
    mse_test=np.round(mean_squared_error(test_data[len(ts)-1,:],predicted_test[len(ts)-1,:]),4)
    test_r2_score = np.round(r2_score(test_data[len(ts)-1],predicted_test[len(ts)-1]),4)

    # keep track of error metrics etc
    q = [SMAPE_test_data,mse_test,test_r2_score]
    error_metrics.append(q)
    predicted_y=predicted_test[len(ts)-1,:]
    
    # x variable  ########################################
    error_metrics = []
    ts=np.array([Y,X])

    #spliting data ###############################################
    ts_len = len(ts[0])
    n_train_samps = int(train_fract*ts_len)
    train_data = np.array(ts[:,0:n_train_samps])
    n_test_samps = ts_len - n_train_samps
    test_data = np.array(ts[:,n_train_samps:])

    metric=[]
    start = time()
    #fitting training data ##################
    k_matrix, A, param, normalize, train_X,train_Y,rho_values,para_hist = fit_data_anl3(train_data,delay_opt,regu_lambda_opt,learning_opt,noptsteps_opt)
    #kerneltype = kernel_used        
    kernel = kernels_dic[kerneltype]
    print(f'Time taken to fit data iterations 25000: {time() - start} seconds')

    
    #the optimal rho is the min of rho_values ######################### instead of using the last one because the rho is decreasing 
    ind_opt=np.nanargmin(rho_values)
    rho_opt=rho_values[ind_opt]

    if approach == "learning":
        para_opt=para_hist[ind_opt]  # the hyperparameters are computed from the KF method
    elif approach == "no_learning":
        para_opt=[1,1]     # set the hyperparameters of the kernel without KFs 
    print("opt para",para_opt)

    k_matrix_opt = kernel(train_X, train_X, para_opt)
    k_matrix_opt += regu_lambda_opt * np.identity(k_matrix_opt.shape[0])
    #A_opt=np.matmul(np.linalg.inv(k_matrix_opt), train_Y)
    A_opt=np.linalg.solve(k_matrix_opt, train_Y)
    
    # Predict and get error on training data ####################################################
    predicted_train = test_fit_anl3(train_data, train_X, delay_opt, k_matrix_opt, A_opt, para_opt, normalize)
    SMAPE_train_data = Calc_SMAPE(train_data[len(ts)-1,:],predicted_train[len(ts)-1,:])
    mse_train=np.round(mean_squared_error(train_data[len(ts)-1,:],predicted_train[len(ts)-1,:]),4)
    train_r2_score = np.round(r2_score(train_data[len(ts)-1],predicted_train[len(ts)-1]),4)

    # Predict and get error on testing data #######################################################
    test_data = np.array(ts[:,n_train_samps-delay_opt:])
    predicted_test = test_fit_anl3(test_data, train_X, delay_opt, k_matrix_opt, A_opt, para_opt, normalize)
    predicted_test = predicted_test[:,delay_opt:] # ignore the first delay points, they are history
    # redefine test data
    test_data = np.array(ts[:,n_train_samps:])
    
    # Visualize the error metrics
    SMAPE_test_data = Calc_SMAPE(test_data[len(ts)-1,:],predicted_test[len(ts)-1,:]) #only calc error from
    mse_test=np.round(mean_squared_error(test_data[len(ts)-1,:],predicted_test[len(ts)-1,:]),4)
    test_r2_score = np.round(r2_score(test_data[len(ts)-1],predicted_test[len(ts)-1]),4)

    # keep track of error metrics etc
    q = [SMAPE_test_data,mse_test,test_r2_score]
    error_metrics.append(q)
    predicted_x=predicted_test[len(ts)-1,:]

    ###### Figures ########################################################
        
    fig = plt.figure(figsize=(15,8))
    #plot of graph true data and predicted data        
    plt.subplot(1,1,1)   
    plt.plot(test_data[1,:],test_data[0,:], '^',alpha = 0.8,markersize=4,color='orange',label='True')        
    plt.plot(predicted_x,predicted_y, '^',alpha = 0.8,  markersize=4,color='b',label='Prediction Test')
    plt.xlim([np.min(ts[1,:])-0.1,np.max(ts[1,:])+0.1])          
    plt.ylim([np.min(ts[0,:])-0.1,np.max(ts[0,:])+0.1])            
    #plt.title('Hénon map attractor no learning'+'  \ndelay=%d reg_lam=%.2f noptstp=%d '  % (delay_opt,regu_lambda_opt, noptsteps_opt) )
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
        
    plt.show()
    #plt.savefig('new_metric_map/henon_map/'+str('new_metrci_attractor_anl3')+'.png')

    return error_metrics

##############################################################################################################################

if __name__ == "__main__":
    """
    #inputs are:
    1/// choose the kernel you want to use from the kernel_functions.py file
    
    2/// choose the approach:
    approach = "learning"    #to learn the kernel with KFs
    approach = "no_learning" # to use the base estimator, ie non learned kernel
        
    example: henon_attractor("RBF","no_learning")
    this instruction provides the attractor approximation of henon map using the base estiamtor
    
    parameters that you need to change manually in this code:
    -> delay_opt             :  the value of delay, it is also the value of the forecasting horizon  
    -> regu_lambda_opt       :  the regularization parameter lambda
    -> learning_opt          :  the step size of SGD   
    -> noptsteps_opt         :  number of iterations if the approach is learning, otherwise no need to change it 
    -> para_opt              :  the hyperparameters of the kernel if the approach is no_learning, example [1,1] if we work with kernel RBF
                                
    important remarks: 
    *if you are using the new metric and you have changed the delay value in this file, don't forget to change it in the "fit_rkhs_stdl" file too. 
    *if you are using the new metric, choose the version of 2 dimension dynamical systems in the "fit_rkhs_stdl" file.
    """

    kernel_used = "RBF"
    approach = "no_learning"
    res=henon_attractor(kernel_used,approach)
    print("result",res)
