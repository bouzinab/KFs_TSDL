from fit_rkhs_stdl import*
import numpy as np
import multiprocess as mp
import matplotlib.pyplot as plt
import time
from time import time
import concurrent.futures
import json
from sklearn.metrics import r2_score

import matplotlib.font_manager as fm, os
from scipy.integrate import odeint
from mpl_toolkits.mplot3d.axes3d import Axes3D
np.random.seed(1)

####### Lorenz system #########################################################################################
from scipy.integrate import odeint
initial_state_1 = [0, 1, 1.05]
initial_state_2 = [0.5, 1.5, 2.5]

#initial_state = [0.5, 1.5, 2.5]
sigma = 10.
rho   = 28.
beta  = 8./3.

start_time = 0
end_time_1 = 10
end_time_2 = 490

time_points_1 = np.linspace(start_time, end_time_1, end_time_1*100)
time_points_2 = np.linspace(start_time, end_time_2, end_time_2*100)

def lorenz_system(current_state, t):
    x, y, z = current_state

    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

xyz = odeint(lorenz_system, initial_state_1, time_points_1)

X1 = xyz[:, 0]
Y1 = xyz[:, 1]
Z1 = xyz[:, 2]    

xyz = odeint(lorenz_system, initial_state_2, time_points_2)


X2 = xyz[:, 0]
Y2 = xyz[:, 1]
Z2 = xyz[:, 2]    

X=np.concatenate((X1,X2))
Y=np.concatenate((Y1,Y2))
Z=np.concatenate((Z1,Z2))

##############################################################################################    
##############################################################################################    
    
def lorenz_attractor(kernel_used,approach):
    # z variable  ##############################################
    error_metrics = []
    ts=np.array([X,Y,Z])
    
    #spliting data ###############################################
    ts_len = len(ts[0])
    train_fract = 0.02
    n_train_samps = int(train_fract*ts_len)
    train_data = np.array(ts[:,0:n_train_samps])
    n_test_samps = ts_len - n_train_samps
    test_data = np.array(ts[:,n_train_samps:])

    # Set parameters #############################################
    delay_opt = 3 
    regu_lambda_opt = 0.00001
    learning_opt = 0.01
    
    if approach == "learning":
        noptsteps_opt = 2000 #number of iterations
    elif approach == "no_learning":
        noptsteps_opt = 2
    else:
        print("error: approach should only be learning or no_learning")    
    metric=[]
    start = time()
    
    # fitting training data ######################################
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
    print("Z",error_metrics)
    predicted_z=predicted_test[len(ts)-1,:]
    
    # y variable  ########################################
    error_metrics = []
    ts=np.array([X,Z,Y])
    
    #spliting data ###############################################
    ts_len = len(ts[0])
    n_train_samps = int(train_fract*ts_len)
    train_data = np.array(ts[:,0:n_train_samps])
    n_test_samps = ts_len - n_train_samps
    test_data = np.array(ts[:,n_train_samps:])

    # Set parameters #################################################
    metric=[]
    start = time()
    #fitting training data ##################
    k_matrix, A, param, normalize, train_X,train_Y,rho_values,para_hist = fit_data_anl3(train_data,delay_opt,regu_lambda_opt,learning_opt,noptsteps_opt)
    #kerneltype = "RBF"        
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
    print("Y",error_metrics)
    predicted_y=predicted_test[len(ts)-1,:]

    # x variable  ########################################
    error_metrics = []
    ts=np.array([Z,Y,X])
    
    #spliting data ###############################################
    ts_len = len(ts[0])
    n_train_samps = int(train_fract*ts_len)
    train_data = np.array(ts[:,0:n_train_samps])
    n_test_samps = ts_len - n_train_samps
    test_data = np.array(ts[:,n_train_samps:])

    # Set parameters #################################################
    learning_opt=0.000001

    metric=[]
    start = time()
    #see how the metrics change while training ##################
    k_matrix, A, param, normalize, train_X,train_Y,rho_values,para_hist = fit_data_anl3(train_data,delay_opt,regu_lambda_opt,learning_opt,noptsteps_opt)
    #kerneltype = "RBF"        
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
    A_opt=np.matmul(np.linalg.inv(k_matrix_opt), train_Y)
    #A_opt=np.linalg.solve(k_matrix_opt, train_Y)

    
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
    #q = [rho_opt,SMAPE_train_data,SMAPE_test_data,train_r2_score,test_r2_score,r,gradient]
    q = [SMAPE_test_data,mse_test,test_r2_score]
    error_metrics.append(q)
    print("X",error_metrics)
    predicted_x=predicted_test[len(ts)-1,:]

    ###### Figures ########################################################

    # plot the lorenz attractor in three-dimensional phase space
    font_family = 'Myriad Pro'
    title_font = fm.FontProperties(family=font_family, style='normal', size=20, weight='normal', stretch='normal')
    fig = plt.figure(figsize=(12, 9))
    ax = fig.gca(projection='3d')
    ax.xaxis.set_pane_color((1,1,1,1))
    ax.yaxis.set_pane_color((1,1,1,1))
    ax.zaxis.set_pane_color((1,1,1,1))
    ax.plot(test_data[2,:],test_data[1,:], test_data[0,:], color='orange', alpha=0.9, linewidth=1.5)
    ax.plot(predicted_x,predicted_y,predicted_z, color='blue', alpha=0.7, linewidth=1)

    ax.set_title('Lorenz attractor and approximation with learned kernel', fontproperties=title_font)

    #plt.savefig('new_metric_plot_report/'+'lorenz_attractor_learning_rate'+'.png')
    plt.show()
    
    #plt.savefig('new_metric/'+s+'index'+str(ind)+str('anl3')+'.png')
    #plt.savefig('new_metric_map/lorenz_map/'+str('new_metrci_attractor_anl3')+'.png')

    return error_metrics

##############################################################################################################################

if __name__ == "__main__":
    """
    #inputs are:
    1/// choose the kernel you want to use from the kernel_functions.py file
    
    2/// choose the approach:
    approach = "learning"    #to learn the kernel with KFs
    approach = "no_learning" # to use the base estimator, ie non learned kernel
        
    example: lorenz_attractor("RBF","no_learning")
    this instruction provides the attractor approximation of lorenz map using the base estiamtor
    
    parameters that you need to change manually in this code:
    -> delay_opt             :  the value of delay, it is also the value of the forecasting horizon  
    -> regu_lambda_opt       :  the regularization parameter lambda
    -> learning_opt          :  the step size of SGD   
    -> noptsteps_opt         :  number of iterations if the approach is learning, otherwise no need to change it 
    -> para_opt              :  the hyperparameters of the kernel if the approach is no_learning, example [1,1] if we work with kernel RBF
    
    important remarks: 
    *if you are using the new metric and you have changed the delay value in this file, don't forget to change it in the "fit_rkhs_stdl" file too. 
    *if you are using the new metric, choose the version of 3 dimension dynamical systems in the "fit_rkhs_stdl" file.
    """

    kernel_used = "RBF"
    approach = "no_learning"
    res=lorenz_attractor(kernel_used,approach)
    print("result",res)
