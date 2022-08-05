from fit_rkhs_stdl import*
import numpy as np
import multiprocess as mp
import matplotlib.pyplot as plt
import time
from time import time
import concurrent.futures
import json
from sklearn.metrics import r2_score
import timesynth as tsl

np.random.seed(1)

# Choose the dynamical system: logistic map, hénon map, lorenz system, synthetic time series

####### Logistic map #########################################################################################
def logistic_map(x,n):
    a=x
    lis= np.zeros(n)

    for i in range(n):
        lis[i]=a
        a=4*a*(1-a)
    return lis            

####### Hénon map #########################################################################################
def henon_attractor(x, y, a=1.4, b=0.3):
    '''Computes the next step in the Henon 
    map for arguments x, y with kwargs a and
    b as constants.
    '''
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

####### Lorenz system #########################################################################################
from scipy.integrate import odeint
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

#######################################################################################################
#######################################################################################################

def dynamic_test(dynamical_system, dim,kernel_used, approach, delay_opt, regu_lambda_opt, learning_opt):

    error_metrics = []
    
    if dynamical_system == "logistic":
        ts1=logistic_map(0.1,200) # x(0)=0.1 and 200 training points
        ts2=logistic_map(0.97,1800) # x(0)=0.97 and 1800 testing points
        ts=np.concatenate((ts1,ts2))
        ts=logistic_map(0.1,1000)
        ts=ts.reshape(1,len(ts))

        train_fract = 0.1  # 10% of the data is used for training (since we have 200 training points and 1800 testing points)

    elif dynamical_system == "henon":

        X1=henon_map(0.9,-0.9,100)[0]
        Y1=henon_map(0.9,-0.9,100)[1]
        X2=henon_map(-0.1,0.1,4900)[0]  
        Y2=henon_map(-0.1,0.1,4900)[1]
        X=np.concatenate((X1,X2))
        Y=np.concatenate((Y1,Y2))

        train_fract = 0.02
        if dim == 1:
            ts=np.array([Y,X])
        elif dim == 2:
            ts=np.array([X,Y])
        else:
            print("Error dim, henon map has 2 dimensions, dim should take 1 or 2")

    elif dynamical_system == "lorenz":

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

        train_fract = 0.02
        if dim == 1:
            ts=np.array([Y,Z,X])
        elif dim == 2:
            ts=np.array([X,Z,Y])
        elif dim == 3:
            ts=np.array([X,Y,Z])
        else:
            print("Error dim, lorenz_system has 3 dimensions, dim should take 1 or 2 or 3")

    elif dynamical_system == "synthetic":
        time_sampler = tsl.TimeSampler(stop_time=150)
        regular_time_samples = time_sampler.sample_regular_time(num_points=1000)

        gp = tsl.signals.GaussianProcess(kernel='SE',lengthscale=1,variance=1)
        white_noise = tsl.noise.GaussianNoise(std=0.1)
        gp_series = tsl.TimeSeries(signal_generator=gp,noise_generator=white_noise)
        ts = gp_series.sample(regular_time_samples)[0]
        ts=ts.reshape(1,len(ts))

        train_fract = 0.25

    else:
        print("dynamical_system should only be one of the following: logistic, henon, lorenz, synthetic ")
    

    # spliting data #####################################################
    ts_len = len(ts[0])
    n_train_samps = int(train_fract*ts_len)
    train_data = np.array(ts[:,0:n_train_samps])
    n_test_samps = ts_len - n_train_samps
    test_data = np.array(ts[:,n_train_samps:])

    # Set nb of iterations #################################################
    
    if approach == "learning":
        noptsteps_opt = 5000 #number of iterations
    elif approach == "no_learning":
        noptsteps_opt = 2
    else:
        print("error: approach should only be learning or no_learning")    
    
    metric=[]
    start = time()
    
    # fitting data ##################
    k_matrix, A, param, normalize, train_X,train_Y,rho_values,para_hist = fit_data_anl3(train_data,delay_opt,regu_lambda_opt,learning_opt,noptsteps_opt)
    # select the kernel #############
    kerneltype = kernel_used        
    kernel = kernels_dic[kerneltype]
    print(f'Time taken to fit data iterations 25000: {time() - start} seconds')

    #the optimal rho is the min of rho_values ######################### instead of using the last one because the rho is not strictly decreasing 
    ind_opt=np.nanargmin(rho_values)
    rho_opt=rho_values[ind_opt]
    
    if approach == "learning":
        para_opt=para_hist[ind_opt]  # the hyperparameters are computed from the KF method
    elif approach == "no_learning":
        para_opt=[1,1]     # set the hyperparameters of the kernel without KFs 
    
    print("para opt",para_opt)
    
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
    #q = [rho_opt,SMAPE_train_data,SMAPE_test_data,train_r2_score,test_r2_score,r,gradient]
    q = [SMAPE_test_data,mse_test,test_r2_score]
    error_metrics.append(q)
    
    ###### Figures ########################################################
    
    fig = plt.figure(figsize=(15,8))
    
    #plot of graph true data and predicted data for train set       
    plt.subplot(2,2,1)   
    plt.plot(np.arange(n_train_samps),train_data[len(ts)-1,:],color='orange',label='True')        
    plt.plot(np.arange(n_train_samps),predicted_train[len(ts)-1,:],color='b',label='Prediction TRAIN')
    plt.ylim([np.min(ts[len(ts)-1,:])-0.5,np.max(ts[len(ts)-1,:])+0.5])           
    plt.title('synthetic time series, new metric  ,'+'  \ndelay=%d reg_lam=%.2f noptstp=%d \nrho=%.2f \nTrain_sMAPE=%.2f'  % (delay_opt,regu_lambda_opt, noptsteps_opt,rho_opt,SMAPE_train_data)+'%' + ' & Test_sMAPE=%.2f' % (SMAPE_test_data)+'%')
    
    plt.xlabel('Time step')
    plt.ylabel('time Series values')
    plt.legend()
        
    #plot of graph true data and predicted data for test set       
    plt.subplot(2,2,2)
    plt.plot(np.arange(n_train_samps,n_train_samps+n_test_samps),test_data[len(ts)-1,:],color='orange',label='True')        
    plt.plot(np.arange(n_train_samps,n_train_samps+n_test_samps),
                     predicted_test[len(ts)-1,:],color='c',label='Prediction TEST')
    
    plt.ylim([np.min(ts[len(ts)-1,:])-0.1,np.max(ts[len(ts)-1,:])+0.1])            
    plt.title('Train r_2 score ='+str(train_r2_score)+'& Test r_2 score ='+str(test_r2_score))
    plt.legend()
    
    #the loss function history
    plt.subplot(2,2,3)
    plt.plot(rho_values,'g')
    plt.grid('on')
    #plt.ylim([0,1])
    plt.ylabel('rho')
    plt.xlabel('SGD Iteration')
    plt.title('Tr mse =%.4f Te mse=%.4f' %(mse_train,mse_test))
    
    #the hyperpatameters history
    plt.subplot(2,2,4)
    plt.plot(para_hist)
    plt.ylabel('Parameter values')
    plt.xlabel('SGD Iteration')
    plt.tight_layout()
    
    ## choose to save the plot or just show it:

    #plt.savefig('new_metric_plot_report/'+'ts_synthetic'+'_learning'+'.png')
    plt.show()
    
    """
    plt.plot(np.arange(n_test_samps),test_data[len(ts)-1,:],color='orange',label='True')        
    plt.plot(np.arange(n_test_samps),predicted_test[len(ts)-1,:],color='c',label='Prediction TEST')
    plt.title('synthetic time series , using the metric=smape+(R2-1)^2, '+'  \ndelay=%d reg_lam=%.2f noptstp=%d \nrho=%.2f \nTrain_sMAPE=%.2f'  % (delay_opt,regu_lambda_opt, noptsteps_opt,rho_opt,SMAPE_train_data)+'%' + ' & Test_sMAPE=%.2f' % (SMAPE_test_data)+'%')
    
    plt.xlabel('Time step')
    plt.ylabel('ltime series values')
    plt.legend()
    plt.savefig('new_metric_plot_report/'+'ts_synthetic_learning'+'.png')

    #plt.show()    
    #plt.savefig('plot_report/'+s+'index'+str(ind)+'_rkhs_metric'+'.png')
    """
    
    print("########################  success time series" )

    return error_metrics

##############################################################################################################################

if __name__ == "__main__":
    
    """
    #inputs are:
    1/// choose the dynamical system you want to study
    dynamical_system= "logistic", "henon", "lorenz", "synthetic"
    
    2/// choose the variable you want to forecast (x, y or z variable for a 3 dimension dynamical system)
    dim = 1 for logistic
        = 1 or 2 for henon
        = 1, 2 or 3 for lorenz
        = 1 for synthetic

    3/// Choose the kernel, example: "RBF"

    4/// choose the approach:
    approach = "learning"    #to learn the kernel with KFs
    approach = "no_learning" # to use the base estimator, ie non learned kernel
    
    5/// delay_opt             :  the value of delay, it is also the value of the forecasting horizon  
    6/// regu_lambda_opt       :  the regularization parameter lambda
    7/// learning_opt          :  the step size of SGD   
    
    example: dynamic_test("henon",1,"RBF","no_learning",3 ,0.00001, 0.01 )
    this instruction provides the forecasting of the x variable of henon map using the base estiamtor
    
    parameters that you need to change manually in this code:
    ->  noptsteps_opt : number of iterations if the approach is learning, otherwise no need to change it 
    ->  para_opt      : the hyperparameters of the kernel if the approach is no_learning, example [1,1] if we work with kernel RBF
    
    important remarks: 
    *if you are using the new metric and you have changed the delay value in this file, don't forget to change it in the "fit_rkhs_stdl" file too. 
    *choose your loss function in the "fit_rkhs_stdl" file.
    """
    
    delay_opt = 3 
    regu_lambda_opt = 0.00001
    learning_opt = 0.01   

    dynamic_test("lorenz",1,"RBF", "no_learning",delay_opt, regu_lambda_opt, learning_opt)
