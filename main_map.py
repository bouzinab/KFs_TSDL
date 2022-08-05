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

# data from https://pkg.yangzhuoranyang.com/tsdl/
# Rob Hyndman and Yangzhuoran Yang (2018). tsdl: Time Series Data Library. v0.1.0. https://pkg.yangzhuoranyang.com/tsdl/

#opening the json file and indices of ts from 0 to 647
# TSDL time series, set the path to the JSON file

with open("C:\\Users\\bbouz\\internship\\tsdl.JSON", 'r') as f:
    data = json.load(f)

def ts_multi(ind):

    error_metrics = []
    # writing all time series as matrix #################################
    ts = data[ind]['values']
    
    if data[ind]['type']=='univariate':
        ts=replace_nan_last(ts)
    else:
        ts=[replace_nan_last(ts[i]) for i in range(len(ts)) ]
    
    ts=np.array(ts)
    if data[ind]['type']=='univariate':
        ts=ts.reshape(1,len(ts))
    
    #spliting data #####################################################
    ts_len = len(ts[0])
    train_fract = 0.85 # 85% of the data is used for training
    n_train_samps = int(train_fract*ts_len)
    train_data = np.array(ts[:,0:n_train_samps])
    n_test_samps = ts_len - n_train_samps
    test_data = np.array(ts[:,n_train_samps:])

    # Set parameters #######################################################
    delay_opt = 3 #14
    regu_lambda_opt = 0.00001
    learning_opt = 0.01
    noptsteps_opt = 1000
    metric=[]
    start = time()
    
    # fitting data ####################################################
    k_matrix, A, param, normalize, train_X,train_Y,rho_values,para_hist = fit_data_anl3(train_data,delay_opt,regu_lambda_opt,learning_opt,noptsteps_opt)
    kerneltype = "RBF"        
    kernel = kernels_dic[kerneltype]
    print(f'Time taken to fit data iterations 25000: {time() - start} seconds')

    #the optimal rho is the min of rho_values ######################### instead of using the last one because the rho is not strictly decreasing 
    ind_opt=np.nanargmin(rho_values)
    rho_opt=rho_values[ind_opt]
    para_opt=para_hist[ind_opt]
    #para_opt=[1,1]
    
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
    #plot of graph true data and predicted data        
    
    plt.subplot(2,2,1)   
    plt.plot(np.arange(n_train_samps+n_test_samps),np.concatenate([train_data[len(ts)-1,:],test_data[len(ts)-1,:]]),color='orange',label='True')        
    plt.plot(np.arange(n_train_samps),predicted_train[len(ts)-1,:],color='b',label='Prediction TRAIN')
    plt.plot(np.arange(n_train_samps,n_train_samps+n_test_samps),
                     predicted_test[len(ts)-1,:],color='c',label='Prediction TEST')
                    
    plt.xlabel('Time step')
    plt.ylabel('time Series values')
    plt.legend()
    plt.title(str(data[ind]['description'])+' (ind=%d) \ndelay=%d reg_lam=%.2f noptstp=%d \nrho=%.2f \nTrain_sMAPE=%d'  % (ind,delay_opt,regu_lambda_opt, noptsteps_opt,rho_opt,SMAPE_train_data)+'%' + ' & Test_sMAPE=%d' % (SMAPE_test_data)+'%')

    plt.subplot(2,2,2)
    plt.scatter(predicted_test[len(ts)-1,:],test_data[len(ts)-1,:],color='r')
    plt.plot(test_data[len(ts)-1,:],test_data[len(ts)-1,:],'black',label='x=y')
    plt.xlabel('Prediction TEST')
    plt.ylabel('TEST True')
    plt.legend()
    #plt.savefig('plot_report/'+'corr'+'index'+str(ind)+'.png')
    plt.title('Train r_2 score ='+str(train_r2_score)+'& Test r_2 score ='+str(test_r2_score))
    
    plt.subplot(2,2,3)
    plt.plot(rho_values,'g')
    plt.grid('on')
    #plt.ylim([0,1])
    plt.ylabel('rho')
    plt.xlabel('SGD Iteration')
    #plt.savefig('plot_report/'+'rho'+'index'+str(ind)+'.png')
    plt.title('Tr mse =%.2f Te mse=%.2f' %(mse_train,mse_test))
    
    plt.subplot(2,2,4)
    plt.plot(para_hist)
    plt.ylabel('Parameter values')
    plt.xlabel('SGD Iteration')
    plt.tight_layout()
    #plt.savefig('plot_report/'+'para'+'index'+str(ind)+'.png')

    s = str(data[ind]['subject'])
    s = s.replace("/","-")
    s = s.replace(".","-")
    
    plt.show()
    #plt.savefig('new_metric/'+s+'index'+str(ind)+str('anl3')+'.png')
    
    """    
    if SMAPE_test_data <= 15 and test_r2_score>=0.2 : # ie model trained well
        plt.savefig('Figs_forecast_new_metric/testing_small_sMAPE/'+s+'index'+str(ind)+'.png')
    else:
        plt.savefig('Figs_forecast_new_metric/testing_big_sMAPE/'+s+'index'+str(ind)+'.png')
        
    #plt.savefig('plot_report/'+s+'index'+str(ind)+'_rkhs_metric'+'.png')
    """
    print("########################  success time series n: %d" %(ind))
    
    return error_metrics

##############################################################################################################################

if __name__ == "__main__":
    
    """
    parameters that you need to change manually in this code:
    
    1/ To read the json file containing the TSDL time series, change the path to your local path
    
    2/ in section:  set parameters  
    ->  delay_opt         : the value of delay, it is also the value of the forecasting horizon
    ->  regu_lambda_opt   : the regularization parameter lambda
    ->  learning_opt      : the step size of SGD
    ->  noptsteps_opt     : number of iterations 
    
    3/ in section: fitting data
    ->  kernel type   : the kernel used in KFs, example "RBF", in the report we use "anl3" kernel
    ->  para_opt      : the hyperparameters of the kernel if the approach is no_learning, example [1,1] if we work with kernel RBF
    
    important remarks: 
    *if you are using the new metric and you have changed the delay value in this file, don't forget to change it in the "fit_rkhs_stdl" file too. 
    *choose your loss function in the "fit_rkhs_stdl" file.
    """

    # the following section enables uss to run time 6 time series at the same (multiprocessing)

    with concurrent.futures.ProcessPoolExecutor(6) as executor: 
        #list1 = range(625,648)
        #unwanted_num = {191,205,207,219,222,225,241,348,417,445,446,447,448,450,458,466,509,511,548,550,561,564,565,566,569,571,572,627,628,633}
        #list1 = [ele for ele in list1 if ele not in unwanted_num]    

        lis=[10]
        result = list(executor.map(ts_multi,lis ))
        
        # use this line to run a set of time series from TSDL
        #result = list(executor.map(ts_multi,range(0,18) )) 
    
    print("result",result)