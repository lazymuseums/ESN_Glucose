'''
    Relationship between number of nodes and error
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from model import ESN, RLS, LMS
import openpyxl

# from model import ESN, Tikhonov

from lowpass import butter_lowpass_filter
from kalman import kalman_filter
np.random.seed(seed=0)


# Read data from file
def read_sugar_data(file_name):
    '''
    :input: file_name
    :output: data
    '''
    time=np.empty(0)
    data = np.empty(0)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split()
            time = np.hstack((time, float(tmp[0])))
            data = np.hstack((data, float(tmp[1])))  # 2rd column
    return time, data

def interpolate_data(time, data):
    # Construct interpolation function
    f = interp1d(time, data, kind='linear')

    # Interpolation operation
    interpolated_time = np.linspace(time[0], time[-1], num=(len(time)-1)*5 + 1)
    interpolated_data = f(interpolated_time)

    return interpolated_time, interpolated_data


if __name__ == '__main__':
    # Blood glucose data
    time, bloodsugar = read_sugar_data(file_name='blood_sugar(9).txt')
    data = bloodsugar
    
    # Make 5 min/step to 1 min/step:
    # time, data = interpolate_data(time, data)
    
    # Delete the # at the beginning of the corresponding line if need to use a filter:
    # Kalman filter
    # process_variance = 1e-2
    # measurement_variance = 0.1
    # data = kalman_filter(bloodsugar, process_variance, measurement_variance)

    # Low-Pass Filter
    # data = butter_lowpass_filter(bloodsugar, cutoff=1/12.5, fs=1/5, order=10)

    # Multi-step ahead prediction
    step = 2

    T_train = 1600
    T_test = data.size - T_train - step


    # Data for training and testing
    train_U = data[:T_train].reshape(-1, 1)
    train_D = data[step:T_train+step].reshape(-1, 1)
    test_U = data[T_train:T_train + T_test].reshape(-1, 1)
    test_D = data[T_train + step:T_train + T_test + step].reshape(-1, 1)

    workbook = openpyxl.Workbook()
    sheet = workbook.active

    # Increase the number of nodes to 300 in steps of 5
    for N_x in range(5,301,5):
        # For each number of nodes take the result of multiple seed
        for seed in range(0,201,10):
            # ESN model
            model = ESN(train_U.shape[1], train_D.shape[1], N_x, density=0.1, leaking_rate=0.9,
                        fb_scale=1,fb_seed=0,input_scale=0.01, rho=0.9,seed=seed)
            #online RLS
            train_Y, Wout_size,Wout = model.adapt(train_U, train_D,
                                             RLS(N_x, train_D.shape[0], delta=1e-4,
                                                 lam=1, update=1))
            # online LMS
            # train_Y, Wout_size,Wout = model.adapt(train_U, train_D,
            #                                  LMS(N_x, train_D.shape[0], step_size=0.003))

            test_Y = model.RLS_predict(test_U,Wout)
  
            # Training error
            RMSE = np.sqrt(((train_D - train_Y) ** 2)
                           .mean())
            NRMSE = RMSE / np.sqrt(np.var(train_D))
            print('node:', N_x)
            print('seed:', seed)
            # print(step, 'step(s) ahead prediction')
            # print('Training error: RMSE =', RMSE)
            # print('Training error: NRMSE =', NRMSE)

            # Testing error
            RMSE = np.sqrt(((test_D - test_Y) ** 2)
                           .mean())
            NRMSE = RMSE / np.sqrt(np.var(test_D))
            print(step, 'step(s) ahead prediction')
            print('Testing error: RMSE =', RMSE)
            print('Testing error: NRMSE =', NRMSE)

            sheet.cell(row=N_x/5, column=seed/10+1, value=RMSE)
    
    # RMSE save to Excel(row=node/5, column=seed/10+1)
    workbook.save('node.xlsx')

    # Data for graph display
    T_disp = (200, 200) # (last 200 of training data, first 200 of testing data)
    print('disp:',T_disp)
    t_axis = np.arange(0,T_disp[0]+T_disp[1], 1)
    disp_D_step1 = np.concatenate((train_D[train_D.size - T_disp[0]:],
                                   test_D[:T_disp[1]]))
    disp_Y_step1 = np.concatenate((train_Y[train_Y.shape[0] - T_disp[0]:],
                                   test_Y[:T_disp[1]]))
    disp_U_step1 = np.concatenate((train_U[train_U.size - T_disp[0]:],
                                   test_U[:T_disp[1]]))


    # Graph display
    plt.rcParams['font.size'] = 18
    fig = plt.figure(figsize=(8, 7))
    plt.subplots_adjust(hspace=0.3)

    # Overall blood glucose
    ax1 = fig.add_subplot(2, 1, 1)
    x=range(0, 250)
    x_scaled = [i / 12 for i in x]
    plt.plot(data, color='k', linewidth=1)
    plt.xlabel('step(/5 min)')
    plt.ylabel('blood glucose(mg/dL)')

    # Prediction graph
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.text(-0.15, 1, '', transform=ax2.transAxes)
    ax2.grid(True)
    plt.plot(t_axis*5/60, disp_D_step1, color='k', linewidth=1, label='target')
    plt.plot(t_axis*5/60, disp_Y_step1[:,0], color='red', linestyle='--', linewidth=1, label='prediction')
    plt.plot(t_axis * 5/60 , disp_U_step1[:, 0], color='b', linestyle=':', linewidth=1, label='filtered data')

    plt.xlabel('time(hours)')
    plt.ylabel('glucose prediction(mg/dL)')

    plt.legend(loc='upper center')

    plt.show()

