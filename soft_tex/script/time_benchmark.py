import sys
sys.path.append('../..')

import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import os
import pandas as pd
import scipy.signal as scs
from tqdm import tqdm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime

# soft_tex imports
from soft_tex.model.parallel_net import ParallelSoftSensingLSTM, ParallelSoftSensingGRU
from soft_tex.common.aux_data import get_dataset_dict, exponential_moving_average


data_dir_path = '../../data/SoftTex/'

"""
1) LOAD DATASETS
"""
# Load datasets for sensor voltage and robot positions
dataset_names = ['dataset_res_sens_pos_1.mat', 'dataset_res_sens_pos_2.mat',  'dataset_res_sens_pos_3.mat']

dataset_dict = get_dataset_dict(data_dir_path=data_dir_path, dataset_names=dataset_names)

# Concatenate pressure dataset
pressure_dataset_names = ['dataset_act_sens_pos_1.mat', 'dataset_act_sens_pos_2.mat', 'dataset_act_sens_pos_3.mat']

for dataset_name, pressure_dataset_name in zip(dataset_names, pressure_dataset_names):
    dataset = sio.loadmat(data_dir_path + pressure_dataset_name)['Dataset']
    pressures = dataset[:, [0,1,2]]

    dataset_dict[dataset_name]['pressure'] = pressures

"""
2) PREPROCESS DATASETS
"""
# Split training and validation set and preprocess # 3 1 2
TR_IDX = 3
VL_IDX = 1
TS_IDX = 2
tr_dataset = dataset_dict['dataset_res_sens_pos_%d.mat' % TR_IDX]
vl_dataset = dataset_dict['dataset_res_sens_pos_%d.mat' % VL_IDX]
ts_dataset = dataset_dict['dataset_res_sens_pos_%d.mat' % TS_IDX]

# 3 time series of shape (time, 3)
tr_sensor, tr_pressure, tr_tip_position = tr_dataset['sensor_resistance'], tr_dataset['pressure'], tr_dataset['tip_position']
vl_sensor, vl_pressure, vl_tip_position = vl_dataset['sensor_resistance'], vl_dataset['pressure'], vl_dataset['tip_position']
ts_sensor, ts_pressure, ts_tip_position = ts_dataset['sensor_resistance'], ts_dataset['pressure'], ts_dataset['tip_position']

# shift minimum resistance to 0
tr_sensor -= np.min(tr_sensor, axis=0)
vl_sensor -= np.min(vl_sensor, axis=0)
ts_sensor -= np.min(ts_sensor, axis=0)

# concatenate pressures and sensor
X_tr_series = np.concatenate((tr_pressure, tr_sensor), axis=1)
X_vl_series = np.concatenate((vl_pressure, vl_sensor), axis=1)
X_ts_series = np.concatenate((ts_pressure, ts_sensor), axis=1)
Y_tr_series = tr_tip_position
Y_vl_series = vl_tip_position
Y_ts_series = ts_tip_position

print("Dataset shapes")
print("TR:", X_tr_series.shape, Y_tr_series.shape)
print("VL:", X_vl_series.shape, Y_vl_series.shape)
print("TS:", X_ts_series.shape, Y_ts_series.shape)

# Create development set, create scalers and fit the scalers
# Fit the scalers on the original data;
# Apply EMA on the original data
# Scale the smoothed data.
X_development = np.concatenate((X_tr_series, X_vl_series, X_ts_series))
Y_development = np.concatenate((Y_tr_series, Y_vl_series, Y_ts_series))

observation_scaler = StandardScaler().fit(X_development)
output_scaler = StandardScaler().fit(Y_development)

# apply EMA (alpha=0.6 for pressures, alpha=0.4 for resistance)
alpha = np.array([0.6, 0.6, 0.6, 0.4, 0.4, 0.4])
X_tr_series = exponential_moving_average(X_tr_series, alpha=alpha)
X_vl_series = exponential_moving_average(X_vl_series, alpha=alpha)

# Scale training set and validation set and transform in Torch Tensors
X_tr_series = th.tensor(observation_scaler.transform(X_tr_series), dtype=th.float32)
Y_tr_series = th.tensor(output_scaler.transform(Y_tr_series), dtype=th.float32)


def count_parameters(net):
    counter = 0
    for parameters in net.parameters():
        counter += np.prod(parameters.shape)
    
    return counter


def compute_dataset_inference_time(net, X, n_trials):
    # reset network states and predict
    net.eval()

    with th.no_grad():
        execution_times = []
        for i in range(n_trials):
            net.reset_states()
            tic = time.time()
            Y_hats = net.predict(X)
            toc = time.time()

            execution_times.append(toc - tic)
    
    inference_time = {
        'mean': np.mean(execution_times),
        'std': np.std(execution_times)}
    
    return inference_time

"""
3) BENCHMARK COMPUTING TIME FOR GPU AND CPU
"""
models = [ParallelSoftSensingGRU, ParallelSoftSensingLSTM]
n_units = [8, 16, 32, 64]
devices = [th.device('cuda'), th.device('cpu')]
n_trials = 10

cpu_results_dict = {model.__name__: dict() for model in models}

for model in cpu_results_dict:
    for n_unit in n_units:
        cpu_results_dict[model][n_unit] = {}

# CPU: compute inference time for whole dataset execution
device = th.device('cpu')
print("Using", device)
for model_class in models:
    for n_unit in n_units:
        # create network
        net = model_class(input_size_1=3, input_size_2=3, output_size=3, 
                            hidden_size_1=n_unit, hidden_size_2=n_unit, num_layers=1)
    
        # device transfer
        net.to(device)
        X_tr_series = X_tr_series.to(device)

        # compute inference time on training set
        inference_time = compute_dataset_inference_time(net, X_tr_series.unsqueeze(1), n_trials)
        cpu_results_dict[model_class.__name__][n_unit]['inference_time'] = inference_time
        # compute number of parameters
        cpu_results_dict[model_class.__name__][n_unit]['parameters_count'] = count_parameters(net)

# GPU: compute inference time for whole dataset execution
device = th.device('cuda')
print("Using", device)

gpu_results_dict = {model.__name__: dict() for model in models}

for model in gpu_results_dict:
    for n_unit in n_units:
        gpu_results_dict[model][n_unit] = {}

for model_class in models:
    for n_unit in n_units:
        # create network
        net = model_class(input_size_1=3, input_size_2=3, output_size=3, 
                            hidden_size_1=n_unit, hidden_size_2=n_unit, num_layers=1)
    
        # device transfer
        net.to(device)
        X_tr_series = X_tr_series.to(device)

        # compute inference time on training set
        inference_time = compute_dataset_inference_time(net, X_tr_series.unsqueeze(1), n_trials)
        gpu_results_dict[model_class.__name__][n_unit]['inference_time'] = inference_time
        # compute number of parameters
        gpu_results_dict[model_class.__name__][n_unit]['parameters_count'] = count_parameters(net)


"""
4) SAVE benchmark
"""
results_dict = {
    'cpu': cpu_results_dict,
    'gpu': gpu_results_dict}

results_path = 'time_benchmark.pth'

th.save(results_dict, results_path)

print("Results saved at", results_path)