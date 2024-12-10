import sys
sys.path.append('/home/carlo/Documents/Lavoro/PhD/Progetti/SoftTex')

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


data_dir_path = '/home/carlo/Documents/Lavoro/PhD/Progetti/SoftTex/data/SoftTex/'
device = th.device('cpu')

print('Device used is:', device)


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
X_tr_series = th.tensor(observation_scaler.transform(X_tr_series), dtype=th.float32).to(device)
Y_tr_series = th.tensor(output_scaler.transform(Y_tr_series), dtype=th.float32).to(device)

X_vl_series = th.tensor(observation_scaler.transform(X_vl_series), dtype=th.float32).to(device)
Y_vl_series = th.tensor(output_scaler.transform(Y_vl_series), dtype=th.float32).to(device)


"""
3) DEFINE NETWORK, LOSS FUNCTION, AND OPTIMIZER
"""
#net = SoftSensingLSTM(input_size=6, output_size=3, hidden_size=64, num_layers=2, dropout=0.5, device=device)
net = ParallelSoftSensingGRU(input_size_1=3, input_size_2=3, output_size=3, 
                            hidden_size_1=50, hidden_size_2=50, num_layers=1, 
                            dropout=0.5, bidirectional=False, input_dropout_1=0.0, input_dropout_2=0.0, device=device)
print(net)

loss_fn = nn.MSELoss()
optimizer = opt.Adam(net.parameters(), lr=1e-4, weight_decay=1.2e-3, betas=(0.999, 0.999)) # betas=(0.9, 0.999)
#weight_decay=1.2e-3 # curriculum 3.4

"""
4 ) TRAINING
"""
# training with randomized sequence length and shift
#sequence_lengths = np.arange(10,200,10) # 200 per 3.9
sequence_shift_factors = [1] #[0.9, 1] per 3.9 e 3.4
history = None
n_epochs = 3
n_randomizations = 2#150 # 150 per 3.9, 180 per 3.4

X_vl_series, Y_vl_series = X_vl_series.unsqueeze(1), Y_vl_series.unsqueeze(1)

for i in tqdm(range(n_randomizations), desc='Randomizations'): 
    # sample sequence parameters (50 e i 20 per 3.9)
    #sequence_len = 20 + 5 * int(i / 20) # curriculum 3.4
    sequence_len = 10 + 5 * int(i / 20) # curriculum 3.4
    #sequence_len = 40

    #20 if i < 20 else np.random.choice(sequence_lengths)
    sequence_shift = int(np.random.choice(sequence_shift_factors) * sequence_len)

    # create dataset with randomized sequence length
    X_unfold, Y_unfold = net.unfold_dataset(X=X_tr_series, Y=Y_tr_series, 
                                            sequence_len=sequence_len, sequence_shift=sequence_shift)

    # fit for some epochs
    history = net.fit(X_unfold, Y_unfold, loss_fn, optimizer, n_epochs, 
                        validation_data=(X_vl_series, Y_vl_series), 
                        history=history, 
                        X_noise_scale=(1e-1, 1e-1, 1e-1, 2e-3, 2e-3, 2e-3), # era 2e-3 per 3.9
                        Y_noise_scale=3e-2)
        
"""
5) SAVE TRAINING RESULTS AND MODEL
"""
# save model and training history
today = datetime.today().strftime("date_%Y.%m.%d_h_%H.%M")
results_path = '/home/carlo/Documents/Lavoro/PhD/Progetti/SoftTex/trainings/'
results_path += 'training' + '_' + today + '.pth'
#history_path = model_path + '_history'

results_dict = {
    'net': net,
    'optimizer': optimizer,
    'history': history
}

th.save(results_dict, results_path)
#np.save(history_path, history)

print("Model saved at:", results_path)
#print("History saved at:", history_path + '.npy')

