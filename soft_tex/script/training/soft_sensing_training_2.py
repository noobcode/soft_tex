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

from soft_tex.model.networks import SoftSensingLSTM, ParallelSoftSensingLSTM
import soft_tex.common.aux_plot as aux_plot
from soft_tex.common.aux_data import get_dataset_dict
from datetime import datetime

data_dir_path = './data/LycraSTIFF-FLOP/'

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
TR_IDX, VL_IDX, TS_IDX = 1, 3, 2 # these datasets have respectively 10801, 16201, 5401 samples
training_dataset = dataset_dict['dataset_res_sens_pos_%d.mat' % TR_IDX]
validation_dataset = dataset_dict['dataset_res_sens_pos_%d.mat' % VL_IDX]
test_dataset = dataset_dict['dataset_res_sens_pos_%d.mat' % TS_IDX]

# Training
tr_sensor = training_dataset['sensor_resistance']  # (time, 3)
tr_pressure = training_dataset['pressure']
tr_tip_position = training_dataset['tip_position'] # (time, 3)

# Validation
vl_sensor = validation_dataset['sensor_resistance']   # (time, 3)
vl_pressure = validation_dataset['pressure']
vl_tip_position = validation_dataset['tip_position'] # (time, 3)

# Test
ts_sensor = test_dataset['sensor_resistance']   # (time, 3)
ts_pressure = test_dataset['pressure']
ts_tip_position = test_dataset['tip_position'] # (time, 3)

## concatenate pressures and sensor
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
X_development = np.concatenate((X_tr_series, X_vl_series, X_ts_series))
Y_development = np.concatenate((Y_tr_series, Y_vl_series, Y_ts_series))

observation_scaler = MinMaxScaler(feature_range=(-1,1)) #MinMaxScaler(feature_range=(-1,1)) #StandardScaler()
output_scaler = MinMaxScaler(feature_range=(-1,1)) #MinMaxScaler(feature_range=(-1,1)) #StandardScaler()

observation_scaler.fit(X_development)
output_scaler.fit(Y_development)

# Scale training set and validation set and transform in Torch Tensors
X_tr_series = th.tensor(observation_scaler.transform(X_tr_series), dtype=th.float32).to(device)
Y_tr_series = th.tensor(output_scaler.transform(Y_tr_series), dtype=th.float32).to(device)

X_vl_series = th.tensor(observation_scaler.transform(X_vl_series), dtype=th.float32).to(device)
Y_vl_series = th.tensor(output_scaler.transform(Y_vl_series), dtype=th.float32).to(device)

# duplicate data creating batches, so that the input dropout does not cause catastrophic forgetting
# Since the datasets have different lengths we do a simple trick to create batches learned in parallel.
# We train on two dataset selected as TR and VL in parallel up to the common number of samples. Then validate on
# the whole VL set which is the longest and the final part of it was not used for training (as expected).
common_n_samples = np.min([tr_pressure.shape[0], vl_pressure.shape[0]])
X_tr_series = th.concatenate([X_tr_series[:common_n_samples].unsqueeze(1), 
                              X_vl_series[:common_n_samples].unsqueeze(1)], dim=1)
Y_tr_series = th.concatenate([Y_tr_series[:common_n_samples].unsqueeze(1), 
                              Y_vl_series[:common_n_samples].unsqueeze(1)], dim=1)

print("Actual training data size:", X_tr_series.shape, Y_tr_series.shape)
print("VL set", X_vl_series.shape, Y_vl_series.shape)
#batch_size = 1
#X_tr_series = th.concatenate([X_tr_series.unsqueeze(1) for _ in range(batch_size)], dim=1)
#Y_tr_series = th.concatenate([Y_tr_series.unsqueeze(1) for _ in range(batch_size)], dim=1)

"""
3) DEFINE NETWORK, LOSS FUNCTION, AND OPTIMIZER
"""
#net = SoftSensingLSTM(input_size=6, output_size=3, hidden_size=64, num_layers=2, dropout=0.5, device=device)
net = ParallelSoftSensingLSTM(input_size_1=3, input_size_2=3, output_size=3, 
                              hidden_size_1=32, hidden_size_2=32, num_layers=1, 
                              dropout=0.5, bidirectional=False, input_dropout_1=0.0, input_dropout_2=0.0, device=device)
print(net)

loss_fn = nn.MSELoss()
optimizer = opt.Adam(net.parameters(), lr=5e-5, weight_decay=1e-4, betas=(0.999, 0.999)) # betas=(0.9, 0.999)

#lr = 5e-5 per 3.9
"""
4) TRAINING
0 -> with fixed sequence length and sequence shift
1 -> with randomized sequence length and sequence shift
"""
TRAINING_MODE = 1

if TRAINING_MODE == 0:
    sequence_len = 50 # (equivalent to 0.5 seconds)
    sequence_shift = 10 # steps to shift sequence

    X_unfold, Y_unfold = net.unfold_dataset(X=X_tr_series, Y=Y_tr_series, 
                                            sequence_len=sequence_len, sequence_shift=sequence_shift)
    X_vl_series, Y_vl_series = X_vl_series.unsqueeze(1), Y_vl_series.unsqueeze(1)

    n_epochs = 10
    history = net.fit(X_unfold, Y_unfold, loss_fn, optimizer, n_epochs, 
                      validation_data=(X_vl_series, Y_vl_series),
                      X_noise_scale=(2e-2, 2e-2, 2e-2, 3e-3, 3e-3, 3e-3), Y_noise_scale=2e-3)
    
if TRAINING_MODE == 1:
    # training with randomized sequence length and shift
    sequence_lengths = np.arange(10,100,10) # 200 per 3.9
    sequence_shift_factors = [0.9, 1]#[0.9, 1] per 3.9
    history = None
    n_epochs = 3
    n_randomizations = 100 # 150 per 3.9

    X_vl_series, Y_vl_series = X_vl_series.unsqueeze(1), Y_vl_series.unsqueeze(1)

    for i in tqdm(range(n_randomizations), desc='Randomizations'): 
        # sample sequence parameters (i 20 per 3.9)
        sequence_len = 50 if i < 20 else np.random.choice(sequence_lengths)
        sequence_shift = 50 if i < 20 else int(np.random.choice(sequence_shift_factors) * sequence_len)

        # create dataset with randomized sequence length
        X_unfold, Y_unfold = net.unfold_dataset(X=X_tr_series, Y=Y_tr_series, 
                                                sequence_len=sequence_len, sequence_shift=sequence_shift)
    
        # fit for some epochs
        history = net.fit(X_unfold, Y_unfold, loss_fn, optimizer, n_epochs, 
                          validation_data=(X_vl_series, Y_vl_series), 
                          history=history, 
                          X_noise_scale=(1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2), # era 2e-3 per 3.9
                          Y_noise_scale=2e-3)
        
"""
5) SAVE TRAINING RESULTS AND MODEL
"""
# save model and training history
today = datetime.today().strftime("date_%Y.%m.%d_h_%H.%M")
model_path = './models/'
model_path += 'trained_model' + '_' + today
history_path = model_path + '_history'

th.save(net, model_path)
np.save(history_path, history)

print("Model saved at:", model_path)
print("History saved at:", history_path + '.npy')

