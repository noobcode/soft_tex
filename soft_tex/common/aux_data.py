import numpy as np
import pandas as pd
import scipy.io as sio
import os


# function to remove outliers in sensors readings, interpolate to remove nans and transform V to Ohms
def prepr_sens(sensors):
    # eliminate outliers
    sensors = np.where(sensors>3, np.nan, sensors) # 3V
    sensors = np.where(sensors<1, np.nan, sensors) # 1V
    _pd_m = pd.DataFrame(sensors)
    _pd_m = _pd_m.interpolate()
    sensors = np.array(_pd_m)
    # from voltage to resistance
    V_in = 3.2 # V (input voltage) 
    R_k = 12   # Ohm (known resistance)
    sens_raw = (sensors*R_k) / (V_in - sensors)

    return sens_raw


def get_dataset_dict(data_dir_path, dataset_names):
    dataset_dict = {}

    for dataset_name in dataset_names:
        # load dataset
        dataset_path = os.path.join(data_dir_path, dataset_name)
        prepr_dataset_circ = sio.loadmat(dataset_path)['Dataset'] # (time, 3 pos and 3 sens)

        # extract positions and sensor values
        pos_out = prepr_dataset_circ[:, 3:6]  # positions, (time, 3) - xyz?
        sens_in = prepr_dataset_circ[:, 0:3]  # sensor, (time, 3)

        # preprocess sensor (remove nan, convert voltage to resistance)
        sens_circ = prepr_sens(sens_in)

        # save dataset in dictionary
        dataset_dict[dataset_name] = {}
        dataset_dict[dataset_name]['sensor_voltage'] = sens_in
        dataset_dict[dataset_name]['sensor_resistance'] = sens_circ
        dataset_dict[dataset_name]['tip_position'] = pos_out

    return dataset_dict


def exponential_moving_average(x, alpha):
    assert(type(alpha) == float or len(alpha) == x.shape[1]) 
    assert(np.all(alpha < 1))
    assert(np.all(alpha >  0))
    
    s = np.zeros_like(x)
    
    s[0,:] = x[0,:]
    for t in range(1, len(x), 1):
        s[t,:] = alpha * x[t,:] + (1 - alpha) * s[t-1,:]

    return s
    