import torch
import matplotlib.pyplot as plt
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from test_data import azimuths, azimuth_idxs

from utils import *

# Load data
train_data = np.load('spike_data.npy', allow_pickle=True)

# Network parameters
extract_time_ms = 100
extract_time_s = extract_time_ms * 1e-3

classify_time_ms = 50
classify_freq = 10e3  # Hz
classify_dt_s = 1 / classify_freq
classify_dt_ms = classify_dt_s * 1e3
classify_time_s = classify_time_ms * 1e-3

# Assume same number of azimuths on left and right, they are all increasing, and
# there is an odd number, the same amount on either side and zero
num_azimuths = len(azimuths)
num_rfs_right = (num_azimuths+1)//2
rfs_freqs_right = np.linspace(20, 40, num_rfs_right)
rfs_widths_right = np.ones(num_rfs_right) * 2

num_rfs_left = (num_azimuths+1)//2
rfs_freqs_left = np.linspace(20, 40, num_rfs_left)
rfs_widths_left = np.ones(num_rfs_left) * 2        

# Get models
network_extract, network_classify, monitors = get_models(extract_time_ms, classify_time_ms,
                                                         classify_dt_ms, num_rfs_right, num_rfs_left)

# Train
for datapoint in train_data:

    # Get data values
    freq = datapoint['freq']
    azimuth = datapoint['azimuth']
    azimuth_num = datapoint['azimuth_num']
    elevation = datapoint['elevation']
    left_spikes = datapoint['left_spikes']
    right_spikes = datapoint['right_spikes']
    print('Training with parameters:\n\tfrequency={}\n\tazimuth={}\n\televation={}'.format(
        freq, azimuth, elevation))

    # Construct input for extraction network 
    inputs = { 'cochlea_left': left_spikes, 'cochlea_right': right_spikes }

    # Run extraction network
    network_extract.run(inputs=inputs, time=extract_time_ms)

    # Get spikes, voltages, and spike frequencies
    spikes = { 
               'cochlea_left': monitors['cochlea_left'].get('s'),
               'cochlea_right': monitors['cochlea_right'].get('s'),
               'lso_left': monitors['lso_left'].get('s'),
               'lso_right': monitors['lso_right'].get('s')
             }
    voltages = { 
                 'lso_left': monitors['lso_left'].get('v'),
                 'lso_right': monitors['lso_right'].get('v')
               }

    left_spikes_freq = (torch.sum(spikes['lso_left']).item() / extract_time_s)
    right_spikes_freq = (torch.sum(spikes['lso_right']).item() / extract_time_s)
    print('left spikes frequency: {} Hz'.format(left_spikes_freq))
    print('right spikes frequency: {} Hz'.format(right_spikes_freq))

    # Construct inputs for classification network
    rfs_inputs_right = get_rf_node_inputs(rfs_freqs_right, rfs_widths_right,
                                              classify_dt_s, classify_time_s, right_spikes_freq)
    rfs_inputs_left = get_rf_node_inputs(rfs_freqs_left, rfs_widths_left,
                                              classify_dt_s, classify_time_s, left_spikes_freq)
    teacher_inputs_left, teacher_inputs_right = get_teacher_inputs(num_rfs_right, azimuth_num,
                                                                   classify_dt_s, classify_time_s)
    # Run classification network
    classify_inputs = { 'rfs_right': rfs_inputs_right, 'rfs_left': rfs_inputs_left,
                        'teacher_right': teacher_inputs_right, 'teacher_left': teacher_inputs_left }
    network_classify.run(inputs=classify_inputs, time=classify_time_ms)

    # Get rfs and output spikes
    spikes_classify = {
                   'rfs_right': monitors['rfs_right'].get('s'),
                   'rfs_left': monitors['rfs_left'].get('s'),
                   'output_right': monitors['output_right'].get('s'),
                   'output_left': monitors['output_left'].get('s'),
                   'teacher_right': monitors['teacher_right'].get('s'),
                   'teacher_left': monitors['teacher_left'].get('s'),
                 }

    plt.ioff()
    plot_spikes(spikes)
    plot_voltages(voltages)
    plot_spikes(spikes_classify)
    plt.show()



