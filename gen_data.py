import torch
import numpy as np
from utils import *

# Sampling frequency
fs = 100e3 # Hz

# Run time
time_ms = 100   # ms
time_s = time_ms / 1e3    # max time in seconds

# Frequencies to test with
#test_freqs = np.arange(100, 441, 50)
test_freqs = [100, 150, 200]

# Angles indices to test with
# -80, -45, 0, 45, 80
#azimuths = [0, 3, 12, 21, 24]
azimuth_idxs = [3, 12, 21]
azimuths = [-45, 0, 45]

# 0 (?)
elevation_idxs = [8]
elevations = [0]


data = []
for freq in test_freqs:
    for azimuth_idx, azimuth in zip(azimuth_idxs, azimuths):
        for elevation_idx, elevation in zip(elevation_idxs, elevations):
            print('Generating data for frequency={}, azimuth={}, elevation={}'.format(freq, azimuth, elevation))
            signal_raw = get_audio_data(fs=fs, max_t=time_s, freq=freq)
            signal_left, signal_right = signal_HRTF_transform(signal_raw, azimuth_idx, elevation_idx)
            signal_left_spikes = cochlear_model(signal_left, fs, time_s)
            signal_right_spikes = cochlear_model(signal_right, fs, time_s)
            signal_left_spikes = torch.tensor(signal_left_spikes).byte().unsqueeze(1)
            signal_right_spikes = torch.tensor(signal_right_spikes).byte().unsqueeze(1)
            data.append({ 'freq': freq, 'azimuth': azimuth, 'elevation': elevation,
                          'left_spikes': signal_left_spikes, 'right_spikes': signal_right_spikes })
        break
    break


data = np.array(data)
np.save('spike_data.npy', data)
