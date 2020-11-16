import sys
import torch
import numpy as np
from scipy import signal
import sounddevice as sd
from utils import *

if len(sys.argv) < 3:
    print("Usage: python gen_data.py [spike data file] [method]")
    sys.exit(1)

spike_file = sys.argv[1]
method = sys.argv[2]
assert(method in ['generate', 'record'])

# Sampling frequencies, hrir is 44.1kHz to match impulse data, cochela is 100kHz to be valid for cochlear model
fs_hrir = 44.1e3 # Hz
fs_cochlea = 100e3 # Hz

# Run time
from test_data import extract_time_ms
time_ms = extract_time_ms   # ms
time_s = time_ms / 1e3    # max time in seconds

# Frequencies to test with
#test_freqs = np.arange(100, 441, 50)
test_freqs = [ 110 ]

# Angles indices to test with
# -80, -45, 0, 45, 80
#azimuths = [0, 3, 12, 21, 24]
from test_data import azimuths, azimuth_idxs

# 0 (?)
elevation_idxs = [8]
elevations = [0]

def generate_spikes(signal_raw, azimuth_idx, elevation_idx):
    # Find ear responses
    signal_left, signal_right = signal_HRTF_transform(signal_raw, azimuth_idx, elevation_idx)

    # Upsample for cochlear model
    #print('HRTF: {}'.format(signal_left.shape))
    signal_left = signal.resample(signal_left, int(fs_cochlea * time_s))
    #print('Resampled: {}'.format(signal_left.shape))
    signal_right = signal.resample(signal_right, int(fs_cochlea * time_s))

    # Find cochlear response
    signal_left_spikes = cochlear_model(signal_left, fs_cochlea, time_s)
    signal_right_spikes = cochlear_model(signal_right, fs_cochlea, time_s)
    signal_left_spikes = torch.tensor(signal_left_spikes).byte().unsqueeze(1)
    signal_right_spikes = torch.tensor(signal_right_spikes).byte().unsqueeze(1)

    return signal_left_spikes, signal_right_spikes

if method == 'record':
    print('Record now')
    signal_raw = sd.rec(int(fs_hrir * time_s), samplerate=fs_hrir, channels=1)
    sd.wait()
    print('Finished recording')
    signal_raw = signal_raw.squeeze()

# azimuth_num is the index in the azimuth angles chosen, between 0 and the number of azimuths
data = []
for azimuth_num, (azimuth_idx, azimuth) in enumerate(zip(azimuth_idxs, azimuths)):
    for elevation_idx, elevation in zip(elevation_idxs, elevations):
        if method == 'generate':
            for freq in test_freqs:
                # Generate audio data
                print('Generating data for frequency={}, azimuth={}, elevation={}'.format(freq, azimuth, elevation))
                signal_raw = get_audio_data(fs=fs_hrir, max_t=time_s, freq=freq)
                signal_left_spikes, signal_right_spikes = generate_spikes(signal_raw, azimuth_idx, elevation_idx)
                data.append({ 'freq': freq, 'azimuth': azimuth, 'azimuth_num': azimuth_num, 'elevation': elevation,
                              'left_spikes': signal_left_spikes, 'right_spikes': signal_right_spikes })
        elif method == 'record':
            signal_left_spikes, signal_right_spikes = generate_spikes(signal_raw, azimuth_idx, elevation_idx)
            data.append({ 'freq': None, 'azimuth': azimuth, 'azimuth_num': azimuth_num, 'elevation': elevation,
                          'left_spikes': signal_left_spikes, 'right_spikes': signal_right_spikes })


data = np.array(data)
np.save(spike_file, data)
