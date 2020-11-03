import torch
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, IFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages

from utils import *

###
## General Setup
###

time_ms = 150   # ms
time_s = time_ms / 1e3    # max time in seconds
fs = 100e3 # Hz
dt = (1/fs)*1e3    # ms
timesteps = int(time_ms/dt)
network_extract = Network(dt=dt)
print('Network timesteps: {}'.format(timesteps))

###
## Layers
###

###
## Input
###
signal_raw = get_audio_data(fs=fs, max_t=time_s)
signal_left, signal_right = signal_HRTF_transform(signal_raw, 24, 1)
signal_left_spikes = cochlear_model(signal_left, fs, time_s)
signal_right_spikes = cochlear_model(signal_right, fs, time_s)

print('Raw lengths: {}'.format(signal_raw.shape[0]))
print('HRTF lengths: left={}, right={}'.format(signal_left.shape[0], signal_right.shape[0]))
print('Spike lengths: left={}, right={}'.format(signal_left_spikes.shape[0], signal_right_spikes.shape[0]))
#assert(False)

###
## Cochlear model
###
l_cochlea_left = Input(n=1, traces=True) # Input for now for testing
l_cochlea_right = Input(n=1, traces=True) # Input for now for testing
network_extract.add_layer(layer=l_cochlea_left, name='cochlea_left')
network_extract.add_layer(layer=l_cochlea_right, name='cochlea_right')

# Temorary dummy input data for cochlea
#input_data_cleft = torch.bernoulli(0.9 * torch.ones(timesteps, 1)).byte()
input_data_cright = torch.bernoulli(0.3 * torch.ones(timesteps, 1)).byte()
#inputs = { 'cochlea_left': input_data_cleft, 'cochlea_right': input_data_cright }

# Get temporary test data from spikes from tone made for testing
#in_data = torch.tensor(cochlear_model(*get_audio_data())).byte().unsqueeze(1)
input_data_left = torch.tensor(signal_left_spikes).byte().unsqueeze(1)
input_data_right = torch.tensor(signal_right_spikes).byte().unsqueeze(1)
inputs = { 'cochlea_left': input_data_left, 'cochlea_right': input_data_right }
print('sum: {}'.format(torch.sum(input_data_left)))
print('bernoulli sum: {}'.format(torch.sum(input_data_cright)))
print('shape: {}'.format(input_data_left.shape))
print('bernoulli shape: {}'.format(input_data_cright.shape))
#assert(False)

"""print(signal_raw.shape)
print(signal_left.shape)
print(signal_left_spikes.shape)

plt.ioff()
fig, axs = plt.subplots(2, 1)
axs[0].plot(input_data_left, '.')
axs[0].set_title('input_data_left')
axs[1].plot(input_data_right, '.')
axs[1].set_title('input_data_right')
plt.show()
assert(False)
"""

###
## LSO
###

# Two LIF neurons with excitatory synapse from ipsilateral cochlear model,
# and one synapse from the contralateral cochlear model
#l_lso_left = LIFNodes(n=1, refrac=0.05, tc_delay=1, traces=True)
#l_lso_right = LIFNodes(n=1, refrac=0.05, tc_delay=1, traces=True)
l_lso_left = LIFNodes(n=1, traces=True)
l_lso_right = LIFNodes(n=1, traces=True)
network_extract.add_layer(layer=l_lso_left, name='lso_left')
network_extract.add_layer(layer=l_lso_right, name='lso_right')

c_cleft_lsoleft = Connection(
    source=l_cochlea_left,
    target=l_lso_left,
    w=torch.tensor([[5.0]]),
    wmin=0
)

c_cright_lsoleft = Connection(
    source=l_cochlea_right,
    target=l_lso_left,
    w=torch.tensor([[-0.5]]),
    wmax=0
)

c_cleft_lsoright = Connection(
    source=l_cochlea_left,
    target=l_lso_right,
    w=torch.tensor([[-0.5]]),
    wmax=0
)

c_cright_lsoright = Connection(
    source=l_cochlea_right,
    target=l_lso_right,
    w=torch.tensor([[5.0]]),
    wmin=0
)

network_extract.add_connection(c_cleft_lsoleft, source='cochlea_left', target='lso_left')
network_extract.add_connection(c_cright_lsoleft, source='cochlea_right', target='lso_left')
network_extract.add_connection(c_cleft_lsoright, source='cochlea_left', target='lso_right')
network_extract.add_connection(c_cright_lsoright, source='cochlea_right', target='lso_right')


mon_lso_left = Monitor(
    obj=l_lso_left,
    state_vars=('s', 'v')
#    time=time_ms
)
mon_lso_right = Monitor(
    obj=l_lso_right,
    state_vars=('s', 'v')
#    time=time_ms
)
mon_cochlea_left = Monitor(
    obj=l_cochlea_left,
    state_vars=('s')
#    time=time_ms
)
mon_cochlea_right = Monitor(
    obj=l_cochlea_right,
    state_vars=('s')
#    time=time_ms
)
"""mon_misc = Monitor(
    obj=l_cochlea_left,
    state_vars=('s'),
    time=time_ms
)"""

network_extract.add_monitor(monitor=mon_lso_left, name='lso_left')
network_extract.add_monitor(monitor=mon_lso_right, name='lso_right')
network_extract.add_monitor(monitor=mon_cochlea_left, name='cochlea_left')
network_extract.add_monitor(monitor=mon_cochlea_right, name='cochlea_right')
#network_extract.add_monitor(monitor=mon_misc, name='cochlea_left')

###
## Run Extraction Network
###

network_extract.run(inputs=inputs, time=time_ms)
spikes = { 'cochlea_left': mon_cochlea_left.get('s'), 'cochlea_right': mon_cochlea_right.get('s'),
           'lso_left': mon_lso_left.get('s'), 'lso_right': mon_lso_right.get('s') }
voltages = { 'lso_left': mon_lso_left.get('v'), 'lso_right': mon_lso_right.get('v') }

left_spikes_freq = (torch.sum(spikes['lso_left']).item() / time_s)
right_spikes_freq = (torch.sum(spikes['lso_right']).item() / time_s)
print('left spikes frequency: {} Hz'.format(left_spikes_freq))
print('right spikes frequency: {} Hz'.format(right_spikes_freq))

###
## Receptive Fields
###
#l_rfs = RFNodes(torch.ones(3), torch.ones(3), n=3)
#network.add_layer(layer=l_rfs, name='rfs')
#
#c_rfs = Connection(
#    source=l_lso_left,
#    target=l_rfs,
#    w=torch.ones(l_lso_left.n, l_rfs.n)
#)
#network.add_connection(c_rfs, source='lso_left', target='rfs')


plt.ioff()
plot_spikes(spikes)
plot_voltages(voltages)
plt.show()
