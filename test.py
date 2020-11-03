import torch
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages

from utils import *


time = 900   # ms (?)
dt = 0.01    # ms
timesteps = int(time/dt)
network = Network()

# Input node
l_in_node = Input(n=1, traces=True) # Input for now for testing
network.add_layer(layer=l_in_node, name='in_node')

in_data = torch.tensor(get_spike_data()).byte().unsqueeze(1)
#in_data = torch.bernoulli(0.3 * torch.ones(timesteps, 1)).byte()
inputs = { 'in_node': in_data }

print(in_data)
print(torch.sum(in_data))

# LIF Node
l_lso_left = LIFNodes(n=1, refrac=0.05, tc_delay=1, traces=True)

mon_in = Monitor(
    obj=l_in_node,
    state_vars=('s'),
    time=time
)
network.add_monitor(monitor=mon_in, name='mon_in')

network.run(inputs=inputs, time=time)
spikes = { 'input': mon_in.get('s') }

plt.ioff()
plot_spikes(spikes)
plt.show()
