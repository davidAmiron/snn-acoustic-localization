import torch
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre, Hebbian

# Simulation time
time = 1500

# Create the network
net = Network()

# Create layers
source_layer = Input(n=100, traces=True)
target_layer = LIFNodes(n=1000, traces=True)

# Add layers
net.add_layer(layer=source_layer, name='A')
net.add_layer(layer=target_layer, name='B')

# Create connection
connection = Connection(
    source=source_layer,
    target=target_layer,
    w=0.01 + 0.1 * torch.randn(source_layer.n, target_layer.n),
    update_rule=PostPre,
    nu=(1e-4, 1e-2)
)

# Add connection
net.add_connection(connection, source='A', target='B')

# Create and add input and output monitors
source_monitor = Monitor(
    obj=source_layer,
    state_vars=('s'), # Record spikes
    time=time
)

target_monitor = Monitor(
    obj=target_layer,
    state_vars=('s', 'v'), # Record spikes and voltages
    time=time
)

net.add_monitor(monitor=source_monitor, name='A')
net.add_monitor(monitor=target_monitor, name='B')

# Create input data
input_data = torch.bernoulli(0.1 * torch.ones(time, source_layer.n)).byte()
inputs = {'A': input_data}

# Run network
net.run(inputs=inputs, time=time)

# Get spike data
spikes = {'A': source_monitor.get('s'), 'B': target_monitor.get('s')}
voltages = {'B': target_monitor.get('v')}

# Plot data
plt.ioff()
plot_spikes(spikes)
plot_voltages(voltages, plot_type='line')
plt.show()
