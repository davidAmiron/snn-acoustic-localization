from bindsnet.network.nodes import Nodes
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, IFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.learning import PostPre
from typing import Union, Optional, Iterable
import torch
import numpy as np
import simpleaudio as sa
from scipy.io import loadmat
import matplotlib.pyplot as plt
import ctypes as ct
import pathlib
import matlab.engine

class RFNodes(Nodes):
    """
    Layer of receptive field nodes. Each node responds with a linear spike train to
    a particular operating frequency and width

    Only takes input size of 1
    """

    def __init__(
        self,
        op_freqs: torch.Tensor,
        op_widths: torch.Tensor,
        freq_window_len: int = 5,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        #traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
    ) -> None:
        """
        op_freqs: tensor of operating frequencies for rf nodes
        op_widths: tensor of operating widths for rf nodes
        freq_window_len: length in ms of window for calculating frequency of inputs
        """

        # freqs and widths should be 1d tensors of the same length
        assert(len(op_freqs.shape) == 1)
        assert(op_freqs.shape[0] == op_widths.shape[0])
        assert(op_freqs.shape[0] == n)
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.op_freqs = op_freqs
        self.op_widths = op_widths
        self.freq_window_len = freq_window_len

        self.out_freqs = torch.zeros_like(self.op_freqs)
        self.first_run = True

    def forward(self, x: torch.Tensor) -> None:
        assert(len(x) == 1) # Only work with inputs of size 1

        # Get frequency calculation window timesteps, only on first run
        if self.first_run:
            self.freq_window_timesteps = int(self.freq_window_len / self.dt)

            # tensor with 1 if spike, 0 if not. Location in the array means nothing, will be written
            # to in a circular fasion to save memory.
            self.spike_buffer = torch.zeros(self.freq_window_timesteps) 

            self.spike_buffer_loc = 0                                   
            self.dt_save = self.dt                                      
            self.first_run = False

        assert(self.dt == self.dt_save) # Make sure dt does not change
        
        # Update input frequencies
        self.spike_buffer[self.spike_buffer_loc] = (x[0, 0] != 0) # Only using one item in x, they should all be the same
        if self.spike_buffer_loc == len(self.spike_buffer) - 1:
            self.spike_buffer_loc = 0
        else:
            self.spike_buffer_loc += 1

        num_spikes = torch.sum(self.spike_buffer)
        print(num_spikes)
        self.in_freq = (num_spikes / self.freq_window_len) * 1e3 # In Hz


        self.s = torch.ones(self.n)

        super().forward(x)

def get_rf_node_inputs(freqs, widths, dt, max_t, in_freq, max_out_freq=40):
    assert(len(freqs) == len(widths))
    freqs = np.array(freqs)
    widths = np.array(widths)
    n = len(freqs)
    out_freqs = np.exp(-np.square((in_freq-freqs)/widths)) * max_out_freq
    out_spikes = []
    for f in range(len(freqs)):
        arr = []
        last_spike = 0
        f_dt = 1 / out_freqs[f]
        for t in np.arange(0, max_t, dt):
            if t >= last_spike:
                arr.append(1)
                last_spike += f_dt
            else:
                arr.append(0)
        out_spikes.append(arr)
    return torch.tensor(out_spikes).byte().permute(1, 0)

    


def get_audio_data(fs=100e3, max_t=1, freq=440):
    # Create dummy tone with frequency freq, max time max_t and sampling frequency fs
    t = np.arange(0, max_t, 1/fs)   # Time period
    signal = np.sin(2*np.pi*freq*t)
    return signal

def play_signal(signal, fs):
    signal = signal * (2**15 - 1) / np.max(np.abs(signal))
    signal = signal.astype(np.int16)
    play_obj = sa.play_buffer(signal, 1, 2, fs)
    play_obj.wait_done()

def signal_HRTF_transform(signal, azn, eln):
    """
    signal_hrtf_transform Transform a signal with a HRTF
       Take in a signal, azimuthal index, and elevation index, and return
       the signal convolved with the HRIRs for subject 040
    """
    hrir_data = loadmat('/Users/davidmiron/Documents/neuromorphic_computing/CIPIC_hrtf_database/standard_hrir_database/subject_040/hrir_final.mat')
    hrir_l = hrir_data['hrir_l']
    hrir_r = hrir_data['hrir_r']

    ir_left = hrir_l[azn, eln, :]
    ir_right = hrir_r[azn, eln, :]
    
    signal_left = np.convolve(ir_left, signal)
    signal_right = np.convolve(ir_right, signal)

    return signal_left, signal_right


"""def cochlear_model(signal_raw, fs, max_t):
    # Get the model
    libname = pathlib.Path().absolute() / "cochlear_model/libcochlea.so"
    lib_cochlea = ct.CDLL(libname)
    lib_cochlea.IHCAN.restype = None
    #lib_cochlea.IHCAN.argtypes = [ ct.POINTER(ct.c_double), ct.c_double, ct.c_int, ct.c_double, ct.c_int,
    #                               ct.c_double, ct.c_double, ct.c_int, ct.POINTER(ct.c_double) ]

    # Args
    signal = signal_raw.tolist()####HERE
    v_pin = (ct.c_double * len(signal))(*signal)     # input sound wave
    v_pinlen = ct.c_int(len(signal))
    v_CF = ct.c_double(1e3)               # characteristic frequency of fiber (Hz)
    nrep = 1
    v_nrep = ct.c_int(nrep)               # number of repititions for psth
    dt = 1/fs
    v_dt = ct.c_double(dt)              # binsize in seconds, reciprocal of sampling rate
    reptime = max_t + 0.1
    v_reptime = ct.c_double(reptime)  # time between stimulus repitions in seconds
    v_coch = ct.c_int(1)               # OHC scaling factor: 1 for normal, 0 for complete dysfunction
    v_cich = ct.c_int(1)               # IHC scaling factor: 1 for normal, 0 for complete dysfunction
    v_species = ct.c_int(2)            # model species: 1 for cat, 2 for human with BM Shera et al., 3 for human with BM tuning from Clasberg & Moore
    ihc_len = int(np.floor(reptime/dt+0.5)) * nrep
    ihc_python = [1] * ihc_len
    v_ihcout = (ct.c_double * ihc_len)(*ihc_python)
    lib_cochlea.wrapper_IHCAN(v_pin, v_pinlen, v_CF, v_nrep, v_dt, v_reptime, v_coch, v_cich, v_species, v_ihcout)

    psth_len = int(np.floor(ihc_len/nrep))

    plt.ioff()
    plt.plot(v_ihcout)
    plt.show()

#    a = ct.POINTER(ct.c_double)()
#    b = ct.POINTER(ct.c_double)()
    #print(dir(a))
    #print(lib_cochlea.IHCAN(a, 1, 0, 0, 0, 0, 0, 0, b))
#    lib_cochlea.argtypes = (ct.c_float)
#    c_float_init = ct.c_float * 2
#    arr = c_float_init(7.2, 3.9)
#    print(lib_cochlea.testfun(ct.c_float(5), arr))
#    print('arr:', arr[0], arr[1])"""

def cochlear_model(signal_raw, fs, max_t):
    eng = matlab.engine.start_matlab()
    eng.addpath('/Users/davidmiron/Documents/neuromorphic_computing/UR_EAR_2020b')

    # Get IHC voltage
    signal = signal_raw.tolist()
    v_pin = matlab.double(signal_raw.tolist())
    v_cf = float(1e3)               # characteristic frequency of fiber (Hz)
    v_nrep = float(1)
    v_dt = float(1/fs)
    v_reptime = float(max_t+0.01)
    v_coch = float(1)               # OHC scaling factor: 1 for normal, 0 for complete dysfunction
    v_cich = float(1)               # IHC scaling factor: 1 for normal, 0 for complete dysfunction
    v_species = float(2)            # model species: 1 for cat, 2 for human with BM Shera et al., 3 for human with BM tuning from Clasberg & Moore
    vihc = eng.model_IHC_BEZ2018(v_pin, v_cf, v_nrep, v_dt, v_reptime, v_coch, v_cich, v_species)
    vihc = np.asarray(vihc[0])
    #vihc.fill(0)

    # Get spike train
    s_vihc = matlab.double(vihc.tolist())
    s_cf = v_cf
    s_nrep = v_nrep
    s_dt = v_dt
    s_noiseType = float(1)
    s_implnt = float(1)
    s_spont = float(10)
    s_tabs = float(0.007e-3)
    s_trel = float(0.006e-3)

    psth = eng.model_Synapse_BEZ2018(s_vihc, s_cf, s_nrep, s_dt, s_noiseType, s_implnt, s_spont, s_tabs, s_trel)
    psth = np.asarray(psth[0])
    #plt.ioff()
    #plt.plot(vihc)
    #plt.show()
    #eng.quit()
    return psth

"""def get_spike_data():
    # For now just return one spike data while I get the rest of the network running
    return loadmat('spike_data/psth3.mat')['psth'][0]"""

def get_models(extract_time_ms, classify_time_ms, classify_dt_ms, num_rfs_right, num_rfs_left, traces=True):

    ###
    ## General Setup
    ###
    time_s = extract_time_ms / 1e3    # max time in seconds
    fs = 100e3 # Hz
    dt = (1/fs)*1e3    # ms
    timesteps = int(extract_time_ms/dt)
    network_extract = Network(dt=dt)
    print('Network timesteps: {}'.format(timesteps))

    monitors = {}

    ###
    ## Cochlear model
    ###
    l_cochlea_left = Input(n=1, traces=traces)
    l_cochlea_right = Input(n=1, traces=traces)
    network_extract.add_layer(layer=l_cochlea_left, name='cochlea_left')
    network_extract.add_layer(layer=l_cochlea_right, name='cochlea_right')

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
    #l_lso_left = LIFNodes(n=1, refrac=0.05, tc_delay=1, traces=traces)
    #l_lso_right = LIFNodes(n=1, refrac=0.05, tc_delay=1, traces=traces)
    l_lso_left = LIFNodes(n=1, traces=traces)
    l_lso_right = LIFNodes(n=1, traces=traces)
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
    )
    mon_lso_right = Monitor(
        obj=l_lso_right,
        state_vars=('s', 'v')
    )
    mon_cochlea_left = Monitor(
        obj=l_cochlea_left,
        state_vars=('s')
    )
    mon_cochlea_right = Monitor(
        obj=l_cochlea_right,
        state_vars=('s')
    )
    """mon_misc = Monitor(
        obj=l_cochlea_left,
        state_vars=('s')
    )"""
    monitors['lso_left'] = mon_lso_left
    monitors['lso_right'] = mon_lso_right
    monitors['cochlea_left'] = mon_cochlea_left
    monitors['cochlea_right'] = mon_cochlea_right

    network_extract.add_monitor(monitor=mon_lso_left, name='lso_left')
    network_extract.add_monitor(monitor=mon_lso_right, name='lso_right')
    network_extract.add_monitor(monitor=mon_cochlea_left, name='cochlea_left')
    network_extract.add_monitor(monitor=mon_cochlea_right, name='cochlea_right')
    #network_extract.add_monitor(monitor=mon_misc, name='cochlea_left')

    ###
    ## Classification Network
    ###

    # Receptive Fields
    classify_freq = 10e3  # Hz
    classify_dt_s = 1 / classify_freq
    classify_dt_ms = classify_dt_s * 1e3
    classify_time_s = classify_time_ms * 1e-3

    network_classify = Network(dt=classify_dt_ms)

    # Add right RFS nodes
    l_rfs_right = Input(n=num_rfs_right, traces=traces)
    network_classify.add_layer(layer=l_rfs_right, name='rfs_right')

    mon_rfs_right = Monitor(
        obj=l_rfs_right,
        state_vars=('s')
    )
    network_classify.add_monitor(monitor=mon_rfs_right, name='rfs_right')
    monitors['rfs_right'] = mon_rfs_right

    # Add left RFS Nodes
    l_rfs_left = Input(n=num_rfs_left, traces=traces)
    network_classify.add_layer(layer=l_rfs_left, name='rfs_left')

    mon_rfs_left = Monitor(
        obj=l_rfs_left,
        state_vars=('s')
    )
    network_classify.add_monitor(monitor=mon_rfs_left, name='rfs_left')
    monitors['rfs_left'] = mon_rfs_left

    # Add teacher nodes
    l_teacher_right = Input(n=num_rfs_right, traces=traces)
    l_teacher_left = Input(n=num_rfs_left, traces=traces)
    network_classify.add_layer(layer=l_teacher_right, name='teacher_right')
    network_classify.add_layer(layer=l_teacher_left, name='teacher_left')

    # Add output nodes
    l_output_right = LIFNodes(n=num_rfs_right, traces=traces)
    l_output_left = LIFNodes(n=num_rfs_left, traces=traces)
    network_classify.add_layer(layer=l_output_right, name='output_right')
    network_classify.add_layer(layer=l_output_left, name='output_left')

    # Add connections to output nodes
    c_rfsright_outputright = Connection(
        source=l_rfs_right,
        target=l_output_right,
        w=torch.ones(num_rfs_right, num_rfs_right),
        update_rule=PostPre
    )
    c_teacherright_outputright = Connection(
        source=l_teacher_right,
        target=l_output_right,
        w=torch.ones(num_rfs_right, num_rfs_right) * 5
    )

    c_rfsleft_outputleft = Connection(
        source=l_rfs_left,
        target=l_output_left,
        w=torch.ones(num_rfs_left, num_rfs_left),
        update_rule=PostPre
    )
    c_teacherleft_outputleft = Connection(
        source=l_teacher_left,
        target=l_output_left,
        w=torch.ones(num_rfs_left, num_rfs_left) * 5
    )

    network_classify.add_connection(c_rfsright_outputright, source='rfs_right', target='output_right')
    network_classify.add_connection(c_teacherright_outputright, source='teacher_right', target='output_right')
    network_classify.add_connection(c_rfsleft_outputleft, source='rfs_left', target='output_left')
    network_classify.add_connection(c_teacherleft_outputleft, source='teacher_left', target='output_left')

    # Add teacher monitors
    mon_teacher_right = Monitor(
        obj=l_teacher_right,
        state_vars=('s')
    )
    mon_teacher_left = Monitor(
        obj=l_teacher_left,
        state_vars=('s')
    )

    monitors['teacher_right'] = mon_teacher_right
    monitors['teacher_left'] = mon_teacher_left
    network_classify.add_monitor(monitor=mon_teacher_right, name='teacher_right')
    network_classify.add_monitor(monitor=mon_teacher_left, name='teacher_left')

    # Add output monitors
    mon_output_right = Monitor(
        obj=l_output_right,
        state_vars=('s')
    )
    mon_output_left = Monitor(
        obj=l_output_left,
        state_vars=('s')
    )

    monitors['output_right'] = mon_output_right
    monitors['output_left'] = mon_output_left
    network_classify.add_monitor(monitor=mon_output_right, name='output_right')
    network_classify.add_monitor(monitor=mon_output_left, name='output_left')

    return network_extract, network_classify, monitors

def get_teacher_inputs(num_outputs_per_side, azimuth_num, dt, max_t, spike_freq=3e3):
    # spike_freq is Hz of teacher signal
    # Put high frequency linear spike train in location for the angle
    # [ l0, l1, l2, middle, r0, r1, r2 ] <- azimuth_num indexes into this
    # num_outputs_per_side is maximum azimuth_num/2, for this would be 4   <-- this might be wrong
    #
    # Frequency ordering
    # left side: [ 0, -this, -max ]
    # right side: [ 0, +this, +max ]
    #
    #

    assert(azimuth_num < (2 * num_outputs_per_side) - 1)
    times = np.arange(0, max_t, dt)

    # Generate teacher spike train
    teacher_spikes = []
    teacher_dt = 1/spike_freq
    last_spike = 0
    for t in times:
        if t >= last_spike:
            teacher_spikes.append(1)
            last_spike += teacher_dt
        else:
            teacher_spikes.append(0)
    teacher_spikes = torch.tensor(teacher_spikes).byte()

    right_spikes = torch.zeros(len(times), num_outputs_per_side).byte()
    left_spikes = torch.zeros(len(times), num_outputs_per_side).byte()

    if azimuth_num < num_outputs_per_side-1:
        left_spikes[:, azimuth_num + 1] = teacher_spikes
    elif azimuth_num == num_outputs_per_side-1:
        left_spikes[:, 0] = teacher_spikes
        right_spikes[:, 0] = teacher_spikes
    else:
        right_spikes[:, azimuth_num - num_outputs_per_side + 1] = teacher_spikes

    return left_spikes, right_spikes




    






if __name__ == '__main__':
    inputs = get_rf_node_inputs([5, 10], [2, 2], 1/100, 2, 6)
    print(inputs)
    print(torch.sum(inputs, axis=1))

    """psth = cochlear_model(*get_audio_data())
    #psth = get_spike_data()
    print(psth)
    print(type(psth))
    plt.ioff()
    plt.plot(psth)
    plt.show()"""


    """tone, sampling_freq = get_audio_data()
    tone_left, tone_right = signal_HRTF_transform(tone, 8, 1)
    play_signal(tone_left, sampling_freq)
    plt.ioff()
    plt.figure(1)
    plt.plot(tone[1000:1500])
    plt.figure(2)
    plt.plot(tone_left[1000:1500])
    plt.show()"""
