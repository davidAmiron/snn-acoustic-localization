from bindsnet.network.nodes import Nodes
from typing import Union, Optional, Iterable
import torch
import numpy as np
import simpleaudio as sa
from scipy.io import loadmat
import matplotlib.pyplot as plt
import ctypes as ct
import pathlib

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
            traces=True,
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

def get_audio_data():
    # Create dummy signal
    max_t = 2           # Max time
    fs = 44100          # Sampling frequency
    freq = 440          # Desired frequency
    t = np.arange(0, max_t, 1/fs)   # Time period
    signal = np.sin(2*np.pi*freq*t)
    return signal, fs, max_t

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
    hrir_data = loadmat('/Users/davidmiron/Downloads/CIPIC_hrtf_database/standard_hrir_database/subject_040/hrir_final.mat')
    hrir_l = hrir_data['hrir_l']
    hrir_r = hrir_data['hrir_r']

    ir_left = hrir_l[azn, eln, :]
    ir_right = hrir_r[azn, eln, :]
    
    signal_left = np.convolve(ir_left, signal)
    signal_right = np.convolve(signal, ir_right)

    return signal_left, signal_right


def cochlear_model(signal_raw, fs, max_t):
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
    print(v_ihcout[200])
    lib_cochlea.wrapper(v_pin, v_pinlen, v_CF, v_nrep, v_dt, v_reptime, v_coch, v_cich, v_species, v_ihcout)
    print(v_ihcout[200])
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
#    print('arr:', arr[0], arr[1])


if __name__ == '__main__':
    cochlear_model(*get_audio_data())

    """tone, sampling_freq = get_audio_data()
    tone_left, tone_right = signal_HRTF_transform(tone, 8, 1)
    play_signal(tone_left, sampling_freq)
    plt.ioff()
    plt.figure(1)
    plt.plot(tone[1000:1500])
    plt.figure(2)
    plt.plot(tone_left[1000:1500])
    plt.show()"""
