% Test program
clear; format short e

addpath("/Users/davidmiron/Documents/neuromorphic_computing/UR_EAR_2020b")

% Create dummy signal
max_t = 50 / 1000;          % Max time
fs = 44100;         % Sampling frequency
freq = 100;         % Desired frequency
t = 0:1/fs:max_t;   % Time period
signal = sin(2*pi*freq*t);

% Get inner hair cell (IHC) voltage over time
v_pin = signal;     % input sound wave
v_CF = 1e3;               % characteristic frequency of fiber (Hz)
v_nrep = 10;               % number of repititions for psth
v_dt = 1/fs;              % binsize in seconds, reciprocal of sampling rate
v_reptime = max_t + 0.1;  % time between stimulus repitions in seconds
v_coch = 1;               % OHC scaling factor: 1 for normal, 0 for complete dysfunction
v_cich = 1;               % IHC scaling factor: 1 for normal, 0 for complete dysfunction
v_species = 2;            % model species: 1 for cat, 2 for human with BM Shera et al., 3 for human with BM tuning from Clasberg & Moore

vihc = model_IHC_BEZ2018(v_pin, v_CF, v_nrep, v_dt, v_reptime, v_coch, v_cich, v_species);
%vihc = model_IHC(v_pin, v_CF, v_nrep, v_dt, v_reptime, v_coch, v_cich, v_species);

% Convert inner hair cell voltage to a spike train with synapse model
s_CF = v_CF;            % characteristic frequency of fiber (Hz)
s_nrep = 1;             % number of repititions for psth
s_dt = 1/fs;            % binsize in seconds, reciprocal of sampling rate
s_noiseType = 1;        % 1 for variable fGn, 0 for fixed (frozen) fGn
s_implnt = 0;           % 0 for approximate implementation of power-law functions, 1 for actual
s_spont = 10;           % spontaneous firing rate in /s
s_tabs = 0.7e-3;        % absolute refractory period in s
s_trel = 0.6e-3;        % baselines mean relative refractory period in s

% psth is the peri-stimulus time histogram (PSTH) (or a spike train if nrep = 1)
psth = model_Synapse_BEZ2018(vihc, s_CF, s_nrep, s_dt, s_noiseType, s_implnt, s_spont, s_tabs, s_trel);
%psth = model_Synapse(vihc, s_CF, s_nrep, s_dt, s_noiseType, s_implnt, s_spont, s_tabs, s_trel);

%save("spike_data/psth2.mat", "psth")
