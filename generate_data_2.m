addpath("/Users/davidmiron/Documents/neuromorphic_computing/UR_EAR_2020b")

CF    = 1.0e3;   % CF in Hz;   
cohc  = 1.0;    % normal ohc function
cihc  = 1.0;    % normal ihc function
species = 1;    % 1 for cat (2 for human with Shera et al. tuning; 3 for human with Glasberg & Moore tuning)
noiseType = 1;  % 1 for variable fGn (0 for fixed fGn)
fiberType = 3;  % spontaneous rate (in spikes/s) of the fiber BEFORE refractory effects; "1" = Low; "2" = Medium; "3" = High
implnt = 0;     % "0" for approximate or "1" for actual implementation of the power-law functions in the Synapse
% stimulus parameters
F0 = CF;     % stimulus frequency in Hz
Fs = 100e3;  % sampling rate in Hz (must be 100, 200 or 500 kHz)
T  = 10e-3;  % stimulus duration in seconds
rt = 2.5e-3; % rise/fall time in seconds
stimdb = 70; % stimulus intensity in dB SPL
% PSTH parameters
nrep = 500;               % number of stimulus repetitions (e.g., 50);
psthbinwidth = 0.05e-3; % binwidth in seconds;

t = 0:1/Fs:T-1/Fs; % time vector
mxpts = length(t);
irpts = rt*Fs;
% Create Stimulus waveform
pin = sqrt(2)*20e-6*10^(stimdb/20)*sin(2*pi*F0*t); % unramped stimulus
pin(1:irpts)= pin(1:irpts).*(0:(irpts-1))/irpts; 
pin((mxpts-irpts):mxpts)=pin((mxpts-irpts):mxpts).*(irpts:-1:0)/irpts;

vihc = model_IHC(pin,CF,nrep,1/Fs,T*2,cohc,cihc,species); % Computes Voltage of inner hair cell
%[vihc,bmout] = model_IHC_BM(pin,CF,nrep,1/Fs,T*2,cohc,cihc,species); % Computes Voltage of inner hair cell
[meanrate,varrate,psth] = model_Synapse(vihc,CF,nrep,1/Fs,fiberType,noiseType,implnt); % Compute synpase response

% Do stupid thing, make psth any number equal to 1
psth = int8(psth > 0);

save("spike_data/psth3.mat", "psth")
