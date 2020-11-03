signal = [1, 2, 3, 4, 5, 6, 7];
fs = 20;
max_t = 0.5;

% Get inner hair cell (IHC) voltage over time
v_pin = signal;     % input sound wave
v_CF = 1e3;               % characteristic frequency of fiber (Hz)
v_nrep = 1;               % number of repititions for psth
v_dt = 1/fs;              % binsize in seconds, reciprocal of sampling rate
v_reptime = max_t + 0.1;  % time between stimulus repitions in seconds
v_coch = 1;               % OHC scaling factor: 1 for normal, 0 for complete dysfunction
v_cich = 1;               % IHC scaling factor: 1 for normal, 0 for complete dysfunction
v_species = 2;            % model species: 1 for cat, 2 for human with BM Shera et al., 3 for human with BM tuning from Clasberg & Moore

vihc = model_IHC_BEZ2018(v_pin, v_CF, v_nrep, v_dt, v_reptime, v_coch, v_cich, v_species);
display(vihc(11))
