%% connect to SLM
LUTlocation = 'PCT/Experiments/LUT.mat';
disp(['Loading SLM Lookup Table from ' LUTlocation])
mfilepath = fileparts(mfilename('fullpath'));
load(fullfile(mfilepath, '../..', LUTlocation));

disp('Connecting to SLM...')
displayPort = 2;
slm = SLM(displayPort);
slm.setData(0,0);
slm.update;

alpha = LUT(end);                   % gray value corresponding to 2pi phase shift
slm.setLUT(LUT');                   % set look up table for specific SLM at specific wavelength
slm.t_idle = 2;                     % time for SLM to start responding to new signal (in number of vsyncs)
slm.t_settle = 2;                   % time for SLM settle to new value (in number of vsyncs)
slm_res = slm.getSize;              % resolution of SLM
