%% connect to SLM
LUTlocation = fullfile(repodir, 'Experiments/LUT.mat');
disp(['Loading SLM Lookup Table from ' LUTlocation])
load(LUTlocation);

disp('Connecting to SLM...')
displayPort = 3;                    % Check this number every time after restarting!
slm = SLM(displayPort);
slm.setData(0,0);
slm.update;

alpha = LUT(end);                   % gray value corresponding to 2pi phase shift
slm.setLUT(LUT');                   % set look up table for specific SLM at specific wavelength
slm.t_idle = 2;                     % time for SLM to start responding to new signal (in number of vsyncs)
slm.t_settle = 2;                   % time for SLM settle to new value (in number of vsyncs)
slm_res = slm.getSize;              % resolution of SLM
