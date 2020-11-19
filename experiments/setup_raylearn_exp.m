% Setup paths and hardware for raylearn experiment

% Define location paths
addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
dirconfig_raylearn

% Setup hardware
setup_cam
setup_slm
setup_steppermotor
fprintf('\n')
