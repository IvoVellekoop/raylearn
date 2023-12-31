% Setup paths and hardware for raylearn experiments in Two Photon Microscope
clear; clc

% Define location paths
addpath(fileparts(fileparts(mfilename('fullpath'))))   % Grand parent dir
dirconfig_raylearn


%% Setup hardware
active_devices.slm = true;
active_devices.galvos = false;
active_devices.cam_img = false;
active_devices.cam_ft = false;
active_devices.cam_slm = false;
active_devices.pmt_gain = true;
active_devices.sample_stage = false;

setup_hardware
fprintf('\n')
