% Setup paths and hardware for raylearn experiments in Two Photon Microscope
clear; close all; clc

% Define location paths
addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
dirconfig_raylearn


%% Setup hardware
active_devices.slm = true;
active_devices.galvos = true;
active_devices.cam_img = true;
active_devices.cam_ft = true;
active_devices.pmt_gain = false;
active_devices.sample_stage = false;

setup_hardware
fprintf('\n')
