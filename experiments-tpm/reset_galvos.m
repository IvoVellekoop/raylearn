% Set Galvos to (0, 0) position

doreset = 0;

if doreset || ~exist('daqs', 'var')
    close all; clear; clc
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Grand parent dir
    dirconfig_raylearn
    
    active_devices.slm = false;
    active_devices.galvos = true;
    active_devices.cam_img = false;
    active_devices.cam_ft = false;
    active_devices.pmt_gain = false;
    active_devices.sample_stage = false;

    setup_hardware
    fprintf('\n')
end

outputSingleScan(daqs, [0, 0]);
