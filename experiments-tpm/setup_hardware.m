%%% set experimental parameters that are the same for all experiments
% clear wfs; clear feedback;
%% add required pathways
% old_dir = cd;
% git_folder = 'C://git/';
% cd(fileparts(mfilename('fullpath')));

% addpath([git_folder,'AO/Fitting wavefront/functions']);
% addpath(genpath([git_folder,'/hardware/matlab']));
% addpath(genpath([git_folder,'wavefront_shaping/']));
% addpath(genpath([git_folder,'scanimage/']));
% addpath(genpath([git_folder,'tpm/']));
% cd(old_dir); % return to original directory

% load custom circular colormap
% load([git_folder,'\utilities\Colormaps\cyclic_colormap.mat']);

%% check which devices need to be connected for the experiment
if ~exist('active_devices','var')
    active_devices = struct;
end
active_devices = setDefaultDevices(active_devices);

%% connect to SLM
if active_devices.slm
    clear sopt;
    
    % SLM options
    lambda = 700;                           % laser wavelength (in nm)
    sopt.slm_patch = 1;                     % patch number used for wavefront correction
    sopt.N_diameter = 18;                   % number of segments along the diagonal of the SLM pattern
    sopt.displayPort = 1;
    
    % setup SLM
    [slm, sopt] = SLMsetup(lambda,sopt);
    slm.setData(1,0); slm.update;
    fprintf('Initialized SLM with ?=%.1f\n', lambda)
end

%% connect to SLM camera
if active_devices.cam_slm
    clear copt2; clear cam_slm;
    copt2.ExposureTime = 1/60*10^6;
    copt2.Id = 'Camera/22797787:Basler';
    copt2.Width = 1024;
    copt2.Height = 1024;
    cam_slm = Camera(copt2);
    
    % set camera ROI to center of sensor
    copt2.OffsetX = (cam_slm.get('WidthMax') - copt2.Width)/2+35;
    copt2.OffsetY = (cam_slm.get('HeightMax') - copt2.Height)/2+10;
    cam_slm.setROI([copt2.OffsetX, copt2.OffsetY, copt2.Width, copt2.Height]);
    
    % set axes of image plane (after demagnification)
    copt2.dx = 5.5*(200/100);                  % pixel_size at image plane (in um)
    copt2.cam_x = copt2.dx*(-copt2.Width/2+1:copt2.Width/2);
    copt2.cam_y = copt2.dx*(-copt2.Height/2+1:copt2.Height/2);
end

%% connect to SLM camera
if active_devices.cam_slm
    clear copt2; clear cam_slm;
    copt2.ExposureTime = 1/60*10^6;
    copt2.Id = 'Camera/22241376:Basler';
    copt2.Width = 1024;
    copt2.Height = 1024;
    cam_slm = Camera(copt2);
    
    % set camera ROI to center of sensor
    copt2.OffsetX = (cam_slm.get('WidthMax') - copt2.Width)/2+35;
    copt2.OffsetY = (cam_slm.get('HeightMax') - copt2.Height)/2+10;
    cam_slm.setROI([copt2.OffsetX, copt2.OffsetY, copt2.Width, copt2.Height]);
    
    % set axes of image plane (after demagnification)
    copt2.dx = 5.5*(200/100);                  % pixel_size at image plane (in um)
    copt2.cam_x = copt2.dx*(-copt2.Width/2+1:copt2.Width/2);
    copt2.cam_y = copt2.dx*(-copt2.Height/2+1:copt2.Height/2);
end

%% connect to SLM camera
if active_devices.cam_slm
    clear copt2; clear cam_slm;
    copt2.ExposureTime = 1/60*10^6;
    copt2.Id = 'Camera/22241376:Basler';
    copt2.Width = 1024;
    copt2.Height = 1024;
    cam_slm = Camera(copt2);
    
    % set camera ROI to center of sensor
    copt2.OffsetX = (cam_slm.get('WidthMax') - copt2.Width)/2+35;
    copt2.OffsetY = (cam_slm.get('HeightMax') - copt2.Height)/2+10;
    cam_slm.setROI([copt2.OffsetX, copt2.OffsetY, copt2.Width, copt2.Height]);
    
    % set axes of image plane (after demagnification)
    copt2.dx = 5.5*(200/100);                  % pixel_size at image plane (in um)
    copt2.cam_x = copt2.dx*(-copt2.Width/2+1:copt2.Width/2);
    copt2.cam_y = copt2.dx*(-copt2.Height/2+1:copt2.Height/2);
end


%% connect to PMT gain readout channel
if active_devices.pmt_gain
    pmt_gain = daq.createSession('ni');
    addAnalogInputChannel(pmt_gain,'Dev3', 1, 'Voltage');
end

%% connect to sample stage
if active_devices.sample_stage
    s = Serial('COM5', 9600);
    z1 = Zaber(s,1); % motor 1
    z2 = Zaber(s,2); % motor 2

    % limit Zaber travel range to prevent stage from hitting top microscope
    % objective
    z1.travel_range = 4.8e4+[-12.5e3, 12.5e3]; 
    z2.travel_range = 4.25e4+[-12.5e3, 12.5e3]; 
    disp(['Current position Motor 1: ',num2str(z1.getPosition())]);
    disp(['Current position Motor 2: ',num2str(z2.getPosition())]);
end

%% Power monitor
% function to convert power at the beam splitter to power at the sample
% plane (based on measurements performed on 190909)
f_power = @(Pbs)(-5.3780e-07*lambda.^2+7.6990e-04*lambda+-0.0490)*Pbs;

%% Connect to Galvos
daqs = daq.createSession('ni');
daqs.addAnalogOutputChannel('Dev3', 'ao0', 'Voltage');
daqs.addAnalogOutputChannel('Dev3', 'ao1', 'Voltage');
