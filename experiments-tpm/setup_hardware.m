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
    error('Specify which devices to activate with a struct named ''active_devices''.')
end


%% connect to SLM
if active_devices.slm
    clear sopt;
    
    % SLM options
    lambda = 715;                           % laser wavelength (in nm)
    sopt.slm_patch = 1;                     % patch number used for wavefront correction
    sopt.N_diameter = 18;                   % number of segments along the diagonal of the SLM pattern
    sopt.displayPort = 1;
    
    % setup SLM
    [slm, sopt] = SLMsetup(lambda,sopt);
    slm.setData(1,0); slm.update;
    fprintf('Initialized SLM with \x3BB=%.1f\n', lambda)
end

%% Connect to Fourier Plane Camera
if active_devices.cam_ft
    copt_ft.ExposureTime = 1/60*10^6;
    copt_ft.Id = 'Camera/22797787:Basler';

    copt_ft.Width = 1088;
    copt_ft.Height = 1088;
    cam_ft = Camera(copt_ft);
    
    % set camera ROI to center of sensor
    copt_ft.OffsetX = (cam_ft.get('WidthMax') - copt_ft.Width)/2;
    copt_ft.OffsetY = (cam_ft.get('HeightMax') - copt_ft.Height)/2;
    cam_ft.setROI([copt_ft.OffsetX, copt_ft.OffsetY, copt_ft.Width, copt_ft.Height]);
    
    % set axes of image plane (after demagnification)
    copt_ft.dx = 5.5*(200/100);                  % pixel_size at image plane (in um)
    copt_ft.cam_x = copt_ft.dx*(-copt_ft.Width/2+1:copt_ft.Width/2);
    copt_ft.cam_y = copt_ft.dx*(-copt_ft.Height/2+1:copt_ft.Height/2);
    fprintf('Initialized Fourier Plane Camera\n')
end

%% Connect to Image Plane Camera
if active_devices.cam_img
    copt_img.ExposureTime = 1/60*10^6;
    copt_img.Id = 'Camera/23572269:Basler';
    copt_img.Width = 1088;
    copt_img.Height = 1088;
    cam_img = Camera(copt_img);
    
    % set camera ROI to center of sensor
    copt_img.OffsetX = (cam_img.get('WidthMax') - copt_img.Width)/2;
    copt_img.OffsetY = (cam_img.get('HeightMax') - copt_img.Height)/2;
    cam_img.setROI([copt_img.OffsetX, copt_img.OffsetY, copt_img.Width, copt_img.Height]);
    
    % set axes of image plane (after demagnification)
    copt_img.dx = 5.5*(200/100);                  % pixel_size at image plane (in um)
    copt_img.cam_x = copt_img.dx*(-copt_img.Width/2+1:copt_img.Width/2);
    copt_img.cam_y = copt_img.dx*(-copt_img.Height/2+1:copt_img.Height/2);
    fprintf('Initialized Image Plane Camera\n')
end


%% connect to PMT gain readout channel
if active_devices.pmt_gain
    pmt_gain = daq.createSession('ni');
    addAnalogInputChannel(pmt_gain,'Dev3', 1, 'Voltage'); %%%%%
    fprintf('Initialized PMT gain readout\n')
end

%% connect to sample stage
if active_devices.sample_stage
    s = Serial('COM5', 9600);
    z1 = Zaber(s,1); % motor 1
    z2 = Zaber(s,2); % motor 2

    % limit Zaber travel range to prevent stage from hitting top microscope
    % objective
    z1.travel_range = [51840 140000];
    z2.travel_range = [640 203520]; 
    disp(['Current position Motor 1: ',num2str(z1.getPosition())]);
    disp(['Current position Motor 2: ',num2str(z2.getPosition())]);
    fprintf('Initialized sample stage\n')
end

%% Power monitor
% function to convert power at the beam splitter to power at the sample
% plane (based on measurements performed on 190909)
f_power = @(Pbs)(-5.3780e-07*lambda.^2+7.6990e-04*lambda+-0.0490)*Pbs;

%% Connect to Galvos
if active_devices.galvos
    daqs = daq.createSession('ni');
    daqs.addAnalogOutputChannel('Dev4', 'ao2', 'Voltage');
    daqs.addAnalogOutputChannel('Dev4', 'ao3', 'Voltage');
    fprintf('Initialized Galvo Mirrors\n')
end

