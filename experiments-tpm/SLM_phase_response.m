%%% This script is used to generate data that can be used 
%      1. To verify whether the software LUT is generating the correct 
%           phase shift(by running with the software LUT enabled)
%      2. To LUT for the SLM

% Notes: 
% SETUP:  MIchelson interferometer after the SLM
% LASER (MaiTai): 
%   ## To not burn the camera
%         = LOWEST POWER (~0.1 mW Green power), continuous mode
%   ## To get an interference pattern even with large path length
%         difference between the 2 arms of Michelson interferometer
%         = CONTINUOUS MODE (Pump optimization : OFF; 
%                           Modelocker Enabled : OFF)
% CAMERA:
%   ## Pylon software
%       -View the intereference pattern in the Pylon viewer.
%       -Change the exposure time / laser power so that the image is not over exposed
%       -Use the same value/60 in the if-condition for active_devices.cam in
%           ./setup.m
%       - Disconnect the camera from the Pylon viewer (otherwise, matlab cannot access it)
%    ## Matlab 
%       C:\git\tpm\setup\setDefaultDevices.m : Enable the camera
%       {make default_devices to be 'true' for the entry corresponding to the 
%       position of cam in the list devices}
%       ./setup.m : if-condition for active_devices.cam is running with the correct
%           serial number (copt.Id = 'Camera/xxxxxxxx:Basler';
%           currently, it is copt.Id = 'Camera/22961593:Basler';)
% SLM:
% Case1: To verify whether the generated phase shift is correct
%       load the appropriate LUT into the SLM using blink 
%           (Eg: C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\LUT Files\corrected_0degrees_0800nm.blt)
%       Use the software LUT as well (with the correct wavelength)
% Case2: To produce data to generate a new LUT for the SLM
%       load the linear LUT into the SLM using blink
%           (Eg: C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\LUT
%           Files\linear.blt)
%       Disable the software LUT by removing (commenting) 
%       slm.setLUT(LUT) in the file C:\git\tpm\setup\SLMsetup.m


% TROUBLE SHOOTS 
%   - make sure that activedevices.cam=1
%       -else, make sure that the camera is enabled (refer the same section of comments)
%       -delete the already existing struct active_devices & rerun the
%         script
%   - make sure that the camera is disconnected from other softwares (pylon Viewer)
%   - use the correct setup file (./setup.m)


%% House keeping
clc;

% clear all; % Matlab geting stuck when you clear the variables!
% close all
addpath('C:\git\utilities');
addpath('C:\git\LayerByLayerRecon\Dependencies');
details = get_details();
% details.result_folder = 'P:\TNW\BMPI\Users\Harish Sasikumar\Data\Exp Results\ZH161_2PhotonSetup\SLM_Camera'; % Name of the subfolder in which results are saved
details.result_folder = fullfile(dirs.expdata, 'raylearn-data/TPM/SLM_LUT_calibration'); % Name of the subfolder in which results are saved


%% Type1 LUT: existing one from the "setup.m" (that you will be opening in the next line)
% Note: Confirm that you are using the correct lambda in the setup file
% [filename, dirname] = uigetfile('*setup.m');
% run([dirname,filename]);

%% Type2 LUT: custom software LUT 
% to test and replace the one used in 'C:\git\tpm\setup\SLMsetup.m'
% Comment this section if you do not want to do this

% lambda = 804;
% GV_skip = 2;                            % first two gray values are skipped because they introduce a 0.4 rad jump
% alpha = 189;          % SLM gray value corresponding with 2pi
% LUT = linspace(GV_skip+1, alpha+GV_skip, 256);
% slm.setLUT(LUT); 

%% Type3 LUT: to test the LUT for the Complete Range
% to calibrate the software LUT - GV_skip, sopt.alpha, etc used 
% in 'C:\git\tpm\setup\SLMsetup.m'
% Comment this section if you do not want to do this

% LUT = 0:255;
% slm.setLUT(LUT); 

%% Experiment parameters
gray_values = (0:255)';    % gray values tested     
n_exp = 4;              % number of times experiment is performed
pause_time = 0.1;       % waiting time between every measurement (in sec)
excluded_from_fft = 5;  % first 'excluded_from_fft' sample is excluded in finding FFT maximum

%% Find frequency corresponding to fringes
% project black image onto the SLM
slm.setData(0,0); slm.update; pause(1);

%% For virtual Camera inputs (remove in actual TPM setup)
% cam.trigger = 'camera triggered';
% cam.getData = ones(512,512)


%% capture frame and 
cam_slm.trigger;
frame = double(cam_slm.getData);

%% separate the top and bottom part of the image
frame_top = frame(1:end/4,:);
frame_bottom = frame(end*3/4+1:end, :);

% find spatial frequencies of fringes in the bottom and top part
frame_top_fft = fft2(frame_top-mean2(frame_top));
frame_bottom_fft = fft2(frame_bottom-mean2(frame_bottom));
max_freq_top = find_max_index(frame_top_fft, excluded_from_fft);
max_freq_bottom = find_max_index(frame_bottom_fft, excluded_from_fft);

%% Perform experiment
% initialization
phase_response = zeros(length(gray_values),n_exp);
modulation_depth = zeros(length(gray_values),n_exp);
frames = zeros(size(frame,1),size(frame,2), length(gray_values));

figure()

for exp_i = 1:n_exp
    % wait for SLM to return to zero voltage
    wavefront = zeros(4);
    slm.setData(0,0); slm.update;
    pause(1);
    disp(['starting experiment: ',num2str(exp_i),'/',num2str(n_exp)])
    
    for gray_i = 1:length(gray_values)
        % set frame to SLM
        pattern = [0,gray_values(gray_i)]';
        slm.setData(0,pattern); slm.update;
%         slm.wait(slm.t_idle+slm.t_settle);
        pause(0.1)
        % To verify the pixel values
        pixels = slm.getPixels();
        pix_min(exp_i,gray_i) = min(min(pixels));
        pix_max(exp_i,gray_i) = max(max(pixels));
       
        % Capture frame and compute mean signal from top and bottom part
        cam_slm.trigger;
        frame = double(cam_slm.getData);        
        frames(:,:,gray_i,exp_i)=frame;
        
        imagesc(frame); title(gray_i); colormap(gray); drawnow;
    end
end

%% Saving the results into a subfolder
% mkdir(details.result_folder);
% details.current_folder = pwd;
% cd(details.result_folder)
% display(['..Saving results in the sub folder:' details.result_folder])
% file_name_to_save = ['SLM_Camera ' details.date_and_time '.mat'];
% save('-v7.3', file_name_to_save,'details','pix_min','pix_max','frames','n_exp','gray_values','excluded_from_fft','max_freq_top','max_freq_bottom');
% cd(details.current_folder)

%% Analysis
for exp_i = 1:n_exp
    disp(['starting the analysis of experiment: ',num2str(exp_i),'/',num2str(n_exp)])
    for gray_i = 1:length(gray_values)
        frame=frames(:,:,gray_i,exp_i);

        frame_top = frame(1:end*1/4, :);
        frame_bottom = frame(end*3/4+1:end, :);
     
        % 2D Fourier transform bottom and top part of the frame
        frame_top_fft = fft2(frame_top);
        frame_bottom_fft = fft2(frame_bottom);
     
        % Compute phase difference between top and bottom
        phase_response(gray_i,exp_i) = angle(frame_top_fft(max_freq_top)) - angle(frame_bottom_fft(max_freq_bottom));
        modulation_depth(gray_i,exp_i) = min(abs(frame_top_fft(max_freq_top)),abs(frame_bottom_fft(max_freq_bottom)));
    end
    phase_response(:,exp_i) = unwrap(phase_response(:,exp_i) - phase_response(1,exp_i));
end

% compute mean and standard deviation of phase response
if phase_response(end) < 0 % make sure that phase is always increasing
    phase_mean = -mean(phase_response,2);
else
    phase_mean = mean(phase_response,2);
end
phase_std = std(phase_response,0,2);

%% retrieve field modulated field
E_s = mean(modulation_depth,2).*exp(1.0i*phase_mean);
E_s = E_s/max(abs(E_s(:)));

%% calculate phase corresponding to 2pi (using linear fit)
f = fit(gray_values,phase_mean,'a*x+b','Start',[0,0]);
alpha = round((2*pi-f.b)/f.a);
p_linear = f.a*gray_values+f.b;
p_error = mean(abs(phase_mean-p_linear));

%% plot phase response

figure(); clf
plot(gray_values,phase_response); hold on
plot([0 255], 2*pi*[1 1], 'g'); hold off
xlabel('Gray value');
ylabel('Measured phase shift (rad)');
title('Phase responses from individual experiments');
axis square; grid on;

figure(); clf; hold on;

% plot data with errorbars
errorbar(gray_values,phase_mean,phase_std,'s'); hold on;

% plot linear fitted response
plot(gray_values,p_linear,'--k','LineWidth',2);
plot([0 255], 2*pi*[1 1], 'g')

% figure layout
legend('Measured','linear fit', '2\pi','Location','SouthEast');
xlabel('Gray value');
ylabel('Measured phase shift (rad)');
title('SLM phase response');
set(gca,'FontSize',14);
xlim([gray_values(1), gray_values(end)]);
grid on


% plot modulated field in complex plane
figure(); clf;
plot(E_s,'-*b','LineWidth',2); hold on;
plot(real(E_s(1)),imag(E_s(1)),'*r',real(E_s(end)),imag(E_s(end)),'*k','LineWidth',2);
xlabel('Re\{E\}'); ylabel('Im\{E\}');
title('Modulated field');
set(gca,'FontSize',14);
xlim([-1.1,1.1]); ylim([-1.1,1.1])
axis square; grid on;

%% Saving the analysis results into the same subfolder
details.Date_and_time = date;
cd(details.result_folder)
display(['..Saving alayzed results in the sub folder:' details.result_folder])
file_name_to_save = ['SLM_Camera_analyzed ' details.Date_and_time '.mat'];
save(file_name_to_save,'phase_mean','phase_std','f','alpha','p_linear','p_error','details','pix_min','pix_max','n_exp','gray_values');
cd(details.current_folder)

