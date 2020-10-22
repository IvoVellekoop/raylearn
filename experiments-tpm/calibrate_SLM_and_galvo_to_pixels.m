%% Calibrate SLM and Galvo to pixels
% The SLM and Galvo can both translate the beam in the image plane
% Both will be calibrated here
doreset = 0;

%% Reset variables and hardware connections
if doreset
    close all; clear; clc
    setup_raylearn_exptpm
end

%% Settings SLM
bg_patch_id = 2;    % SLM background patch id
pppmaxinv = 0.050;  % 1/maximum ppp value
Nppp = 12;          % Number of ppp values

% Initialize constants
ppps = 1 ./ (linspace(-pppmaxinv, pppmaxinv, Nppp));    % Array of pixels per period
framesSLMx = zeros(copt_img.Width, copt_img.Height, Nppp, 'single');
framesSLMy = framesSLMx;

%% Measure displacement from SLM gradients

% Set background grating
[NxSLM, NySLM] = size(slm.getPixels);
slm.setRect(bg_patch_id, [0 0 1 1]);

starttime = now;
for n = 1:Nppp
    % SLM gradient X direction
    xblaze = bg_grating('blaze', ppps(n), 0, 255, NxSLM)';   % Generate gradient
    slm.setData(bg_patch_id, xblaze);
    slm.update
    framesSLMx(:,:,n) = single(cam_img.HDR());              % Camera frame X direction
    
    % SLM gradient Y direction
    yblaze = bg_grating('blaze', ppps(n), 0, 255, NySLM);   % Generate gradient
    slm.setData(bg_patch_id, yblaze);
    slm.update
    framesSLMy(:,:,n) = single(cam_img.HDR());              % Camera frame Y direction
    
    eta(n, Nppp, starttime, 'console', 'Performing SLM displacement scan...', 0);
end

slm.setData(bg_patch_id, 0); slm.update;                    % Reset SLM data
framesSLMbg = single(cam_img.HDR());                        % Camera background frame

%% Settings Galvo Mirrors
NXY = 12;
XYmax = 0.1;
Xconst = 0;
Yconst = 0;

% Initialize constants
XYs = linspace(-XYmax, XYmax, NXY);
framesGalvox = zeros(copt_img.Width, copt_img.Height, NXY, 'single');
framesGalvoy = framesGalvox;

%% Measure displacement from Galvo tilt

starttime = now;
for n = 1:NXY
    % Galvo X direction
    outputSingleScan(daqs, [XYs(n), Yconst]);
    pause(0.01)
    framesGalvox(:,:,n) = single(cam_img.HDR());        % Camera frame X direction
    
    % Galvo Y direction
    outputSingleScan(daqs, [Xconst, XYs(n)]);
    pause(0.01)
    framesGalvoy(:,:,n) = single(cam_img.HDR());        % Camera frame Y direction
    
    eta(n, NXY, starttime, 'console', 'Performing Galvo displacement scan...', 0);
end

outputSingleScan(daqs, [0, 0]);                         % Reset Galvos

%% Save
script_name = mfilename('fullpath');
[~, script_version] = system(['git --git-dir="' repodir '\.git" rev-parse HEAD']);
save_time = now;

savedir = [expdatadir '\raylearn-data\TPM\'];
savepath = fullfile(savedir, 'raylearn_calibration_SLM_Galvo_raw.mat');
fprintf('\nSaving to %s\n', savepath)
save(savepath, '-v7.3',...
    'framesSLMbg', 'framesSLMx', 'framesSLMy', 'framesGalvox', 'framesGalvoy',...
    'ppps', 'XYs', 'Xconst', 'Yconst',...
    'copt_ft', 'copt_img', 'script_name', 'script_version', 'save_time')

