%% Calibrate SLM and Galvo to pixels
% The SLM and Galvo can both translate the beam in the image plane
% Both will be calibrated here
doreset = 1;

%% Reset variables and hardware connections
if doreset
    close all; clear; clc
    setup_raylearn_exptpm
end

%% Settings SLM
bg_patch_id = 2;    % SLM background patch id
pppmaxinv = 0.025;  % 1/maximum ppp value
Nppp = 12;          % Number of ppp values

% Initialize constants
ppps = 1 ./ (linspace(-pppmaxinv, pppmaxinv, Nppp));    % Array of pixels per period
framesSLMx = zeros(Nxcam, Nycam, Nppp);
framesSLMy = framesSLMx;

%% Measure displacement from SLM gradients

NySLM = 1000; %%%===%%%
% % Set background grating
% [NxSLM, NySLM] = size(slm.getPixels);
% slm.setRect(bg_patch_id, [0 0 1 1]);

for n = 1:Nppp
    xblaze = bg_grating('blaze', ppps(n), 0, 255, NySLM);
    % slm.setData(bg_patch_id, xblaze);
    % slm.update
    
    % Camera x
    
    yblaze = bg_grating('blaze', ppps(n), 0, 255, NySLM);
    % slm.setData(bg_patch_id, yblaze);
    % slm.update

    % Camera y
end


%% Settings Galvo Mirrors
NXY = 10;
XYmax = 0.1;
Xconst = 0;
Yconst = 0;

% Initialize constants
XYs = linspace(-XYmax, XYmax, NXY);
framesGalvox = zeros(Nxcam, Nycam, Nppp);
framesGalvoy = framesGalvox;

%% Measure displacement from Galvo tilt

for n = 1:NXY
    outputSingleScan(daqs, [XYs(n), Yconst]);
    pause(0.1)
    
    % Camera x
%     framesGalvox(:,:,n) = 
    
    outputSingleScan(daqs, [Xconst, XYs(n)]);
    pause(0.1)
    
    % Camera y
end


