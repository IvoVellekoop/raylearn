%% Calibrate aperture Galvos field of view

doreset = 1;

if doreset
    close all; clear; clc
    setup_raylearn_exptpm
end

%% Settings
doshowcams = 1;                     % Toggle show what the cameras see

% SLM Settings
ppp = 3;                            % Pixels per period on SLM grating
bg_patch_id = 2;                    % Background grating Patch ID

% Galvo Mirror settings
p.GalvoXcenter = single(0.00);      % Galvo center x
p.GalvoYcenter = single(0.00);      % Galvo center y
p.GalvoXmax    = single(0.70);      % Galvo center to outer, x
p.GalvoYmax    = single(0.25);      % Galvo center to outer, y
p.GalvoNX      = single(80);       % Number of Galvo steps, x
p.GalvoNY      = single(30);        % Number of Galvo steps, y

%% Initialization
% Set background grating
[NxSLM, NySLM] = size(slm.getPixels);
slm_pattern = bg_grating('blaze', ppp, 0, 255, NySLM);
slm.setRect(bg_patch_id, [0 0 1 1]);
slm.setData(bg_patch_id, slm_pattern);
slm.update

% Create set of Galvo tilts
p.galvoXs1d = linspace(p.GalvoXcenter-p.GalvoXmax, p.GalvoXcenter+p.GalvoXmax, p.GalvoNX);
p.galvoYs1d = linspace(p.GalvoYcenter-p.GalvoYmax, p.GalvoYcenter+p.GalvoYmax, p.GalvoNY);
G = p.GalvoNX * p.GalvoNY;
galvoscan = zeros(p.GalvoNX, p.GalvoNY);

frames_ft = zeros(copt_img.Width, copt_img.Height, G, 'single');


%% Loop over galvo tilts
starttime = now;
g = 1;
update_plot_every = 35;

if doshowcams
    fig_galvoscan = figure;
    fig_resize(350,3)
end

for gx = 1:p.GalvoNX                        % Loop over Galvo tilts
    for gy = 1:p.GalvoNY                    % Loop over Galvo tilts
        outputSingleScan(daqs, [p.galvoXs1d(gx), p.galvoYs1d(gy)]);   % Set Galvo Mirror
        pause(0.005)

        % Capture camera frames and mean as scanpixel
        cam_ft.trigger;
        frame_ft = single(cam_ft.getData);
        frames_ft(:,:,g) = frame_ft;
        galvoscan(gx, gy) = mean(frame_ft, 'all');

        % Show what's on the cameras and check for overexposure
        if doshowcams && ~mod(g, update_plot_every)
            figure(fig_galvoscan)
            subplot(1,2,1)
            imagesc(frame_ft)
            saturation_ft = 100 * max(frame_ft,[],'all') / 65520;   % 12bit -> 2^16 - 2^4 = 65520
            title(sprintf('Fourier Plane Cam | Tilt %i/%i\nsaturation = %.0f%%', g, G, saturation_ft))
            axis image
            colorbar

            subplot(1,2,2)
            imagesc(sqrt(galvoscan'), 'XData', p.galvoXs1d, 'YData', p.galvoYs1d)
            hold on
            plot(p.galvoXs1d(gx), p.galvoYs1d(gy), 'or')
            hold off
            title('Galvo Scan sqrt(Mean Intensity)')
            xlabel('Galvo voltage X')
            ylabel('Galvo voltage Y')
            axis image
            grid on
            colorbar
            drawnow
        end
        
        eta(g, G, starttime, 'console', 'Scanning galvo tilt passthrough...', 1);
        g = g+1;
    end
end

outputSingleScan(daqs, [0, 0]);   % Set Galvo Mirror

%% Plot end result
figure(2)
imagesc(sqrt(galvoscan'), 'XData', p.galvoXs1d, 'YData', p.galvoYs1d)
title('Galvo Scan sqrt(Mean Intensity)')
xlabel('Galvo voltage X')
ylabel('Galvo voltage Y')
axis image
grid on
colorbar
