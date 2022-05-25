%% Continuous Galvo Scan

doreset = 0;

if doreset || ~exist('daqs', 'var')
    close all; clear; clc
    setup_raylearn_exptpm
end

%% Settings
% Galvo Mirror settings
p.GalvoXcenter = single(0.00);      % Galvo center x
p.GalvoYcenter = single(0.00);      % Galvo center y
p.GalvoXmax    = single(0.00);      % Galvo center to outer, x
p.GalvoYmax    = single(0.30);      % Galvo center to outer, y
p.GalvoNX      = single(10);        % Number of Galvo steps, x
p.GalvoNY      = single(10);        % Number of Galvo steps, y

%% Initialization
% Create set of Galvo tilts
p.galvoXs1d = linspace(p.GalvoXcenter-p.GalvoXmax, p.GalvoXcenter+p.GalvoXmax, p.GalvoNX);
p.galvoYs1d = linspace(p.GalvoYcenter-p.GalvoYmax, p.GalvoYcenter+p.GalvoYmax, p.GalvoNY);
G = p.GalvoNX * p.GalvoNY;
galvoscan = zeros(p.GalvoNX, p.GalvoNY);


%% Loop over galvo tilts
while true
    starttime = now;
    g = 1;
    for gx = 1:p.GalvoNX                        % Loop over Galvo tilts
        for gy = 1:p.GalvoNY                    % Loop over Galvo tilts
            outputSingleScan(daqs, [p.galvoXs1d(gx), p.galvoYs1d(gy)]);   % Set Galvo Mirror
            pause(0.05)
            
            console_text = sprintf('Galvo tilt: (%.2fV, %.2fV)', p.galvoXs1d(gx), p.galvoYs1d(gy));
            eta(g, G, starttime, 'console', console_text, 1);
            g = g+1;
        end
    end
end

%% Run this section to reset Galvo mirrors
outputSingleScan(daqs, [0, 0]);   % Set Galvo Mirror
