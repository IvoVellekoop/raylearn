forcereset = 0;

if forcereset || ~exist('dirs', 'var')
    close all; clear; clc

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn
end

%%

tiffolder = '/mnt/bmpi/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/TPM-3D-scans/23-Jun-2023_tube-500nL/';

M = 5;  % Number of measurements
percentiles = 99.991:0.002:99.999;
high_signals = zeros(M, numel(percentiles));

for index = 1:M
    framestack_raw = load_framestack([tiffolder sprintf('tube-500nL-zoom8-zstep1.400um-no-correction-%i_00001.tif', index)], index, M);
    framestack = framestack_raw - mean(framestack_raw(:));
    high_signals(index, :) = prctile(framestack(:), percentiles)';
end


%% Plot
plot(high_signals, '.-');
% xticks([1 2 3])
% xticklabels(["Before", "Halfway", "After"])
ylabel('Signal')
ylim([0 max(high_signals(:))])
title('Percentiles of flat pattern measurements')
legend({}, 'Location', 'Best')
legend(strcat(num2str(percentiles'), repmat("%tile", [numel(percentiles) 1])))
set(gca, 'FontSize', 14)

%%
function framestack_raw = load_framestack(tifpath, index, M)
    %% Load frames
    tifinfo = imfinfo(tifpath);         % Get info about image
    num_of_frames = length(tifinfo);    % Number of frames
    framestack = zeros(tifinfo(1).Width, tifinfo(1).Height, num_of_frames); % Initialize
    
    starttime = now;
    for n = 1:num_of_frames
        framestack(:,:,n) = imread(tifpath, n);
        eta(n, num_of_frames, starttime, 'cmd', sprintf('Framestack %i/%i\n', index, M), 10);
    end
    
    framestack_raw = single(framestack);
end