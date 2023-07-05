forcereset = 0;

if forcereset || ~exist('dirs', 'var')
    close all; clear; clc

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn
end

%%
disp('Loading calibration...')
calibration_data = load('/mnt/bmpi/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/calibration/calibration_matrix_parabola/calibration_values.mat');
tiffolder = '/mnt/bmpi/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/TPM-3D-scans/23-Jun-2023_tube-500nL/';

M = 5;  % Number of measurements
percentiles = 99.991:0.002:99.999;
high_signals = zeros(M, numel(percentiles));
stds = zeros(M, 1);
offsets = zeros(M, 1);

for index = 1:M
    tiff = load_tiff([tiffolder sprintf('tube-500nL-zoom8-zstep1.400um-no-correction-%i_00001.tif', index)], ...
        calibration_data, sprintf('Loading frame %i/%i', index, M), true);
    high_signals(index, :) = prctile(tiff.framestack(:), percentiles)';
    stds(index) = std_framestack_noisecorrect(tiff.framestack, 1:1024, 1:140, 100:120);
    offsets(index) = tiff.offset;
end

%% Report
perc_index = 4;
mean_perc = mean(high_signals(:, perc_index));
std_perc = std(high_signals(:, perc_index));
fprintf('\n%.3f%%tile   = %.3g\xB1%.3g\n', percentiles(perc_index), mean_perc, std_perc)
fprintf('Signal \x3C3      = %.3g\xB1%.3g\n', mean(stds), std(stds))
fprintf('Signal offset = %.3g\xB1%.3g\n', mean(offsets), std(offsets))


% If we assume that the std error scales with the signal, then the error fraction
% for both measurements is the same, hence the error fraction of the enhancement is:
enhancement_error = sqrt(2) * std(stds) ./ mean(stds);

%% Plot
figure;
plot(high_signals, '.-');
% xticks([1 2 3])
% xticklabels(["Before", "Halfway", "After"])
ylabel('Signal')
ylim([0 max(high_signals(:))])
title('Percentiles of flat pattern measurements')
legend({}, 'Location', 'Best')
legend(strcat(num2str(percentiles'), repmat("%tile", [numel(percentiles) 1])))
set(gca, 'FontSize', 14)

figure;
plot(stds, '.-')
ylabel('Signal STD')
title('Signal STD (noise corrected)')
set(gca, 'FontSize', 14)
