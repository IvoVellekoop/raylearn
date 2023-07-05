% Plot and Report from framestacks

forcereset = 0;

if forcereset || ~exist('dirs', 'var')
    close all; clear; clc

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn
end

do_save = 0;

%%
disp('Loading calibration...')
calibration_data = load(fullfile(dirs.expdata, 'raylearn-data/TPM/calibration/calibration_matrix_parabola/calibration_values.mat'));
tiffolder = fullfile(dirs.expdata, '/raylearn-data/TPM/TPM-3D-scans/23-Jun-2023_tube-500nL/');

M = 5;  % Number of measurements
percentiles = 99.991:0.002:99.999;
flatpattern_high_signals = zeros(M, numel(percentiles));
flatpattern_stds = zeros(M, 1);
flatpattern_offsets = zeros(M, 1);

starttime = now;
for index = 1:M                     % Loop over 'no correction framestack' index
    % Load data
    tiff = load_tiff([tiffolder sprintf('tube-500nL-zoom8-zstep1.400um-no-correction-%i_00001.tif', index)], ...
        calibration_data, sprintf('Loading frame %i/%i', index, M), true);

    % Compute values from data
    flatpattern_high_signals(index, :) = prctile(tiff.framestack(:), percentiles)';
    flatpattern_stds(index) = std_framestack_noisecorrect(tiff.framestack, 1:1024, 1:140, 100:120);
    flatpattern_offsets(index) = tiff.offset;

    eta(index, M, starttime, 'cmd', sprintf('Loading framestacks... (%i/%i)', index, M), 0)
end

%% Report on flat patterns, photobleaching analysis and signal stability

% High signals (percentiles)
perc_index = 4;
flatpattern_perc_mean = mean(flatpattern_high_signals(:, perc_index));
flatpattern_perc_std = std(flatpattern_high_signals(:, perc_index));
fprintf('\n%.3f%%tile   = %.3g\xB1%.2g\n', percentiles(perc_index), flatpattern_perc_mean, flatpattern_perc_std)

% Signal std
fprintf('Signal \x3C3      = %.3g\xB1%.2g\n', mean(flatpattern_stds), std(flatpattern_stds))

% Signal offset
fprintf('Signal offset = %.3g\xB1%.2g\n', mean(flatpattern_offsets), std(flatpattern_offsets))


%% Plot photobleaching analysis and signal stability
figure;
plot(flatpattern_high_signals, '.-');
ylabel('Signal')
ylim([0 max(flatpattern_high_signals(:))])
title('Percentiles of flat pattern measurements')
legend({}, 'Location', 'Best')
legend(strcat(num2str(percentiles'), repmat("%tile", [numel(percentiles) 1])))
set(gca, 'FontSize', 14)

figure;
plot(flatpattern_stds, '.-')
ylabel('Signal \sigma')
title('Signal \sigma noise corrected, flat pattern')
set(gca, 'FontSize', 14)

%% Process signals with correction patterns

process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-top-RT-1_00001.tif'], calibration_data, flatpattern_stds, 'Top RT 1');
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-top-RT-2_00001.tif'], calibration_data, flatpattern_stds, 'Top RT 2');
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-top-AO-1_00001.tif'], calibration_data, flatpattern_stds, 'Top AO 1');
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-top-AO-2_00001.tif'], calibration_data, flatpattern_stds, 'Top AO 2');
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-bottom-RT-1_00001.tif'], calibration_data, flatpattern_stds, 'Bottom RT 1');
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-bottom-RT-2_00001.tif'], calibration_data, flatpattern_stds, 'Bottom RT 2');
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-bottom-AO-1_00001.tif'], calibration_data, flatpattern_stds, 'Bottom AO 1');
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-bottom-AO-2_00001.tif'], calibration_data, flatpattern_stds, 'Bottom AO 2');



% If we assume that the std error scales with the signal, then the error fraction
% for both measurements is the same, hence the error fraction of the enhancement is:

function process_tiff(tiffpath, calibration_data, flatpattern_stds, titlestr)
	tiff = load_tiff(tiffpath, calibration_data);
	
    %%%%%%%%%%% Onderscheid maken boven en onder
	std_framestack = std_framestack_noisecorrect(tiff.framestack, 1:1024, 1:140, 100:120);
    enhancement = std_framestack ./ mean(flatpattern_stds);

    % Note: this is based on the assumption that the error scaling with the signal
    enhancement_error = sqrt(2) * std(flatpattern_stds) ./ mean(flatpattern_stds);

    fprintf('\n%s\n', titlestr)
	fprintf('Signal \x3C3 = %.3g\n', std_framestack)
    fprintf('Signal enhancement = %.3g\xB1%.2g\n', enhancement, enhancement_error)
end


