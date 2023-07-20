% Plot and Report from framestacks

forcereset = 0;

if forcereset || ~exist('dirs', 'var')
    close all; clear; clc

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn
end

do_save = 1;
savedir = fullfile(dirs.repo, 'plots/3D-scans/');
mkdir(savedir);

%%
disp('Loading calibration...')
calibration_data = load(fullfile(dirs.expdata, 'raylearn-data/TPM/calibration/calibration_matrix_parabola/calibration_values.mat'));
tiffolder = fullfile(dirs.expdata, '/raylearn-data/TPM/TPM-3D-scans/23-Jun-2023_tube-500nL/');

noise_index = {1:1024, 1:140, 100:120};
signal_index_top = {1:1024, 1:1024, 1:30};
signal_index_bottom = {1:1024, 1:1024, 91:120};

M = 4;  % Number of measurements
percentiles = 99.991:0.002:99.999;
flatpattern_high_signals = zeros(M, numel(percentiles));
flatpattern_stds_top    = zeros(M, 1);
flatpattern_stds_bottom = zeros(M, 1);
flatpattern_offsets = zeros(M, 1);

disp('Loading flat pattern measurements...')
starttime = now;
for index = 1:M                     % Loop over 'no correction framestack' index
    % Load data
    tiff = load_tiff([tiffolder sprintf('tube-500nL-zoom8-zstep1.400um-no-correction-%i_00001.tif', index)], ...
        calibration_data, '', true);

    % Compute values from data
    flatpattern_high_signals(index, :) = prctile(tiff.framestack(:), percentiles)';
    flatpattern_stds_top(index)    = std_framestack_noisecorrect(tiff.framestack, noise_index, signal_index_top);
    flatpattern_stds_bottom(index) = std_framestack_noisecorrect(tiff.framestack, noise_index, signal_index_bottom);
    flatpattern_offsets(index) = tiff.offset;

    eta(index, M, starttime, 'cmd', sprintf('Loading framestacks... (%i/%i)', index, M), 0);
end

%% Report on flat patterns, photobleaching analysis and signal stability
disp('Flat patterns')

% High signals (percentiles)
perc_index = 4;
flatpattern_perc_mean = mean(flatpattern_high_signals(:, perc_index));
flatpattern_perc_std = std(flatpattern_high_signals(:, perc_index));
fprintf('%.3f%%tile   = %.3g\xB1%.2g\n', percentiles(perc_index), flatpattern_perc_mean, flatpattern_perc_std)

% Signal std
fprintf('Signal top \x3C3      = %.3g\xB1%.2g\n', mean(flatpattern_stds_top), std(flatpattern_stds_top))
fprintf('Signal bottom \x3C3   = %.3g\xB1%.2g\n', mean(flatpattern_stds_bottom), std(flatpattern_stds_bottom))

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
plot(flatpattern_stds_top, '.-'); hold on
plot(flatpattern_stds_bottom, '.-'); hold off
ylabel('Signal \sigma')
title('Signal \sigma noise corrected, flat pattern')
legend('\sigma top', '\sigma bottom')
set(gca, 'FontSize', 14)

%% Process/plot/save signals with correction patterns

figure;
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-top-RT-1_00001.tif'], calibration_data, flatpattern_stds_top, noise_index, signal_index_top, 'Top RT 1', savedir, do_save);
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-top-RT-2_00001.tif'], calibration_data, flatpattern_stds_top, noise_index, signal_index_top, 'Top RT 2', savedir, do_save);
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-top-AO-1_00001.tif'], calibration_data, flatpattern_stds_top, noise_index, signal_index_top, 'Top AO 1', savedir, do_save);
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-top-AO-2_00001.tif'], calibration_data, flatpattern_stds_top, noise_index, signal_index_top, 'Top AO 2', savedir, do_save);
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-bottom-RT-1_00001.tif'], calibration_data, flatpattern_stds_bottom, noise_index, signal_index_bottom, 'Bottom RT 1', savedir, do_save);
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-bottom-RT-2_00001.tif'], calibration_data, flatpattern_stds_bottom, noise_index, signal_index_bottom, 'Bottom RT 2', savedir, do_save);
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-bottom-AO-1_00001.tif'], calibration_data, flatpattern_stds_bottom, noise_index, signal_index_bottom, 'Bottom AO 1', savedir, do_save);
process_tiff([tiffolder 'tube-500nL-zoom8-zstep1.400um-bottom-AO-2_00001.tif'], calibration_data, flatpattern_stds_bottom, noise_index, signal_index_bottom, 'Bottom AO 2', savedir, do_save);

%% Process/plot/save signals without correction patterns
for index = 1:M                     % Loop over 'no correction framestack' index
    process_tiff([tiffolder sprintf('tube-500nL-zoom8-zstep1.400um-no-correction-%i_00001.tif', index)], calibration_data, flatpattern_stds_top, noise_index, signal_index_top, sprintf('No correction %i', index), savedir, do_save);
end


% If we assume that the std error scales with the signal, then the error fraction
% for both measurements is the same, hence the error fraction of the enhancement is:

function process_tiff(tiffpath, calibration_data, flatpattern_stds, noise_index, signal_index, titlestr, savedir, do_save)
	tiff = load_tiff(tiffpath, calibration_data);
	
	std_framestack = std_framestack_noisecorrect(tiff.framestack, noise_index, signal_index);
    enhancement = std_framestack ./ mean(flatpattern_stds);

    % Note: this is based on the assumption that the error scales with the signal
    enhancement_error = sqrt(2) * std(flatpattern_stds) ./ mean(flatpattern_stds);

    fprintf('\n%s\n', titlestr)
	fprintf('Signal \x3C3 = %.3g\n', std_framestack)
    fprintf('Signal enhancement = %.3g\xB1%.2g\n', enhancement, enhancement_error)


    % === Plot ===
    um = [' (' 181 'm)'];
    axesoptions = struct('fontsize', 14);
    
    % Plot max intensity projection
    fig = im3(log(flip(tiff.framestack, 3)), ...
        'slicedim', 1,...
        'maxprojection', true,...
        'title', ['0.5' 181 'm beads in 0.5' 181 'L tube, ' titlestr], ...
        'dimdata', tiff.dimdata, ...
        'dimlabels', {['y' um], ['x' um], ['z' um]}, ...
        'clim', [5 9], ...
        'axesoptions', axesoptions);
    colormap inferno
    
    fig_resize(550, 1.05);
    movegui('center')
    
    cb = colorbar;
    ylabel(cb, 'log(signal)')

    fig.UserData = struct();        % Remove UserData before saving (as this contains the whole framestack!)

    drawnow
    pause(0.05)

    % Save max intensity projection
    if do_save
        [~, filename, ~] = fileparts(tiff.filepath);
        savepath = fullfile(savedir, [filename '_max-proj.fig']);
        savefig(savepath);
        fprintf('Saved as: %s\n', savepath)
    end

    % Plot horizontal slice
    fig = im3(log(flip(tiff.framestack, 3)), ...
        'slicedim', 3,...
        'slicenum', 17,...
        'maxprojection', false,...
        'title', ['0.5' 181 'm beads in 0.5' 181 'L tube, ' titlestr], ...
        'dimdata', tiff.dimdata, ...
        'dimlabels', {['y' um], ['x' um], ['z' um]}, ...
        'clim', [5 9], ...
        'axesoptions', axesoptions);
    colormap inferno
    
    fig_resize(550, 1.05);
    movegui('center')
    
    cb = colorbar;
    ylabel(cb, 'log(signal)')

    fig.UserData = struct();        % Remove UserData before saving (as this contains the whole framestack!)

    drawnow
    pause(0.05)

    % Save horizontal slice
    if do_save
        [~, filename, ~] = fileparts(tiff.filepath);
        savepath = fullfile(savedir, [filename '_hori-slice.fig']);
        savefig(savepath);
        fprintf('Saved as: %s\n', savepath)
    end
end
