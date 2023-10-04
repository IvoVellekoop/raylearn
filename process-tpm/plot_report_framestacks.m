% Plot and Report from framestacks
% This script processes the 3D scans (tiff files) measured with experiments-tpm/compare_correction_scanimage

forcereset = 0;

if forcereset || ~exist('dirs', 'var')
    close all; clear; clc

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn
end

do_save = 1;
savedir_base = fullfile(dirs.repo, 'plots/3D-scans');
reportname = 'report.txt';
mkdir(savedir_base);

%% === Initial analysis === %%
% Define directories
disp('Loading calibration...')
calibration_data = load(fullfile(dirs.expdata, 'raylearn-data/TPM/calibration/calibration_matrix_parabola/calibration_values.mat'));
tiffolder = fullfile(dirs.expdata, '/raylearn-data/TPM/TPM-3D-scans/01-Oct-2023_tube-500nL/');
tif_input_pattern = [tiffolder 'tube-500nL-*'];
[~,measurement_foldername,~] = fileparts(fileparts(tiffolder));
savedir = fullfile(savedir_base, measurement_foldername);
mkdir(savedir);

% Load a tiff file to get some metadata
tiff = load_tiff([tiffolder 'tube-500nL-zoom8-zstep0.500um-no-correction-1_00001.tif'], ...
    calibration_data, '', true);


%% === Define volumes === %
% Note: manually selected volumes

% In micrometers
% volume_size_y_um = tiff.FOV_y_um / 4;   % In AO scan, a quarter of the full FOV_y was used
% volume_size_z_um = 15;                  % In AO scan, a volume of 15um in z was used for feedback
volume_size_y_um = 20;
volume_size_z_um = 20;                  % In AO scan, a volume of 15um in z was used for feedback

center_z_rt_um = 5;                     % Center of tube
top_z_rt_um = 76 - 20;                  % Center of Top volume
bottom_z_rt_um = -66 + 20;              % Center of Bottom volume
side_z_rt_um = center_z_rt_um;

center_z_ao_um = 0;                     % Center of tube
top_z_ao_um = 71 - 20;                  % Center of Top volume
bottom_z_ao_um = -71 + 20;              % Center of Bottom volume
side_z_ao_um = center_z_ao_um;
side_y_um = -67+18;                     % Center of Side volume (y)
offset_y_um = 3;                        % Offset of tube in y direction

% Convert to pixels
Nx = size(tiff.framestack, 1);
Ny = size(tiff.framestack, 2);
Nz = size(tiff.framestack, 3);
Vyhalf = round(volume_size_y_um / 2 * tiff.pixels_per_um);
Vzhalf = round(volume_size_z_um / 2 / tiff.zstep_um);

top_z_rt = Nz/2 + top_z_rt_um / tiff.zstep_um;
center_z_rt = Nz/2 + center_z_rt_um / tiff.zstep_um;
bottom_z_rt = Nz/2 + bottom_z_rt_um / tiff.zstep_um;
side_z_rt = Nz/2 + side_z_rt_um / tiff.zstep_um;

top_z_ao = Nz/2 + top_z_ao_um / tiff.zstep_um;
center_z_ao = Nz/2 + center_z_ao_um / tiff.zstep_um;
bottom_z_ao = Nz/2 + bottom_z_ao_um / tiff.zstep_um;
side_z_ao = Nz/2 + center_z_ao_um / tiff.zstep_um;
side_y = Ny/2 + round(side_y_um * tiff.pixels_per_um);
offset_y = round(offset_y_um * tiff.pixels_per_um);

% Define volume index arrays
volume_indices = struct();
volume_indices.noise =     {1:Nx, 1:100, 1:50};
volume_indices.top_rt =    {1:Nx, offset_y + (Ny/2-Vyhalf:Ny/2+Vyhalf), top_z_rt-Vzhalf:top_z_rt+Vzhalf};
volume_indices.center_rt = {1:Nx, offset_y + (Ny/2-Vyhalf:Ny/2+Vyhalf), center_z_rt-Vzhalf:center_z_rt+Vzhalf};
volume_indices.bottom_rt = {1:Nx, offset_y + (Ny/2-Vyhalf:Ny/2+Vyhalf), bottom_z_rt-Vzhalf:bottom_z_rt+Vzhalf};
volume_indices.side_rt =   {1:Nx, (side_y-Vyhalf:side_y+Vyhalf), side_z_rt-Vzhalf:side_z_rt+Vzhalf};

volume_indices.top_ao =    {1:Nx, offset_y + (Ny/2-Vyhalf:Ny/2+Vyhalf), top_z_ao-Vzhalf:top_z_ao+Vzhalf};
volume_indices.center_ao = {1:Nx, offset_y + (Ny/2-Vyhalf:Ny/2+Vyhalf), center_z_ao-Vzhalf:center_z_ao+Vzhalf};
volume_indices.bottom_ao = {1:Nx, offset_y + (Ny/2-Vyhalf:Ny/2+Vyhalf), bottom_z_ao-Vzhalf:bottom_z_ao+Vzhalf};
volume_indices.side_ao =   {1:Nx, (side_y-Vyhalf:side_y+Vyhalf), side_z_ao-Vzhalf:side_z_ao+Vzhalf};

% Define location markers
location_markers = struct();
location_markers.top_rt =       [offset_y_um top_z_rt_um];
location_markers.center_rt =    [offset_y_um center_z_rt_um];
location_markers.bottom_rt =    [offset_y_um bottom_z_rt_um];
location_markers.side_rt =      [side_y_um side_z_rt_um];

location_markers.top_ao =       [offset_y_um top_z_ao_um];
location_markers.center_ao =    [offset_y_um center_z_ao_um];
location_markers.bottom_ao =    [offset_y_um bottom_z_ao_um];
location_markers.side_ao =      [side_y_um side_z_ao_um];

location_markers.no_correction = [nan nan];


%% === Initialize signal vars === %
M = 3;  % Number of measurements
percentiles = 99.991:0.002:99.999;
flatpattern_high_signals = zeros(M, numel(percentiles));
flatpattern_offsets = zeros(M, 1);

% === Analyse flat pattern measurements === %
disp('Loading flat pattern measurements...')
flatpattern_stds = struct();
starttime = now;
for index = 1:M                     % Loop over 'no correction framestack' index
    % Load data
    tiff = load_tiff([tiffolder sprintf('tube-500nL-zoom8-zstep0.500um-no-correction-%i_00001.tif', index)], ...
        calibration_data, '', true);

    % Compute values from data
    flatpattern_high_signals(index, :) = prctile(tiff.framestack(:), percentiles)';
    flatpattern_offsets(index) = tiff.offset;

    
    % Compute corrected signal std for every defined volume
    volumes_names = fields(volume_indices);
    for i = 1:numel(volumes_names)
        volume = volume_indices.(volumes_names{i});
        flatpattern_stds(index).(volumes_names{i}) = std_framestack_noisecorrect(tiff.framestack, volume_indices.noise, volume);
    end

    eta(index, M, starttime, 'cmd', sprintf('Loading framestacks... (%i/%i)', index, M), 0);
end


%% Report on flat patterns, photobleaching analysis and signal stability
file_report = fopen(fullfile(savedir, reportname), 'w+');
fprintf(file_report, 'Flat patterns:\n');

% High signals (percentiles)
perc_index = 4;
flatpattern_perc_mean = mean(flatpattern_high_signals(:, perc_index));
flatpattern_perc_std = std(flatpattern_high_signals(:, perc_index));
fprintf(file_report, '%.3f%%tile   = %.3g\xB1%.2g\n', percentiles(perc_index), flatpattern_perc_mean, flatpattern_perc_std);

% Signal offset
fprintf(file_report, 'Signal offset = %.3g\xB1%.2g\n', mean(flatpattern_offsets), std(flatpattern_offsets));

%% Plot photobleaching analysis
figure;
plot(flatpattern_high_signals, '.-');
ylabel('Signal')
ylim([0 max(flatpattern_high_signals(:))])
title('Percentiles of flat pattern measurements')
legend({}, 'Location', 'Best')
legend(strcat(num2str(percentiles'), repmat("%tile", [numel(percentiles) 1])))
set(gca, 'FontSize', 14)
drawnow
if do_save
    [~, filename, ~] = fileparts(tiff.filepath);
    savepath = fullfile(savedir, [filename '_percentiles.fig']);
    savefig(savepath);
    fprintf('Saved as: %s\n', savepath)
end

clear tiff

%% Process/plot/save signals with correction patterns

figure;

% Process
assert(exist(tiffolder, 'dir'), 'Directory "%s" doesn''t exist', tiffolder)
filelist = dir(tif_input_pattern);      % List of files to be processed
num_files = length(filelist);                   % Number of measurements
assert(num_files > 0, 'Directory list is empty. Please check input pattern.')

starttime = now;
for index_file = 1:num_files
    % Prepare file info
    filename = filelist(index_file).name;                           % File name
    filepath = fullfile(filelist(index_file).folder, filename);     % File path
    filename_values_cell = regexp(filename, 'tube-500nL-zoom([.\d]+)-zstep([.\d]+)um-([^_]+)_\d+.tif', 'tokens');
    titlestr = filename_values_cell{:}{3};

    % Process the file
    process_tiff(filepath, calibration_data, flatpattern_stds, volume_indices, location_markers, titlestr, savedir, do_save, file_report);
    eta(index_file, num_files, starttime, 'cmd', sprintf('Processing framestacks... (%i/%i)', index_file, num_files), 0);
end

fclose(file_report);


function process_tiff(tiffpath, calibration_data, flatpattern_stds, volume_indices, location_markers, titlestr, savedir, do_save, file_report)
	tiff = load_tiff(tiffpath, calibration_data);

    percentile = 99.5;  % Percentile for the projection
	
    fprintf(file_report, '\n%s\n', titlestr);

    volumes_names = fields(volume_indices);
    for i = 1:numel(volumes_names)
        volume = volume_indices.(volumes_names{i});

        std_framestack = std_framestack_noisecorrect(tiff.framestack, volume_indices.noise, volume);
        enhancement = std_framestack ./ flatpattern_stds(1).(volumes_names{i});
    
        % Note: this is based on the assumption that the error scales with the signal
%         enhancement_error = sqrt(2) * std(flatpattern_stds) ./ flatpattern_stds.(volumes_names{i});
    
	    fprintf(file_report, 'Signal at %s \x3C3 = %.3g, ', volumes_names{i}, std_framestack);
%         fprintf(file_report, 'contrast enhancement = %.3g\xB1%.2g\n', volumes_names{i}, enhancement, enhancement_error);
        fprintf(file_report, 'contrast enhancement = %.3g\n', enhancement);
    end


    % === Plot ===
    um = [' (' 181 'm)'];
    axesoptions = struct('fontsize', 14);
    
    location_mark_field = replace(lower(titlestr(1:end-2)), '-', '_');
    location_mark = location_markers.(location_mark_field);
    % Plot percentile projection
    fig = im3(log(flip(tiff.framestack, 3)), ...
        'slicedim', 1,...
        'projection', percentile,...
        'title', ['0.5' 181 'm beads in 0.5' 181 'L tube, ' titlestr], ...
        'dimdata', tiff.dimdata, ...
        'dimlabels', {['y' um], ['x' um], ['z' um]}, ...
        'clim', [5 10], ...
        'axesoptions', axesoptions);
    colormap inferno
    hold on
    drawnow

    % Plot marker indicating focus location
    plot(location_mark(1), location_mark(2), 'xc', 'MarkerSize', 10, 'LineWidth', 1.5)
    hold off
    
    fig_resize(550, 1.05);
    movegui('center')
    
    cb = colorbar;
    ylabel(cb, 'log_{10}(signal)')

    fig.UserData = struct();        % Remove UserData before saving (as this contains the whole framestack!)

    drawnow
    pause(0.05)

    % Save max intensity projection
    if do_save
        [~, filename, ~] = fileparts(tiff.filepath);
        savepath = fullfile(savedir, [filename sprintf('_%.1fprctile-proj.fig', percentile)]);
        savefig(savepath);
        fprintf('Saved as: %s\n', savepath)
    end

    % Plot horizontal slice
    fig = im3(log(flip(tiff.framestack, 3)), ...
        'slicedim', 3,...
        'slicenum', 85,...
        'projection', -1,...
        'title', ['0.5' 181 'm beads in 0.5' 181 'L tube, ' titlestr], ...
        'dimdata', tiff.dimdata, ...
        'dimlabels', {['y' um], ['x' um], ['z' um]}, ...
        'clim', [5 10], ...
        'axesoptions', axesoptions);
    colormap inferno
    
    fig_resize(550, 1.05);
    movegui('center')
    
    cb = colorbar;
    ylabel(cb, 'log_{10}(signal)')

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

