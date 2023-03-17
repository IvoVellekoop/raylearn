% Plot TPM 3D scan

% Note: combine plots with imagemagick:
% magick montage 0* -tile 3x -geometry +0+0 montage_0.5um-beads-in-25uL-tube.png

forcereset = 0;

if forcereset || ~exist('dirs', 'var')
    close all; clear; clc

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn
end

dosave = 0;

%%
% BMPI/Projects/WAVEFRONTSHAPING/data/TPM/4th gen/calibration/calibration_values.mat
%%%%% Automate: M matrix, pixels in frame / pixels per line, zoom

tifpath = '/mnt/bmpi/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/TPM-3D-scans/tube-0.5uL-zoom10-with-correction_00001.tif';
% tifpath = "/home/dani/LocalData/raylearn-data/TPM/TPM-3D-scans/beads-0.5um-in-25uL-cyl-zoom30-zstep0.4um_00002.tif";
% tifpath = "D:\ScientificData\TPM-3D-scans\beads-0.5um-in-25uL-cyl-zoom30-zstep0.4um_00002.tif";
zoom = 10;
factor = 4;
zstep_um = 2;

% % tifpath = "/home/dani/LocalData/raylearn-data/TPM/TPM-3D-scans/beads-0.5um-in-25uL-cyl-zoom15-zstep0.5_00001.tif";
% tifpath = "D:\ScientificData\TPM-3D-scans\beads-0.5um-in-25uL-cyl-zoom15-zstep0.5_00001.tif";
% zoom = 15;
% factor = 4;
% zstep_um = 0.5;

pixels_per_um = 0.2033 * factor * zoom;

FOV_um = 1/pixels_per_um * 1024;

%% Load frames
tifinfo = imfinfo(tifpath);         % Get info about image
num_of_frames = length(tifinfo);    % Number of frames
framestack = zeros(tifinfo(1).Width, tifinfo(1).Height, num_of_frames); % Initialize

for n = 1:num_of_frames
    framestack(:,:,n) = imread(tifpath, n);
end


framestack_raw = framestack;
framestack(framestack<0) = 0;
stack_depth_um = zstep_um * size(framestack, 3);

%% Plot
um = [' (' 181 'm)'];
fig = figure;
dimdata = {[-FOV_um/2 FOV_um/2], [-FOV_um/2 FOV_um/2], [-stack_depth_um/2 stack_depth_um/2]};
% axesoptions = struct('xlim', [-4 8], 'ylim', [-8 4], 'fontsize', 16);
axesoptions = struct('fontsize', 14);

% Plot with im3
im3(framestack, ...
    'title', ['0.5' 181 'm beads in 25' 181 'L tube (cropped)'], ...
    'dimdata', dimdata, ...
    'dimlabels', {['y' um], ['x' um], ['z' um]}, ...
    'axesoptions', axesoptions)
colormap inferno

fig_resize(550, 1.05);
movegui('center')
savename = ['../plots/0.5' 181 'm-beads-in-25' 181 'L-tube-cropped-slice%03i.png'];

fig.UserData.slicenum = 5;

%% Save plots of a few different slices
fig.UserData.changeslice(fig, 0)
ax = gca;

if dosave
    for s = 1:6
        zposition_um = zstep_um * fig.UserData.slicenum - stack_depth_um/2;
        subtitle = ax.Title.String{2,1};
        subtitle = sprintf(['%s, z=%.2f' 181 'm'], subtitle, zposition_um);
        ax.Title.String{2,1} = subtitle;
        
        saveas(fig, sprintf(savename, fig.UserData.slicenum))
        fig.UserData.changeslice(fig, 4)
    end
end


