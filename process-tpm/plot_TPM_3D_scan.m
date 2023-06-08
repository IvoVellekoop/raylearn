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

% tifpath = fullfile(dirs.expdata, 'raylearn-data/TPM/TPM-3D-scans/2023-03-24_tube-0.5uL-zoom10-no-correction_00003.tif');
tifpath = fullfile(dirs.expdata, 'raylearn-data/TPM/TPM-3D-scans/2023-03-24_tube-0.5uL-zoom10-with-correction_00001.tif');
zoom = 10;
factor = 4;
zstep_um = 1;

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

starttime = now;
for n = 1:num_of_frames
    framestack(:,:,n) = imread(tifpath, n);
    eta(n, num_of_frames, starttime, 'console', 'Loading tif...', 5);
end


framestack_raw = framestack;
framestack(framestack<0) = 0;
stack_depth_um = zstep_um * size(framestack, 3);

%% Plot
um = [' (' 181 'm)'];
fig = figure;
<<<<<<< HEAD
dimdata = {[-FOV_um/2 FOV_um/2], [-FOV_um/2 FOV_um/2], [stack_depth_um/2 -stack_depth_um/2]};
=======
dimdata = {[-FOV_um/2 FOV_um/2], [-FOV_um/2 FOV_um/2], [-stack_depth_um/2 stack_depth_um/2]};
>>>>>>> 94f03aa (Update settings)
% axesoptions = struct('xlim', [-4 8], 'ylim', [-8 4], 'fontsize', 16);
axesoptions = struct('fontsize', 14);

% Plot with im3
<<<<<<< HEAD
im3(framestack, ...
    'title', ['0.5' 181 'm beads in 0.5' 181 'L tube'], ...
    'slicedim', 3, ...
    'slicenum', 160, ...
=======
im3(min(framestack, 15000), ...
    'title', ['0.5' 181 'm beads in 25' 181 'L tube (cropped)'], ...
>>>>>>> 94f03aa (Update settings)
    'dimdata', dimdata, ...
    'dimlabels', {['x' um], ['y' um], ['z' um]}, ...
    'maxprojection', 0, ...
    'axesoptions', axesoptions)
colormap inferno

pause(0.5)

title(sprintf(['0.5' 181 'm beads in 0.5' 181 'L tube\nslice at z=%.0f' 181 'm'], stack_depth_um/2 - fig.UserData.slicenum * zstep_um))

fig_resize(600, 1.1);
movegui('center')
% caxis([0 8e3])
ylabel(fig.Children(1), 'Intensity')

%% Plot
um = [' (' 181 'm)'];
fig = figure;
dimdata = {[-FOV_um/2 FOV_um/2], [-FOV_um/2 FOV_um/2], [stack_depth_um/2 -stack_depth_um/2]};
% axesoptions = struct('xlim', [-4 8], 'ylim', [-8 4], 'fontsize', 16);
axesoptions = struct('fontsize', 14);

% Plot with im3
im3(framestack, ...
    'title', ['0.5' 181 'm beads in 0.5' 181 'L tube'], ...
    'slicedim', 1, ...
    'dimdata', dimdata, ...
    'dimlabels', {['x' um], ['y' um], ['z' um]}, ...
    'maxprojection', 1, ...
    'axesoptions', axesoptions)
colormap inferno

pause(0.5)

title(sprintf(['0.5' 181 'm beads in 0.5' 181 'L tube\nmax intensity projection'], stack_depth_um/2 - fig.UserData.slicenum * zstep_um))

fig_resize(600, 0.9);
movegui('center')
% caxis([0 8e3])
ylabel(fig.Children(1), 'Intensity')


%% Determine FWHM
figure;
imagesc(framestack(310:370,487:492,164));
caxis([0 5e3]);
axis image

figure;
frameline = mean(framestack_raw(310:370,487:492,164), 2);
x = (1:length(frameline))/pixels_per_um;
f = fit(x',frameline,'gauss1');

xfit = linspace(min(x), max(x), 200);
yfit = f.a1 * exp(-((xfit - f.b1) ./ f.c1).^2);
plot(x, frameline, '.-'); hold on
plot(xfit, yfit, 'k')
xlabel('x (um)')
set(gca, 'fontsize', 14)

sigma = f.c1 / sqrt(2);
FWHM = 2*sqrt(2*log(2)) * sigma;
plot(f.b1 - (FWHM/2 * [-1 1]), f.a1 * [1 1]/2, '.-r')
hold off
legend('image line', 'fit', 'FWHM')
title(sprintf('FWHM = %.3fum', FWHM))

%% Save?
savename = ['../plots/0.5' 181 'm-beads-in-0.5' 181 'L-tube-cropped-slice%03i.png'];

if dosave
    fig.UserData.slicenum = 5;

    %% Save plots of a few different slices
    fig.UserData.changeslice(fig, 0)
    ax = gca;

    for s = 1:6
        zposition_um = zstep_um * fig.UserData.slicenum - stack_depth_um/2;
        subtitle = ax.Title.String{2,1};
        subtitle = sprintf(['%s, z=%.2f' 181 'm'], subtitle, zposition_um);
        ax.Title.String{2,1} = subtitle;
        
        saveas(fig, sprintf(savename, fig.UserData.slicenum))
        fig.UserData.changeslice(fig, 4)
    end
end


