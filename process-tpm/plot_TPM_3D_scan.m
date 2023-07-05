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
calibration_data = load(fullfile(dirs.expdata, 'raylearn-data/TPM/calibration/calibration_matrix_parabola/calibration_values.mat'));

tiffolder = fullfile(dirs.expdata, '/raylearn-data/TPM/TPM-3D-scans/23-Jun-2023_tube-500nL/');
% tifpath = [tiffolder 'tube-500nL-zoom8-zstep1.400um-top-RT-1_00001.tif']; titlestr = 'Top RT 1';
% tifpath = [tiffolder 'tube-500nL-zoom8-zstep1.400um-top-RT-2_00001.tif']; titlestr = 'Top RT 2';
% tifpath = [tiffolder 'tube-500nL-zoom8-zstep1.400um-top-AO-1_00001.tif']; titlestr = 'Top AO 1';
% tifpath = [tiffolder 'tube-500nL-zoom8-zstep1.400um-top-AO-2_00001.tif']; titlestr = 'Top AO 2';
tifpath = [tiffolder 'tube-500nL-zoom8-zstep1.400um-bottom-RT-1_00001.tif']; titlestr = 'Bottom RT 1';
% tifpath = [tiffolder 'tube-500nL-zoom8-zstep1.400um-bottom-RT-2_00001.tif']; titlestr = 'Bottom RT 2';
% tifpath = [tiffolder 'tube-500nL-zoom8-zstep1.400um-bottom-AO-1_00001.tif']; titlestr = 'Bottom AO 1';
% tifpath = [tiffolder 'tube-500nL-zoom8-zstep1.400um-bottom-AO-2_00001.tif']; titlestr = 'Bottom AO 2';
% tifpath = [tiffolder 'tube-500nL-zoom8-zstep1.400um-no-correction-1_00001.tif']; titlestr = 'No correction 1';

tiff = load_tiff(tifpath, calibration_data);

%% Plot
um = [' (' 181 'm)'];
fig = figure;
% axesoptions = struct('xlim', [-4 8], 'ylim', [-8 4], 'fontsize', 16);
axesoptions = struct('fontsize', 14);

% Plot with im3
im3(log(flip(tiff.framestack, 3)), ...
    'slicedim', 1,...
    'maxprojection', true,...
    'title', ['0.5' 181 'm beads in 0.5' 181 'L tube, ' titlestr], ...
    'dimdata', tiff.dimdata, ...
    'dimlabels', {['y' um], ['x' um], ['z' um]}, ...
    'axesoptions', axesoptions)
colormap inferno

fig_resize(550, 1.05);
movegui('center')
savename = ['../plots/0.5' 181 'm-beads-in-25' 181 'L-tube-cropped-slice%03i.png'];

fig.UserData.slicenum = 5;
cb = colorbar;
ylabel(cb,'log(signal)')
drawnow
pause(0.1)
% title(titlestr)
caxis([5.5 8.0])

%% Save plots of a few different slices

if dosave
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


