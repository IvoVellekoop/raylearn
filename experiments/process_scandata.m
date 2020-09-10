clear; close all; clc

% Load stuff - adjust according to data files/folders
addpath /home/daniel/NTFS-HDD/Code-big/utilities/
cd /home/daniel/NTFS-HDD/ScientificData/raylearn-data/

reference_file = 'raylearn_laser_2x170um_21-Aug-2020_738024.657875.mat';
measurement_file = 'raylearn_led_1x400um_21-Aug-2020_738024.796382.mat';

% Parameters
framenum = 455;             %%% Manually choose this for now
xysize = 60;                % Match raylearn setting
pixsize = 4.8;              % Camera pixel size (µm)

% Compute magnification from focal distances
fobj = 1.65;                % Objective effective focal distance (mm)
f3 = 75;
f4 = 150;
f5 = 50;
Mtr  = f3*f5 / (fobj*f4);   % Transverse Magnification
Mlat = Mtr^2;               % Lateral magnification


% Find chosen measurement plane offset (selected by setting framenum)
ref = load(reference_file);
[~, maxindex] = max(ref.scan3D, [], 'all', 'linear');
[i1, i2, i3] = ind2sub(size(ref.scan3D), maxindex);
focalpos_ref_magnified_um = ref.p.scanrange_um(i3);     % Reference focus position (magnified space) (µm)

led = load(measurement_file);
measurepos_magnified_um = led.p.scanrange_um(framenum); % Measurement plane position (magnified space) (µm)  

% Offset of measurement plane (real space) (µm)
measure_offset_um = (focalpos_ref_magnified_um - measurepos_magnified_um) / Mlat;

% % Background
% background = mean(scan3D(:, 256:end, 1), [1 2]);

% Crop to area around focus
frame = led.scan3D(:,:,framenum);
frame(frame < 0) = 0;                                   % Remove sub SNR values

maxintensity = max(frame(:));
frame_threshed = frame;
frame_threshed(frame_threshed < maxintensity*0.4) = 0;

[col, row] = img_center_of_mass(frame_threshed);
cols_crop = ceil(col - xysize/2):floor(col + xysize/2);
rows_crop = ceil(row - xysize/2):floor(row + xysize/2);
% [~, maxindex] = max(frame, [], 'all', 'linear');
% [i1, i2] = ind2sub(size(led.scan3D), maxindex);
% cols_crop = ceil(i1 - xysize/2):floor(i1 + xysize/2);
% rows_crop = ceil(i2 - xysize/2):floor(i2 + xysize/2);
frame_cropped = frame(cols_crop, rows_crop);

% Compute real size in µm
xymax_um = 0.5 * xysize * pixsize / Mtr;            % Maximum x or y from center
colrange_um = linspace(-xymax_um, xymax_um, xysize);
rowrange_um = linspace(-xymax_um, xymax_um, xysize);
save('frame-led.mat', '-v7.3', 'frame_cropped', 'xymax_um', 'colrange_um', 'rowrange_um', 'measure_offset_um')

% Plot result
imagesc(colrange_um, rowrange_um, frame_cropped)
axis image
colorbar

title(sprintf('Measurement plane offset: %.2fµm (real space)\n(distance from original focal point)', measure_offset_um))
xlabel('x (µm)')
ylabel('y (µm)')


%%

% max_per_frame = max(max(scan));
% overHM = zeros(size(scan));
% overHM(scan > max_per_frame/2) = 1;
% FWHM = squeeze(sum(sum(overHM)));
% plot(FWHM)