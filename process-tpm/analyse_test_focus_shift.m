% Analyse tested focus shift raytraced patterns

forcereset = 1;

if forcereset || ~exist('dirs', 'var')
    close all; clear; clc

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn
end


do_save = 1;

%% Define directories
disp('Loading calibration...')
calibration_data = load(fullfile(dirs.expdata, 'raylearn-data/TPM/calibration/calibration_matrix_parabola/calibration_values.mat'));
tiffolder = fullfile(dirs.expdata, '/raylearn-data/TPM/TPM-3D-scans/28-Sep-2023_beads_testshift_xscale0.985200_yscale0.973700/');

tiff_flat = load_tiff([tiffolder 'beads-zoom10-zstep0.333um-no-correction_00001.tif'], calibration_data, '', true); disp('Loaded flat')
tiff_xshift = load_tiff([tiffolder 'beads-zoom10-zstep0.333um-xshift_00001.tif'], calibration_data, '', true); disp('Loaded x-shift')
tiff_yshift = load_tiff([tiffolder 'beads-zoom10-zstep0.333um-yshift_00001.tif'], calibration_data, '', true); disp('Loaded y-shift')

%% Analyze x&y

Minv = calibration_data.M^-1;           % TPMpix vector to zaber_um vector conversion matrix, for zoom=1
bg_median = median(tiff_flat.framestack_raw, [1 2 3]);

xshifts_in_um = [];
yshifts_in_um = [];
xshifts_tpmpix = [];
yshifts_tpmpix = [];
slice_start = 10;
slice_range = slice_start:size(tiff_flat.framestack_raw, 3);
count = 1;
starttime = now;
for i_zslice = slice_range
    index = i_zslice - slice_start + 1;
    [xshifts_in_um(index, :), xshifts_tpmpix(index, :)] = compute_shift_in_um(bg_median, i_zslice, tiff_flat, tiff_xshift, Minv);
    [yshifts_in_um(index, :), yshifts_tpmpix(index, :)] = compute_shift_in_um(bg_median, i_zslice, tiff_flat, tiff_yshift, Minv);
    eta(count, length(slice_range), starttime, 'cmd', 'Computing x- and y-shifts...', 0);
    count = count+1;
end

xshift_in_um = median(xshifts_in_um, 1);
yshift_in_um = median(yshifts_in_um, 1);
xshift_in_um_std = std(vecnorm(xshifts_in_um'));
yshift_in_um_std = std(vecnorm(yshifts_in_um'));

%% Plot
figure;
quiver(0, 0, xshift_in_um(1), xshift_in_um(2)); hold on
text(xshift_in_um(1), xshift_in_um(2), sprintf('\n\nx-shift: %.2f\xB1%.2f um', norm(xshift_in_um), xshift_in_um_std), 'FontSize', 14)
quiver(0, 0, yshift_in_um(1), yshift_in_um(2))
text(yshift_in_um(1), yshift_in_um(2), sprintf('y-shift: %.2f\xB1%.2f um', norm(yshift_in_um), yshift_in_um_std), 'FontSize', 14); hold off
axis image
title('Image shifts by ray traced gradient', 'of 30um')
drawnow
xlim([-40 40])
ylim([-40 40])
xlabel('zaber-x (um)')
ylabel('zaber-y (um)')
set(gca, 'FontSize', 14)
drawnow

%% Analyze z
clear tiff_xshift
clear tiff_yshift
tiff_zshift_with_obj = load_tiff([tiffolder 'beads-zoom10-zstep0.333um-zshift_with_obj_zshift_00001.tif'], calibration_data, '', true);  disp('Loaded z-shift')
thresh = 2000;
[zshift_in_um, zshift_index, shift_products, shift_steps] = compute_zshift_in_um(bg_median, tiff_flat, tiff_zshift_with_obj, thresh);
figure;

%% Plot z
[max_innerproduct, i_max_innerproduct] = max(shift_products);
z_shift_um = tiff_flat.zstep_um .* shift_steps(i_max_innerproduct);
zshift_step = shift_steps(i_max_innerproduct);

plot(tiff_flat.zstep_um .* shift_steps, shift_products, '.-'); hold on
plot(z_shift_um, max_innerproduct, 'or'); hold off
text(z_shift_um, max_innerproduct, sprintf('  z-shift: %.2fum', z_shift_um))
ylabel('Inner product')
xlabel('(circular) z-shift (um)')
title('30um SLM defocus', 'and 30um objective piezo zshift to compensate')
set(gca, 'FontSize', 14)

%% Analyze x&y shift of defocus pattern (z-shift)
stackflat = tiff_flat.framestack_raw - bg_median;
stackshift = tiff_zshift_with_obj.framestack_raw - bg_median;
stackflat(stackflat < thresh) = 0;
stackshift(stackshift < thresh) = 0;
stackflat_shifted = circshift(stackflat, [0 0 zshift_step]);         % Fill in slice shift for z-shift match

defocus_xyshifts_in_um = [];

slice_start = 1;
slice_range = slice_start:size(tiff_flat.framestack_raw, 3);
starttime = now;
count = 1;
for i_zslice = slice_range
    index = i_zslice - slice_start + 1;
    defocus_xyshifts_in_um(index, :) = compute_shift_defocus_in_um(i_zslice, stackflat_shifted, stackshift, Minv, tiff_flat.zoom);
    eta(count, length(slice_range), starttime, 'cmd', 'Computing x- and y-shifts from defocus pattern...', 4);
    count = count+1;
end

%%
defocus_xyshift_in_um = median(defocus_xyshifts_in_um, 1);
defocus_xyshift_in_um_std = std(vecnorm(defocus_xyshifts_in_um'));

%% Plot
figure;
quiver(0, 0, defocus_xyshift_in_um(1), defocus_xyshift_in_um(2)); hold on
text(defocus_xyshift_in_um(1), defocus_xyshift_in_um(2), sprintf('\n\nshift: %.2f\xB1%.2f um', norm(defocus_xyshift_in_um), defocus_xyshift_in_um_std), 'FontSize', 14)
axis image
title('Image shifts by ray traced defocus', 'of 30um')
drawnow
xlim([-1 1])
ylim([-1 1])
xlabel('zaber-x (um)')
ylabel('zaber-y (um)')
set(gca, 'FontSize', 14)
drawnow

%% Functions
function [shift_in_um, shift_tpmpix] = compute_shift_in_um(bg_median, i_zslice, tiff_flat, tiff_shift, Minv)
    wienerkinchin_array = wiener_khinchin(tiff_flat.framestack(:,:,i_zslice) - bg_median, tiff_shift.framestack(:,:,i_zslice) - bg_median);
    shift_tpmpix = calculate_offset_in_peak(wienerkinchin_array);
    shift_in_um = Minv * shift_tpmpix' / tiff_flat.zoom;

%     imagesc(fftshift(wienerkinchin_array)'); hold on
%     plot(shift_tpmpix(1) + size(wienerkinchin_array, 1)/2, shift_tpmpix(2) + size(wienerkinchin_array, 2)/2, 'sw', 'MarkerSize', 15); hold off
%     title(strcat('Shift: ', num2str(shift_in_um), ' um'))
%     colorbar
%     pause(0.3)
end

function [shift_in_um, shift_tpmpix] = compute_shift_defocus_in_um(i_zslice, array1, array2, Minv, zoom)
    wienerkinchin_array = wiener_khinchin(array1(:,:,i_zslice), array2(:,:,i_zslice));
    shift_tpmpix = calculate_offset_in_peak(wienerkinchin_array);
    shift_in_um = Minv * shift_tpmpix' / zoom;
end

function [zshift_in_um, zshift_index, shift_products, shift_steps] = compute_zshift_in_um(bg_median, tiff_flat, tiff_shift, thresh)
    stackflat = tiff_flat.framestack_raw - bg_median;
    stackshift = tiff_shift.framestack_raw - bg_median;
    stackflat(stackflat < thresh) = 0;
    stackshift(stackshift < thresh) = 0;

    Nz = size(stackflat, 3);
    shift_products = zeros(1, Nz);
    shift_steps = floor(-Nz/2)+1:floor(Nz/2);
    starttime = now;

    for i_shift = 1:length(shift_steps)
        shift_step = shift_steps(i_shift);
        stackflat_shifted = circshift(stackflat, [0 0 shift_step]);
        shift_products(i_shift) = mean(stackflat_shifted.*stackshift, [1 2 3]);
        eta(i_shift, Nz, starttime, 'cmd', 'Computing shifted inner products', 0);
    end

    [~, zshift_index] = max(shift_products);
    zshift_in_um = zshift_index .* tiff_flat.zstep_um;
end





