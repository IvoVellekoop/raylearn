% Zernike decompose correction patterns
close all;

forcereset = 0;

if forcereset || ~exist('dirs', 'var')
    close all; clear; clc

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn
end

N_modes = 36;
all_amplitudes = zeros(N_modes, 4);


% Top
pattern_data = load('/home/dani/LocalData/raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-top-λ808.0nm.mat'); titlestr = 'a. RT top';
all_amplitudes(:, 1) = zernike_decomposition(pattern_data, dirs, titlestr, N_modes);

% Center
pattern_data = load('/home/dani/LocalData/raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-center-λ808.0nm.mat'); titlestr = 'b. RT center';
all_amplitudes(:, 2) = zernike_decomposition(pattern_data, dirs, titlestr, N_modes);

% Bottom
pattern_data = load('/home/dani/LocalData/raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-bottom-λ808.0nm.mat'); titlestr = 'c. RT bottom';
all_amplitudes(:, 3) = zernike_decomposition(pattern_data, dirs, titlestr, N_modes);

% Side
pattern_data = load('/home/dani/LocalData/raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-side-λ808.0nm.mat'); titlestr = 'd. RT side';
all_amplitudes(:, 4) = zernike_decomposition(pattern_data, dirs, titlestr, N_modes);

function amplitude = zernike_decomposition(pattern_data, dirs, titlestr, N_modes)
    phase_pattern = -(pattern_data.phase_SLM)';                  % Unwrapped phase pattern

    % Initialization
    amplitude = zeros(N_modes, 1);
    modes = zernike_order(N_modes);
    coord_x = linspace(-1, 1, size(phase_pattern, 1));
    coord_y = linspace(-1, 1, size(phase_pattern, 2))';
    rebuilt_pattern = zeros(size(phase_pattern));
    
    % Prepare pattern
    circmask = (coord_x.^2 + coord_y.^2) < 1;
    phase_pattern_circ = phase_pattern .* circmask;
    slm_rotation_deg = 0;

    separate_patterns = [];
    
    % Compute Zernike amplitudes and test them
    starttime = now;
    for j_mode = 1:N_modes
        Zcart = circmask .* imrotate(zernfun_cart(coord_x, coord_y, modes(j_mode).n, modes(j_mode).m, false), slm_rotation_deg, "bilinear", "crop");
        Zcart_normsq = Zcart ./ sum(Zcart(:).^2);
        amplitude(j_mode) = sum(Zcart_normsq(:) .* phase_pattern_circ(:));
    
        % Verify
        rebuilt_pattern = rebuilt_pattern + (amplitude(j_mode) .* Zcart);
        separate_patterns(:, :, j_mode) = (amplitude(j_mode) .* Zcart);
%         eta(j_mode, N_modes, starttime, 'cmd', 'Computing Zernike coefficients...', 5);

%         max(Zcart(:) - min(Zcart(:)))
%         var(Zcart(circmask))
    end
    
    
    %% Plot patterns
    disp(titlestr)
    figure;
    imagesc(rebuilt_pattern);
    correlation = sum(phase_pattern_circ(:) .* rebuilt_pattern(:)) / sum(phase_pattern_circ(:).^2);
    title(sprintf('Rebuilt pattern, correlation: %.4f', correlation))
    colorbar
    
    figure;
    imagesc(phase_pattern_circ)
    title('Original pattern')
    colorbar

    %% Overlap coefficient
    % Compute gaussian field amplitude
    wavelength = 808e-9;    % Vacuum wavelength
    n_2 = 1.3290;           % Refractive index of focus medium @ 808nm, https://refractiveindex.info/?shelf=main&book=H2O&page=Hale
    f_obj1 = 12.5e-3;       % Objective focal length
    W_m_at_SLM = 5.9e-3;    % Beam profile gaussian width (= c-coefficient from gaussian fit) at SLM in meter
    W_m = 2 * W_m_at_SLM;   % Beam profile gaussian width at back pupil in meter, note: magnification 2x from SLM to back pupil
    x_max = 9.2e-6 * 1152;  % Half-height of the SLM, projected onto objective back pupil (factor 2x)
    x_m = linspace(-x_max, x_max, size(phase_pattern_circ, 1));
    y_m = x_m';
    r2_m2 = x_m.*x_m + y_m.*y_m;
    field_amplitude = exp(-r2_m2 ./ (W_m.^2)) .* pattern_data.NA_mask_SLM;

    % Compute fields
    field_original = exp(1i * phase_pattern_circ) .* field_amplitude;
    separate_patterns_lowfreq = sum(separate_patterns(:,:,1:6), 3);
    phase_low_11 = separate_patterns_lowfreq + separate_patterns(:,:,11);
    phase_low_12 = separate_patterns_lowfreq + separate_patterns(:,:,12);
    field_low_11 = exp(1i * (phase_low_11)) .* field_amplitude;
    field_low_12 = exp(1i * (phase_low_12)) .* field_amplitude;
    field_rebuilt = exp(1i * (rebuilt_pattern)) .* field_amplitude;

    phase_normalized_inner_prod_11 = sum(phase_pattern_circ(:) .* phase_low_11(:)) / sum(phase_pattern_circ(:).^2)
    phase_normalized_inner_prod_12 = sum(phase_pattern_circ(:) .* phase_low_12(:)) / sum(phase_pattern_circ(:).^2)
    phase_normalized_inner_prod_rebuilt = sum(phase_pattern_circ(:) .* rebuilt_pattern(:)) / sum(phase_pattern_circ(:).^2)
    
    % Compute k-vector
    k0 = 2*pi / wavelength;
    k_xy = k0 * sqrt(r2_m2) ./ f_obj1;  % k-vector (x,y)-component
    kz_2 = sqrt((k0*n_2)^2 - k_xy.^2);  % k-vector z-component

    % Compute overlap coefficient, optimized for defocus
    zmin_m = -3e-6;
    zmax_m =  3e-6;
    zstep_m = 0.1e-6;
    [c_D_optimal, z_D_optimal, i_D_optimal, E_D_optimal, c_D_scan, z_D_scan] = ...
        field_corr_optimal_defocus(field_original, field_low_11, kz_2, zmin_m, zmax_m, zstep_m);
    overlap_with_field_11 = abs(c_D_optimal).^2

    [c_D_optimal, z_D_optimal, i_D_optimal, E_D_optimal, c_D_scan, z_D_scan] = ...
        field_corr_optimal_defocus(field_original, field_low_12, kz_2, zmin_m, zmax_m, zstep_m);
    overlap_with_field_12 = abs(c_D_optimal).^2

    [c_D_optimal, z_D_optimal, i_D_optimal, E_D_optimal, c_D_scan, z_D_scan] = ...
        field_corr_optimal_defocus(field_original, field_rebuilt, kz_2, zmin_m, zmax_m, zstep_m);
    overlap_with_rebuilt = abs(c_D_optimal).^2
    
    %% Plot zernike decomposition
    % Plot settings
    do_save = 1;
    subdir = fullfile(dirs.repo, 'plots/zernike-decompositions/');
    suffix = replace(replace(titlestr, ' ', '-'), '.', '');
    pdffilepath = fullfile([subdir '/zernike-decomposition-tube-500nL-' suffix '.pdf']);
    fontsize_labels = 16;
    fontsize_axis = 16;
    fontsize_title = 18;
    
    font_ratio = fontsize_labels/8;
    mkdir(subdir)
    
    % Plot
    fig = figure;
    ax = gca();
    bar(amplitude)
    xlabel('Zernike mode j')
    ylabel('Amplitude (rad)')
    hold on
    text(4, amplitude(4), "  4: Vertical Astigmatism (aligned)", 'FontSize', fontsize_labels)
%     text(5, amplitude(5) + sign(amplitude(5))*font_ratio, '5: Defocus', 'FontSize', fontsize_labels)
    y_text = max(amplitude(11:13));
    text(11, y_text + 3.0 * font_ratio, "11: Vertical quadrafoil (aligned)", 'FontSize', fontsize_labels)
    text(12, y_text + 1.5 * font_ratio, "12: 2nd Astigmatism (aligned)", 'FontSize', fontsize_labels)
    text(13, y_text + 0.0 * font_ratio, "13: Primary spherical", 'FontSize', fontsize_labels)
    hold off
    set(ax, 'FontSize', fontsize_axis)
    title(titlestr, 'FontSize', fontsize_title)
    ylim([-5 45])
    
    if do_save
        drawnow
        pause(0.1)
        fig_resize(440, 1.4, 0, fig);
        pause(0.1)
        save_figtopdf(pdffilepath)
    end
end
