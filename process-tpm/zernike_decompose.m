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
phase_pattern = -(pattern_data.phase_SLM)';                  % Unwrapped phase pattern
all_amplitudes(:, 1) = zernike_decomposition(phase_pattern, dirs, titlestr, N_modes);

% Center
pattern_data = load('/home/dani/LocalData/raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-center-λ808.0nm.mat'); titlestr = 'b. RT center';
phase_pattern = -(pattern_data.phase_SLM)';                  % Unwrapped phase pattern
all_amplitudes(:, 2) = zernike_decomposition(phase_pattern, dirs, titlestr, N_modes);

% Bottom
pattern_data = load('/home/dani/LocalData/raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-bottom-λ808.0nm.mat'); titlestr = 'c. RT bottom';
phase_pattern = -(pattern_data.phase_SLM)';                  % Unwrapped phase pattern
all_amplitudes(:, 3) = zernike_decomposition(phase_pattern, dirs, titlestr, N_modes);

% Side
pattern_data = load('/home/dani/LocalData/raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-side-λ808.0nm.mat'); titlestr = 'd. RT side';
phase_pattern = -(pattern_data.phase_SLM)';                  % Unwrapped phase pattern
all_amplitudes(:, 4) = zernike_decomposition(phase_pattern, dirs, titlestr, N_modes);

function amplitude = zernike_decomposition(phase_pattern, dirs, titlestr, N_modes)
    
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
    
    % Compute Zernike amplitudes and test them
    starttime = now;
    for j_mode = 1:N_modes
        Zcart = circmask .* imrotate(zernfun_cart(coord_x, coord_y, modes(j_mode).n, modes(j_mode).m, false), slm_rotation_deg, "bilinear", "crop");
        Zcart_normsq = Zcart ./ sum(Zcart(:).^2);
        amplitude(j_mode) = sum(Zcart_normsq(:) .* phase_pattern_circ(:));
    
        % Verify
        rebuilt_pattern = rebuilt_pattern + (amplitude(j_mode) .* Zcart);
%         eta(j_mode, N_modes, starttime, 'cmd', 'Computing Zernike coefficients...', 5);

%         max(Zcart(:) - min(Zcart(:)))
%         var(Zcart(circmask))
    end
    
    
    %% Plot patterns
    figure;
    imagesc(rebuilt_pattern);
    correlation = sum(phase_pattern_circ(:) .* rebuilt_pattern(:)) / sum(phase_pattern_circ(:).^2);
    title(sprintf('Rebuilt pattern, correlation: %.4f', correlation))
    colorbar
    
    figure;
    imagesc(phase_pattern_circ)
    title('Original pattern')
    colorbar
    
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