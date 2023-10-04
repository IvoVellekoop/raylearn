close all;

forcereset = 0;

if forcereset || ~exist('dirs', 'var')
    close all; clear; clc

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn
end

% pattern_data = load('/home/dani/LocalData/raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-bottom-λ808.0nm.mat'); titlestr = 'Bottom RT';
% phase_pattern = (pattern_data.phase_SLM);                  % Unwrapped phase pattern

pattern_data = load('/home/dani/LocalData/raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-top-λ808.0nm.mat'); titlestr = 'Top RT';
phase_pattern = (pattern_data.phase_SLM);                  % Unwrapped phase pattern

% pattern_data = load('/home/dani/LocalData/raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-center-λ808.0nm.mat'); titlestr = 'Center RT';
% phase_pattern = (pattern_data.phase_SLM);                  % Unwrapped phase pattern

% pattern_data = load('/home/dani/LocalData/raylearn-data/TPM/slm-patterns/pattern-0.5uL-tube-side-λ808.0nm.mat'); titlestr = 'Side RT';
% phase_pattern = (pattern_data.phase_SLM);                  % Unwrapped phase pattern

% pattern_data = load('/mnt/bmpi/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/adaptive-optics/230620-tube500nL-top/tube_ao_739059.828687_tube500nL-top/tube_ao_739059.828687_tube500nL-top_optimal_pattern.mat'); titlestr = 'Top AO';
% phase_pattern = pattern_data.slm_pattern_2pi_optimal;                  % Unwrapped phase pattern

% pattern_data = load('/mnt/bmpi/Data/Daniel Cox/ExperimentalData/raylearn-data/TPM/adaptive-optics/230620-tube500nL-bottom/tube_ao_739059.875192_tube500nL-bottom/tube_ao_739059.875192_tube500nL-bottom_optimal_pattern.mat'); titlestr = 'Bottom AO';
% phase_pattern = pattern_data.slm_pattern_2pi_optimal;                  % Unwrapped phase pattern


N_modes = 36;

amplitude = zeros(N_modes, 1);

modes = zernike_order(N_modes);

coord_x = linspace(-1, 1, size(phase_pattern, 1));
coord_y = linspace(-1, 1, size(phase_pattern, 2))';

circmask = (coord_x.^2 + coord_y.^2) < 1;

phase_pattern_circ = phase_pattern .* circmask;

slm_rotation_deg = 0;

rebuilt_pattern = zeros(size(phase_pattern));

starttime = now;
for j_mode = 1:N_modes
    Zcart = circmask .* imrotate(zernfun_cart(coord_x, coord_y, modes(j_mode).n, modes(j_mode).m, false), slm_rotation_deg, "bilinear", "crop");
    Zcart_normsq = Zcart ./ sum(Zcart(:).^2);
    amplitude(j_mode) = sum(Zcart_normsq(:) .* phase_pattern_circ(:));

    % Verify
    rebuilt_pattern = rebuilt_pattern + (amplitude(j_mode) .* Zcart);
    eta(j_mode, N_modes, starttime, 'cmd', 'Computing Zernike coefficients...', 4);
end


%%
figure;
imagesc(rebuilt_pattern);
correlation = sum(phase_pattern_circ(:) .* rebuilt_pattern(:)) / sum(phase_pattern_circ(:).^2);
title(sprintf('Rebuilt pattern, correlation: %.4f', correlation))
colorbar

figure;
imagesc(phase_pattern_circ)
title('Original pattern')
colorbar

%%
figure;
bar(amplitude)
xlabel('Zernike mode')
ylabel('Amplitude (rad)')
hold on
text(4, amplitude(4)-1, ["  4: Vertical Astigmatism (aligned)"])
text(5, 1, '5: Defocus')
text(11, 7, ["11: Vertical quadrafoil (aligned)"])
text(12, 5.5, ["12: 2nd Astigmatism (aligned)"])
text(13, 4, "13: Primary spherical")
hold off
set(gca, 'FontSize', 14)
title(sprintf('Zernike decomposition %s', titlestr))
