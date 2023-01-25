% First load field_SLM
clear; close all; clc

forcereset = 0;

if forcereset || ~exist('dirs', 'var')
%     close all; clear; clc

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn
end

% Compute defocusfree spherical aberrations
num_pixels = 800;
wavelength = 804e-9;    % Vacuum wavelength
n_1 = 1.3290;           % Refractive index immersion water
n_2 = 1.5171;           % Refractive index sample
d_act = 1000e-6;        % Actual focus depth
NA = 0.8;               % Numerical Aperture

[field_SLM, phase_SLM, kz_2] = defocusfree_spherical_aberrations(...
    n_1, n_2, d_act, wavelength, NA, num_pixels);

% Compute gaussian field amplitude
W_m = 10e-3;            % Beam profile gaussian width in m
x_m = linspace(-10e-3, 10e-3, size(field_SLM, 1));
y_m = x_m';
r2_m2 = x_m.*x_m + y_m.*y_m;
field_amplitude = exp(-r2_m2 ./ (W_m.^2));

% Compute field and correlation
E = field_SLM .* field_amplitude;
c = field_corr(E, field_amplitude);

% Plot beam profile
figure(1)
plot(x_m, field_amplitude(end/2, :).^2);
ylim([0 1]);
set(gca, 'fontsize', 14)
xlabel('x (m)')
ylabel('Relative intensity')
title('Intensity Beam profile')

% Straight out of [Salter et al. 2014]
fprintf('Theoretical enhancements using [Salter et al. 2014]')
theoretical_enhancement(c);

plot_field(E, NA)
title({'Defocus-free spherical aberrations: field', '[Salter et al. 2014]'})


%% Defocus scan
z_D_scan = linspace(-50e-6, 29e-6, 300);
c_D_scan = zeros(size(z_D_scan));

% Scan range of different defoci
starttime = now;
c_D_optimal = 0;
for i_D = 1:length(z_D_scan)

    % Compute defocused field and field correlation
    z_D = z_D_scan(i_D);
    E_D = E .* exp(1i .* kz_2 .* z_D);                  % Defocused electric field
    c_D_scan(i_D) = field_corr(E_D, field_amplitude);   % Correlation with flat field
    
    % Check for optimal correlation
    if abs(c_D_scan(i_D)) > abs(c_D_optimal)
        % Save optimal values
        c_D_optimal = c_D_scan(i_D);
        i_D_optimal = i_D;
        E_D_optimal = E_D;
        z_D_optimal = z_D_scan(i_D_optimal);
    end
end


%% Plot defocus scan
figure;
plot(z_D_scan*1e6, abs(c_D_scan)); hold on
plot(z_D_optimal*1e6, abs(c_D_optimal), '.k', 'markersize', 12);
plot(0*1e6, abs(c), '+r', 'markersize', 14); hold off
legend('Defocus w.r.t. [Salter et al. 2014]', 'Optimal defocus', '[Salter et al. 2014]')
set(gca, 'fontsize', 14)
xlabel(['z (' 181 'm)'])
ylabel('Field correlation')
title('Optimal defocus')
fig_resize(500, 1.6);
movegui('center')

% Optimal defocus
fprintf('\nTheoretical enhancements using optimal defocus')
theoretical_enhancement(c_D_optimal);

plot_field(E_D_optimal, NA)
title({'Optimally defocus-free', 'spherical aberration: field'})


%% Subfunctions
function theoretical_enhancement(c)
    % Compute theoretical enhancements from field correlation c
    fprintf('\n|Field correlation|:  % 5.1f%%  -> ', abs(c)*100)
    fprintf('Field Enhancement: %.2fx\n', 1/abs(c))
    
    fprintf('|Field correlation|%c: % 5.1f%%  -> ', 178, abs(c).^2 *100)
    fprintf('Intensity Enhancement: %.2fx\n', 1/abs(c).^2)
    
    fprintf('|Field correlation|%c:  % 5.2f%% -> ', 8308, abs(c).^4 *100)
    fprintf('2P Enhancement: %.1fx\n', 1/abs(c).^4)
end


function plot_field(E, NA)
    % Plot field
    figure;
    imagesc([-NA NA], [-NA NA], complex2rgb(E))
    ax = gca;
    ax.FontSize = 14;
    xlabel('k_x/k_0')
    ylabel('k_y/k_0')
    fig_resize(450, 1, ax);
    movegui('center')
    complexcolorwheel('position', [0.01, 0.01, 0.14, 0.14])
    axis image
end
